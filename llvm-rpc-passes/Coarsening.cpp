// ============================================================================
// Copyright (c) Richard Rohac, 2019, All rights reserved.
// ============================================================================
// CUDA Coarsening Transformation pass
// -> Based on Alberto's Magni OpenCL coarsening pass algorithm
//    available at https://github.com/HariSeldon/coarsening_pass
// ============================================================================

#include <llvm/Pass.h>
#include <llvm/IR/Instructions.h>

#include "Common.h"
#include "CUDACoarsening.h"
#include "Util.h"
#include "RegionBounds.h"
#include "DivergentRegion.h"
#include "DivergenceAnalysisPass.h"
#include "GridAnalysisPass.h"

Instruction *getAddInstNSW(Value *firstValue, Value *secondValue) {
    Instruction *add =
        BinaryOperator::Create(Instruction::Add, firstValue, secondValue);
    add->setName(firstValue->getName() + "..AddNSW");
    add->setHasNoSignedWrap();
    return add;
}


void CUDACoarseningPass::coarsenKernel()
{
    RegionVector& regions = m_blockLevel ?
                            m_divergenceAnalysisBL->getOutermostRegions() :
                            m_divergenceAnalysisTL->getOutermostRegions();
    InstVector& insts = m_blockLevel ?
                        m_divergenceAnalysisBL->getOutermostInstructions() :
                        m_divergenceAnalysisTL->getOutermostInstructions();

    // Replicate instructions.
    for(InstVector::iterator it = insts.begin(); it != insts.end(); ++it) {
        replicateInstruction(*it);
    }

    // Replicate regions.
    std::for_each(regions.begin(),
                  regions.end(),
                  [this](DivergentRegion *region) {
                      replicateRegion(region); 
                  });
}

void CUDACoarseningPass::replacePlaceholders()
{
  // Replace placeholders.
  for (auto &mapIter : m_phMap) {
      InstVector &phs = mapIter.second;
      // Iteate over placeholder vector.
      for (auto ph : phs) {
          Value *replacement = m_phReplacementMap[ph];
          if (replacement != nullptr && ph != replacement) {
              ph->replaceAllUsesWith(replacement);
          }
      }
  }
}

void CUDACoarseningPass::replicateInstruction(Instruction *inst)
{
    InstVector current;
    current.reserve(m_factor - 1);
    Instruction *bookmark = inst;

    for (unsigned int index = 0; index < m_factor - 1; ++index) {
        // Clone.
        Instruction *newInst = inst->clone();
        Util::renameValueWithFactor(newInst, inst->getName(), index);
        applyCoarseningMap(newInst, index);
        // Insert the new instruction.
        newInst->insertAfter(bookmark);

        bookmark = newInst;
        // Add the new instruction to the coarsening map.
        current.push_back(newInst);
    }
    m_coarseningMap.insert(std::pair<Instruction *, InstVector>(inst, current));

    updatePlaceholderMap(inst, current);
}

void CUDACoarseningPass::applyCoarseningMap(DivergentRegion &region,
                                          unsigned int index)
{
  std::for_each(region.begin(), region.end(), [index, this](BasicBlock *block) {
    applyCoarseningMap(block, index);
  });
}

void CUDACoarseningPass::applyCoarseningMap(BasicBlock *block,
                                          unsigned int index)
{
  for (auto iter = block->begin(), iterEnd = block->end();
       iter != iterEnd; ++iter) {
    applyCoarseningMap(&(*iter), index);
  }
}

void CUDACoarseningPass::applyCoarseningMap(Instruction  *inst,
                                            unsigned int  index)
{
    if (m_coarseningMap.find(inst) != m_coarseningMap.end()) {
        return;
    }

    for (unsigned int i = 0; i < inst->getNumOperands(); ++i) {
        if (!isa<Instruction>(inst->getOperand(i))) {
            continue;
        }

        Instruction *pOP = cast<Instruction>(inst->getOperand(i));
        Instruction *newOp = getCoarsenedInstruction(inst, pOP, index);
        if (newOp == nullptr) {
            continue;
        }
        inst->setOperand(i, newOp);
    }

 // for (unsigned int opIndex = 0, opEnd = inst->getNumOperands();
  //     opIndex != opEnd; ++opIndex) {
 //   Instruction *operand = dyn_cast<Instruction>(inst->getOperand(opIndex));
  //  if (operand == nullptr) {
  //    continue;
   // }
   // if (m_coarseningMap[operand].size() > 0) {
  //    errs() << "Skipping as is def..\n";
   //   continue;
   // }
    //errs() << "Operand $i: " << operand->getName() << "\n";

 //   Instruction *newOp = getCoarsenedInstruction(operand, index);
 //   if (newOp == nullptr) {
  //    continue;
 //   }

  //  inst->setOperand(opIndex, newOp);
 // }
}

void CUDACoarseningPass::updatePlaceholderMap(Instruction *inst,
                                              InstVector&  coarsenedInsts)
{
    // Update placeholder replacement map.
    auto phIter = m_phMap.find(inst);
    if (phIter != m_phMap.end()) {
        InstVector &coarsenedPhs = phIter->second;
        for (unsigned int index = 0; index < coarsenedPhs.size(); ++index) {
            m_phReplacementMap[coarsenedPhs[index]] = coarsenedInsts[index];
        }
    }
}

//------------------------------------------------------------------------------
Instruction *
CUDACoarseningPass::getCoarsenedInstruction(Instruction *ret, Instruction *inst,
                                            unsigned int coarseningIndex) {
  CoarseningMap::iterator It = m_coarseningMap.find(inst);
  // The instruction is in the map.
  if (It != m_coarseningMap.end()) {
    InstVector &entry = It->second;
    Instruction *result = entry[coarseningIndex];

    bool skip = false;
    for (Instruction * is : entry) {
      if (is == ret) {
        skip = true;
      }
    }

    if (ret == result || skip) {
        return nullptr;
    }
    return result;
  } else {
    // The instruction is divergent.
    if (m_blockLevel ? m_divergenceAnalysisBL->isDivergent(inst) :
                       m_divergenceAnalysisTL->isDivergent(inst)) {
      // Look in placeholder map.
      CoarseningMap::iterator phIt = m_phMap.find(inst);
      
      Instruction *result = nullptr;
      if (phIt != m_phMap.end()) {
        // The instruction is in the placeholder map.
        InstVector &entry = phIt->second;
        result = entry[coarseningIndex];
      }
      // The instruction is not in the placeholder map.
      else {
        // Make an entry in the placeholder map.
        InstVector newEntry;
        for (unsigned int counter = 0; counter < m_factor - 1; ++counter) {
          Instruction *ph = inst->clone();
          ph->insertAfter(inst);
          Util::renameValueWithFactor(
              ph, (inst->getName() + Twine(".place.holder")).str(),
              coarseningIndex);
          newEntry.push_back(ph);
        }
        m_phMap.insert(std::pair<Instruction *, InstVector>(inst, newEntry));
        // Return the appropriate placeholder.
        result = newEntry[coarseningIndex];
      }
      return result;
    }
  }
  return nullptr;
}