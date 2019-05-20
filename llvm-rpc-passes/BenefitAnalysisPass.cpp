// ============================================================================
// Copyright (c) Richard Rohac, 2019, All rights reserved.
// ============================================================================

#include <llvm/Pass.h>

#include <llvm/IR/Function.h>

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"

#include "Common.h"
#include "Util.h"
#include "GridAnalysisPass.h"
#include "DivergenceAnalysisPass.h"
#include "BenefitAnalysisPass.h"
#include "RegionBounds.h"
#include "DivergentRegion.h"

extern cl::opt<std::string> CLCoarseningDimension;

// DATA
char BenefitAnalysisPass::ID = 0;

// PUBLIC CONSTRUCTORS
BenefitAnalysisPass::BenefitAnalysisPass()
: FunctionPass(ID)
{
    clear();
}

// PUBLIC MANIPULATORS
void BenefitAnalysisPass::getAnalysisUsage(llvm::AnalysisUsage& AU) const
{
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequired<ScalarEvolutionWrapperPass>();
    AU.addRequired<DivergenceAnalysisPassTL>();
    AU.addRequired<DivergenceAnalysisPassBL>();
    AU.addRequired<GridAnalysisPass>();
    AU.setPreservesAll();
}

bool BenefitAnalysisPass::runOnFunction(llvm::Function& F)
{
    if (!Util::isKernelFunction(F) || F.isDeclaration()) {
        return false;
    }

    clear();

    m_loopInfo = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    m_scalarEvolution = &getAnalysis<ScalarEvolutionWrapperPass>().getSE();
    m_divergenceAnalysisTL = &getAnalysis<DivergenceAnalysisPassTL>();
    m_divergenceAnalysisBL = &getAnalysis<DivergenceAnalysisPassBL>();
    m_gridAnalysis = &getAnalysis<GridAnalysisPass>();

    uint64_t divergentCost = 0;
    uint64_t totalCost = 0;

    InstVector& insts = m_divergenceAnalysisTL->getOutermostInstructions();
    RegionVector& regions = m_divergenceAnalysisTL->getOutermostRegions();

    for(InstVector::iterator it = insts.begin(); it != insts.end(); ++it) {
        divergentCost += getCostForInstruction(*it);
    }

    std::for_each(regions.begin(),
                  regions.end(),
                  [&, this](DivergentRegion *region) {
                      for (BasicBlock *pB : region->getBlocks()) {
                          for (Instruction &I: *pB) {
                              divergentCost += getCostForInstruction(&I);
                          }
                      }
                  });

    for (BasicBlock &B : F) {
        for (Instruction &I: B) {
            Instruction *pI = &I;
            uint64_t instCost = getCostForInstruction(pI);
            if (m_divergenceAnalysisTL->isDivergent(pI))
                totalCost += 2 * instCost;
            else
                totalCost += instCost;
        }
    }

    errs() << "\n\n";
    errs() << "CUDA Coarsening Benefit Analysis Pass results: \n";
    errs() << "===================================================== \n";
    errs() << "==== Mode ========= Factor ========= Duplication ==== \n";

    std::vector<unsigned int> factors = {2, 4, 8, 16};
    for (unsigned int factor : factors) {
         errs() << "==== THREAD =======";
         errs() << " " << factor << "x";
         if (factor < 10) {
             errs() << " ";
         }

         errs() << "               ";

         std::stringstream tmp;
         uint64_t dupCost = duplicationCost(divergentCost, false, factor);
         tmp << dupCost << " / " << totalCost << " = " << std::setprecision(4) << 
                         (double)dupCost / (double)totalCost;
         errs() << tmp.str();
         errs() << "\n";
    }

    divergentCost = 0;
    totalCost = 0;

    insts = m_divergenceAnalysisBL->getOutermostInstructions();
    regions = m_divergenceAnalysisBL->getOutermostRegions();

    for(InstVector::iterator it = insts.begin(); it != insts.end(); ++it) {
        divergentCost += getCostForInstruction(*it);
    }

    std::for_each(regions.begin(),
                  regions.end(),
                  [&, this](DivergentRegion *region) {
                      for (BasicBlock *pB : region->getBlocks()) {
                          for (Instruction &I: *pB) {
                              divergentCost += getCostForInstruction(&I);
                          }
                      }
                  });

    for (BasicBlock &B : F) {
        for (Instruction &I: B) {
            Instruction *pI = &I;
            uint64_t instCost = getCostForInstruction(pI);
            if (m_divergenceAnalysisBL->isDivergent(pI))
                totalCost += 2 * instCost;
            else
                totalCost += instCost;
        }
    }

    for (unsigned int factor : factors) {
         errs() << "==== BLOCK  =======";
         errs() << " " << factor << "x";
         if (factor < 10) {
             errs() << " ";
         }

         errs() << "               ";

         std::stringstream tmp;
         uint64_t dupCost = duplicationCost(divergentCost, true, factor);
         tmp << dupCost << " / " << totalCost << " = " << std::setprecision(4) << 
                         (double)dupCost / (double)totalCost;
         errs() << tmp.str();
         errs() << "\n";
    }
    errs() << "===================================================== \n";

    return false;
}

// PRIVATE ACCESSORS
uint64_t BenefitAnalysisPass::getCostForInstruction(Instruction *pI)
{
    uint64_t retVal = 1;

    BasicBlock *parent = pI->getParent();
    Loop *loop = m_loopInfo->getLoopFor(parent);
    if (loop != nullptr) {
        // Instruction considered resides within a loop. To amplify the
        // this fact within the measured metric, we try to compute how many
        // times the instruction executes. This relies on two factors:
        // a) loop depth
        // b) (total) trip count
        // The latter can only be computed for some of the loops (where the
        // trip count is known at the compile time).

        uint64_t totalCost = 1;
        uint64_t depth = m_loopInfo->getLoopDepth(parent);
        for (uint64_t i = 0; i < m_loopInfo->getLoopDepth(parent); ++i) {
            uint64_t loopCost = this->loopCost(loop);
            if (!loopCost) {
                return depth;
            }

            totalCost *= loopCost;
            loop = loop->getParentLoop();
        }

        //errs() << "Got trip count: " << totalCost << " ";
        //        pI->dump();

        retVal = totalCost;
    }

    return retVal;
}

uint64_t BenefitAnalysisPass::loopCost(Loop *loop)
{
    ScalarEvolution *SE = m_scalarEvolution;

    if (SE->hasLoopInvariantBackedgeTakenCount(loop)) {
        const SCEV *takenCount = SE->getBackedgeTakenCount(loop);
        if (!isa<SCEVCouldNotCompute>(takenCount)) {
            // We were able to find statically known trip count for this
            // loop.

            if (!isa<SCEVConstant>(takenCount)) {
                return 0; 
            }

            auto val = cast<SCEVConstant>(takenCount)->getAPInt();
            unsigned int tripCount = val.getLimitedValue(UINT64_MAX - 1);

            return tripCount;
        }
    }

    // Static analysis is not possible.
    return 0;
}

uint64_t BenefitAnalysisPass::duplicationCost(uint64_t     divergentCost,
                                              bool         blockLevel,
                                              unsigned int factor)
{
    uint64_t result = 0;
    unsigned int dimension = Util::numeralDimension(CLCoarseningDimension);

    InstVector sizeInsts = 
                blockLevel
                ? m_gridAnalysis->getGridSizeDependentInstructions(dimension)
                : m_gridAnalysis->getBlockSizeDependentInstructions(dimension);

    result += sizeInsts.size() * COST_MUL_INST;

    InstVector tids = 
                blockLevel
                ? m_gridAnalysis->getBlockIDDependentInstructions(dimension)
                : m_gridAnalysis->getThreadIDDependentInstructions(dimension);
    
    // origTid = [newTid / st] * (cf * st) + newTid % st + subid * st

    result += tids.size() * COST_DIV_INST; // newTid / st [div]
    result += tids.size() * COST_MUL_INST; // * (cf * st) [mul]
    result += tids.size() * COST_MOD_INST; // newTid % st [mod]
    result += tids.size() * COST_ADD_INST; // [mul] + [mod]

    // subIds
    for (unsigned int index = 2; index <= factor; ++index) {
        result += tids.size() * COST_ADD_INST;
    }

    // duplication
    result += divergentCost * (factor - 1);

    return result;
}

// PRIVATE MANIPULATORS
void BenefitAnalysisPass::clear()
{
    //originalCost = 0;
}

static RegisterPass<BenefitAnalysisPass> X("cuda-benefit-analysis-pass",
                                           "CUDA Benefit Analysis Pass",
                                           false, // Only looks at CFG
                                           true // Analysis pass
                                           );