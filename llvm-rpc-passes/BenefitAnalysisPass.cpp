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
extern cl::opt<std::string> CLKernelName;
extern cl::opt<std::string> CLCoarseningMode;

std::unordered_map<unsigned int, unsigned int> g_opcodeCostMap = {
    {Instruction::UDiv,  COST_DIV_NPOW2},
    {Instruction::SDiv,  COST_DIV_NPOW2},
    {Instruction::FDiv,  COST_DIV_NPOW2},
    {Instruction::URem,  COST_MOD_NPOW2},
    {Instruction::SRem,  COST_MOD_NPOW2},
    {Instruction::FRem,  COST_MOD_NPOW2},
    {Instruction::Br,    COST_BRANCH_DIV},
    {Instruction::Store, COST_STORE_GLOBAL},
    {Instruction::Load,  COST_LOAD_GLOBAL}
};

std::unordered_map<unsigned int, unsigned int>::iterator opcCostMapIt_t;

// DATA
char BenefitAnalysisPass::ID = 0;

// PUBLIC CONSTRUCTORS
BenefitAnalysisPass::BenefitAnalysisPass()
: FunctionPass(ID)
{
    clear();
}

// PUBLIC ACCESSORS
void BenefitAnalysisPass::printStatistics() const
{
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

        // coarseningBenefit& cb = m_benefitMapTL[factor]->second;

         std::stringstream tmp;
         uint64_t dupCost = duplicationCost(m_totalTL - m_costTL, false, factor);
         tmp << dupCost << " / " << m_totalTL << " = " << std::setprecision(4) << 
                         (double)dupCost / (double)m_totalTL;
         errs() << tmp.str();
         errs() << "\n";
    }

    for (unsigned int factor : factors) {
         errs() << "==== BLOCK  =======";
         errs() << " " << factor << "x";
         if (factor < 10) {
             errs() << " ";
         }

         errs() << "               ";

         std::stringstream tmp;
         uint64_t dupCost = duplicationCost(m_totalBL - m_costBL, true, factor);
         tmp << dupCost << " / " << m_totalBL << " = " << std::setprecision(4) << 
                         (double)dupCost / (double)m_totalBL;
         errs() << tmp.str();
         errs() << "\n";
    }
    errs() << "===================================================== \n";
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
    if (F.getParent()->getTargetTriple() != CUDA_TARGET_TRIPLE) {
        // Run analysis only on device code.
        return false;
    }

    // Apply the pass to kernels only.
    if (!Util::shouldCoarsen(F,
                             CLKernelName,
                             false,
                             CLCoarseningMode == "dynamic")) {
        return false;
    }

    clear();

    m_loopInfo = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    m_scalarEvolution = &getAnalysis<ScalarEvolutionWrapperPass>().getSE();
    m_divergenceAnalysisTL = &getAnalysis<DivergenceAnalysisPassTL>();
    m_divergenceAnalysisBL = &getAnalysis<DivergenceAnalysisPassBL>();
    m_gridAnalysis = &getAnalysis<GridAnalysisPass>();

    m_totalTL = 0;
    m_costTL = 0;

    InstVector& insts = m_divergenceAnalysisTL->getOutermostInstructions();
    RegionVector& regions = m_divergenceAnalysisTL->getOutermostRegions();

    for(InstVector::iterator it = insts.begin(); it != insts.end(); ++it) {

        m_costTL += getCostForInstruction(*it);
    }

    std::for_each(regions.begin(),
                  regions.end(),
                  [&, this](DivergentRegion *region) {
                      for (BasicBlock *pB : region->getBlocks()) {
                          for (Instruction &I: *pB) {
                              m_costTL += getCostForInstruction(&I);
                          }
                      }
                  });

    for (BasicBlock &B : F) {
        for (Instruction &I: B) {
            Instruction *pI = &I;
            m_totalTL += getCostForInstruction(pI);
        }
    }

    m_totalBL = 0;
    m_costBL = 0;

    insts = m_divergenceAnalysisBL->getOutermostInstructions();
    regions = m_divergenceAnalysisBL->getOutermostRegions();

    for(InstVector::iterator it = insts.begin(); it != insts.end(); ++it) {
        m_costBL += getCostForInstruction(*it);
    }

    std::for_each(regions.begin(),
                  regions.end(),
                  [&, this](DivergentRegion *region) {
                      for (BasicBlock *pB : region->getBlocks()) {
                          for (Instruction &I: *pB) {
                              m_costBL += getCostForInstruction(&I);
                          }
                      }
                  });

    for (BasicBlock &B : F) {
        for (Instruction &I: B) {
            Instruction *pI = &I;
            m_totalBL += getCostForInstruction(pI);
        }
    }

    return false;
}

// PRIVATE ACCESSORS
bool isPow2(int i) {
    if ( i <= 0 ) {
        return 0;
    }

    return !(i & (i - 1));
}

uint64_t BenefitAnalysisPass::getCostForInstruction(Instruction *pI)
{
    uint64_t instCost = COST_DEFAULT;

    auto costMapIt = g_opcodeCostMap.find(pI->getOpcode());
    if (costMapIt != g_opcodeCostMap.end()) {
        instCost = costMapIt->second;
    }

    // Handle special cases
    if (pI->getOpcode() == Instruction::UDiv ||
        pI->getOpcode() == Instruction::SDiv ||
        pI->getOpcode() == Instruction::FDiv) {
            ConstantInt *c = dyn_cast<ConstantInt>(pI->getOperand(1));
            if (c) {
                if (isPow2(c->getLimitedValue())) {
                    instCost = COST_DIV_POW2;
                }
            }
    }

    if (pI->getOpcode() == Instruction::URem ||
        pI->getOpcode() == Instruction::SRem ||
        pI->getOpcode() == Instruction::FRem) {
            ConstantInt *c = dyn_cast<ConstantInt>(pI->getOperand(1));
            if (c) {
                if (isPow2(c->getLimitedValue())) {
                    instCost = COST_MOD_POW2;
                }
            }
    }

    //if (isa<StoreInst>(pI)) {
    //    StoreInst *store = dyn_cast<StoreInst>(pI);
    //    if (store->getSt)
    //}

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

        uint64_t depth = m_loopInfo->getLoopDepth(parent);
        for (uint64_t i = 0; i < m_loopInfo->getLoopDepth(parent); ++i) {
            uint64_t loopCost = this->loopCost(loop);
            if (!loopCost) {
                return depth * instCost;
            }

            instCost *= loopCost;
            loop = loop->getParentLoop();
        }

        //errs() << "Got trip count: " << totalCost << " ";
        //        pI->dump();

        //retVal = totalCost;
    }

    return instCost;
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
                                              unsigned int factor) const
{
    uint64_t result = 0;
    unsigned int dimension = Util::numeralDimension(CLCoarseningDimension);

    InstVector sizeInsts = 
                blockLevel
                ? m_gridAnalysis->getGridSizeDependentInstructions(dimension)
                : m_gridAnalysis->getBlockSizeDependentInstructions(dimension);

    result += sizeInsts.size() * COST_DEFAULT;

    InstVector tids = 
                blockLevel
                ? m_gridAnalysis->getBlockIDDependentInstructions(dimension)
                : m_gridAnalysis->getThreadIDDependentInstructions(dimension);
    
    // origTid = [newTid / st] * (cf * st) + newTid % st + subid * st

    result += tids.size() * COST_DIV_POW2; // newTid / st [div]
    result += tids.size() * COST_DEFAULT; // * (cf * st) [mul]
    result += tids.size() * COST_MOD_POW2; // newTid % st [mod]
    result += tids.size() * COST_DEFAULT; // [mul] + [mod]

    // subIds
    for (unsigned int index = 2; index <= factor; ++index) {
        result += tids.size() * COST_DEFAULT;
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