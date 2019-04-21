// ============================================================================
// Copyright (c) Richard Rohac, 2019, All rights reserved.
// ============================================================================
// Divergence analysis pass
// -> Based on Alberto's Magni OpenCL coarsening pass algorithm
//    available at https://github.com/HariSeldon/coarsening_pass
// ============================================================================

#include "llvm/Pass.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PostDominators.h"

#include "Common.h"
#include "Util.h"
#include "GridAnalysisPass.h"
#include "RegionBounds.h"
#include "DivergentRegion.h"
#include "DivergenceAnalysisPass.h"

extern cl::opt<int> CLCoarseningDirection;

using namespace llvm;

// DATA
char DivergenceAnalysisPass::ID = 0;

// CREATORS
DivergenceAnalysisPass::DivergenceAnalysisPass()
: FunctionPass(ID) 
{
}

// PUBLIC MANIPULATORS
void DivergenceAnalysisPass::getAnalysisUsage(AnalysisUsage& AU) const
{
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addPreserved<LoopInfoWrapperPass>(); //! not in multi-dim?
    AU.addRequired<PostDominatorTreeWrapperPass>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<GridAnalysisPass>();
    AU.setPreservesAll();
}

bool DivergenceAnalysisPass::runOnFunction(Function& F)
{
    errs() << "--  INFO  -- Divergence analysis invoked on: ";
    errs().write_escaped(F.getName()) << '\n';

    clear();

    m_loopInfo = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    m_postDomT = &getAnalysis<PostDominatorTreeWrapperPass>().getPostDomTree();
    m_domT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    m_grid = &getAnalysis<GridAnalysisPass>();

    analyse();
    findDivergentBranches();
    findRegions();

    return false;
}

// PRIVATE MANIPULATORS
void DivergenceAnalysisPass::clear()
{
    m_divergent.clear();
    m_divergentBranches.clear();
    m_regions.clear();
}

void DivergenceAnalysisPass::analyse()
{
    InstVector seeds =
        m_grid->getGridIDDependentInstructions(CLCoarseningDirection);

    InstSet worklist(seeds.begin(), seeds.end());

    while (!worklist.empty()) {
        auto iter = worklist.begin();
        Instruction *inst = *iter;
        worklist.erase(iter);
        m_divergent.push_back(inst);

        InstSet users;

        // Manage branches.
        if (isa<BranchInst>(inst)) {
            BasicBlock *block = Util::findImmediatePostDom(inst->getParent(),
                                                           m_postDomT);
            for (auto it = block->begin(); isa<PHINode>(it); ++it) {
                users.insert(&*it);
            }
        }

        Util::findUsesOf(inst, users);
        // Add users of the current instruction to the work list.
        for (InstSet::iterator iter = users.begin();
             iter != users.end();
             ++iter) {
             if (!isPresent(*iter, m_divergent)) {
                worklist.insert(*iter);
            }
        }
    }
}

void DivergenceAnalysisPass::findDivergentBranches()
{
    std::copy_if(m_divergent.begin(),
                 m_divergent.end(),
                 std::back_inserter(m_divergentBranches),
                 [](Instruction *pI) {
                        return isa<BranchInst>(pI);
                 });
}

void DivergenceAnalysisPass::findRegions()
{
    for (Instruction *divBranch : m_divergentBranches) {
        BasicBlock *header = divBranch->getParent();
        BasicBlock *exiting = Util::findImmediatePostDom(header, m_postDomT);

        if (m_loopInfo->isLoopHeader(header)) {
            Loop *loop = m_loopInfo->getLoopFor(header);
            if (loop == m_loopInfo->getLoopFor(exiting))
                exiting = loop->getExitBlock();
        }

        m_regions.push_back(new DivergentRegion(header, exiting));
    }

    // Remove redundant regions. The ones coming from loops.
    m_regions = cleanUpRegions(m_regions, m_domT);
}

RegionVector DivergenceAnalysisPass::cleanUpRegions(RegionVector&        regions,
                                                    const DominatorTree *dt)
{
    RegionVector result;

    for (size_t index1 = 0; index1 < regions.size(); ++index1) {
        DivergentRegion *region1 = regions[index1];
        BlockVector &blocks1 = region1->getBlocks();
  
        bool toAdd = true;

        for (size_t index2 = 0; index2 < regions.size(); ++index2) {
            if(index2 == index1) 
                break;

            DivergentRegion *region2 = regions[index2];
            BlockVector &blocks2 = region2->getBlocks();

            if (is_permutation(blocks1.begin(),
                               blocks1.end(),
                               blocks2.begin()) &&
                dt->dominates(region2->getHeader(), region1->getHeader())) {
                toAdd = false;
                break; 
            }
        }

        if(toAdd) 
        result.push_back(region1); 
    }

    return result;
}

static RegisterPass<DivergenceAnalysisPass> X("cuda-divergence-analysis-pass",
                                              "CUDA Divergence Analysis Pass",
                                              false, // Only looks at CFG
                                              true // Analysis pass
                                              );