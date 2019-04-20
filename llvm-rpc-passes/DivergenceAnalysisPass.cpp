// ============================================================================
// Copyright (c) Richard Rohac, 2019, All rights reserved.
// ============================================================================
// Divergence analysis pass
// -> Based on Alberto's Magni OpenCL coarsening pass algorithm
//    available at https://github.com/HariSeldon/coarsening_pass
// ============================================================================

#include "llvm/Pass.h"
#include "llvm/IR/Function.h"

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PostDominators.h"

#include "Common.h"
#include "GridAnalysisPass.h"
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

    return false;
}

// PRIVATE MANIPULATORS
void DivergenceAnalysisPass::clear()
{
    m_divergent.clear();
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
        //if (isa<BranchInst>(inst)) {
        //    BasicBlock *block = findImmediatePostDom(inst->getParent(), m_postDomTree);
        //    for (auto inst = block->begin(); isa<PHINode>(inst); ++inst) {
        //        users.insert(inst);
        //    }
        //}

        //findUsesOf(inst, users);
        // Add users of the current instruction to the work list.
        //for (InstSet::iterator iter = users.begin(), iterEnd = users.end();
        //    iter != iterEnd; ++iter) {
        //    if (!isPresent(*iter, m_divergent)) {
        //        worklist.insert(*iter);
        //    }
        //}
    }
}

static RegisterPass<DivergenceAnalysisPass> X("cuda-divergence-analysis-pass",
                                              "CUDA Divergence Analysis Pass",
                                              false, // Only looks at CFG
                                              true // Analysis pass
                                              );