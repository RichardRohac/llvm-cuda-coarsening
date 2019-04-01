// ============================================================================
// Copyright (c) Richard Rohac, 2019, All rights reserved.
// ============================================================================
// Thread Level Coarsening Transformation pass
// -> Based on Alberto's Magni OpenCL coarsening pass algorithm
//    available at https://github.com/HariSeldon/coarsening_pass
// ============================================================================

#include "ThreadLevel.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PostDominators.h"
#include "DivergenceInfo.h"

bool isKernelFunction(const Function &F);

ThreadLevel::ThreadLevel()
: FunctionPass(ID) {
}

void ThreadLevel::getAnalysisUsage(AnalysisUsage &Info) const {
    Info.addRequired<LoopInfoWrapperPass>();
    Info.addRequired<PostDominatorTreeWrapperPass>();
    Info.addRequired<DominatorTreeWrapperPass>();
    Info.addRequired<DivergenceInfo>();
    //Info.addPreserved<DivergenceInfo>();
}

bool ThreadLevel::runOnFunction(Function &F) {
    const llvm::NamedMDNode *kernels = 
                F.getParent()->getNamedMetadata("nvvm.annotations");

    if (!kernels) {
        return false;
    }

    if (!isKernelFunction(F)) {
        return false;
    }

    errs() << "Thread level analysis invoked at kernel: ";
    errs().write_escaped(F.getName()) << '\n';

    // Perform initial analysis.
    m_loopInfo = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    m_postDomTree = &getAnalysis<PostDominatorTreeWrapperPass>().getPostDomTree();
    m_domTree = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    m_divergenceInfo = &getAnalysis<DivergenceInfo>();

    //applyTransformation();

    return true;
}

void ThreadLevel::applyTransformation() {
    
}

char ThreadLevel::ID = 0;
static RegisterPass<ThreadLevel> X("cuda-thread-level-coarsening",
                                   "CUDA Thread Level Coarsening Pass",
                                   false /* Only looks at CFG */,
                                   true /* Analysis Pass */);

//static void registerThreadLevelPass(const PassManagerBuilder &,
//                                    legacy::PassManagerBase  &PM) {
//    PM.add(new ThreadLevel());
//}
//static RegisterStandardPasses RegisterMyPass(
//                   PassManagerBuilder::EP_EarlyAsPossible,
//                   registerThreadLevelPass);