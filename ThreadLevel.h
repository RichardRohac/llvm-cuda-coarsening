// ============================================================================
// Copyright (c) Richard Rohac, 2019, All rights reserved.
// ============================================================================
// Thread Level Coarsening Transformation pass
// -> Based on Alberto's Magni OpenCL coarsening pass algorithm
//    available at https://github.com/HariSeldon/coarsening_pass
// ============================================================================

#ifndef LLVM_LIB_TRANSFORMS_CUDA_COARSENING_THREADLEVEL_H
#define LLVM_LIB_TRANSFORMS_CUDA_COARSENING_THREADLEVEL_H

#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/raw_ostream.h"
#include "DivergenceInfo.h"

using namespace llvm;

namespace llvm {
    class LoopInfo;
    class PostDominatorTree;
    class DominatorTree;
}

// Coarsening pass to run on thread (kernel) level
class ThreadLevel : public FunctionPass {
public:
    // CREATORS
    ThreadLevel();

    void getAnalysisUsage(AnalysisUsage &Info) const override;
    bool runOnFunction(Function &F) override;

    // DATA
    static char ID;

private:
    void applyTransformation();

    // DATA
    LoopInfo          *m_loopInfo;
    PostDominatorTree *m_postDomTree;
    DominatorTree     *m_domTree;
    DivergenceInfo    *m_divergenceInfo;
};

#endif