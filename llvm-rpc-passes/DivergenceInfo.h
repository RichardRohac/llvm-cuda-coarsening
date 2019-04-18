/* // ============================================================================
// Copyright (c) Richard Rohac, 2019, All rights reserved.
// ============================================================================
// Divergence information
// -> Based on Alberto's Magni OpenCL coarsening pass algorithm
//    available at https://github.com/HariSeldon/coarsening_pass
// ============================================================================

#ifndef LLVM_LIB_TRANSFORMS_CUDA_COARSENING_DIVERGENCEINFO_H
#define LLVM_LIB_TRANSFORMS_CUDA_COARSENING_DIVERGENCEINFO_H

#include "llvm/Pass.h"
#include "llvm/IR/Function.h"

#include "CUDACoarsening.h"

using namespace llvm;

namespace llvm {
    class PostDominatorTree;
    class DominatorTree;
}

class DivergenceInfo : public FunctionPass {
public:
    // CREATORS
    DivergenceInfo();

    // MANIPULATORS
    void getAnalysisUsage(AnalysisUsage &Info) const override;
    bool runOnFunction(Function &F) override;

    // DATA
    static char ID;

private:
    void clear();
    void analyse();

    // DATA
    InstVector         m_divergent;

    PostDominatorTree *m_postDomTree;
    DominatorTree     *m_domTree;
};

#endif

 */