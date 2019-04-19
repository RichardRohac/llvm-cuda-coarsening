// ============================================================================
// Copyright (c) Richard Rohac, 2019, All rights reserved.
// ============================================================================
// CUDA Grid Analysis Pass
// ============================================================================

#ifndef LLVM_LIB_TRANSFORMS_CUDA_COARSENING_GRIDANALYSISPASS_H
#define LLVM_LIB_TRANSFORMS_CUDA_COARSENING_GRIDANALYSISPASS_H

using namespace llvm;

namespace llvm {
}

class GridAnalysisPass : public FunctionPass {
public:
    // CREATORS
    GridAnalysisPass();

    // ACCESSORS
    InstVector getGridDependentInstructions(int direction) const;

    // MANIPULATORS
    void getAnalysisUsage(AnalysisUsage& AU) const override;
    bool runOnFunction(Function& F) override;

    // DATA
    static char ID;

private:
    void clear();
    void analyse();

    // DATA
};

#endif
