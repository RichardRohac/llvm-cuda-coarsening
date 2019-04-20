// ============================================================================
// Copyright (c) Richard Rohac, 2019, All rights reserved.
// ============================================================================
// CUDA Grid Analysis Pass
// ============================================================================

#ifndef LLVM_LIB_TRANSFORMS_CUDA_COARSENING_GRIDANALYSISPASS_H
#define LLVM_LIB_TRANSFORMS_CUDA_COARSENING_GRIDANALYSISPASS_H

using namespace llvm;

#define CUDA_THREAD_ID_VAR  "threadIdx"
#define CUDA_BLOCK_ID_VAR   "blockIdx"
#define CUDA_BLOCK_DIM_VAR  "blockDim"
#define CUDA_GRID_DIM_VAR   "gridDim"

#define CUDA_MAX_DIM        3

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
    // PRIVATE TYPES
    typedef std::unordered_map<std::string, InstVector> varInstructions_t;

    // PRIVATE MANIPULATORS
    void init();

    // DATA
    std::vector<varInstructions_t> gridInstructions; 
};

#endif
