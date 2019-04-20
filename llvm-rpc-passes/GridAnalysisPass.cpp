// ============================================================================
// Copyright (c) Richard Rohac, 2019, All rights reserved.
// ============================================================================
// CUDA Grid Analysis Pass
// ============================================================================

#include "llvm/Pass.h"
#include "llvm/IR/Function.h"

#include "Common.h"
#include "GridAnalysisPass.h"

using namespace llvm;

// DATA
char GridAnalysisPass::ID = 0;

// CREATORS
GridAnalysisPass::GridAnalysisPass()
: FunctionPass(ID) 
{
}

// PUBLIC ACCESSORS
InstVector GridAnalysisPass::getGridDependentInstructions(int direction) const
{
    InstVector result;



    return result;
}

// PUBLIC MANIPULATORS
void GridAnalysisPass::getAnalysisUsage(AnalysisUsage& AU) const
{
    AU.setPreservesAll();
}

bool GridAnalysisPass::runOnFunction(Function& F)
{
    errs() << "--  INFO  -- Grid analysis invoked on: ";
    errs().write_escaped(F.getName()) << '\n';

    init();

    return false;
}

// PRIVATE MANIPULATORS
void GridAnalysisPass::init()
{
    // Clear data, as pass can run multiple times
    gridInstructions.clear();
    gridInstructions.reserve(CUDA_MAX_DIM);

    for (unsigned int i = 0; i < CUDA_MAX_DIM; ++i) {
        gridInstructions.push_back(varInstructions_t());
        gridInstructions[i][CUDA_THREAD_ID_VAR] = InstVector();
        gridInstructions[i][CUDA_BLOCK_ID_VAR] = InstVector();
        gridInstructions[i][CUDA_BLOCK_DIM_VAR] = InstVector();
        gridInstructions[i][CUDA_GRID_DIM_VAR] = InstVector();
    }
}

static RegisterPass<GridAnalysisPass> X("cuda-grid-analysis-pass",
                                        "CUDA Grid Analysis Pass",
                                        false, // Only looks at CFG
                                        true // Analysis pass
                                        );