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
void GridAnalysisPass::getAnalysisUsage(AnalysisUsage& AU) const {
    AU.setPreservesAll();
}

bool GridAnalysisPass::runOnFunction(Function& F) {
    errs() << "--  INFO  -- Grid analysis invoked on: ";
    errs().write_escaped(F.getName()) << '\n';

    return false;
}

// PRIVATE MANIPULATORS
static RegisterPass<GridAnalysisPass> X("cuda-grid-analysis-pass",
                                        "CUDA Grid Analysis Pass",
                                        false, // Only looks at CFG
                                        true // Analysis pass
                                        );