// ============================================================================
// Copyright (c) Richard Rohac, 2019, All rights reserved.
// ============================================================================

#include <llvm/Pass.h>

#include <llvm/IR/Function.h>

#include "Common.h"
#include "Util.h"
#include "DivergenceAnalysisPass.h"
#include "BenefitAnalysisPass.h"

// DATA
char BenefitAnalysisPass::ID = 0;

// PUBLIC CONSTRUCTORS
BenefitAnalysisPass::BenefitAnalysisPass()
: FunctionPass(ID)
{
    clear();
}

// PUBLIC MANIPULATORS
void BenefitAnalysisPass::getAnalysisUsage(llvm::AnalysisUsage& AU) const
{
    AU.addRequired<DivergenceAnalysisPass>();
    AU.setPreservesAll();
}

bool BenefitAnalysisPass::runOnFunction(llvm::Function& F)
{
    if (!Util::isKernelFunction(F) || F.isDeclaration()) {
        return true;
    }

    clear();

    // Apply cost metric over the original kernel function.
    for (BasicBlock &B : F) {
        for (Instruction& I : B) {
            originalCost += getCostForInstruction(&I);
        }
    }

    errs() << "\n";
    errs() << "Benefit analysis pass results: " << "\n";
    errs() << " -- -- Measured cost of the original kernel "
           << F.getName() << " : " << originalCost << "\n";
    errs() << "\n";

    return true;
}

// PRIVATE ACCESSORS
unsigned int BenefitAnalysisPass::getCostForInstruction(Instruction *pI)
{
    return 1; // TODO
}

// PRIVATE MANIPULATORS
void BenefitAnalysisPass::clear()
{
    originalCost = 0;
}

static RegisterPass<BenefitAnalysisPass> X("cuda-benefit-analysis-pass",
                                           "CUDA Benefit Analysis Pass",
                                           false, // Only looks at CFG
                                           true // Analysis pass
                                           );