// ============================================================================
// Copyright (c) Richard Rohac, 2019, All rights reserved.
// ============================================================================

#ifndef LLVM_LIB_TRANSFORMS_CUDA_COARSENING_BENEFITANALYSISPASS_H
#define LLVM_LIB_TRANSFORMS_CUDA_COARSENING_BENEFITANALYSISPASS_H

class BenefitAnalysisPass : public llvm::FunctionPass {
  public:
    // CREATORS
    BenefitAnalysisPass();

    // MANIPULATORS
    void getAnalysisUsage(llvm::AnalysisUsage& AU) const override;
    bool runOnFunction(llvm::Function& F) override;

    // DATA
    static char ID;

  private:
    // PRIVATE ACCESSORS
    unsigned int getCostForInstruction(llvm::Instruction *pI);

    // PRIVATE MANIPULATORS
    void clear();

    // PRIVATE DATA
    unsigned int originalCost;
};

#endif // LLVM_LIB_TRANSFORMS_CUDA_COARSENING_BENEFITANALYSISPASS_H
