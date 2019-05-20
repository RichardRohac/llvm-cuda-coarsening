// ============================================================================
// Copyright (c) Richard Rohac, 2019, All rights reserved.
// ============================================================================

#ifndef LLVM_LIB_TRANSFORMS_CUDA_COARSENING_BENEFITANALYSISPASS_H
#define LLVM_LIB_TRANSFORMS_CUDA_COARSENING_BENEFITANALYSISPASS_H

#define COST_ADD_INST 1
#define COST_MUL_INST 1
#define COST_DIV_INST 2
#define COST_MOD_INST 2

// TODO add cost to stores/loads

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
    uint64_t getCostForInstruction(llvm::Instruction *pI);
    uint64_t loopCost(llvm::Loop *loop);
    uint64_t duplicationCost(uint64_t     divergentCost,
                             bool         blockLevel,
                             unsigned int factor);

    // PRIVATE MANIPULATORS
    void clear();

    // PRIVATE DATA
    LoopInfo               *m_loopInfo;
    ScalarEvolution        *m_scalarEvolution;
    GridAnalysisPass       *m_gridAnalysis;
    DivergenceAnalysisPass *m_divergenceAnalysisTL;
    DivergenceAnalysisPass *m_divergenceAnalysisBL;
};

#endif // LLVM_LIB_TRANSFORMS_CUDA_COARSENING_BENEFITANALYSISPASS_H
