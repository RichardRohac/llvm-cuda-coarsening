// ============================================================================
// Copyright (c) Richard Rohac, 2019, All rights reserved.
// ============================================================================

#ifndef LLVM_LIB_TRANSFORMS_CUDA_COARSENING_BENEFITANALYSISPASS_H
#define LLVM_LIB_TRANSFORMS_CUDA_COARSENING_BENEFITANALYSISPASS_H

#define COST_DEFAULT      100   /* Default instruction cost                   */
#define COST_DIV_POW2     200   /* Division cost, divisor is power of two     */
#define COST_DIV_NPOW2    300   /* Division cost, divisor is not power of two */
#define COST_MOD_POW2     150   /* Modulo cost, divisor is power of two       */
#define COST_MOD_NPOW2    350   /* Modulo cost, divisor is not power of two   */
#define COST_LOAD_SHARED  150   /* Cost of shared memory load                 */
#define COST_STORE_SHARED 150   /* Cost of shared memory store                */
#define COST_LOAD_GLOBAL  200   /* Cost of global memory load                 */
#define COST_STORE_GLOBAL 200   /* Cost of global memory store                */
#define COST_BRANCH_DIV   150   /* Cost of divergent branch                   */
#define COST_MATH_FUNC_F  200   /* Cost of FP32 built-in math function        */
#define COST_MATH_FUNC_D  300   /* Cost of FP64 built-in math function        */

/* struct coarseningBenefit {
  uint64_t benefit;
  uint64_t cost;
};*/

//typedef std::unordered_map<unsigned int, coarseningBenefit> benefitMap_t; 

class BenefitAnalysisPass : public llvm::FunctionPass {
  public:
    // CREATORS
    BenefitAnalysisPass();

    // ACCESSORS
    void printStatistics() const;

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
                             unsigned int factor) const;

    // PRIVATE MANIPULATORS
    void clear();

    // PRIVATE DATA
    LoopInfo               *m_loopInfo;
    ScalarEvolution        *m_scalarEvolution;
    GridAnalysisPass       *m_gridAnalysis;
    DivergenceAnalysisPass *m_divergenceAnalysisTL;
    DivergenceAnalysisPass *m_divergenceAnalysisBL;

    uint64_t                m_totalTL;
    uint64_t                m_costTL;
    uint64_t                m_totalBL;
    uint64_t                m_costBL;

    //benefitMap_t            m_benefitMapTL;
    //benefitMap_t            m_benefitMapBL;
};

#endif // LLVM_LIB_TRANSFORMS_CUDA_COARSENING_BENEFITANALYSISPASS_H
