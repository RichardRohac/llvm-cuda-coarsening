// ============================================================================
// Branch extraction pass
// -> Based on Alberto's Magni OpenCL coarsening pass algorithm
//    available at https://github.com/HariSeldon/coarsening_pass
// ============================================================================

#ifndef LLVM_LIB_TRANSFORMS_CUDA_COARSENING_BRANCHEXTRACTIONPASS_H
#define LLVM_LIB_TRANSFORMS_CUDA_COARSENING_BRANCHEXTRACTIONPASS_H

class BranchExtractionPass : public FunctionPass {
  public:
    // CREATORS
    BranchExtractionPass();

    // MANIPULATORS
    void getAnalysisUsage(AnalysisUsage& AU) const override;
    bool runOnFunction(Function& F) override;
    
    // DATA
    static char ID;

  private:
    void extractBranches(DivergentRegion *region);
    void isolateRegion(DivergentRegion *region);

  private:
    LoopInfo *loopInfo;
    DominatorTree *dt;
    PostDominatorTree *pdt;
    DivergenceAnalysisPassBL *divBL;
    DivergenceAnalysisPassTL *divTL;
};

#endif