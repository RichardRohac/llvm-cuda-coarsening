// ============================================================================
// Copyright (c) Richard Rohac, 2019, All rights reserved.
// ============================================================================
// Divergence analysis pass
// -> Based on Alberto's Magni OpenCL coarsening pass algorithm
//    available at https://github.com/HariSeldon/coarsening_pass
// ============================================================================

#ifndef LLVM_LIB_TRANSFORMS_CUDA_COARSENING_DIVERGENCEANALYSISPASS_H
#define LLVM_LIB_TRANSFORMS_CUDA_COARSENING_DIVERGENCEANALYSISPASS_H

using namespace llvm;

namespace llvm {
    class LoopInfo;
    class PostDominatorTree;
    class DominatorTree;
}

class GridAnalysisPass;
class DivergentRegion;

class DivergenceAnalysisPass : public FunctionPass {
public:
    // CREATORS
    DivergenceAnalysisPass();

    // MANIPULATORS
    void getAnalysisUsage(AnalysisUsage& AU) const override;
    bool runOnFunction(Function& F) override;

    // DATA
    static char ID;

private:
    // PRIVATE MANIPULATORS
    void clear();
    void analyse();
    void findDivergentBranches();
    void findRegions();

    // DATA
    InstVector         m_divergent;
    InstVector         m_divergentBranches;
    RegionVector       m_regions;

    LoopInfo          *m_loopInfo;
    PostDominatorTree *m_postDomT;
    DominatorTree     *m_domT;
    GridAnalysisPass  *m_grid;
};

#endif
