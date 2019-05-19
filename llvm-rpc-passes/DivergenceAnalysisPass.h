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

class DivergenceAnchorPass : public FunctionPass {
  public:
    // CREATORS
    DivergenceAnchorPass();

    // MANIPULATORS
    bool runOnFunction(Function& F) override;
    
    // DATA
    static char ID;
};

class DivergenceAnalysisPass : public FunctionPass {
public:
    // CREATORS
    DivergenceAnalysisPass();

    // ACCESSORS
    RegionVector& getOutermostRegions();
    InstVector& getOutermostInstructions();

    bool isDivergent(Instruction *inst);

    // MANIPULATORS
    void getAnalysisUsage(AnalysisUsage& AU) const override;
    bool runOnFunction(Function& F) override;

    // DATA
    static char ID;

private:
    // PRIVATE MANIPULATORS
    void clear();
    void analyse(Function& F);
    void findOutermost(InstVector&   insts,
                       RegionVector& regions,
                       InstVector&   result);
    void findDivergentBranches();
    void findRegions();
    void findOutermostRegions();

    void findUsers(InstVector& seeds, InstVector *out, bool skipBranches);

    RegionVector cleanUpRegions(RegionVector& regions, const DominatorTree *dt);

    // DATA
    InstVector         m_divergent;
    InstVector         m_outermostDivergent;
    InstVector         m_divergentBranches;
    RegionVector       m_regions;
    RegionVector       m_outermostRegions;

    LoopInfo          *m_loopInfo;
    PostDominatorTree *m_postDomT;
    DominatorTree     *m_domT;
    GridAnalysisPass  *m_grid;

    bool               m_blockLevel;
    unsigned int       m_dimension;
};

#endif
