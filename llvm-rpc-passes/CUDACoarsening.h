// ============================================================================
// Copyright (c) Richard Rohac, 2019, All rights reserved.
// ============================================================================
// CUDA Coarsening Transformation pass
// -> Based on Alberto's Magni OpenCL coarsening pass algorithm
//    available at https://github.com/HariSeldon/coarsening_pass
// ============================================================================


#ifndef LLVM_LIB_TRANSFORMS_CUDA_COARSENING_H
#define LLVM_LIB_TRANSFORMS_CUDA_COARSENING_H

using namespace llvm;

namespace llvm {
    class LoopInfo;
    class PostDominatorTree;
    class DominatorTree;
}

class DivergenceAnalysisPass;

class CUDACoarseningPass : public ModulePass {
  public:
    // CREATORS
    CUDACoarseningPass();

    // ACCESSORS
    bool runOnModule(Module& M) override;
    void getAnalysisUsage(AnalysisUsage& AU) const override;

    // DATA
    static char ID;

  private:
    // MODIFIERS
    bool handleDeviceCode(Module& M);
    bool handleHostCode(Module& M);
    
    void analyzeKernel(Function& F);
    void scaleGrid(BasicBlock *configBlock, CallInst *configCall);

    CallInst *amendConfiguration(Module& M, BasicBlock *configOKBlock);

    void insertCudaConfigureCallScaled(Module& M);

    // DATA
    LoopInfo               *m_loopInfo;
    PostDominatorTree      *m_postDomT;
    DominatorTree          *m_domT;
    DivergenceAnalysisPass *m_divergenceAnalysis;

    Function               *m_cudaConfigureCallScaled;

    // CL config
    std::string             m_kernelName;
    unsigned int            m_factor;
    unsigned int            m_stride;
    bool                    m_blockLevel;
    bool                    m_dimX;
    bool                    m_dimY;
    bool                    m_dimZ;
};

#endif