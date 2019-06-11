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

typedef std::unordered_map<Function *, bool> coarsenedKernelMap_t;

namespace llvm {
    class LoopInfo;
    class PostDominatorTree;
    class DominatorTree;
}

class DivergenceAnalysisPassTL;
class DivergenceAnalysisPassBL;
class GridAnalysisPass;
class BenefitAnalysisPass;

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
    bool parseConfig();

    bool handleDeviceCode(Module& M);
    bool handleHostCode(Module& M);

    void generateVersions(Function& F, bool deviceCode);
    void generateVersion(Function&     F,
                         bool          deviceCode,
                         unsigned int  factor,
                         unsigned int  stride,
                         unsigned int  dimension,
                         bool          blockMode,
                         CallInst     *cudaRegFuncCall);
    std::string namedKernelVersion(std::string kernel, int d, int b, int t, int s);
    
    void analyzeKernel(Function& F);
    void scaleKernelGrid();
    void scaleKernelGridSizes(unsigned int dimension);
    void scaleKernelGridIDs(unsigned int dimension);
    void scaleGrid(BasicBlock  *configBlock,
                   CallInst    *configCall,
                   std::string  kernelName);

    void coarsenKernel(Function& F);
    void replacePlaceholders();

    void replicateInstruction(Instruction *inst);
    void replicateGlobal(GlobalVariable *gv);
    void replicateRegion(DivergentRegion *region);
    void replicateRegionClassic(DivergentRegion *region);

    void replicateRegionImpl(DivergentRegion *region, CoarseningMap& aliveMap);

    void initAliveMap(DivergentRegion *region, CoarseningMap& aliveMap);
    void updateAliveMap(CoarseningMap& aliveMap, Map& regionMap);
    void updatePlaceholdersWithAlive(CoarseningMap& aliveMap);

    void applyCoarseningMap(DivergentRegion& region, unsigned int index);
    void applyCoarseningMap(BasicBlock *block, unsigned int index);
    void applyCoarseningMap(Instruction *inst, unsigned int index);
    Instruction *getCoarsenedInstruction(Instruction *ret, Instruction *inst,
                                         unsigned int coarseningIndex);

    void updatePlaceholderMap(Instruction *inst, InstVector& coarsenedInsts);

    void insertRPCFunctions(Module& M);
    void deleteRPCFunctions(Module& M);
    void insertRPCLaunchKernel(Module& M);
    void insertRPCRegisterFunction(Module& M);

    // PRIVATE ACCESSORS
    bool shouldCoarsen(Function& F, bool hostCode = false) const;
      // Returns true if and only if this function is to be coarsened according
      // to the current pass configuration.

    CallInst *cudaRegistrationCallForKernel(Module&     M,
                                            std::string kernelName) const;
      // Retrieves call to the CUDA runtime responsible for the fat binary
      // registration of the kernel function specified by 'kernelName'.
      // If no such registration call is found, function returns 'nullptr'.

    // DATA
    LoopInfo               *m_loopInfo;
    PostDominatorTree      *m_postDomT;
    DominatorTree          *m_domT;
    DivergenceAnalysisPassTL *m_divergenceAnalysisTL;
    DivergenceAnalysisPassBL *m_divergenceAnalysisBL;
    GridAnalysisPass       *m_gridAnalysis;
    BenefitAnalysisPass    *m_benefitAnalysis;

    CoarseningMap           m_coarseningMap;
    CoarseningMap           m_phMap;
    Map                     m_phReplacementMap;
    GlobalsSet              m_divergentGlobals;
    GlobalsCMap             m_globalsCoarseningMap;

    Function               *m_rpcLaunchKernel;
    Function               *m_rpcRegisterFunction;

    Function               *m_readEnvConfig;

    coarsenedKernelMap_t    m_coarsenedKernelMap;

    // CL config
    std::string             m_kernelName;
    unsigned int            m_factor;
    unsigned int            m_stride;
    bool                    m_blockLevel;
    bool                    m_dynamicMode;
    unsigned int            m_dimension;
};

#endif