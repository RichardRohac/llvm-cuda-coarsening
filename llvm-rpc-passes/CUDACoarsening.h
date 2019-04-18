#ifndef LLVM_LIB_TRANSFORMS_CUDA_COARSENING_H
#define LLVM_LIB_TRANSFORMS_CUDA_COARSENING_H

//#include <vector>
//#include <set>

//#include "llvm/IR/InstrTypes.h"
//#include "llvm/IR/Instructions.h"

//typedef std::vector<llvm::Instruction *> InstVector;
//typedef std::set<llvm::Instruction *> InstSet;

using namespace llvm;

namespace llvm {
    class LoopInfo;
    class PostDominatorTree;
    class DominatorTree;
}

namespace {

class CUDACoarseningPass : public ModulePass {
  public:
    // CREATORS
    CUDACoarseningPass();

    // ACCESSORS
    bool runOnModule(Module& M) override;
    void getAnalysisUsage(AnalysisUsage& Info) const override;

    // DATA
    static char ID;

  private:
    // MODIFIERS
    bool handleDeviceCode(Module& M);
    bool handleHostCode(Module& M);
    
    void analyzeKernel(Function& F);

    // DATA
    LoopInfo          *m_loopInfo;
    PostDominatorTree *m_postDomT;
    DominatorTree     *m_domT;
};

} // end anonymous namespace

#endif