#ifndef LLVM_LIB_TRANSFORMS_CUDA_COARSENING_H
#define LLVM_LIB_TRANSFORMS_CUDA_COARSENING_H

//#include <vector>
//#include <set>

//#include "llvm/IR/InstrTypes.h"
//#include "llvm/IR/Instructions.h"

//typedef std::vector<llvm::Instruction *> InstVector;
//typedef std::set<llvm::Instruction *> InstSet;

using namespace llvm;

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
};

} // end anonymous namespace

#endif