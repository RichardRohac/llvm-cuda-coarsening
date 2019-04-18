#ifndef LLVM_LIB_TRANSFORMS_CUDA_COARSENING_UTIL_H
#define LLVM_LIB_TRANSFORMS_CUDA_COARSENING_UTIL_H

namespace llvm {
    class Function;
}

class Util {
  public:
    static bool isKernelFunction(llvm::Function& F);
};

#endif // LLVM_LIB_TRANSFORMS_CUDA_COARSENING_UTIL_H