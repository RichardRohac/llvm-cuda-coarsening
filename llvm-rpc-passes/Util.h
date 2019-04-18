#ifndef LLVM_LIB_TRANSFORMS_CUDA_COARSENING_UTIL_H
#define LLVM_LIB_TRANSFORMS_CUDA_COARSENING_UTIL_H

#define CUDA_TARGET_TRIPLE  "nvptx64-nvidia-cuda"
#define CUDA_RUNTIME_LAUNCH "cudaLaunch"

namespace llvm {
    class Function;
}

class Util {
  public:
    static bool isKernelFunction(llvm::Function& F);
};

#endif // LLVM_LIB_TRANSFORMS_CUDA_COARSENING_UTIL_H