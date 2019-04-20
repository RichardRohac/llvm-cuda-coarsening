#ifndef LLVM_LIB_TRANSFORMS_CUDA_COARSENING_UTIL_H
#define LLVM_LIB_TRANSFORMS_CUDA_COARSENING_UTIL_H

#define CUDA_TARGET_TRIPLE  "nvptx64-nvidia-cuda"
#define CUDA_RUNTIME_LAUNCH "cudaLaunch"

#define CUDA_THREAD_ID_VAR  "threadIdx"
#define CUDA_BLOCK_ID_VAR   "blockIdx"
#define CUDA_BLOCK_DIM_VAR  "blockDim"
#define CUDA_GRID_DIM_VAR   "gridDim"

#define CUDA_MAX_DIM        3

#define LLVM_PREFIX            "llvm"
#define CUDA_READ_SPECIAL_REG  "nvvm.read.ptx.sreg"
#define CUDA_THREAD_ID_REG     "tid"
#define CUDA_BLOCK_ID_REG      "ctaid"
#define CUDA_BLOCK_DIM_REG     "ntid"
#define CUDA_GRID_DIM_REG      "nctaid"

namespace llvm {
    class Function;
}

class Util {
  public:
    static bool isKernelFunction(llvm::Function& F);
    static std::string directionToString(int direction);
    static std::string cudaVarToRegister(std::string var);
};

#endif // LLVM_LIB_TRANSFORMS_CUDA_COARSENING_UTIL_H