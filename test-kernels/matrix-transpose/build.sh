#!/bin/bash

LLVM_BUILD_DIR=/DATA/LLVM/build_debug/
LLVM_BIN_DIR=/DATA/LLVM/build_debug/bin
CUDA_BIN_DIR=/opt/cuda-9.2/bin
DEVICE_ARCH=sm_61
DEVICE_COMPUTE_ARCH=compute_61

# -----------------------------------------------------------------------------
# Manual method (using opt):
# -----------------------------------------------------------------------------

# Compile the input hostcode into the LLVM IR using CUDA 9.2
$LLVM_BIN_DIR/clang++ -c -emit-llvm -O3 ./matrix-transpose.cu                 \
                      --cuda-path=/opt/cuda-9.2                               \
                      --cuda-gpu-arch=$DEVICE_ARCH                            \
                      --cuda-host-only -o host.bc

# Compile the input device code into the LLVM IR using CUDA 9.2
$LLVM_BIN_DIR/clang++ -c -emit-llvm -O3 ./matrix-transpose.cu                 \
                      --cuda-path=/opt/cuda-9.2                               \
                      --cuda-gpu-arch=$DEVICE_ARCH                            \
                      --cuda-device-only -o device.bc

# Generate readable versions
$LLVM_BIN_DIR/llvm-dis ./host.bc
$LLVM_BIN_DIR/llvm-dis ./device.bc

# Modify host kernel launch routines
$LLVM_BIN_DIR/opt -load $LLVM_BUILD_DIR/lib/LLVMCUDACoarsening.so             \
                  -cuda-coarsening-pass                                       \
                  -coarsened-kernel transposeNaive                            \
                  -coarsening-dimension x                                     \
                  -coarsening-factor 2                                        \
                  -debug-pass=Structure < host.bc > host_coarsened.bc

# Optimize the device code using our pass
$LLVM_BIN_DIR/opt -load $LLVM_BUILD_DIR/lib/LLVMCUDACoarsening.so             \
                  -cuda-coarsening-pass                                       \
                  -coarsened-kernel transposeNaive                            \
                  -coarsening-dimension x                                     \
                  -coarsening-factor 2                                        \
                  -debug-pass=Structure < device.bc > device_coarsened.bc

# Generate readable versions
$LLVM_BIN_DIR/llvm-dis ./host_coarsened.bc
$LLVM_BIN_DIR/llvm-dis ./device_coarsened.bc

# Assemble hostcode
$LLVM_BIN_DIR/llc -O0 -debugger-tune=gdb -filetype=obj -o hostcode.o          \
                  host_coarsened.bc

# Produce PTX
$LLVM_BIN_DIR/llc -mcpu=$DEVICE_ARCH -o kernel.$DEVICE_ARCH.ptx               \
                  device_coarsened.bc

# Assemble
$CUDA_BIN_DIR/ptxas -m64                                                      \
                    --gpu-name=$DEVICE_ARCH kernel.$DEVICE_ARCH.ptx           \
                    --output-file kernel.$DEVICE_ARCH.cubin

# Produce fat binary
$CUDA_BIN_DIR/fatbinary --cuda -64 --create device.fatbin                     \
          "--image=profile=$DEVICE_ARCH,file=kernel.$DEVICE_ARCH.cubin"       \
          "--image=profile=$DEVICE_COMPUTE_ARCH,file=kernel.$DEVICE_ARCH.ptx"

# Combine
$LLVM_BIN_DIR/clang-8 -cc1 -triple x86_64-unknown-linux-gnu                   \
                      -aux-triple nvptx64-nvidia-cuda -emit-obj -mrelax-all   \
                      -disable-free -main-file-name matrix-transpose.cu       \
                      -mrelocation-model static -mthread-model posix          \
                      -mdisable-fp-elim -fmath-errno -masm-verbose            \
                      -mconstructor-aliases -munwind-tables -fuse-init-array  \
                      -target-cpu x86-64 -dwarf-column-info                   \
                      -debugger-tuning=gdb                                    \
                      -resource-dir /DATA/LLVM/build_debug/lib/clang/8.0.1    \
                      -internal-isystem /DATA/LLVM/build_debug/lib/clang/8.0.1/include/cuda_wrappers -internal-isystem /opt/cuda-9.2/include -include __clang_cuda_runtime_wrapper.h -I/opt/intel/composerxe/linux/ipp/include -I/opt/intel/composerxe/linux/mkl/include -ISUBSTITUTE_INSTALL_DIR_HERE/include -I/opt/intel/composerxe/linux/tbb/include -internal-isystem /usr/lib64/gcc/x86_64-pc-linux-gnu/8.2.1/../../../../include/c++/8.2.1 -internal-isystem /usr/lib64/gcc/x86_64-pc-linux-gnu/8.2.1/../../../../include/c++/8.2.1/x86_64-pc-linux-gnu -internal-isystem /usr/lib64/gcc/x86_64-pc-linux-gnu/8.2.1/../../../../include/c++/8.2.1/backward -internal-isystem /usr/lib64/gcc/x86_64-pc-linux-gnu/8.2.1/../../../../include/c++/8.2.1 -internal-isystem /usr/lib64/gcc/x86_64-pc-linux-gnu/8.2.1/../../../../include/c++/8.2.1/x86_64-pc-linux-gnu -internal-isystem /usr/lib64/gcc/x86_64-pc-linux-gnu/8.2.1/../../../../include/c++/8.2.1/backward -internal-isystem /usr/local/include -internal-isystem /DATA/LLVM/build_debug/lib/clang/8.0.1/include -internal-externc-isystem /include -internal-externc-isystem /usr/include -internal-isystem /usr/local/include -internal-isystem /DATA/LLVM/build_debug/lib/clang/8.0.1/include -internal-externc-isystem /include -internal-externc-isystem /usr/include \
                      -fdeprecated-macro -fdebug-compilation-dir /home/richard/CUDACoarsening/cuda-coarsening/test-kernels/matrix-transpose -ferror-limit 19 -fmessage-length 0 -fobjc-runtime=gcc -fcxx-exceptions -fexceptions -fdiagnostics-show-option -o combined.o -x cuda ./matrix-transpose.cu -fcuda-include-gpubinary device.fatbin -faddrsig

# Link
$LLVM_BIN_DIR/clang -L/opt/cuda-9.2/lib64 -lcudart -o matrix-transpose combined.o

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------