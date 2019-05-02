#!/bin/bash

LLVM_BUILD_DIR=/DATA/LLVM/build_debug/
LLVM_BIN_DIR=/DATA/LLVM/build_debug/bin
CUDA_BIN_DIR=/opt/cuda-9.2/bin
DEVICE_ARCH=sm_61
DEVICE_COMPUTE_ARCH=compute_61
STRIDE=32
FACTOR=4

# -----------------------------------------------------------------------------
# Manual method (using opt):
# -----------------------------------------------------------------------------

# Compile the input device code into the LLVM IR using CUDA 9.2
$LLVM_BIN_DIR/clang++ -c -emit-llvm -O3 ./warp-shuffle.cu                     \
                      --cuda-path=/opt/cuda-9.2                               \
                      --cuda-gpu-arch=$DEVICE_ARCH                            \
                      --cuda-device-only -o device.bc

# # Generate readable version
$LLVM_BIN_DIR/llvm-dis ./device.bc

# Optimize the device code using our pass
$LLVM_BIN_DIR/opt -load $LLVM_BUILD_DIR/lib/LLVMCUDACoarsening.so             \
                  -cuda-coarsening-pass                                       \
                  -coarsened-kernel reduceShfl                                \
                  -coarsening-dimension x                                     \
                  -coarsening-factor $FACTOR                                  \
                  -coarsening-stride $STRIDE                                  \
                  -coarsening-mode thread                                     \
                  -debug-pass=Structure < device.bc > device_coarsened.bc

# Generate readable versions
$LLVM_BIN_DIR/llvm-dis ./device_coarsened.bc

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

# Produce host code combined with the fatbinary
$LLVM_BIN_DIR/clang-8 -cc1 -O0 -emit-llvm -triple x86_64-unknown-linux-gnu    \
                      -aux-triple nvptx64-nvidia-cuda -mrelax-all             \
                      -disable-free -main-file-name warp-shuffle.cu       \
                      -mrelocation-model static -mthread-model posix          \
                      -mdisable-fp-elim -fmath-errno -masm-verbose            \
                      -mconstructor-aliases -munwind-tables -fuse-init-array  \
                      -target-cpu x86-64 -dwarf-column-info                   \
                      -debugger-tuning=gdb                                    \
                      -resource-dir /usr/lib/clang/8.0.0 -internal-isystem /usr/lib/clang/8.0.0/include/cuda_wrappers -internal-isystem /opt/cuda-9.2/include -include __clang_cuda_runtime_wrapper.h -I/opt/intel/composerxe/linux/ipp/include -I/opt/intel/composerxe/linux/mkl/include -ISUBSTITUTE_INSTALL_DIR_HERE/include -I/opt/intel/composerxe/linux/tbb/include -internal-isystem /usr/bin/../lib64/gcc/x86_64-pc-linux-gnu/8.3.0/../../../../include/c++/8.3.0 -internal-isystem /usr/bin/../lib64/gcc/x86_64-pc-linux-gnu/8.3.0/../../../../include/c++/8.3.0/x86_64-pc-linux-gnu -internal-isystem /usr/bin/../lib64/gcc/x86_64-pc-linux-gnu/8.3.0/../../../../include/c++/8.3.0/backward -internal-isystem /usr/bin/../lib64/gcc/x86_64-pc-linux-gnu/8.3.0/../../../../include/c++/8.3.0 -internal-isystem /usr/bin/../lib64/gcc/x86_64-pc-linux-gnu/8.3.0/../../../../include/c++/8.3.0/x86_64-pc-linux-gnu -internal-isystem /usr/bin/../lib64/gcc/x86_64-pc-linux-gnu/8.3.0/../../../../include/c++/8.3.0/backward -internal-isystem /usr/local/include -internal-isystem /usr/lib/clang/8.0.0/include -internal-externc-isystem /include -internal-externc-isystem /usr/include -internal-isystem /usr/local/include -internal-isystem /usr/lib/clang/8.0.0/include -internal-externc-isystem /include -internal-externc-isystem /usr/include \
                      -fdeprecated-macro -fdebug-compilation-dir /home/richard/CUDACoarsening/cuda-coarsening/test-kernels/warp-shuffle -ferror-limit 19 -fmessage-length 0 -fobjc-runtime=gcc -fcxx-exceptions -fexceptions -fdiagnostics-show-option -o combined.ll -x cuda ./warp-shuffle.cu -fcuda-include-gpubinary device.fatbin -faddrsig

# Modify host kernel launch routines
$LLVM_BIN_DIR/opt -load $LLVM_BUILD_DIR/lib/LLVMCUDACoarsening.so             \
                  -cuda-coarsening-pass                                       \
                  -coarsened-kernel reduceShfl                                \
                  -coarsening-dimension x                                     \
                  -coarsening-factor $FACTOR                                  \
                  -coarsening-stride $STRIDE                                  \
                  -coarsening-mode thread                                     \
                  -debug-pass=Structure < combined.ll > combined_coarsened.bc

# Generate readable versions
$LLVM_BIN_DIR/llvm-dis ./combined_coarsened.bc

$LLVM_BIN_DIR/llc -filetype=obj combined_coarsened.bc

# # Link
$LLVM_BIN_DIR/clang -L/opt/cuda-9.2/lib64 -lcudart -o warp-shuffle combined_coarsened.o

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------