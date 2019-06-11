#!/bin/bash

# ------------------------------------------------------------------------------
# Manual CUDA Coarsening Compilation
# ------------------------------------------------------------------------------
# Local environment variables required by the script:
#
# RPC_LLVM_BUILD_DIR=.../LLVM/build_debug/
# RPC_LLVM_BIN_DIR=.../LLVM/build_debug/bin
# RPC_DEVICE_ARCH=sm_61
# RPC_COMPUTE_ARCH=compute_61
# CUDA_PATH=/opt/cuda
# CUDA_VERSION=10.1
#
# ------------------------------------------------------------------------------
# Coarsening configuration is to be supplied through the environment variable
# as well:
#
# RPC_CONFIG=<kernelName (or "all" for all to be coarsened in dynamic mode)>,
#            <dimension (x/y/z)>,
#            <mode (thread,block,dynamic)>,
#            <coarsening factor>,
#            <coarsening stride>
#
# For example, RPC_CONFIG=matrixTranspose,x,thread,2,32
#
# ------------------------------------------------------------------------------
# General script usage format:
# RPC_CONFIG="..." m3c.sh <input> <output> <builddir> <incdir>
# ------------------------------------------------------------------------------

# Stop executing on error
set -e

# Make sure input and output are passed into the script
if [ "$#" -ne 4 ]; then
    echo "Usage: RPC_CONFIG=\"...\" m3c.sh <input> <output> <builddir> <incdir>"
    exit 1
fi

# Parse coarsening configuration
IFS=',' read -r -a RPC_TOKENS <<< "$RPC_CONFIG"

INPUT_FILE=$1
OUTPUT_FILE=$2
BUILD_DIR=$3
INCLUDE_DIR=$4
KERNEL_NAME=${RPC_TOKENS[0]}
COARSENING_DIMENSION=${RPC_TOKENS[1]}
COARSENING_MODE=${RPC_TOKENS[2]}
COARSENING_FACTOR=${RPC_TOKENS[3]}
COARSENING_STRIDE=${RPC_TOKENS[4]}
OPT=-O3

printf '%s\n' "$RPC_LLVM_BUILD_DIR"
printf '%s\n' "$RPC_LLVM_BIN_DIR"
printf '%s\n' "$RPC_DEVICE_ARCH"
printf '%s\n' "$RPC_COMPUTE_ARCH"
printf '%s\n' "$RPC_RUNTIME_LIB"
printf '%s\n' "$CUDA_PATH"
printf '%s\n' "$CUDA_VERSION"

printf '%s\n' "$INPUT_FILE"
printf '%s\n' "$OUTPUT_FILE"
printf '%s\n' "$BUILD_DIR"
printf '%s\n' "$KERNEL_NAME"
printf '%s\n' "$COARSENING_DIMENSION"
printf '%s\n' "$COARSENING_MODE"
printf '%s\n' "$COARSENING_FACTOR"
printf '%s\n' "$COARSENING_STRIDE"

# ------------------------------------------------------------------------------
# Compile the input device code into the LLVM IR
$RPC_LLVM_BIN_DIR/clang++ -x cuda -c -emit-llvm $OPT $INPUT_FILE               \
                          -I$INCLUDE_DIR                                       \
                          -Xclang -disable-O0-optnone                          \
                          --cuda-path=$CUDA_PATH                               \
                          --cuda-gpu-arch=$RPC_DEVICE_ARCH                     \
                          --cuda-device-only -o $BUILD_DIR/rpc_device.bc

$RPC_LLVM_BIN_DIR/clang++ -x cuda -c -emit-llvm $OPT $INPUT_FILE              \
                      -I$INCLUDE_DIR                                          \
                      --cuda-path=$CUDA_PATH                                  \
                      --cuda-gpu-arch=$RPC_DEVICE_ARCH                        \
                      --cuda-host-only -o $BUILD_DIR/rpc_host.bc

# Generate readable version
$RPC_LLVM_BIN_DIR/llvm-dis $BUILD_DIR/rpc_device.bc -o $BUILD_DIR/rpc_device.ll
$RPC_LLVM_BIN_DIR/llvm-dis $BUILD_DIR/rpc_host.bc -o $BUILD_DIR/rpc_host.ll

# Optimize the device code using our pass
$RPC_LLVM_BIN_DIR/opt -load $RPC_LLVM_BUILD_DIR/lib/LLVMCUDACoarsening.so     \
                      -mem2reg -indvars -structurizecfg -be                   \
                      -cuda-coarsening-pass                                   \
                      -coarsened-kernel $KERNEL_NAME                          \
                      -coarsening-dimension $COARSENING_DIMENSION             \
                      -coarsening-factor $COARSENING_FACTOR                   \
                      -coarsening-stride $COARSENING_STRIDE                   \
                      -coarsening-mode $COARSENING_MODE                       \
                      -o $BUILD_DIR/rpc_device_coarsened.bc                   \
                       < $BUILD_DIR/rpc_device.bc

# Generate readable versions
$RPC_LLVM_BIN_DIR/llvm-dis $BUILD_DIR/rpc_device_coarsened.bc                  \
                       -o $BUILD_DIR/rpc_device_coarsened.ll

# Produce PTX
$RPC_LLVM_BIN_DIR/llc $OPT -mcpu=$RPC_DEVICE_ARCH                              \
                  -o $BUILD_DIR/rpc_kernel.$RPC_DEVICE_ARCH.ptx                \
                  $BUILD_DIR/rpc_device_coarsened.bc

# Assemble
$CUDA_PATH/bin/ptxas -m64                                                     \
                    --gpu-name=$RPC_DEVICE_ARCH                               \
                    $BUILD_DIR/rpc_kernel.$RPC_DEVICE_ARCH.ptx                \
                    --output-file $BUILD_DIR/rpc_kernel.$RPC_DEVICE_ARCH.cubin

# Produce fat binary
$CUDA_PATH/bin/fatbinary -64 --create $BUILD_DIR/rpc_device.fatbin                   \
"--image=profile=$RPC_DEVICE_ARCH,file=$BUILD_DIR/rpc_kernel.$RPC_DEVICE_ARCH.cubin" \
"--image=profile=$RPC_COMPUTE_ARCH,file=$BUILD_DIR/rpc_kernel.$RPC_DEVICE_ARCH.ptx"

# Produce host code combined with the fatbinary
$RPC_LLVM_BIN_DIR/clang-8                                                      \
   -cc1 $OPT -emit-llvm -triple x86_64-unknown-linux-gnu                       \
   -x cuda                                                                     \
   -target-sdk-version=$CUDA_VERSION                                           \
   -aux-triple nvptx64-nvidia-cuda -mrelax-all                                 \
   -disable-free -main-file-name $INPUT_FILE                                   \
   -mrelocation-model static -mthread-model posix                              \
   -mdisable-fp-elim -fmath-errno -masm-verbose                                \
   -mconstructor-aliases -munwind-tables -fuse-init-array                      \
   -target-cpu x86-64 -dwarf-column-info                                       \
   -debugger-tuning=gdb                                                        \
   -resource-dir $RPC_LLVM_BUILD_DIR/lib/clang/9.0.0                           \
   -internal-isystem $INCLUDE_DIR -internal-externc-isystem $INCLUDE_DIR       \
   -internal-isystem $RPC_LLVM_BUILD_DIR/lib/clang/9.0.0/include/cuda_wrappers \
   -internal-isystem $CUDA_PATH/include                                        \
   -include __clang_cuda_runtime_wrapper.h                                     \
   -internal-isystem /usr/include/c++/8.3.0/                                   \
   -internal-isystem /usr/include/c++/8.3.0/x86_64-pc-linux-gnu                \
   -internal-isystem /usr/include/c++/8.3.0/backward                           \
   -internal-isystem /usr/local/include                                        \
   -internal-isystem $RPC_LLVM_BUILD_DIR/lib/clang/9.0.0/include               \
   -internal-externc-isystem /usr/include                                      \
   -internal-isystem /usr/local/include                                        \
   -fdeprecated-macro                                                          \
   -fdebug-compilation-dir $BUILD_DIR                                          \
   -ferror-limit 19 -fmessage-length 0 -fobjc-runtime=gcc                      \
   -fcxx-exceptions -fexceptions -fdiagnostics-show-option                     \
   -o $BUILD_DIR/rpc_combined.ll $INPUT_FILE                                   \
   -fcuda-include-gpubinary $BUILD_DIR/rpc_device.fatbin                       \
   -faddrsig

# Modify host kernel launch routines
$RPC_LLVM_BIN_DIR/opt -load $RPC_LLVM_BUILD_DIR/lib/LLVMCUDACoarsening.so      \
                      -cuda-coarsening-pass                                    \
                      -coarsened-kernel $KERNEL_NAME                           \
                      -coarsening-dimension $COARSENING_DIMENSION              \
                      -coarsening-factor $COARSENING_FACTOR                    \
                      -coarsening-stride $COARSENING_STRIDE                    \
                      -coarsening-mode $COARSENING_MODE                        \
                      -o $BUILD_DIR/rpc_combined_coarsened.bc                  \
                       < $BUILD_DIR/rpc_combined.ll

# Generate readable versions
$RPC_LLVM_BIN_DIR/llvm-dis $BUILD_DIR/rpc_combined_coarsened.bc                \
                       -o $BUILD_DIR/rpc_combined_coarsened.ll

# Build
$RPC_LLVM_BIN_DIR/llc $OPT -filetype=obj $BUILD_DIR/rpc_combined_coarsened.bc  \
                  -o $OUTPUT_FILE
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------