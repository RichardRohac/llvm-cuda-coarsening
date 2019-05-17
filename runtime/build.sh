#!/bin/bash

# Local environment variables
LLVM_BUILD_DIR=/DATA/LLVM/build_debug/
LLVM_BIN_DIR=/DATA/LLVM/build_debug/bin
CUDA_DIR=/opt/cuda
DEVICE_ARCH=sm_61
DEVICE_COMPUTE_ARCH=compute_61
STRIDE=1
FACTOR=4
MODE=dynamic

# -----------------------------------------------------------------------------

$LLVM_BIN_DIR/clang++ -c -O3 ./dynamic.cpp

# -----------------------------------------------------------------------------