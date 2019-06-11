#!/bin/bash

# ------------------------------------------------------------------------------
# Manual CUDA Coarsening Compilation Link
# ------------------------------------------------------------------------------
# Local environment variables required by the script:
#
# RPC_LLVM_BIN_DIR=.../LLVM/build_debug/bin
# RPC_RUNTIME_LIB=.../cuda-coarsening/runtime/rpc_dynamic.o
# CUDA_PATH=/opt/cuda
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
# RPC_CONFIG="..." m3c_link.sh <input> <output>
# ------------------------------------------------------------------------------

# Stop executing on error
set -e

INPUT_FILES=$1
OUTPUT_FILE=$2

# Make sure input and output are passed into the script
if [ "$#" -ne 2 ]; then
    echo "Usage: RPC_CONFIG=\"...\" m3c_link.sh <input> <output>"
    exit 1
fi

# Parse coarsening configuration
IFS=',' read -r -a RPC_TOKENS <<< "$RPC_CONFIG"

COARSENING_MODE=${RPC_TOKENS[2]}

RUNTIME_LIB=""

if [[ "$COARSENING_MODE" == dynamic ]]; then
	RUNTIME_LIB=$RPC_RUNTIME_LIB
fi

$RPC_LLVM_BIN_DIR/clang -L$CUDA_PATH/lib64 -lcudart -ldl -lm -lstdc++          \
                        $RUNTIME_LIB $INPUT_FILES                              \
                        -o $OUTPUT_FILE