#ifndef LLVM_LIB_TRANSFORMS_CUDA_COARSENING_H
#define LLVM_LIB_TRANSFORMS_CUDA_COARSENING_H

#include <vector>
#include <set>

#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"

typedef std::vector<llvm::Instruction *> InstVector;
typedef std::set<llvm::Instruction *> InstSet;

#endif