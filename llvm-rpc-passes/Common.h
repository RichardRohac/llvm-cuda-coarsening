#ifndef LLVM_LIB_TRANSFORMS_CUDA_COARSENING_COMMON_H
#define LLVM_LIB_TRANSFORMS_CUDA_COARSENING_COMMON_H

#include <string>
#include <vector>
#include <set>
#include <unordered_map>
#include <deque>
#include <numeric>
#include <functional>
#include <algorithm>
#include <map>

#include "llvm/IR/ValueMap.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

namespace llvm {
    class Instruction;
    class Value;
    class BasicBlock;
    class BranchInst;
    class WeakTrackingVH;
}

class DivergentRegion;

// ===========================================================================
// DATA TYPES
// ===========================================================================
typedef llvm::ValueMap<const llvm::Value *, llvm::WeakTrackingVH> Map;

typedef std::vector<llvm::Instruction *> InstVector;
typedef std::vector<const llvm::Instruction *> ConstInstVector;
typedef std::set<llvm::Instruction *> InstSet;
typedef std::set<const llvm::Instruction *> ConstInstSet;

typedef std::vector<llvm::BasicBlock *> BlockVector;
typedef std::deque<llvm::BasicBlock *> BlockDeque;

typedef std::vector<llvm::Value *> ValueVector;
typedef std::vector<const llvm::Value *> ConstValueVector;

typedef std::vector<llvm::BranchInst *> BranchVector;
typedef std::set<llvm::BranchInst *> BranchSet;

typedef std::vector<DivergentRegion *> RegionVector;

typedef std::map<llvm::Instruction *, InstVector> CoarseningMap;

// ===========================================================================
// HELPER FUNCTIONS
// ===========================================================================

// ---------------------------------------------------------------------------
// isPresent
//
// Check if the given element is present in the given container.
// ---------------------------------------------------------------------------
template <class T>
bool isPresent(const T *value, const std::vector<T *>& vector);
template <class T>
bool isPresent(const T *value, const std::vector<const T *>& vector);
template <class T>
bool isPresent(const T *value, const std::set<T *>& vector);
template <class T>
bool isPresent(const T *value, const std::set<const T *>& vector);
template <class T>
bool isPresent(const T *value, const std::deque<T *>& deque);

bool isPresent(const llvm::Instruction *inst, const BlockVector& value);
bool isPresent(const llvm::Instruction *inst, std::vector<BlockVector *>& value);

#endif 