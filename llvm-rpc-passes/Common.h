#ifndef LLVM_LIB_TRANSFORMS_CUDA_COARSENING_COMMON_H
#define LLVM_LIB_TRANSFORMS_CUDA_COARSENING_COMMON_H

#include <vector>
#include <set>

namespace llvm {
    class Instruction;
}

// ===========================================================================
// DATA TYPES
// ===========================================================================
typedef std::vector<llvm::Instruction *> InstVector;
typedef std::set<llvm::Instruction *> InstSet;

/*
BasicBlock *findImmediatePostDom(BasicBlock *block,
                                 const PostDominatorTree *pdt);

void findUsesOf(Instruction *inst, InstSet &result);

// Container management.
// Check if the given element is present in the given container.
template <class T>
bool isPresent(const T *value, const std::vector<T *> &vector);
template <class T>
bool isPresent(const T *value, const std::vector<const T *> &vector);
template <class T> bool isPresent(const T *value, const std::set<T *> &vector);
template <class T>
bool isPresent(const T *value, const std::set<const T *> &vector);
template <class T> bool isPresent(const T *value, const std::deque<T *> &deque);

bool isPresent(const Instruction *inst, const BlockVector &value);
bool isPresent(const Instruction *inst, std::vector<BlockVector *> &value);
*/

#endif 