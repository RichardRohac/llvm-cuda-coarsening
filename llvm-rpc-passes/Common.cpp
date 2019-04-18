/*#include "llvm/IR/BasicBlock.h"

#include "llvm/Analysis/PostDominators.h"

#include "CUDACoarsening.h"

using namespace llvm;

BasicBlock *findImmediatePostDom(BasicBlock *block,
                                 const PostDominatorTree *pdt) {
    return pdt->getNode(block)->getIDom()->getBlock();
}

void findUsesOf(Instruction *inst, InstSet &result) {
    for (auto userIter = inst->user_begin(); userIter != inst->user_end(); ++userIter) {
        if (Instruction *userInst = dyn_cast<Instruction>(*userIter)) {
            result.insert(userInst);
        }
    }
}

// isPresent.
//------------------------------------------------------------------------------
template <class T>
bool isPresent(const T *value, const std::vector<T *> &values) {
  auto result = std::find(values.begin(), values.end(), value);
  return result != values.end();
}

template bool isPresent(const Instruction *value, const InstVector &values);
template bool isPresent(const Value *value, const ValueVector &values);

template <class T>
bool isPresent(const T *value, const std::vector<const T *> &values) {
  auto result = std::find(values.begin(), values.end(), value);
  return result != values.end();
}

template bool isPresent(const Instruction *value,
                        const ConstInstVector &values);
template bool isPresent(const Value *value, const ConstValueVector &values);

template <class T> bool isPresent(const T *value, const std::set<T *> &values) {
  auto result = std::find(values.begin(), values.end(), value);
  return result != values.end();
}

template bool isPresent(const Instruction *value, const InstSet &values);

template <class T>
bool isPresent(const T *value, const std::set<const T *> &values) {
  auto result = std::find(values.begin(), values.end(), value);
  return result != values.end();
}

template bool isPresent(const Instruction *value, const ConstInstSet &values);

template <class T> bool isPresent(const T *value, const std::deque<T *> &d) {
  auto result = std::find(d.begin(), d.end(), value);
  return result != d.end();
}

template bool isPresent(const BasicBlock *value, const BlockDeque &deque);

bool isPresent(const Instruction *inst, const BlockVector &value) {
  const BasicBlock *BB = inst->getParent();
  return isPresent<BasicBlock>(BB, value);
}

bool isPresent(const Instruction *inst, std::vector<BlockVector *> &value) {
  for (auto Iter = value.begin(), E = value.end(); Iter != E; ++Iter)
    if (isPresent(inst, **Iter))
      return true;
  return false;
}
*/