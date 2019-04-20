#include "Common.h"
#include "llvm/IR/Instruction.h"

// ===========================================================================
// HELPER FUNCTIONS
// ===========================================================================

// ---------------------------------------------------------------------------
// isPresent
//
// Check if the given element is present in the given container.
// ---------------------------------------------------------------------------
template <class T>
bool isPresent(const T *value, const std::vector<T *>& values) {
    auto result = std::find(values.begin(), values.end(), value);
    return result != values.end();
}

template bool isPresent(const llvm::Instruction *value,
                        const InstVector&        values);
template bool isPresent(const llvm::Value  *value,
                        const ValueVector&  values);

template <class T>
bool isPresent(const T *value, const std::vector<const T *>& values) {
    auto result = std::find(values.begin(), values.end(), value);
    return result != values.end();
}

template bool isPresent(const llvm::Instruction *value,
                        const ConstInstVector&   values);
template bool isPresent(const llvm::Value       *value,
                        const ConstValueVector&  values);

template <class T> bool isPresent(const T *value, const std::set<T *>& values) {
    auto result = std::find(values.begin(), values.end(), value);
    return result != values.end();
}

template bool isPresent(const llvm::Instruction *value, const InstSet& values);

template <class T>
bool isPresent(const T *value, const std::set<const T *>& values) {
    auto result = std::find(values.begin(), values.end(), value);
    return result != values.end();
}

template bool isPresent(const llvm::Instruction *value,
                        const ConstInstSet&      values);

template <class T> bool isPresent(const T *value, const std::deque<T *>& d) {
    auto result = std::find(d.begin(), d.end(), value);
    return result != d.end();
}

template bool isPresent(const llvm::BasicBlock *value,
                        const BlockDeque&       deque);

bool isPresent(const llvm::Instruction *inst, const BlockVector &value) {
    const llvm::BasicBlock *BB = inst->getParent();
    return isPresent<llvm::BasicBlock>(BB, value);
}

bool isPresent(const llvm::Instruction     *inst,
               std::vector<BlockVector *>&  value) {
    for (auto Iter = value.begin(), E = value.end(); Iter != E; ++Iter) {
        if (isPresent(inst, **Iter))
            return true;
    }
    return false;
}
