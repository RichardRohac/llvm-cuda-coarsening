// ============================================================================
// Region bounds
// -> Based on Alberto's Magni OpenCL coarsening pass algorithm
//    available at https://github.com/HariSeldon/coarsening_pass
// ============================================================================

#ifndef LLVM_LIB_TRANSFORMS_CUDA_COARSENING_REGIONBOUNDS_H
#define LLVM_LIB_TRANSFORMS_CUDA_COARSENING_REGIONBOUNDS_H

namespace llvm {
    class BasicBlock;
}

class RegionBounds {
  public:
    RegionBounds(llvm::BasicBlock *header, llvm::BasicBlock *exiting);
    RegionBounds();

  public:
    llvm::BasicBlock *getHeader();
    llvm::BasicBlock *getExiting();
    const llvm::BasicBlock *getHeader() const;
    const llvm::BasicBlock *getExiting() const;

    void setHeader(llvm::BasicBlock *Header);
    void setExiting(llvm::BasicBlock *Exiting);

    void listBlocks(BlockVector& result);

    void dump(const std::string& prefix = "") const;

  private:
    llvm::BasicBlock *header;
    llvm::BasicBlock *exiting;
};

void listBlocks(RegionBounds *bounds, BlockVector& result);
void listBlocks(llvm::BasicBlock *header,
                llvm::BasicBlock *exiting,
                BlockVector&      result);

#endif // LLVM_LIB_TRANSFORMS_CUDA_COARSENING_REGIONBOUNDS_H