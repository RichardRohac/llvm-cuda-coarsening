// ============================================================================
// Divergent Region
// -> Based on Alberto's Magni OpenCL coarsening pass algorithm
//    available at https://github.com/HariSeldon/coarsening_pass
// ============================================================================

#ifndef LLVM_LIB_TRANSFORMS_CUDA_COARSENING_DIVERGENTREGION_H
#define LLVM_LIB_TRANSFORMS_CUDA_COARSENING_DIVERGENTREGION_H

namespace llvm {
    class Instruction;
    class Twine;
    class LoopInfo;
    class BasicBlock;
  //  class DominatorTree;
  //  class PostDominatorTree;
}

class RegionBounds;

class DivergentRegion {
  public:
    DivergentRegion(llvm::BasicBlock *header, llvm::BasicBlock *exiting);
    DivergentRegion(llvm::BasicBlock *header,
                    llvm::BasicBlock *exiting,
                    InstVector&       alive);
    DivergentRegion(RegionBounds& bounds);

    // Getter and Setter.
    llvm::BasicBlock *getHeader();
    llvm::BasicBlock *getExiting();

    const llvm::BasicBlock *getHeader() const;
    const llvm::BasicBlock *getExiting() const;

    RegionBounds& getBounds();
    BlockVector& getBlocks();
    InstVector& getAlive();
    InstVector& getIncoming();

    void setHeader(llvm::BasicBlock *Header);
    void setExiting(llvm::BasicBlock *Exiting);
    void setAlive(const InstVector& alive);
    void setIncoming(const InstVector& incoming);

    void fillRegion();
    void findAliveValues();
    void findIncomingValues();

    void analyze();
    bool areSubregionsDisjoint();

    DivergentRegion *clone(const llvm::Twine&   suffix,
                           llvm::DominatorTree *dt,
                           Map&                 valueMap);
    llvm::BasicBlock *getSubregionExiting(unsigned int branchIndex);

    unsigned int size();
    void dump();

  private:
    void updateBounds(llvm::DominatorTree *dt, llvm::PostDominatorTree *pdt); 

  private:
    RegionBounds bounds;
    BlockVector blocks;
    InstVector alive;
    InstVector incoming;

  public:
    // Iterator class.
    //---------------------------------------------------------------------------
    class iterator
        : public std::iterator<std::forward_iterator_tag, llvm::BasicBlock *> {
      public:
        iterator();
        iterator(const DivergentRegion &region);
        iterator(const iterator &original);

      private:
        BlockVector blocks;
        size_t currentBlock;

      public:
        // Pre-increment.
        iterator &operator++();
        // Post-increment.
        iterator operator++(int);
        llvm::BasicBlock *operator*() const;
        bool operator!=(const iterator &iter) const;
        bool operator==(const iterator &iter) const;

        static iterator end();

      private:
        void toNext();
    };

    DivergentRegion::iterator begin();
    DivergentRegion::iterator end();

    // Const iterator class.
    //---------------------------------------------------------------------------
    class const_iterator
        : public std::iterator<std::forward_iterator_tag, llvm::BasicBlock *> {
      public:
        const_iterator();
        const_iterator(const DivergentRegion &region);
        const_iterator(const const_iterator &original);

      private:
        BlockVector blocks;
        size_t currentBlock;

      public:
        // Pre-increment.
        const_iterator &operator++();
        // Post-increment.
        const_iterator operator++(int);
        const llvm::BasicBlock *operator*() const;
        bool operator!=(const const_iterator &iter) const;
        bool operator==(const const_iterator &iter) const;

        static const_iterator end();

      private:
        void toNext();
    };

    DivergentRegion::const_iterator begin() const;
    DivergentRegion::const_iterator end() const;
};

// Non member functions.
// -----------------------------------------------------------------------------
llvm::BasicBlock *getExit(DivergentRegion &region);
llvm::BasicBlock *getPredecessor(DivergentRegion *region, llvm::LoopInfo *loopInfo);
RegionBounds cloneRegion(RegionBounds &bounds, const llvm::Twine &suffix,
                         llvm::DominatorTree *dt);
bool contains(const DivergentRegion &region, const llvm::Instruction *inst);
bool containsInternally(const DivergentRegion &region, const llvm::Instruction *inst);
bool contains(const DivergentRegion &region, const llvm::BasicBlock *block);
bool containsInternally(const DivergentRegion &region, const llvm::BasicBlock *block);
bool containsInternally(const DivergentRegion &region,
                        const DivergentRegion *innerRegion);
llvm::BasicBlock *getSubregionExiting(DivergentRegion *region,
                                      unsigned int     branchIndex);
void getSubregionAlive(DivergentRegion *region,
                       const llvm::BasicBlock *subregionExiting, InstVector &result);

#endif // LLVM_LIB_TRANSFORMS_CUDA_COARSENING_DIVERGENTREGION_H