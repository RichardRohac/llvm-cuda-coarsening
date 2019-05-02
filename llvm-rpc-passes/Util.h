#ifndef LLVM_LIB_TRANSFORMS_CUDA_COARSENING_UTIL_H
#define LLVM_LIB_TRANSFORMS_CUDA_COARSENING_UTIL_H

#define CUDA_TARGET_TRIPLE         "nvptx64-nvidia-cuda"
#define CUDA_RUNTIME_CONFIGURECALL "cudaConfigureCall"
#define CUDA_RUNTIME_LAUNCH        "cudaLaunch"

#define CUDA_THREAD_ID_VAR  "threadIdx"
#define CUDA_BLOCK_ID_VAR   "blockIdx"
#define CUDA_BLOCK_DIM_VAR  "blockDim"
#define CUDA_GRID_DIM_VAR   "gridDim"

#define CUDA_MAX_DIM        3

#define LLVM_PREFIX            "llvm"
#define CUDA_READ_SPECIAL_REG  "nvvm.read.ptx.sreg"
#define CUDA_THREAD_ID_REG     "tid"
#define CUDA_BLOCK_ID_REG      "ctaid"
#define CUDA_BLOCK_DIM_REG     "ntid"
#define CUDA_GRID_DIM_REG      "nctaid"

#define CUDA_SHUFFLE_DOWN      "nvvm.shfl.down"
#define CUDA_SHUFFLE_UP        "nvvm.shfl.up"
#define CUDA_SHUFFLE_BFLY      "nvvm.shfl.bfly"
#define CUDA_SHUFFLE_IDX       "nvvm.shfl.idx"

namespace llvm {
    class Function;
    class Instruction;
    class BasicBlock;
    class DominatorTree;
    class PostDominatorTree;
    class BranchInst;
    class PHINode;
    class Value;
    class StringRef;
}

class Util {
  public:
    static bool isKernelFunction(llvm::Function& F);
    static std::string directionToString(int direction);
    static std::string cudaVarToRegister(std::string var);
    static void findUsesOf(llvm::Instruction *inst,
                           InstSet&           result,
                           bool               skipBranches = false);
    static llvm::BasicBlock *findImmediatePostDom(
                                           llvm::BasicBlock              *block,
                                           const llvm::PostDominatorTree *pdt);

    // Domination -------------------------------------------------------------
    static bool isDominated(const llvm::Instruction   *inst,
                            BranchSet&                 blocks,
                            const llvm::DominatorTree *dt);

    static bool isDominated(const llvm::Instruction   *inst,
                            BranchVector&              blocks,
                            const llvm::DominatorTree *dt);

    static bool isDominated(const llvm::BasicBlock    *block,
                            const BlockVector&         blocks,
                            const llvm::DominatorTree *dt);

    static bool dominatesAll(const llvm::BasicBlock    *block,
                             const BlockVector&         blocks,
                             const llvm::DominatorTree *dt);

    static bool postdominatesAll(const llvm::BasicBlock        *block,
                                 const BlockVector&             blocks,
                                 const llvm::PostDominatorTree *pdt);

    // Cloning support --------------------------------------------------------
    static void cloneDominatorInfo(llvm::BasicBlock    *block,
                                   Map&                 map,
                                   llvm::DominatorTree *dt);

    // Map management ---------------------------------------------------------
    static void applyMap(llvm::Instruction *Inst, Map& map);
    static void applyMap(llvm::BasicBlock *block, Map& map);
    static void applyMapToPHIs(llvm::BasicBlock *block, Map& map);
    static void applyMapToPhiBlocks(llvm::PHINode *Phi, Map& map);
    //void applyMap(llvm::Instruction *Inst, CoarseningMap &map, unsigned int CF);
    static void applyMap(InstVector& insts, Map& map, InstVector& result);

    static void replaceUses(llvm::Value *oldValue, llvm::Value *newValue);

    // Regions ---------------------------------------------------------------
    static bool isOutermost(llvm::Instruction *inst, RegionVector& regions);
    static bool isOutermost(DivergentRegion *region, RegionVector& regions);

    static void renameValueWithFactor(llvm::Value     *value,
                                      llvm::StringRef  oldName,
                                      unsigned int     index);

    static void changeBlockTarget(llvm::BasicBlock   *block,
                                  llvm::BasicBlock   *newTarget,
                                  unsigned int        branchIndex = 0);


    static void remapBlocksInPHIs(llvm::BasicBlock *block,
                                  llvm::BasicBlock *oldBlock,
                                  llvm::BasicBlock *newBlock);
};

#endif // LLVM_LIB_TRANSFORMS_CUDA_COARSENING_UTIL_H