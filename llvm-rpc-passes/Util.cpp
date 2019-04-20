#include "Common.h"
#include "Util.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Support/MutexGuard.h"

using namespace llvm;

bool findOneNVVMAnnotation(const GlobalValue  *gv,
                           const std::string&  prop,
                           unsigned int&       retval);

bool Util::isKernelFunction(llvm::Function& F)
{
    unsigned int x = 0;
    bool retval = findOneNVVMAnnotation(&F, "kernel", x);

    if (!retval) {
        // There is no NVVM metadata, check the calling convention
        return F.getCallingConv() == CallingConv::PTX_Kernel;
    }

    return (x == 1);
}

std::string Util::directionToString(int direction)
{
    assert(direction >= 0 &&
           direction <= 2 &&
           "directionToString(): Direction out of bounds!");

    std::string strDirection;
    switch (direction) {
        case 0:
            strDirection = "x";
            break;
        
        case 1:
            strDirection = "y";
            break;

        case 2:
            strDirection = "z";
            break;
    }

    return strDirection;
}

std::string Util::cudaVarToRegister(std::string var)
{
    if (var == CUDA_THREAD_ID_VAR) {
        return CUDA_THREAD_ID_REG;
    }
    else if (var == CUDA_BLOCK_ID_VAR) {
        return CUDA_BLOCK_ID_REG;
    }
    else if (var == CUDA_BLOCK_DIM_VAR) {
        return CUDA_BLOCK_DIM_REG;
    }
    else if (var == CUDA_GRID_DIM_VAR) {
        return CUDA_GRID_DIM_REG;
    }
    else {
        assert(0 && "cudaVarToRegister(): Unknown CUDA variable");
    }
}

void Util::findUsesOf(Instruction *inst, InstSet &result) {
    for (auto userIter = inst->user_begin();
         userIter != inst->user_end();
         ++userIter) {
        if (Instruction *userInst = dyn_cast<Instruction>(*userIter)) {
            result.insert(userInst);
        }
    }
}

BasicBlock *Util::findImmediatePostDom(BasicBlock              *block,
                                       const PostDominatorTree *pdt) {
    return pdt->getNode(block)->getIDom()->getBlock();
}

// Domination ----------------------------------------------------------------
bool Util::isDominated(const Instruction   *inst,
                       BranchVector&        branches,
                       const DominatorTree *dt)
{
    const BasicBlock *block = inst->getParent();
    return std::any_of(branches.begin(), branches.end(),
                        [inst, block, dt](BranchInst *branch) {
        const BasicBlock *currentBlock = branch->getParent();
        return inst != branch && dt->dominates(currentBlock, block);
    });
}

bool Util::isDominated(const Instruction   *inst,
                       BranchSet&           branches,
                       const DominatorTree *dt) 
{
    const BasicBlock *block = inst->getParent();
    return std::any_of(branches.begin(), branches.end(),
                        [inst, block, dt](BranchInst *branch) {
        const BasicBlock *currentBlock = branch->getParent();
        return inst != branch && dt->dominates(currentBlock, block);
    });
}

bool Util::isDominated(const BasicBlock    *block,
                       const BlockVector&   blocks,
                       const DominatorTree *dt)
{
    return std::any_of(blocks.begin(), blocks.end(),
                        [block, dt](BasicBlock *iter) {
        return block != iter && dt->dominates(iter, block);
    });
}

bool Util::dominatesAll(const BasicBlock    *block,
                        const BlockVector&   blocks,
                        const DominatorTree *dt)
{
    return std::all_of(
        blocks.begin(), blocks.end(),
        [block, dt](BasicBlock *iter) { return dt->dominates(block, iter); });
}

bool Util::postdominatesAll(const BasicBlock        *block,
                            const BlockVector&       blocks,
                            const PostDominatorTree *pdt) {
    return std::all_of(
        blocks.begin(), blocks.end(),
        [block, pdt](BasicBlock *iter) { return pdt->dominates(block, iter); });
}

// Cloning support ------------------------------------------------------------
// This is black magic. Don't touch it.
void Util::cloneDominatorInfo(BasicBlock *BB, Map &map, DominatorTree *DT)
{
    assert(DT && "DominatorTree is not available");
    Map::iterator BI = map.find(BB);
    assert(BI != map.end() && "BasicBlock clone is missing");
    BasicBlock *NewBB = cast<BasicBlock>(BI->second);

    // NewBB already got dominator info.
    if (DT->getNode(NewBB))
        return;

    assert(DT->getNode(BB) && "BasicBlock does not have dominator info");
    // Entry block is not expected here. Infinite loops are not to cloned.
    assert(DT->getNode(BB)->getIDom() &&
            "BasicBlock does not have immediate dominator");
    BasicBlock *BBDom = DT->getNode(BB)->getIDom()->getBlock();

    // NewBB's dominator is either BB's dominator or BB's dominator's clone.
    BasicBlock *NewBBDom = BBDom;
    Map::iterator BBDomI = map.find(BBDom);
    if (BBDomI != map.end()) {
        NewBBDom = cast<BasicBlock>(BBDomI->second);
        if (!DT->getNode(NewBBDom))
        cloneDominatorInfo(BBDom, map, DT);
    }
    DT->addNewBlock(NewBB, NewBBDom);
}

// Map management ---------------------------------------------------------
/*void Util::applyMap(Instruction *Inst, CoarseningMap &map, unsigned int CF) {
  for (unsigned op = 0, opE = Inst->getNumOperands(); op != opE; ++op) {
    Instruction *Op = dyn_cast<Instruction>(Inst->getOperand(op));
    CoarseningMap::iterator It = map.find(Op);

    if (It != map.end()) {
      InstVector &instVector = It->second;
      Value *NewValue = instVector.at(CF);
      Inst->setOperand(op, NewValue);
    }
  }
}*/

void Util::applyMap(Instruction *Inst, Map& map) {
    for (unsigned op = 0, opE = Inst->getNumOperands(); op != opE; ++op) {
        Value *Op = Inst->getOperand(op);

        Map::const_iterator It = map.find(Op);
        if (It != map.end())
        Inst->setOperand(op, It->second);
    }

    if (PHINode *Phi = dyn_cast<PHINode>(Inst))
        Util::applyMapToPhiBlocks(Phi, map);
}

void Util::applyMap(BasicBlock *block, Map &map) {
    for (auto iter = block->begin(), end = block->end(); iter != end; ++iter)
        Util::applyMap(&*iter, map);
}

void Util::applyMapToPHIs(BasicBlock *block, Map &map) {
    for (auto phi = block->begin(); isa<PHINode>(phi); ++phi)
        Util::applyMap(&*phi, map);
}

void Util::applyMapToPhiBlocks(PHINode *Phi, Map &map) {
    for (unsigned int index = 0; index < Phi->getNumIncomingValues(); ++index) {
        BasicBlock *OldBlock = Phi->getIncomingBlock(index);
        Map::const_iterator It = map.find(OldBlock);

        if (It != map.end()) {
            // I am not proud of this.
            BasicBlock *NewBlock =
                const_cast<BasicBlock *>(cast<BasicBlock>(It->second));
            Phi->setIncomingBlock(index, NewBlock);
        }
    }
}

void Util::applyMap(InstVector &insts, Map &map, InstVector &result) {
    result.clear();
    result.reserve(insts.size());

    for (auto inst : insts) {
        Value *newValue = map[inst];
        if (newValue != nullptr) {
            if (Instruction *newInst = dyn_cast<Instruction>(newValue)) {
                result.push_back(newInst);
            }
        }
    }
}

// ============================================================================
// Function annotation helper functions, taken from NVPTX back-end
// ============================================================================
typedef std::map<std::string, std::vector<unsigned> > key_val_pair_t;
typedef std::map<const GlobalValue *, key_val_pair_t> global_val_annot_t;
typedef std::map<const Module *, global_val_annot_t> per_module_annot_t;

static ManagedStatic<per_module_annot_t> annotationCache;
static sys::Mutex Lock;

void clearAnnotationCache(const Module *Mod) 
{
    MutexGuard Guard(Lock);
    annotationCache->erase(Mod);
}

static void cacheAnnotationFromMD(const MDNode *md, key_val_pair_t& retval)
{
    MutexGuard Guard(Lock);
    assert(md && "Invalid mdnode for annotation");
    assert((md->getNumOperands() % 2) == 1 && "Invalid number of operands");
    // start index = 1, to skip the global variable key
    // increment = 2, to skip the value for each property-value pairs
    for (unsigned i = 1, e = md->getNumOperands(); i != e; i += 2) {
        // property
        const MDString *prop = dyn_cast<MDString>(md->getOperand(i));
        assert(prop && "Annotation property not a string");
  
        // value
        ConstantInt *Val = mdconst::dyn_extract<ConstantInt>(
                                                         md->getOperand(i + 1));
        assert(Val && "Value operand not a constant int");
  
        std::string keyname = prop->getString().str();
        if (retval.find(keyname) != retval.end()) {
            retval[keyname].push_back(Val->getZExtValue());
        }
        else {
            std::vector<unsigned> tmp;
            tmp.push_back(Val->getZExtValue());
            retval[keyname] = tmp;
        }
    }
}

static void cacheAnnotationFromMD(const Module *m, const GlobalValue *gv) 
{
    MutexGuard Guard(Lock);
    NamedMDNode *NMD = m->getNamedMetadata("nvvm.annotations");
    if (!NMD)
        return;
    key_val_pair_t tmp;
    for (unsigned i = 0, e = NMD->getNumOperands(); i != e; ++i) {
        const MDNode *elem = NMD->getOperand(i);
  
        GlobalValue *entity =
            mdconst::dyn_extract_or_null<GlobalValue>(elem->getOperand(0));
        // entity may be null due to DCE
        if (!entity)
            continue;
        if (entity != gv)
            continue;
  
        // accumulate annotations for entity in tmp
        cacheAnnotationFromMD(elem, tmp);
    }
  
    if (tmp.empty()) // no annotations for this gv
        return;
  
    if ((*annotationCache).find(m) != (*annotationCache).end()) {
        (*annotationCache)[m][gv] = std::move(tmp);
    }
    else {
      global_val_annot_t tmp1;
      tmp1[gv] = std::move(tmp);
      (*annotationCache)[m] = std::move(tmp1);
    }
}

bool findOneNVVMAnnotation(const GlobalValue *gv, const std::string &prop,
                           unsigned &retval) 
{
    MutexGuard Guard(Lock);
    const Module *m = gv->getParent();
    if ((*annotationCache).find(m) == (*annotationCache).end())
        cacheAnnotationFromMD(m, gv);
    else if ((*annotationCache)[m].find(gv) == (*annotationCache)[m].end())
        cacheAnnotationFromMD(m, gv);
    if ((*annotationCache)[m][gv].find(prop) == (*annotationCache)[m][gv].end())
        return false;
    retval = (*annotationCache)[m][gv][prop][0];
    return true;
}