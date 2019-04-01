// ============================================================================
// Copyright (c) Richard Rohac, 2019, All rights reserved.
// ============================================================================
// CUDA Coarsening Transformation pass
// ============================================================================

#include "llvm/ADT/Statistic.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MutexGuard.h"
#include "llvm/IR/Constants.h"

#include "ThreadLevel.h"

using namespace llvm;

#define DEBUG_TYPE "CUDACoarsening"

namespace {
typedef std::map<std::string, std::vector<unsigned> > key_val_pair_t;
typedef std::map<const GlobalValue *, key_val_pair_t> global_val_annot_t;
typedef std::map<const Module *, global_val_annot_t> per_module_annot_t;
} // anonymous namespace

static ManagedStatic<per_module_annot_t> annotationCache;
static sys::Mutex Lock;

void clearAnnotationCache(const Module *Mod) {
  MutexGuard Guard(Lock);
  annotationCache->erase(Mod);
}

static void cacheAnnotationFromMD(const MDNode *md, key_val_pair_t &retval) {
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
    ConstantInt *Val = mdconst::dyn_extract<ConstantInt>(md->getOperand(i + 1));
    assert(Val && "Value operand not a constant int");

    std::string keyname = prop->getString().str();
    if (retval.find(keyname) != retval.end())
      retval[keyname].push_back(Val->getZExtValue());
    else {
      std::vector<unsigned> tmp;
      tmp.push_back(Val->getZExtValue());
      retval[keyname] = tmp;
    }
  }
}

static void cacheAnnotationFromMD(const Module *m, const GlobalValue *gv) {
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

  if ((*annotationCache).find(m) != (*annotationCache).end())
    (*annotationCache)[m][gv] = std::move(tmp);
  else {
    global_val_annot_t tmp1;
    tmp1[gv] = std::move(tmp);
    (*annotationCache)[m] = std::move(tmp1);
  }
}

bool findOneNVVMAnnotation(const GlobalValue *gv, const std::string &prop,
                           unsigned &retval) {
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

bool isKernelFunction(const Function &F) {
  unsigned x = 0;
  bool retval = findOneNVVMAnnotation(&F, "kernel", x);
  if (!retval) {
    // There is no NVVM metadata, check the calling convention
    return F.getCallingConv() == CallingConv::PTX_Kernel;
  }
  return (x == 1);
}

struct CUDACoarsening : public ModulePass {
    static char ID;

    ThreadLevel *m_threadLevel;

    CUDACoarsening() : ModulePass(ID) {}

    bool runOnModule(Module &M) override {
        
        errs() << "\nInvoked CUDA COARSENING PASS (MODULE LEVEL) ";
        errs() << "in module called: " << M.getName() << "\n";

        const llvm::NamedMDNode *kernels = 
                    M.getNamedMetadata("nvvm.annotations");

        if (!kernels) {
            errs() << "--! STOP !-- No CUDA kernels in this module.\n";
            return false;
        }


        for (auto &F : M) {
            if (isKernelFunction(F)) {
                errs() << "Found CUDA kernel: " << F.getName() << "\n";
            }

          // ThreadLevel *threadLevel = &getAnalysis<ThreadLevel>(F);
        }
        
        return true;
    }

    void getAnalysisUsage(AnalysisUsage &Info) const {
        Info.addRequired<ThreadLevel>();
        Info.addPreserved<ThreadLevel>();
    }
};

char CUDACoarsening::ID = 0;
static RegisterPass<CUDACoarsening> X("cuda-coarsening", "CUDA Coarsening Pass", false, false);

static void registerCUDACoarseningPass(const PassManagerBuilder &,
                                       legacy::PassManagerBase  &PM) {
    PM.add(new DivergenceInfo());
    PM.add(new ThreadLevel());
    PM.add(new CUDACoarsening());
}
static RegisterStandardPasses RegisterMyPass(
                   PassManagerBuilder::EP_ModuleOptimizerEarly,
                   registerCUDACoarseningPass);

static RegisterStandardPasses RegisterMyPass0(
                   PassManagerBuilder::EP_EnabledOnOptLevel0,
                   registerCUDACoarseningPass);