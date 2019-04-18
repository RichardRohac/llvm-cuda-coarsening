// ============================================================================
// Copyright (c) Richard Rohac, 2019, All rights reserved.
// ============================================================================
// CUDA Coarsening Transformation pass
// ============================================================================

/* #include "llvm/ADT/Statistic.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MutexGuard.h"
#include "llvm/IR/Constants.h" */

//#include "ThreadLevel.h"

#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"

#include "CUDACoarsening.h"
#include "Util.h"

using namespace llvm;

namespace {

char CUDACoarseningPass::ID = 0;

// CREATORS
CUDACoarseningPass::CUDACoarseningPass()
: ModulePass(ID)
{

}

bool CUDACoarseningPass::runOnModule(Module& M)
{
    errs() << "\nInvoked CUDA COARSENING PASS (MODULE LEVEL) ";
    errs() << "on module: " << M.getName() << "\n";

    const llvm::NamedMDNode *kernels = M.getNamedMetadata("nvvm.annotations");
    if (!kernels) {
        errs() << "--! STOP !-- No CUDA kernels in this module.\n";
        return false;
    }

    for (auto& F : M) {
        if (Util::isKernelFunction(F)) {
            errs() << "Found CUDA kernel: " << F.getName() << "\n";
        }

        // ThreadLevel *threadLevel = &getAnalysis<ThreadLevel>(F);
    }

    errs() << std::endl;

    return true;
}

void CUDACoarseningPass::getAnalysisUsage(AnalysisUsage& Info) const
{
}

} // end anonymous namespace

static RegisterPass<CUDACoarseningPass> X("cuda-coarsening-pass",
                                          "CUDA Coarsening Pass",
                                          false, // Only looks at CFG,
                                          false // Analysis pass
                                          );


}
/*
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
                   registerCUDACoarseningPass);*/