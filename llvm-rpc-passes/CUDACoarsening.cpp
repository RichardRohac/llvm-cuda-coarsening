// ============================================================================
// Copyright (c) Richard Rohac, 2019, All rights reserved.
// ============================================================================
// CUDA Coarsening Transformation pass
// -> Based on Alberto's Magni OpenCL coarsening pass algorithm
//    available at https://github.com/HariSeldon/coarsening_pass
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

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/CallSite.h"

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PostDominators.h"

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

    bool result = false;

    if (M.getTargetTriple() == CUDA_TARGET_TRIPLE) {
        // -----------------------------------------------------------------
        // Device code gets extended with coarsened versions of the kernels
        // For example:
        // -----------------------------------------------------------------
        // kernelXYZ -> kernelXYZ_1x_2x kernelXYZ_1x_4x kernelXYZ_1x_8x ...
        //              kernelXYZ_2x_1x
        //              kernelXYZ_4x_1x
        //              kernelXYZ_8x_1x
        //              ...
        // -----------------------------------------------------------------
        // Where the numbering in the kernel names is defined as follows:
        // <block_level_coarsening_factor>_<thread_level_coarsening_factor>
        // -----------------------------------------------------------------
        result = handleDeviceCode(M);
    }
    else {
        // -----------------------------------------------------------------
        // Host code gets either extended with a dispatcher function
        // to support more versions of coarsened grids, or, for optimization
        // purposes, specific one can be selected as well.
        // -----------------------------------------------------------------
        result = handleHostCode(M);
    }
    errs() << "End of CUDA coarsening pass!" << "\n\n";

    return result;
}

void CUDACoarseningPass::getAnalysisUsage(AnalysisUsage& Info) const
{
    Info.addRequired<LoopInfoWrapperPass>();
    Info.addRequired<PostDominatorTreeWrapperPass>();
    Info.addRequired<DominatorTreeWrapperPass>();
}

bool CUDACoarseningPass::handleDeviceCode(Module& M)
{
    errs() << "--  INFO  -- Running on device code" << "\n";

    const llvm::NamedMDNode *nvmmAnnot = M.getNamedMetadata("nvvm.annotations");
    if (!nvmmAnnot) {
        errs() << "--  STOP  -- Missing nvvm.annotations in this module.\n";
        return false;
    }

    bool foundKernel = false;
    for (auto& F : M) {
        if (Util::isKernelFunction(F) && !F.isDeclaration()) {
            foundKernel = true;
            errs() << "--  INFO  -- Found CUDA kernel: " << F.getName() << "\n";

            analyzeKernel(F);
        }

        // ThreadLevel *threadLevel = &getAnalysis<ThreadLevel>(F);
    }

    return foundKernel;
}

bool CUDACoarseningPass::handleHostCode(Module& M)
{
    errs() << "--  INFO  -- Running on host code" << "\n";

    bool foundGrid = false;

    // In case of the host code, look for "cudaLaunch" invocations
    for (Function& F : M) {
        // Function consists of basic blocks, which in turn consist of
        // instructions.
        for (BasicBlock& B : F) {
            for (Instruction& I : B) {
                Instruction *pI = &I;
                if (CallInst *callInst = dyn_cast<CallInst>(pI)) {
                    Function *calledF = callInst->getCalledFunction();

                    if (calledF->getName() == CUDA_RUNTIME_LAUNCH) {
                        foundGrid = true;
                        errs() << callInst->getCalledFunction()->getName();
                        callInst->print(errs());
                        errs() << "\n";                        
                    }
                }
            }
        }
    }

    return foundGrid;
}

void CUDACoarseningPass::analyzeKernel(Function& F)
{
    // Perform initial analysis.
    m_loopInfo = &getAnalysis<LoopInfoWrapperPass>(F).getLoopInfo();
    m_postDomT = &getAnalysis<PostDominatorTreeWrapperPass>(F).getPostDomTree();
    m_domT = &getAnalysis<DominatorTreeWrapperPass>(F).getDomTree();
}

} // end anonymous namespace

static RegisterPass<CUDACoarseningPass> X("cuda-coarsening-pass",
                                          "CUDA Coarsening Pass",
                                          false, // Only looks at CFG,
                                          false // Analysis pass
                                          );


//}
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