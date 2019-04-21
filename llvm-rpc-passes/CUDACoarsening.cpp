// ============================================================================
// Copyright (c) Richard Rohac, 2019, All rights reserved.
// ============================================================================
// CUDA Coarsening Transformation pass
// -> Based on Alberto's Magni OpenCL coarsening pass algorithm
//    available at https://github.com/HariSeldon/coarsening_pass
// ============================================================================

#include "llvm/Pass.h"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/CallSite.h"

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PostDominators.h"

#include "Common.h"
#include "CUDACoarsening.h"
#include "Util.h"
#include "DivergenceAnalysisPass.h"

#include <cxxabi.h>

// https://gcc.gnu.org/onlinedocs/libstdc++/manual/ext_demangling.html
inline std::string demangle(std::string mangledName)
{
    int status = -1;

    std::unique_ptr<char, decltype(std::free) *> result{
        abi::__cxa_demangle(mangledName.c_str(), NULL, NULL, &status),
        std::free
    };

    return (status == 0) ? result.get() : mangledName;
}

// Command line parameters
cl::opt<std::string> CLKernelName("coarsened-kernel",
                                  cl::init(""),
                                  cl::Hidden,
                                  cl::desc("Name of the kernel to coarsen"));

cl::opt<unsigned int> CLCoarseningFactor("coarsening-factor",
                                         cl::init(1),
                                         cl::Hidden,
                                         cl::desc("Coarsening factor"));

cl::opt<unsigned int> CLCoarseningStride("coarsening-stride",
                                         cl::init(1),
                                         cl::Hidden,
                                         cl::desc("Coarsening stride"));

cl::opt<int> CLCoarseningDirection("coarsening-direction",
                                   cl::init(0),
                                   cl::Hidden,
                                   cl::desc("Coarsening direction"));

using namespace llvm;

char CUDACoarseningPass::ID = 0;

// CREATORS
CUDACoarseningPass::CUDACoarseningPass()
: ModulePass(ID)
{

}

bool CUDACoarseningPass::runOnModule(Module& M)
{
    errs() << "\nInvoked CUDA COARSENING PASS (MODULE LEVEL) "
           << "on module: " << M.getName()
           << " -- kernel: " << CLKernelName << " " << CLCoarseningFactor
           << "x" << " with stride " << CLCoarseningStride << "\n";

    bool result = false;

    if (M.getTargetTriple() == CUDA_TARGET_TRIPLE) {
        // -----------------------------------------------------------------
        // Device code gets extended with coarsened versions of the kernels.
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

void CUDACoarseningPass::getAnalysisUsage(AnalysisUsage& AU) const
{
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequired<DivergenceAnalysisPass>();
    AU.addRequired<PostDominatorTreeWrapperPass>();
    AU.addRequired<DominatorTreeWrapperPass>();
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

            std::string name = demangle(F.getName());
            errs() << "--  INFO  -- Found CUDA kernel: " << name << "\n";

            analyzeKernel(F);
        }
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
                       // std::string name = 
                      //      demangle(callInst->getCalledFunction()->getgetName());
                     //   errs() << name;
                     //   callInst->print(errs());
                      //  errs() << "\n";                        
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
    m_divergenceAnalysis = &getAnalysis<DivergenceAnalysisPass>(F);
}

static RegisterPass<CUDACoarseningPass> X("cuda-coarsening-pass",
                                          "CUDA Coarsening Pass",
                                          false, // Only looks at CFG,
                                          false // Analysis pass
                                          );