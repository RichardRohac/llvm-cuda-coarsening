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
#include "llvm/IR/IRBuilder.h"

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

cl::opt<std::string> CLCoarseningDimension("coarsening-dimension",
                                           cl::init("y"),
                                           cl::Hidden,
                                           cl::desc("Coarsening dimension"));

cl::opt<std::string> CLCoarseningMode("coarsening-mode",
                                      cl::init("thread"),
                                      cl::Hidden,
                                      cl::desc("Coarsening mode (thread/block)"));

using namespace llvm;

char CUDACoarseningPass::ID = 0;

// CREATORS
CUDACoarseningPass::CUDACoarseningPass()
: ModulePass(ID)
{

}

bool CUDACoarseningPass::runOnModule(Module& M)
{
    // Parse command line configuration
    m_kernelName = CLKernelName;
    if (m_kernelName == "") {
        // As the kernel name was not specified, generate runtime dispatcher
        // TODO
    }

    m_blockLevel = CLCoarseningMode == "block";
    m_factor = CLCoarseningFactor;
    m_stride = CLCoarseningStride;
    m_dimX = CLCoarseningDimension.find('x') != std::string::npos;
    m_dimY = CLCoarseningDimension.find('y') != std::string::npos;
    m_dimZ = CLCoarseningDimension.find('z') != std::string::npos;

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

    LLVMContext& ctx = M.getContext();
    Constant *rpcScaleDim = M.getOrInsertFunction(
        "rpcScaleDim",
        Type::getVoidTy(ctx),
        // *XY, *Z, factor, x, y, z
        Type::getInt64PtrTy(ctx),
        Type::getInt32PtrTy(ctx),
        Type::getInt32Ty(ctx),
        Type::getInt8Ty(ctx),
        Type::getInt8Ty(ctx),
        Type::getInt8Ty(ctx)
    );

    Function *rpcScaleDimF = cast<Function>(rpcScaleDim);
    rpcScaleDimF->setCallingConv(CallingConv::C);

    Function::arg_iterator args = rpcScaleDimF->arg_begin();
    Value *xy = args++;
    xy->setName("xy");
    Value *z = args++;
    z->setName("z");
    Value *factor = args++;
    factor->setName("factor");
    Value *dimX = args++;
    dimX->setName("dimX");
    Value *dimY = args++;
    dimY->setName("dimY");
    Value *dimZ = args++;
    dimZ->setName("dimZ");

    BasicBlock* block = BasicBlock::Create(ctx, "entry", rpcScaleDimF);
    BasicBlock *blockX = BasicBlock::Create(ctx, "blockX", rpcScaleDimF);
    IRBuilder<> builder(block);

    Value *enableX = builder.CreateICmpEQ(dimX, builder.getInt8(1));
    BasicBlock *blockXEnd = BasicBlock::Create(ctx, "blockXEnd", rpcScaleDimF);
    Value *bX = builder.CreateCondBr(enableX, blockX, blockXEnd);

    builder.SetInsertPoint(blockX);
   // Value *castX = builder.CreateBitCast(xy, Type::getInt32PtrTy(ctx), "x");
    Value *loadX = builder.CreateLoad(xy, builder.getInt64Ty(), "x_val");
    Value *factorCast = builder.CreateIntCast(factor, builder.getInt64Ty(), false);
    Value *divX = builder.CreateUDiv(loadX, factorCast, "scaled_x");
    //Value *castDivX = builder.CreateIntCast(divX, Type::getInt64Ty(ctx), false, "cast_div_x");
    Value *savedX = builder.CreateAlignedStore(divX, xy, 4);
    builder.CreateBr(blockXEnd);
    
    builder.SetInsertPoint(blockXEnd);
    ReturnInst *ret = builder.CreateRetVoid();

    //Value *loadX = builder.CreateLoad(builder.getInt32Ty(), xy, "x");
    //Value *divX = builder.CreateUDiv()

    for (Function& F : M) {
        for (BasicBlock& B: F) {
            for (Instruction& I : B) {
                Instruction *pI = &I;
                if (CallInst *callInst = dyn_cast<CallInst>(pI)) {
                    Function *calledF = callInst->getCalledFunction();

                    if (calledF->getName() == CUDA_RUNTIME_LAUNCH) {
                        // cudaLaunch receives function pointer as an argument.
                        Constant *castPtr = 
                                        cast<Constant>(callInst->getOperand(0));
                        Function *kernelF =
                                         cast<Function>(castPtr->getOperand(0));
                        std::string kernel = kernelF->getName();
                        
                        kernel = demangle(kernel);
                        kernel = kernel.substr(0, kernel.find_first_of('('));

                        if (kernel != m_kernelName) {
                            continue;
                        }

                        errs() << "Found cudaLaunch of " << kernel << "\n";
                        foundGrid = true;

                        // Call to cudaLaunch is preceded by "numArgs()" of
                        // blocks, where the very first one is referenced by
                        // the unconditional branch instruction that checks
                        // for valid configuration (call to cudaConfigureCall).
                        BasicBlock *configOKBlock = &B;
                        for (unsigned int i = 0;
                             i < kernelF->arg_size();
                             ++i) {
                                 configOKBlock = configOKBlock->getPrevNode();
                        }

                        // FIXED!
                        // Depending on the optimization level, we might be
                        // in a _kernelname_() function call.
                        std::string pn = demangle(
                                         configOKBlock->getParent()->getName());
                        pn = pn.substr(0, pn.find_first_of('('));

                        if (pn == kernel) {
                            for (Function& xF : M) {
                                for (BasicBlock& xB: xF) {
                                    for (Instruction& xI : xB) {
                                        Instruction *pxI = &xI;
                                        if (CallInst *callInst = dyn_cast<CallInst>(pxI)) {
                                            Function *calledxF = callInst->getCalledFunction();
                                            if (calledxF == configOKBlock->getParent()) {
                                                amendConfiguration(M, callInst->getParent());
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        else {
                            amendConfiguration(M, configOKBlock);
                        }
                    }
                }
            }
        }
    }

     //   errs() << "Found invokation to: " << kernelInvokation->getCalledFunction()->getName() << "\n";

        // Function consists of basic blocks, which in turn consist of
        // instructions.
        // for (BasicBlock& B : F) {
        //     for (Instruction& I : B) {
        //         Instruction *pI = &I;
        //         if (CallInst *callInst = dyn_cast<CallInst>(pI)) {
        //             Function *calledF = callInst->getCalledFunction();

        //             if (calledF->getName() == CUDA_RUNTIME_CONFIGURECALL) {
        //                 foundGrid = true;



        //                 Instruction *resultCheck = callInst->getNextNode();
        //                 assert(isa<ICmpInst>(resultCheck) &&
        //                        "Result comparison instruction expected!");
        //                 Instruction *branchI = resultCheck->getNextNode();
        //                 assert(isa<BranchInst>(branchI) &&
        //                        "Branch instruction expected!");
        //                 BranchInst *branch = dyn_cast<BranchInst>(branchI);
        //                 assert(branch->getNumOperands() == 3);

        //                 Value *pathOK = branch->getOperand(1);
        //                 assert(isa<BasicBlock>(pathOK) &&
        //                        "Expected basic block!");
                        
        //                 bool foundKernelName = false;
        //                 BasicBlock *blockOK = dyn_cast<BasicBlock>(pathOK);
        //                 for(Instruction& curI : *blockOK) {
        //                     if (isa<CallInst>(&curI)) {
        //                         CallInst *pcsf = dyn_cast<CallInst>(&curI); 
        //                         std::string kernel =
        //                          demangle(pcsf->getCalledFunction()->getName());
        //                         foundKernelName = true;

        //                         // HACKZ HACKZ HACKZ HACKZ HACKZ HACKZ HACKZ
        //                         kernel =
        //                             kernel.substr(0, kernel.find_first_of('('));

        //                         if (kernel == CLKernelName) {
        //                             scaleGrid(&B, callInst, rpcScaleDim);
        //                         }
        //                     }
        //                 }

        //                 assert(foundKernelName && "Kernel name not found!");                      
        //             }
        //         }
        //     }
    //}

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

void CUDACoarseningPass::scaleGrid(BasicBlock *configBlock,
                                   CallInst   *configCall,
                                   Constant   *scaleFunc)
{
    // operand #0 -> load ... gridDim (64b (x, y))
    // operand #1 -> load ... gridDim (32b (z))
    // operand #2 -> load ... blockDim (64b (x, y))
    // operand #3 -> load ... blockDim (32b (z))

    if (m_blockLevel) {
        // blockLevel(); TODO
        return;
    }

    // Thread level coarsening scaling
    // TODO non pow2
    LoadInst *loadXY = dyn_cast<LoadInst>(configCall->getOperand(2));
    LoadInst *loadZ = dyn_cast<LoadInst>(configCall->getOperand(3));
    IRBuilder<> builder(loadZ);
    builder.SetInsertPoint(configBlock, ++builder.GetInsertPoint());
    Value* args[] = {loadXY->getPointerOperand(),
                     loadZ->getPointerOperand(),
                     builder.getInt32(m_factor),
                     builder.getInt8(m_dimX),
                     builder.getInt8(m_dimY),
                     builder.getInt8(m_dimZ)};
    CallInst *scaleCall = builder.CreateCall(scaleFunc, args);
    loadXY->moveAfter(scaleCall);
    loadZ->moveAfter(loadXY);
}

void CUDACoarseningPass::amendConfiguration(Module&     M,
                                            BasicBlock *configOKBlock)
{
    if (configOKBlock == nullptr) {
        assert(0 && "Found cudaLaunch without corresponding config block!");
    }

    // Find branch instruction jumping to the "configOK" block.
    // This instruction is located within a block that handles
    // cudaConfigureCall().
    BasicBlock *configBlock = nullptr;
    for (Function& F : M) {
        for (BasicBlock& B: F) {
            for (Instruction& I : B) {
                if (isa<BranchInst>(&I)) {
                    BranchInst *bI = cast<BranchInst>(&I);
                    if (bI->getNumOperands() == 3) {
                        if(isa<BasicBlock>(bI->getOperand(1))) {
                            BasicBlock *targetBlock =
                                            cast<BasicBlock>(bI->getOperand(1));

                            if (targetBlock == configOKBlock) {
                                configBlock = bI->getParent();
                            }
                        }
                        
                        if(isa<BasicBlock>(bI->getOperand(2))) {
                            BasicBlock *targetBlock =
                                            cast<BasicBlock>(bI->getOperand(2));

                            if (targetBlock == configOKBlock) {
                                configBlock = bI->getParent();
                            }
                        }
                    }
                }
            }
        }
    }

    assert(configBlock != nullptr);

    for (Instruction& I : *configBlock) {
        Instruction *pI = &I;
        if (CallInst *callInst = dyn_cast<CallInst>(pI)) {
            Function *calledF = callInst->getCalledFunction();

            if (calledF->getName() == CUDA_RUNTIME_CONFIGURECALL) {
                scaleGrid(configBlock, callInst, rpcScaleDim);
            }
        } 
    }
}

static RegisterPass<CUDACoarseningPass> X("cuda-coarsening-pass",
                                          "CUDA Coarsening Pass",
                                          false, // Only looks at CFG,
                                          false // Analysis pass
                                          );