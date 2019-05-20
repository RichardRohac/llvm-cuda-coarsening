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

#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/ScalarEvolution.h"

#include "Common.h"
#include "CUDACoarsening.h"
#include "Util.h"
#include "DivergenceAnalysisPass.h"
#include "GridAnalysisPass.h"
#include "BenefitAnalysisPass.h"

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
                                           cl::init("x"),
                                           cl::Hidden,
                                           cl::desc("Coarsening dimension"));

cl::opt<std::string> CLCoarseningMode(
                            "coarsening-mode",
                            cl::init("block"),
                            cl::Hidden,
                            cl::desc("Coarsening mode (thread/block/dynamic)"));

using namespace llvm;

char CUDACoarseningPass::ID = 0;

// CREATORS
CUDACoarseningPass::CUDACoarseningPass()
: ModulePass(ID)
{
}

bool CUDACoarseningPass::runOnModule(Module& M)
{
    if (!parseConfig()) {
        return false;
    }

    bool result = false;

    if (M.getTargetTriple() == CUDA_TARGET_TRIPLE) {
        // -----------------------------------------------------------------
        // Device code gets extended with coarsened versions of the kernels.
        // For example:
        // -----------------------------------------------------------------
        // XYZ -> XYZ_1_2_<stride> XYZ_1_4_<stride> XYZ_1_8_<stride> ...
        //        XYZ_2_1_1
        //        XYZ_4_1_1
        //        XYZ_8_1_1
        //        ...
        // -----------------------------------------------------------------
        // Where the numbering in the kernel names is defined as follows:
        // <block_factor>_<thread_factor>_<stride_factor>
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

    errs() << "--  INFO  -- End of CUDA coarsening pass!" << "\n\n";

    return result;
}

void CUDACoarseningPass::getAnalysisUsage(AnalysisUsage& AU) const
{
    AU.addRequired<DivergenceAnchorPass>();
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequired<PostDominatorTreeWrapperPass>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<GridAnalysisPass>();
    AU.addRequired<DivergenceAnalysisPassTL>();
    AU.addRequired<DivergenceAnalysisPassBL>();
    AU.addRequired<BenefitAnalysisPass>();
}

bool CUDACoarseningPass::parseConfig()
{
    // Parse command line configuration
    m_kernelName = CLKernelName;
    if (m_kernelName.empty()) {
        errs() << "CUDA Coarsening Pass Error: no kernel specified "
               << "(parameter: coarsened-kernel)\n";
        
        return false;
    }

    m_dynamicMode = false;
    m_blockLevel = false;
    
    if (CLCoarseningMode == "dynamic") {
        m_dynamicMode = true;
    }
    else if (CLCoarseningMode == "block") {
        m_blockLevel = true;
    }
    else if (CLCoarseningMode != "thread") {
        errs() << "CUDA Coarsening Pass Error: wrong coarsening mode specified "
               << "(parameter: coarsening-mode)\n";
    }

    if (!m_dynamicMode) {
        m_factor = CLCoarseningFactor;
        m_stride = CLCoarseningStride;
    }

    if (!(CLCoarseningDimension == "x" ||
        CLCoarseningDimension == "y" ||
        CLCoarseningDimension == "z" )) {
        errs() << "CUDA Coarsening Pass Error: unknown dimension specified "
            << "(parameter: coarsening-dimension)\n";
    }

    m_dimension = Util::numeralDimension(CLCoarseningDimension);

    errs() << "\nCUDA Coarsening Pass configuration:";
    errs() << " kernel: " << CLKernelName;
    errs() << ", dimension: " << CLCoarseningDimension;
    errs() << ", mode: " << CLCoarseningMode;
    if (!m_dynamicMode) {
        errs() << CLCoarseningFactor << "x";
        errs() << ", (stride: " << CLCoarseningStride << ")";
    }
    errs() << "\n";

    return true;
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

            std::string name = Util::demangle(F.getName());
            name = name.substr(0, name.find_first_of('('));

            if (name != m_kernelName) {
                continue;
            }

            errs() << "--  INFO  -- Found CUDA kernel: " << name << "\n";

            analyzeKernel(F);

            if (m_dynamicMode) {
                generateVersions(F, true);
                continue;
            }

            scaleKernelGrid();
            coarsenKernel();
            replacePlaceholders();
        }
    }

    return foundKernel;
}

bool CUDACoarseningPass::handleHostCode(Module& M)
{
    errs() << "--  INFO  -- Running on host code" << "\n";

    bool foundGrid = false;

    m_cudaConfigureCallScaled = nullptr;
    m_cudaLaunchDynamic = nullptr;
    insertCudaConfigureCallScaled(M);
    insertCudaLaunchDynamic(M);

    std::vector<CallInst*> toRemove;

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
                        
                        kernel = Util::demangle(kernel);
                        kernel = kernel.substr(0, kernel.find_first_of('('));

                        if (kernel != m_kernelName) {
                            continue;
                        }

                        errs() << "--  INFO  -- Found cudaLaunch of " << kernel << "\n";
                        foundGrid = true;

                        if (m_dynamicMode) {
                            callInst->setCalledFunction(m_cudaLaunchDynamic);
                        }

                        BasicBlock *configOKBlock = &B;

                        #ifndef CUDA_USES_NEW_LAUNCH
                        // Call to cudaLaunch is preceded by "numArgs()" of
                        // blocks, where the very first one is referenced by
                        // the unconditional branch instruction that checks
                        // for valid configuration (call to cudaConfigureCall).
                        for (unsigned int i = 0;
                             i < kernelF->arg_size();
                             ++i) {
                                 configOKBlock = configOKBlock->getPrevNode();
                        }
                        #endif

                        // FIXED!
                        // Depending on the optimization level, we might be
                        // in a _kernelname_() function call.
                        std::string pn = Util::demangle(
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
                                                CallInst *rem = amendConfiguration(M, callInst->getParent());
                                                assert(rem != nullptr);
                                                toRemove.push_back(rem);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        else {
                            CallInst *rem = amendConfiguration(M, configOKBlock);
                            assert(rem != nullptr);
                            toRemove.push_back(rem);
                        }
                    }
                }
            }
        }
    }

    if (m_dynamicMode) {
        for (Function& F : M) {
            std::string funcName = Util::demangle(F.getName());
            funcName = funcName.substr(0, funcName.find_first_of('('));
            if (funcName == m_kernelName) {
                generateVersions(F, false);
                break;
            }
        }
    }

    for(CallInst *rem : toRemove) {
        rem->eraseFromParent();
    }

    if (!foundGrid) {
        if (m_cudaConfigureCallScaled != nullptr) {
            m_cudaConfigureCallScaled->eraseFromParent();
        }

        if (m_cudaLaunchDynamic != nullptr) {
            m_cudaLaunchDynamic->eraseFromParent();
        }
    }

    return foundGrid;
}

void CUDACoarseningPass::generateVersions(Function& F, bool deviceCode)
{
    std::vector<unsigned int> factors = {2, 4, 8, 16};
    std::vector<unsigned int> strides = {1, 2, 4, 8, 16, 32, 64};

    CallInst *cudaRegFuncCall = nullptr;

    for (Function& xF : *F.getParent()) {
        for (BasicBlock& B : xF) {
          for (Instruction& I : B) {
            Instruction *pI = &I;
            if (CallInst *callInst = dyn_cast<CallInst>(pI)) {
              Function *calledF = callInst->getCalledFunction();

              if (calledF->getName() == CUDA_REGISTER_FUNC) {
                Constant *castPtr = cast<Constant>(callInst->getOperand(1));
                Function *stubF = cast<Function>(castPtr->getOperand(0));
                if (stubF->getName() == F.getName()) {
                  cudaRegFuncCall = callInst;
                  break;
                }
              }
            }
          }
        }
    }

    if(!deviceCode) {
      assert(cudaRegFuncCall != nullptr &&
        "Missing CUDA fatbinary registration routine!");
    }

    for (auto factor : factors) {
        for (auto stride : strides) {
            generateVersion(F,
                            deviceCode,
                            factor,
                            stride,
                            false,
                            cudaRegFuncCall);
        }
        generateVersion(F, deviceCode, factor, 1, true, cudaRegFuncCall);
    }
}

void CUDACoarseningPass::generateVersion(Function&     F,
                                         bool          deviceCode,
                                         unsigned int  factor,
                                         unsigned int  stride,
                                         bool          blockMode,
                                         CallInst     *cudaRegFuncCall)
{
    LLVMContext& ctx = F.getContext();

    llvm::ValueToValueMapTy vMap;
    Function *cloned = llvm::CloneFunction(&F, vMap);
    std::string kn = namedKernelVersion(F.getName(),
                                        blockMode ? factor : 1,
                                        blockMode ? 1 : factor,
                                        stride);
    cloned->setName(kn);

    if (!deviceCode) {
        GEPOperator *origGEP = 
                    dyn_cast<GEPOperator>(cudaRegFuncCall->getOperand(2));
        llvm::GlobalVariable *origGKN =
                        dyn_cast<GlobalVariable>(origGEP->getOperand(0));

        StringRef knWithNull(kn.c_str(), kn.size() + 1);

        llvm::Constant *ckn = 
                     llvm::ConstantDataArray::getString(ctx, knWithNull, false);

        llvm::GlobalVariable *gkn = new llvm::GlobalVariable(
                                           *F.getParent(),
                                           ckn->getType(),
                                           true,
                                           llvm::GlobalVariable::PrivateLinkage,
                                           ckn);

        gkn->setAlignment(origGKN->getAlignment());
        gkn->setUnnamedAddr(origGKN->getUnnamedAddr());

        CallInst *newRegCall =
                                   dyn_cast<CallInst>(cudaRegFuncCall->clone());
        newRegCall->setCalledFunction(m_rpcRegisterFunction);
        CastInst *ptrCast = CastInst::CreatePointerCast(
                                    cloned,
                                    Type::getInt8PtrTy(F.getContext()),
                                    "");
        ptrCast->insertAfter(cudaRegFuncCall);

        SmallVector<Value *, 8> idx = {
            ConstantInt::get(Type::getInt64Ty(ctx), 0),
            ConstantInt::get(Type::getInt64Ty(ctx), 0)
        };

        GetElementPtrInst *gep = GetElementPtrInst::CreateInBounds(
                gkn,
                idx,
                "",
                ptrCast);

        newRegCall->setOperand(1, ptrCast);
        newRegCall->setOperand(2, gep);
        newRegCall->setOperand(3, gep);

        newRegCall->insertAfter(ptrCast);

        // Host code consists of stub functions only, no coarsening
        // is required there.
        return;
    }
    
    unsigned int savedFactor = m_factor;
    unsigned int savedStride = m_stride;
    bool savedBlockLevel = m_blockLevel;

    m_factor = factor;
    m_stride = stride;
    m_blockLevel = blockMode;

    analyzeKernel(*cloned);
    scaleKernelGrid();
    coarsenKernel();
    replacePlaceholders();

    SmallVector<Metadata *, 3> operandsMD;
    operandsMD.push_back(llvm::ValueAsMetadata::getConstant(cloned));
    operandsMD.push_back(llvm::MDString::get(F.getContext(), "kernel"));
    operandsMD.push_back(llvm::ValueAsMetadata::getConstant(
            llvm::ConstantInt::get(llvm::Type::getInt32Ty(F.getContext()),
                                                        1)));

    llvm::NamedMDNode *nvvmMetadataNode =
            F.getParent()->getOrInsertNamedMetadata("nvvm.annotations");
    
    nvvmMetadataNode->addOperand(MDTuple::get(F.getContext(),
                                    operandsMD));

    m_factor = savedFactor;
    m_stride = savedStride;
    m_blockLevel = savedBlockLevel;
}

std::string CUDACoarseningPass::namedKernelVersion(std::string kernel,
                                                   int b, int t, int s)
{
    // Generate <kernel>_<blockfactor>_<threadfactor>_<stride> name
    // TODO other mangling schemes
    // C code?

    std::string demangled = Util::demangle(kernel);
    demangled = demangled.substr(0, demangled.find_first_of('('));

    std::string suffix = "_";
    suffix.append(std::to_string(b));
    suffix.append("_");
    suffix.append(std::to_string(t));
    suffix.append("_");
    suffix.append(std::to_string(s));

    std::string name = "_Z";
    name.append(std::to_string(demangled.length() + suffix.length()));
   // std::string name = "rpc_";
    name.append(demangled);
    name.append(suffix);

    size_t pos = kernel.find(demangled) + demangled.length();
    size_t len = kernel.length() - pos;
    name.append(kernel.substr(pos, len));

    return name;
}

void CUDACoarseningPass::analyzeKernel(Function& F)
{
    m_coarseningMap.clear();
    m_phMap.clear();
    m_phReplacementMap.clear();

    // Perform initial analysis.
    //m_benefitAnalysis = &getAnalysis<BenefitAnalysisPass>(F);
    m_loopInfo = &getAnalysis<LoopInfoWrapperPass>(F).getLoopInfo();
    m_postDomT = &getAnalysis<PostDominatorTreeWrapperPass>(F).getPostDomTree();
    m_domT = &getAnalysis<DominatorTreeWrapperPass>(F).getDomTree();
    m_divergenceAnalysisTL = &getAnalysis<DivergenceAnalysisPassTL>(F);
    m_divergenceAnalysisBL = &getAnalysis<DivergenceAnalysisPassBL>(F);
    m_gridAnalysis = &getAnalysis<GridAnalysisPass>(F);
}

void CUDACoarseningPass::scaleGrid(BasicBlock *configBlock,
                                   CallInst   *configCall)
{
    IRBuilder<> builder(configCall);
    SmallVector<Value *, 12> args(configCall->arg_begin(),
                                  configCall->arg_end());

    // When running dynamic mode, the runtime takes care of
    // coarsening factor scaling.
    if (!m_dynamicMode) {
        uint8_t scaleGrid[CUDA_MAX_DIM];
        uint8_t scaleBlock[CUDA_MAX_DIM];

        scaleGrid[0] = ((m_dimension == 0) && m_blockLevel) ? m_factor : 1;
        scaleGrid[1] = ((m_dimension == 1) && m_blockLevel) ? m_factor : 1;
        scaleGrid[2] = ((m_dimension == 2) && m_blockLevel) ? m_factor : 1;

        scaleBlock[0] = ((m_dimension == 0) && !m_blockLevel) ? m_factor : 1;
        scaleBlock[1] = ((m_dimension == 1) && !m_blockLevel) ? m_factor : 1;
        scaleBlock[2] = ((m_dimension == 2) && !m_blockLevel) ? m_factor : 1;

        args.push_back(builder.getInt8(scaleGrid[0])); // scale grid X
        args.push_back(builder.getInt8(scaleGrid[1])); // scale grid Y
        args.push_back(builder.getInt8(scaleGrid[2])); // scale grid Z
        args.push_back(builder.getInt8(scaleBlock[0])); // scale block X
        args.push_back(builder.getInt8(scaleBlock[1])); // scale block Y
        args.push_back(builder.getInt8(scaleBlock[2])); // scale block Z
    }

    CallInst *newCall = builder.CreateCall(m_cudaConfigureCallScaled, args);
    newCall->setCallingConv(m_cudaConfigureCallScaled->getCallingConv());
    if (!configCall->use_empty()) {
        configCall->replaceAllUsesWith(newCall);
    }
}

CallInst *CUDACoarseningPass::amendConfiguration(Module&     M,
                                                 BasicBlock *configOKBlock)
{
    CallInst *ret = nullptr;

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
                scaleGrid(configBlock, callInst);
                ret = callInst;
            }
        } 
    }

    return ret;
}

AllocaInst *CreateAllocaA(IRBuilder<> *b, Type *Ty, Value *ArraySize = nullptr,
                         const Twine &Name = "", unsigned int align = 8) {
        const DataLayout &DL = b->GetInsertBlock()->getParent()->getParent()->getDataLayout();
return b->Insert(new AllocaInst(Ty, DL.getAllocaAddrSpace(), ArraySize, align), Name);
}

void CUDACoarseningPass::insertCudaConfigureCallScaled(Module& M)
{
    LLVMContext& ctx = M.getContext();

    Function *original = M.getFunction(CUDA_RUNTIME_CONFIGURECALL);
    assert(original != nullptr);

    FunctionType *origFT = original->getFunctionType();

    Function *ptrF;

    // In case of dynamic mode, we use configuration function provided
    // externally.
    if (m_dynamicMode) {
        FunctionCallee scaled = M.getOrInsertFunction(
            "cudaConfigureCallScaled",
            Type::getInt32Ty(ctx),   // return
            Type::getInt64Ty(ctx),   // gridXY
            Type::getInt32Ty(ctx),   // gridZ
            Type::getInt64Ty(ctx),   // blockXY
            Type::getInt32Ty(ctx),   // blockZ
            origFT->getParamType(4), // size
            origFT->getParamType(5)  // ptrStream
        );

        ptrF = cast<Function>(scaled.getCallee());
        ptrF->setCallingConv(original->getCallingConv());

        m_cudaConfigureCallScaled = ptrF;

        return;
    }

    SmallVector<Type *, 16> scaledArgs;
    for (auto& arg : origFT->params()) {
        scaledArgs.push_back(arg);
    }

    assert(original->arg_size() == 6 && "This ABI is not supported yet!");

    scaledArgs.push_back(Type::getInt8Ty(ctx));
    scaledArgs.push_back(Type::getInt8Ty(ctx));
    scaledArgs.push_back(Type::getInt8Ty(ctx));
    scaledArgs.push_back(Type::getInt8Ty(ctx));
    scaledArgs.push_back(Type::getInt8Ty(ctx));
    scaledArgs.push_back(Type::getInt8Ty(ctx));

    FunctionCallee scaled = M.getOrInsertFunction(
        "cudaConfigureCallScaled",
        FunctionType::get(original->getReturnType(), scaledArgs, false)
    );

    ptrF = cast<Function>(scaled.getCallee());
    ptrF->setCallingConv(original->getCallingConv());

    Function::arg_iterator argIt = ptrF->arg_begin();
    Value *gridXY = argIt++; gridXY->setName("gridXY");
    Value *gridZ = argIt++; gridZ->setName("gridZ");
    Value *blockXY = argIt++; blockXY->setName("blockXY");
    Value *blockZ = argIt++; blockZ->setName("blockZ");
    Value *sharedMem = argIt++; sharedMem->setName("sharedMem");
    Value *cudaStream = argIt++; cudaStream->setName("cudaStream");
    Value *scaleGridX = argIt++; scaleGridX->setName("sgX");
    Value *scaleGridY = argIt++; scaleGridY->setName("sgY");
    Value *scaleGridZ = argIt++; scaleGridZ->setName("sgZ");
    Value *scaleBlockX = argIt++; scaleBlockX->setName("sbX");
    Value *scaleBlockY = argIt++; scaleBlockY->setName("sbY");
    Value *scaleBlockZ = argIt++; scaleBlockZ->setName("sbZ");

    BasicBlock* block = BasicBlock::Create(ctx, "entry", ptrF);
    IRBuilder<> builder(block);

    // Allocate space for amended parameters
    AllocaInst *sgXY =
              CreateAllocaA(&builder, builder.getInt64Ty(), nullptr, "sgXY", 8);
    AllocaInst *sgZ =
               CreateAllocaA(&builder, builder.getInt32Ty(), nullptr, "sgZ", 8);
    AllocaInst *sbXY =
              CreateAllocaA(&builder, builder.getInt64Ty(), nullptr, "sbXY", 8);
    AllocaInst *sbZ =
               CreateAllocaA(&builder, builder.getInt32Ty(), nullptr, "sbZ", 8);
    AllocaInst *ssm =
            CreateAllocaA(&builder, origFT->getParamType(4), nullptr, "ssm", 8);
    AllocaInst *scs =
            CreateAllocaA(&builder, origFT->getParamType(5), nullptr, "scs", 8);

    builder.CreateAlignedStore(gridXY, sgXY, 8, false);
    builder.CreateAlignedStore(gridZ, sgZ, 8, false);
    builder.CreateAlignedStore(blockXY, sbXY, 8, false);
    builder.CreateAlignedStore(blockZ, sbZ, 8, false);
    builder.CreateAlignedStore(sharedMem, ssm, 8, false);
    builder.CreateAlignedStore(cudaStream, scs, 8, false);

    // Scale grid X
    Value *ptrGridX = builder.CreatePointerCast(sgXY, Type::getInt32PtrTy(ctx));
    ptrGridX = builder.CreateInBoundsGEP(ptrGridX,
                                         ConstantInt::get(builder.getInt64Ty(),
                                                          0));
    Value *valGridX = builder.CreateAlignedLoad(ptrGridX, 4);
    Value *valScaledGridX = 
        builder.CreateUDiv(valGridX,
                           builder.CreateIntCast(scaleGridX,
                                                 builder.getInt32Ty(),
                                                 false));
    builder.CreateAlignedStore(valScaledGridX, ptrGridX, 4, false);

    // Scale grid Y
    Value *ptrGridY = builder.CreatePointerCast(sgXY, Type::getInt32PtrTy(ctx));
    ptrGridY = builder.CreateInBoundsGEP(ptrGridY,
                                         ConstantInt::get(builder.getInt64Ty(),
                                                          1));
    Value *valGridY = builder.CreateAlignedLoad(ptrGridY, 4);
    Value *valScaledGridY = 
        builder.CreateUDiv(valGridY,
                           builder.CreateIntCast(scaleGridY,
                                                 builder.getInt32Ty(),
                                                 false));
    builder.CreateAlignedStore(valScaledGridY, ptrGridY, 4, false);

    // Scale grid Z
    Value *valGridZ = builder.CreateAlignedLoad(sgZ, 8);
    Value *valScaledGridZ = 
        builder.CreateUDiv(valGridZ,
                           builder.CreateIntCast(scaleGridZ,
                                                 builder.getInt32Ty(),
                                                 false));
    builder.CreateAlignedStore(valScaledGridZ, sgZ, 8, false);

    // Scale BLOCK X
    Value *ptrBlockX = builder.CreatePointerCast(sbXY, Type::getInt32PtrTy(ctx));
    ptrBlockX = builder.CreateInBoundsGEP(ptrBlockX,
                                          ConstantInt::get(builder.getInt64Ty(),
                                                           0));
    Value *valBlockX = builder.CreateAlignedLoad(ptrBlockX, 4);
    Value *valScaledBlockX = 
        builder.CreateUDiv(valBlockX,
                           builder.CreateIntCast(scaleBlockX,
                                                 builder.getInt32Ty(),
                                                 false));
    builder.CreateAlignedStore(valScaledBlockX, ptrBlockX, 4, false);

    // Scale BLOCK Y
    Value *ptrBlockY = builder.CreatePointerCast(sbXY, Type::getInt32PtrTy(ctx));
    ptrBlockY = builder.CreateInBoundsGEP(ptrBlockY,
                                          ConstantInt::get(builder.getInt64Ty(),
                                                           1));
    Value *valBlockY = builder.CreateAlignedLoad(ptrBlockY, 4);
    Value *valScaledBlockY = 
        builder.CreateUDiv(valBlockY,
                           builder.CreateIntCast(scaleBlockY,
                                                 builder.getInt32Ty(),
                                                 false));
    builder.CreateAlignedStore(valScaledBlockY, ptrBlockY, 4, false);

    // Scale BLOCK Z
    Value *valBlockZ = builder.CreateAlignedLoad(sbZ, 8);
    Value *valScaledBlockZ = 
        builder.CreateUDiv(valBlockZ,
                           builder.CreateIntCast(scaleBlockZ,
                                                 builder.getInt32Ty(),
                                                 false));
    builder.CreateAlignedStore(valScaledBlockZ, sbZ, 8, false);

    Value *c_sgXY = builder.CreateAlignedLoad(sgXY, 8, "c_sgXY");
    Value *c_sgZ = builder.CreateAlignedLoad(sgZ, 8, "c_sgZ");
    Value *c_sbXY = builder.CreateAlignedLoad(sbXY, 8, "c_sbXY");
    Value *c_sbZ = builder.CreateAlignedLoad(sbZ, 8, "c_sbZ");
    Value *c_ssm = builder.CreateAlignedLoad(ssm, 8, "c_ssm");
    Value *c_scs = builder.CreateAlignedLoad(scs, 8, "c_scs"); 

    SmallVector<Value *, 6> callArgs;
    callArgs.push_back(c_sgXY); callArgs.push_back(c_sgZ);
    callArgs.push_back(c_sbXY); callArgs.push_back(c_sbZ);
    callArgs.push_back(c_ssm); callArgs.push_back(c_scs);

    CallInst *cudaCall = builder.CreateCall(original, callArgs);

    builder.CreateRet(cudaCall);

    m_cudaConfigureCallScaled = ptrF;
}

void CUDACoarseningPass::insertCudaLaunchDynamic(Module& M)
{
    LLVMContext& ctx = M.getContext();

    Function *original = M.getFunction(CUDA_RUNTIME_LAUNCH);
    assert(original != nullptr);

    FunctionType *origFT = original->getFunctionType();

    Function *ptrF;

    if (m_dynamicMode) {
        FunctionCallee dynLaunch = M.getOrInsertFunction(
            "cudaLaunchDynamic",
            origFT->getReturnType(),
            origFT->getParamType(0), // ptrKernel
            origFT->getParamType(1), // gridXY
            origFT->getParamType(2), // gridZ
            origFT->getParamType(3), // blockXY
            origFT->getParamType(4), // blockZ
            origFT->getParamType(5), // ptrptrArgs
            origFT->getParamType(6), // size
            origFT->getParamType(7)  // ptrStream
        );

        ptrF = cast<Function>(dynLaunch.getCallee());
        ptrF->setCallingConv(original->getCallingConv());

        m_cudaLaunchDynamic = ptrF;

        FunctionCallee registerFunction = M.getOrInsertFunction(
            "rpcRegisterFunction",
            Type::getInt32Ty(ctx),
            Type::getInt8PtrTy(ctx)->getPointerTo(),
            Type::getInt8PtrTy(ctx),
            Type::getInt8PtrTy(ctx),
            Type::getInt8PtrTy(ctx),
            Type::getInt32Ty(ctx),
            Type::getInt8PtrTy(ctx),
            Type::getInt8PtrTy(ctx),
            Type::getInt8PtrTy(ctx),
            Type::getInt8PtrTy(ctx),
            Type::getInt32PtrTy(ctx)
        );

        ptrF = cast<Function>(registerFunction.getCallee());
        //ptrF->setCallingConv(M.getFunction(CUDA_REGISTER_FUNC)->getCallingConv());
        //ptrF->setDSOLocal(M.getFunction(CUDA_REGISTER_FUNC)->isDSOLocal());

        m_rpcRegisterFunction = ptrF;

        return;
    } 
}

static RegisterPass<CUDACoarseningPass> X("cuda-coarsening-pass",
                                          "CUDA Coarsening Pass",
                                          false, // Only looks at CFG,
                                          false // Analysis pass
                                          );