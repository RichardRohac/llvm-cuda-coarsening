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

// IR helpers -----------------------------------------------------------------
AllocaInst *CreateAlignedAlloca(llvm::Module&      M,
                                llvm::IRBuilder<> *builder,
                                llvm::Type        *type,
                                unsigned int       alignment,
                                const Twine&       name = "",
                                llvm::Value       *ArraySize = nullptr)
{
    const DataLayout &DL = M.getDataLayout();
    return builder->Insert(
            new AllocaInst(type, DL.getAllocaAddrSpace(), ArraySize, alignment),
            name);
}

char CUDACoarseningPass::ID = 0;

// CREATORS
CUDACoarseningPass::CUDACoarseningPass()
: ModulePass(ID)
{
}

bool CUDACoarseningPass::runOnModule(Module& M)
{
    m_coarsenedKernelMap.clear();

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
   // AU.addRequired<DivergenceAnchorPass>();
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequired<PostDominatorTreeWrapperPass>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<GridAnalysisPass>();
    AU.addRequired<DivergenceAnalysisPassTL>();
    AU.addRequired<DivergenceAnalysisPassBL>();
    //AU.addRequired<BenefitAnalysisPass>();
}

bool CUDACoarseningPass::parseConfig()
{
    // Parse command line configuration
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

    m_kernelName = CLKernelName;
    if (m_kernelName.empty() && !m_dynamicMode) {
        errs() << "CUDA Coarsening Pass Error: no kernel specified "
               << "(parameter: coarsened-kernel)\n";
        
        return false;
    }

    if (!(CLCoarseningDimension == "x" ||
          CLCoarseningDimension == "y" ||
          CLCoarseningDimension == "z" )) {
        errs() << "CUDA Coarsening Pass Error: unknown dimension specified "
                << "(parameter: coarsening-dimension)\n";
    }

    if (!m_dynamicMode) {
        // In regular mode, configuration parameters need to be set.
        m_factor = CLCoarseningFactor;
        m_stride = CLCoarseningStride;
        m_dimension = Util::numeralDimension(CLCoarseningDimension);
    }

    errs() << "\nCUDA Coarsening Pass configuration:";
    errs() << " kernel: " << (CLKernelName.empty() ? "<all>" : m_kernelName);
    errs() << ", mode: " << CLCoarseningMode << " ";
    if (!m_dynamicMode) {
        errs() << CLCoarseningFactor << "x";
        errs() << ", (stride: " << CLCoarseningStride;
        errs() << ", dimension: " << CLCoarseningDimension << ")";
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
        if (shouldCoarsen(F)) {
            std::string name = Util::demangle(F.getName());
            name = Util::nameFromDemangled(name);

            foundKernel = true;

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

    insertRPCFunctions(M);

    // We are replacing function call instructions; this array will hold the
    // original functions calls which will get removed from the IR.
    std::vector<CallInst *> forRemoval;

    for (Function& F : M) {
        for (BasicBlock& B: F) {
            for (Instruction& I : B) {
                Instruction *pI = &I;

                // Find cudaLaunchKernel call instruction.
                if (CallInst *callInst = dyn_cast<CallInst>(pI)) {
                    Function *calledF = callInst->getCalledFunction();
                    if (!calledF) {
                        // Indirect function invocation, should not matter.
                        continue;
                    }

                    if (calledF->getName() != CUDA_RUNTIME_LAUNCH) {
                        // Not a call to cudaLaunchKernel.
                        continue;
                    }

                    // cudaLaunchKernel receives function pointer as first
                    // parameter. This function pointer can be used to identify
                    // if this is the launch of the kernel we want to coarsen.
                    Constant *cptr = dyn_cast<Constant>(callInst->getOperand(0));
                    if (!cptr) {
                        // TODO investigate this - sometimes this is of Value
                        // type, but seems that for uninteresting calls for us.
                        continue;
                    }
                    Function *kernelF = dyn_cast<Function>(cptr->getOperand(0));

                    std::string kernel = Util::demangle(kernelF->getName());
                    kernel = Util::nameFromDemangled(kernel);

                    if (!shouldCoarsen(*kernelF, true)) {
                        continue;
                    }

                    errs() << "--  INFO  -- Found cudaLaunch of " << kernel;
                    errs() << "\n";
                    foundGrid = true;

                    if (m_dynamicMode) {
                        // In dynamic mode, replace the launch call with
                        // the dispatcher function.
                        callInst->setCalledFunction(m_rpcLaunchKernel);
                        continue;
                    }

                    scaleGrid(callInst->getParent(),
                              callInst,
                              kernelF->getName());

                    forRemoval.push_back(callInst);
                }
            }
        }
    }

    for(CallInst *rem : forRemoval) {
        // Remove original calls to cudaLaunchKernel. These are now replaced
        // with our version.
        rem->eraseFromParent();
    }

    if (!foundGrid) {
        // If no kernel invocation was found, remove the previously inserted
        // helper functions.
        deleteRPCFunctions(M);
    }

    if (m_dynamicMode) {
        for (Function& F : M) {
            if (shouldCoarsen(F, true)) {
                generateVersions(F, false);
            }
        }
    }

    return foundGrid;
}

void CUDACoarseningPass::generateVersions(Function& F, bool deviceCode)
{
    std::vector<unsigned int> factors = {2, 4, 8, 16};
    std::vector<unsigned int> strides = {1, 2, 32};
    std::vector<unsigned int> dimensions = {0, 1};

    CallInst *cudaRegFuncCall = cudaRegistrationCallForKernel(*F.getParent(),
                                                              F.getName());
    if(!deviceCode) {
        if (!cudaRegFuncCall) {
            return;
        }
    }

    for (auto dimension : dimensions) {
        for (auto factor : factors) {
            for (auto stride : strides) {
                generateVersion(F,
                                deviceCode,
                                factor,
                                stride,
                                dimension,
                                false, // Thread-level.
                                cudaRegFuncCall);
            }
            generateVersion(F,
                            deviceCode,
                            factor,
                            1,         // Stride in block-level mode is ignored.
                            dimension,
                            true,      // Block-level.
                            cudaRegFuncCall);
        }
    }
}

void CUDACoarseningPass::generateVersion(Function&     F,
                                         bool          deviceCode,
                                         unsigned int  factor,
                                         unsigned int  stride,
                                         unsigned int  dimension,
                                         bool          blockMode,
                                         CallInst     *cudaRegFuncCall)
{
    LLVMContext& ctx = F.getContext();

    llvm::ValueToValueMapTy vMap;
    Function *cloned = llvm::CloneFunction(&F, vMap);
    std::string kn = namedKernelVersion(F.getName(),
                                        dimension,
                                        blockMode ? factor : 1,
                                        blockMode ? 1 : factor,
                                        stride);
    cloned->setName(kn);
    m_coarsenedKernelMap[cloned] = true;

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

        CallInst *newRegCall = dyn_cast<CallInst>(cudaRegFuncCall->clone());
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

        newRegCall->setOperand(3, newRegCall->getOperand(1));
        newRegCall->setOperand(1, ptrCast);
        newRegCall->setOperand(2, gep);

        newRegCall->insertAfter(ptrCast);

        // Host code consists of stub functions only, no coarsening
        // is required there.
        return;
    }
    
    unsigned int savedFactor = m_factor;
    unsigned int savedStride = m_stride;
    unsigned int savedDimension = m_dimension;
    bool savedBlockLevel = m_blockLevel;

    m_factor = factor;
    m_stride = stride;
    m_blockLevel = blockMode;
    m_dimension = dimension;

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
    m_dimension = savedDimension;
}

std::string CUDACoarseningPass::namedKernelVersion(std::string kernel,
                                                   int d, int b, int t, int s)
{
    // Generate <kernel>_<dimension>_<blockfactor>_<threadfactor>_<stride> name
    // TODO other mangling schemes
    // C code?

    std::string demangled = Util::demangle(kernel);
    demangled = Util::nameFromDemangled(demangled);

    std::string suffix = "_";
    suffix.append(std::to_string(d));
    suffix.append("_");
    suffix.append(std::to_string(b));
    suffix.append("_");
    suffix.append(std::to_string(t));
    suffix.append("_");
    suffix.append(std::to_string(s));

    std::string name = "_Z";
    name.append(std::to_string(demangled.length() + suffix.length()));
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
                                   CallInst   *configCall,
                                   std::string kernelName)
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
    else {
        llvm::CallInst *regFunc = cudaRegistrationCallForKernel(
                             *configCall->getParent()->getParent()->getParent(),
                             kernelName);
        GEPOperator *origGEP = dyn_cast<GEPOperator>(regFunc->getOperand(2));
        llvm::GlobalVariable *origGKN = dyn_cast<GlobalVariable>(
                                                        origGEP->getOperand(0));

        SmallVector<Value *, 8> idx = {
            ConstantInt::get(Type::getInt64Ty(configCall->getContext()), 0),
            ConstantInt::get(Type::getInt64Ty(configCall->getContext()), 0)
        };

        GetElementPtrInst *gep = GetElementPtrInst::CreateInBounds(origGKN,
                                                                   idx,
                                                                   "",
                                                                   configCall);

        args.insert(args.begin(), gep);
    }

    CallInst *newCall = builder.CreateCall(m_rpcLaunchKernel, args);
    newCall->setCallingConv(m_rpcLaunchKernel->getCallingConv());
    if (!configCall->use_empty()) {
        configCall->replaceAllUsesWith(newCall);
    }
}

void CUDACoarseningPass::insertRPCFunctions(Module& M)
{
    m_rpcLaunchKernel = nullptr;
    m_rpcRegisterFunction = nullptr;

    insertRPCLaunchKernel(M);
    if (m_dynamicMode) {
        insertRPCRegisterFunction(M);
    }
}

void CUDACoarseningPass::deleteRPCFunctions(Module& M)
{
    if (m_rpcLaunchKernel) {
        m_rpcLaunchKernel->eraseFromParent();
    }

    if (m_rpcRegisterFunction) {
        m_rpcRegisterFunction->eraseFromParent();
    }
}

void CUDACoarseningPass::insertRPCLaunchKernel(Module& M)
{
    LLVMContext& ctx = M.getContext();

    Function *original = M.getFunction(CUDA_RUNTIME_LAUNCH);
    assert(original != nullptr);

    FunctionType *origFT = original->getFunctionType();

    Function *ptrF;

    // In the dynamic mode we use configuration function provided
    // externally.
    if (m_dynamicMode) {
        FunctionCallee scaled = M.getOrInsertFunction(
            "rpcLaunchKernel",
            Type::getInt32Ty(ctx),   // return type
            Type::getInt8PtrTy(ctx), // deviceFun
            Type::getInt64Ty(ctx),   // gridXY
            Type::getInt32Ty(ctx),   // gridZ
            Type::getInt64Ty(ctx),   // blockXY
            Type::getInt32Ty(ctx),   // blockZ
            origFT->getParamType(5), // args
            origFT->getParamType(6), // sharedMemory
            origFT->getParamType(7)  // cudaStream
        );

        ptrF = cast<Function>(scaled.getCallee());
        ptrF->setCallingConv(original->getCallingConv());

        m_rpcLaunchKernel = ptrF;

        return;
    }

    // Copy the original argument types.
    SmallVector<Type *, 16> scaledArgs;
    for (auto& arg : origFT->params()) {
        scaledArgs.push_back(arg);
    }

    assert(original->arg_size() == 8 && "This ABI is not supported yet!");

    // Append grid and block scale arguments' types.
    scaledArgs.push_back(Type::getInt8Ty(ctx));
    scaledArgs.push_back(Type::getInt8Ty(ctx));
    scaledArgs.push_back(Type::getInt8Ty(ctx));
    scaledArgs.push_back(Type::getInt8Ty(ctx));
    scaledArgs.push_back(Type::getInt8Ty(ctx));
    scaledArgs.push_back(Type::getInt8Ty(ctx));

    // Insert the function prototype.
    FunctionCallee scaled = M.getOrInsertFunction(
        "rpcLaunchKernel",
        FunctionType::get(original->getReturnType(), scaledArgs, false)
    );

    ptrF = cast<Function>(scaled.getCallee());
    ptrF->setCallingConv(original->getCallingConv());

    // Name the function arguments.
    Function::arg_iterator argIt = ptrF->arg_begin();
    Value *argFuncPtr = argIt++; argFuncPtr->setName("funcPtr");
    Value *argGridXY = argIt++; argGridXY->setName("gridXY");
    Value *argGridZ = argIt++; argGridZ->setName("gridZ");
    Value *argBlockXY = argIt++; argBlockXY->setName("blockXY");
    Value *argBlockZ = argIt++; argBlockZ->setName("blockZ");
    Value *argArgs = argIt++; argArgs->setName("args");
    Value *argSharedMem = argIt++; argSharedMem->setName("sharedMem");
    Value *argCudaStream = argIt++; argCudaStream->setName("cudaStream");
    // New arguments:
    Value *argScaleGridX = argIt++; argScaleGridX->setName("scaleGridX");
    Value *argScaleGridY = argIt++; argScaleGridY->setName("scaleGridY");
    Value *argScaleGridZ = argIt++; argScaleGridZ->setName("scaleGridZ");
    Value *argScaleBlockX = argIt++; argScaleBlockX->setName("scaleBlockX");
    Value *argScaleBlockY = argIt++; argScaleBlockY->setName("scaleBlockY");
    Value *argScaleBlockZ = argIt++; argScaleBlockZ->setName("scaleBlockZ");

    // Build the function body.
    BasicBlock* block = BasicBlock::Create(ctx, "entry", ptrF);
    IRBuilder<> builder(block);

    // Allocate space for function parameters.
    AllocaInst *localFuncPtr =
        CreateAlignedAlloca(M, &builder, builder.getInt8PtrTy(), 8, "l_ptr");
    AllocaInst *localGridXY =
        CreateAlignedAlloca(M, &builder, builder.getInt64Ty(), 8, "l_gXY");
    AllocaInst *localGridZ =
        CreateAlignedAlloca(M, &builder, builder.getInt32Ty(), 8, "l_gZ");
    AllocaInst *localBlockXY =
        CreateAlignedAlloca(M, &builder, builder.getInt64Ty(), 8, "l_bXY");
    AllocaInst *localBlockZ =
        CreateAlignedAlloca(M, &builder, builder.getInt32Ty(), 8, "l_bZ");
    AllocaInst *localArgs =
        CreateAlignedAlloca(M, &builder, origFT->getParamType(5), 8, "l_args");
    AllocaInst *localSharedMemory =
        CreateAlignedAlloca(M, &builder, origFT->getParamType(6), 8, "l_sm");
    AllocaInst *localCudaStream =
        CreateAlignedAlloca(M, &builder, origFT->getParamType(7), 8, "l_st");

    // Store the local variables.
    builder.CreateAlignedStore(argFuncPtr, localFuncPtr, 8, false);
    builder.CreateAlignedStore(argGridXY, localGridXY, 8, false);
    builder.CreateAlignedStore(argGridZ, localGridZ, 8, false);
    builder.CreateAlignedStore(argBlockXY, localBlockXY, 8, false);
    builder.CreateAlignedStore(argBlockZ, localBlockZ, 8, false);
    builder.CreateAlignedStore(argArgs, localArgs, 8, false);
    builder.CreateAlignedStore(argSharedMem, localSharedMemory, 8, false);
    builder.CreateAlignedStore(argCudaStream, localCudaStream, 8, false);

    // Scale grid X
    Value *ptrGridX = builder.CreatePointerCast(localGridXY,
                                                Type::getInt32PtrTy(ctx));
    ptrGridX = builder.CreateInBoundsGEP(ptrGridX,
                                         ConstantInt::get(builder.getInt64Ty(),
                                                          0));
    Value *valGridX = builder.CreateAlignedLoad(ptrGridX, 4);
    Value *valScaledGridX = 
        builder.CreateUDiv(valGridX,
                           builder.CreateIntCast(argScaleGridX,
                                                 builder.getInt32Ty(),
                                                 false));
    builder.CreateAlignedStore(valScaledGridX, ptrGridX, 4, false);

    // Scale grid Y
    Value *ptrGridY = builder.CreatePointerCast(localGridXY,
                                                Type::getInt32PtrTy(ctx));
    ptrGridY = builder.CreateInBoundsGEP(ptrGridY,
                                         ConstantInt::get(builder.getInt64Ty(),
                                                          1));
    Value *valGridY = builder.CreateAlignedLoad(ptrGridY, 4);
    Value *valScaledGridY = 
        builder.CreateUDiv(valGridY,
                           builder.CreateIntCast(argScaleGridY,
                                                 builder.getInt32Ty(),
                                                 false));
    builder.CreateAlignedStore(valScaledGridY, ptrGridY, 4, false);

    // Scale grid Z
    Value *valGridZ = builder.CreateAlignedLoad(localGridZ, 8);
    Value *valScaledGridZ = 
        builder.CreateUDiv(valGridZ,
                           builder.CreateIntCast(argScaleGridZ,
                                                 builder.getInt32Ty(),
                                                 false));
    builder.CreateAlignedStore(valScaledGridZ, localGridZ, 8, false);

    // Scale BLOCK X
    Value *ptrBlockX = builder.CreatePointerCast(localBlockXY,
                                                 Type::getInt32PtrTy(ctx));
    ptrBlockX = builder.CreateInBoundsGEP(ptrBlockX,
                                          ConstantInt::get(builder.getInt64Ty(),
                                                           0));
    Value *valBlockX = builder.CreateAlignedLoad(ptrBlockX, 4);
    Value *valScaledBlockX = 
        builder.CreateUDiv(valBlockX,
                           builder.CreateIntCast(argScaleBlockX,
                                                 builder.getInt32Ty(),
                                                 false));
    builder.CreateAlignedStore(valScaledBlockX, ptrBlockX, 4, false);

    // Scale BLOCK Y
    Value *ptrBlockY = builder.CreatePointerCast(localBlockXY,
                                                 Type::getInt32PtrTy(ctx));
    ptrBlockY = builder.CreateInBoundsGEP(ptrBlockY,
                                          ConstantInt::get(builder.getInt64Ty(),
                                                           1));
    Value *valBlockY = builder.CreateAlignedLoad(ptrBlockY, 4);
    Value *valScaledBlockY = 
        builder.CreateUDiv(valBlockY,
                           builder.CreateIntCast(argScaleBlockY,
                                                 builder.getInt32Ty(),
                                                 false));
    builder.CreateAlignedStore(valScaledBlockY, ptrBlockY, 4, false);

    // Scale BLOCK Z
    Value *valBlockZ = builder.CreateAlignedLoad(localBlockZ, 8);
    Value *valScaledBlockZ = 
        builder.CreateUDiv(valBlockZ,
                           builder.CreateIntCast(argScaleBlockZ,
                                                 builder.getInt32Ty(),
                                                 false));
    builder.CreateAlignedStore(valScaledBlockZ, localBlockZ, 8, false);

    Value *c_localPtr = builder.CreateAlignedLoad(localFuncPtr, 8, "c_ptr");
    Value *c_localGridXY = builder.CreateAlignedLoad(localGridXY, 8, "c_gXY");
    Value *c_logalGridZ = builder.CreateAlignedLoad(localGridZ, 8, "c_gZ");
    Value *c_localBlockXY = builder.CreateAlignedLoad(localBlockXY, 8, "c_bXY");
    Value *c_localBlockZ = builder.CreateAlignedLoad(localBlockZ, 8, "c_bZ");
    Value *c_localArgs = builder.CreateAlignedLoad(localArgs, 8, "c_args");
    Value *c_localSharedMemory =
                        builder.CreateAlignedLoad(localSharedMemory, 8, "c_sm");
    Value *c_localCudaStream =
                         builder.CreateAlignedLoad(localCudaStream, 8, "c_scs"); 

    SmallVector<Value *, 8> callArgs;
    callArgs.push_back(c_localPtr);
    callArgs.push_back(c_localGridXY); callArgs.push_back(c_logalGridZ);
    callArgs.push_back(c_localBlockXY); callArgs.push_back(c_localBlockZ);
    callArgs.push_back(c_localArgs);
    callArgs.push_back(c_localSharedMemory);
    callArgs.push_back(c_localCudaStream);

    CallInst *cudaCall = builder.CreateCall(original, callArgs);
    builder.CreateRet(cudaCall);
    m_rpcLaunchKernel = ptrF;
}

void CUDACoarseningPass::insertRPCRegisterFunction(Module& M)
{
    LLVMContext& ctx = M.getContext();

    Function *original = M.getFunction(CUDA_RUNTIME_LAUNCH);
    assert(original != nullptr);;

    Function *ptrF;

    if (m_dynamicMode) {
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

// PRIVATE ACCESSORS
bool CUDACoarseningPass::shouldCoarsen(Function& F, bool hostCode) const
{
    if (m_coarsenedKernelMap.find(&F) != m_coarsenedKernelMap.end()) {
        // Function was already coarsened.
        return false;
    }

    if (hostCode) {
        CallInst *cudaRegFuncCall = cudaRegistrationCallForKernel(*F.getParent(),
                                                                  F.getName());
        if (!cudaRegFuncCall) {
            // We can only coarsen host code functions that were originally
            // registered.
            return false;
        }

        if (m_kernelName == "all" && m_dynamicMode) {
            // In dynamic mode we can coarsen all
            // the available kernels.
            return true;
        }
    }

    return Util::shouldCoarsen(F, m_kernelName, hostCode, m_dynamicMode);
}

CallInst *
CUDACoarseningPass::cudaRegistrationCallForKernel(Module&     M,
                                                  std::string kernelName) const
{
    for (Function& F : M) {
        for (BasicBlock& B : F) {
            for (Instruction& I : B) {
                Instruction *pI = &I;
                if (CallInst *callInst = dyn_cast<CallInst>(pI)) {
                    Function *calledF = callInst->getCalledFunction();
                    if (!calledF) {
                        // Indirect function invocation, skip.
                        continue;
                    }

                    if (calledF->getName() == CUDA_REGISTER_FUNC) {
                        Constant *castPtr = cast<Constant>(callInst->getOperand(1));
                        Function *stubF = cast<Function>(castPtr->getOperand(0));

                        if (stubF->getName() == kernelName) {
                            return callInst;
                        }
                    }
                }
            }
        }
    }

    return nullptr;
}

static RegisterPass<CUDACoarseningPass> X("cuda-coarsening-pass",
                                          "CUDA Coarsening Pass",
                                          false, // Only looks at CFG,
                                          false // Analysis pass
                                          );