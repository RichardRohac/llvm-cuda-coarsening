// ============================================================================
// Copyright (c) Richard Rohac, 2019, All rights reserved.
// ============================================================================
// CUDA Grid Analysis Pass
// ============================================================================

#include "llvm/Pass.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"

#include "Common.h"
#include "Util.h"
#include "GridAnalysisPass.h"

using namespace llvm;

// DATA
char GridAnalysisPass::ID = 0;

// CREATORS
GridAnalysisPass::GridAnalysisPass()
: FunctionPass(ID) 
{
}

// PUBLIC ACCESSORS
InstVector GridAnalysisPass::getThreadIDDependentInstructions() const
{
    // Get all instructions that look at threadIdx

    InstVector resultX = getThreadIDDependentInstructions(0);
    InstVector resultY = getThreadIDDependentInstructions(1);
    InstVector resultZ = getThreadIDDependentInstructions(2);

    InstVector result;
    result.insert(result.end(), resultX.begin(), resultX.end());
    result.insert(result.end(), resultY.begin(), resultY.end());
    result.insert(result.end(), resultZ.begin(), resultZ.end());

    return result;
}

InstVector GridAnalysisPass::getThreadIDDependentInstructions(int direction) const
{
    // Get all instructions that look at threadIdx

    InstVector result;

    const varInstructions_t& varInstructions = gridInstructions[direction];
    auto tidx = varInstructions.find(CUDA_THREAD_ID_VAR);
    assert(tidx != varInstructions.end());

    result.insert(result.end(), tidx->second.begin(), tidx->second.end());

    return result;
}

InstVector GridAnalysisPass::getBlockSizeDependentInstructions() const
{
    // Get all instructions that look at blockDim

    InstVector resultX = getBlockSizeDependentInstructions(0);
    InstVector resultY = getBlockSizeDependentInstructions(1);
    InstVector resultZ = getBlockSizeDependentInstructions(2);

    InstVector result;
    result.insert(result.end(), resultX.begin(), resultX.end());
    result.insert(result.end(), resultY.begin(), resultY.end());
    result.insert(result.end(), resultZ.begin(), resultZ.end());

    return result;
}

InstVector GridAnalysisPass::getBlockSizeDependentInstructions(int direction) const
{
    // Get all instructions that look at blockDim

    InstVector result;

    const varInstructions_t& varInstructions = gridInstructions[direction];

    auto instrs = varInstructions.find(CUDA_BLOCK_DIM_VAR);
    assert(instrs != varInstructions.end());

    result.insert(result.end(), instrs->second.begin(), instrs->second.end());

    return result;
}

InstVector GridAnalysisPass::getBlockIDDependentInstructions() const
{
    // Get all instructions that look at blockDim

    InstVector resultX = getBlockIDDependentInstructions(0);
    InstVector resultY = getBlockIDDependentInstructions(1);
    InstVector resultZ = getBlockIDDependentInstructions(2);

    InstVector result;
    result.insert(result.end(), resultX.begin(), resultX.end());
    result.insert(result.end(), resultY.begin(), resultY.end());
    result.insert(result.end(), resultZ.begin(), resultZ.end());

    return result;
}

InstVector GridAnalysisPass::getBlockIDDependentInstructions(int direction) const
{
    // Get all instructions that look at blockIdx

    InstVector result;

    const varInstructions_t& varInstructions = gridInstructions[direction];
    auto instrs = varInstructions.find(CUDA_BLOCK_ID_VAR);
    assert(instrs != varInstructions.end());

    result.insert(result.end(), instrs->second.begin(), instrs->second.end());

    return result;
}

InstVector GridAnalysisPass::getGridSizeDependentInstructions() const
{
    // Get all instructions that look at gridDim

    InstVector resultX = getGridSizeDependentInstructions(0);
    InstVector resultY = getGridSizeDependentInstructions(1);
    InstVector resultZ = getGridSizeDependentInstructions(2);

    InstVector result;
    result.insert(result.end(), resultX.begin(), resultX.end());
    result.insert(result.end(), resultY.begin(), resultY.end());
    result.insert(result.end(), resultZ.begin(), resultZ.end());

    return result;
}

InstVector GridAnalysisPass::getGridSizeDependentInstructions(int direction) const
{
    // Get all instructions that look at gridDim

    InstVector result;

    const varInstructions_t& varInstructions = gridInstructions[direction];
    auto instrs = varInstructions.find(CUDA_GRID_DIM_VAR);
    assert(instrs != varInstructions.end());

    result.insert(result.end(), instrs->second.begin(), instrs->second.end());

    return result;
}

// PUBLIC MANIPULATORS
void GridAnalysisPass::getAnalysisUsage(AnalysisUsage& AU) const
{
    AU.setPreservesAll();
}

bool GridAnalysisPass::runOnFunction(Function& F)
{
    //errs() << "--  INFO  -- Grid analysis invoked on: ";
    //errs().write_escaped(F.getName()) << '\n';

    init();
    analyse(&F);

    return false;
}

// PRIVATE MANIPULATORS
void GridAnalysisPass::init()
{
    // Clear data, as pass can run multiple times
    gridInstructions.clear();
    gridInstructions.reserve(CUDA_MAX_DIM);

    for (unsigned int i = 0; i < CUDA_MAX_DIM; ++i) {
        gridInstructions.push_back(varInstructions_t());
        gridInstructions[i][CUDA_THREAD_ID_VAR] = InstVector();
        gridInstructions[i][CUDA_BLOCK_ID_VAR] = InstVector();
        gridInstructions[i][CUDA_BLOCK_DIM_VAR] = InstVector();
        gridInstructions[i][CUDA_GRID_DIM_VAR] = InstVector();
    }
}

void GridAnalysisPass::analyse(Function *pF)
{
    findInstructionsByName(CUDA_THREAD_ID_VAR, pF);
    findInstructionsByName(CUDA_BLOCK_ID_VAR, pF);
    findInstructionsByName(CUDA_BLOCK_DIM_VAR, pF);
    findInstructionsByName(CUDA_GRID_DIM_VAR, pF);
}

void GridAnalysisPass::findInstructionsByName(std::string name, Function *pF)
{
    for (unsigned int i = 0; i < CUDA_MAX_DIM; ++i) {
        varInstructions_t& varInstructions = gridInstructions[i];

        findInstructionsByName(name, pF, i, &varInstructions[name]);
    }
}

void GridAnalysisPass::findInstructionsByName(std::string  name,
                                              Function    *pF,
                                              int          direction,
                                              InstVector  *out)
{
    // CUDA variables (like threadIdx) are accessed by invoking calls to read
    // special registers.
    std::string calleeName = LLVM_PREFIX;
    calleeName.append(".");
    calleeName.append(CUDA_READ_SPECIAL_REG);
    calleeName.append(".");
    calleeName.append(Util::cudaVarToRegister(name));
    calleeName.append(".");
    calleeName.append(Util::directionToString(direction));

    Function *pCalleeF = pF->getParent()->getFunction(calleeName);
    if (!pCalleeF) {
        return;
    }

    // Iterate over the uses of the function.
    for (auto user = pCalleeF->user_begin();
         user !=  pCalleeF->user_end();
         ++user) {
        if (CallInst *callInst = dyn_cast<CallInst>(*user)) {
            if (pF == callInst->getParent()->getParent()) {
                out->push_back(callInst);
            }
        }
    }
}

static RegisterPass<GridAnalysisPass> X("cuda-grid-analysis-pass",
                                        "CUDA Grid Analysis Pass",
                                        false, // Only looks at CFG
                                        true // Analysis pass
                                        );