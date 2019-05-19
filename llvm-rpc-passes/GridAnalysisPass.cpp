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

InstVector
GridAnalysisPass::getThreadIDDependentInstructions(unsigned int dimension) const
{
    // Get all instructions that look at threadIdx

    InstVector result;

    const varInstructions_t& varInstructions = gridInstructions[dimension];
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

InstVector
GridAnalysisPass::getBlockSizeDependentInstructions(unsigned int dimension) const
{
    // Get all instructions that look at blockDim

    InstVector result;

    const varInstructions_t& varInstructions = gridInstructions[dimension];

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

InstVector
GridAnalysisPass::getBlockIDDependentInstructions(unsigned int dimension) const
{
    // Get all instructions that look at blockIdx

    InstVector result;

    const varInstructions_t& varInstructions = gridInstructions[dimension];
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

InstVector
GridAnalysisPass::getGridSizeDependentInstructions(unsigned int dimension) const
{
    // Get all instructions that look at gridDim

    InstVector result;

    const varInstructions_t& varInstructions = gridInstructions[dimension];
    auto instrs = varInstructions.find(CUDA_GRID_DIM_VAR);
    assert(instrs != varInstructions.end());

    result.insert(result.end(), instrs->second.begin(), instrs->second.end());

    return result;
}

InstVector GridAnalysisPass::getShuffleInstructions() const
{
    return shuffleInstructions;
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
    shuffleInstructions.clear();

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
    findInstructionsByVar(CUDA_THREAD_ID_VAR, pF);
    findInstructionsByVar(CUDA_BLOCK_ID_VAR, pF);
    findInstructionsByVar(CUDA_BLOCK_DIM_VAR, pF);
    findInstructionsByVar(CUDA_GRID_DIM_VAR, pF);

    std::string base = LLVM_PREFIX;
    base.append(".");
    findInstructionsByName(base + CUDA_SHUFFLE_DOWN + ".i32", pF, &shuffleInstructions);
    findInstructionsByName(base + CUDA_SHUFFLE_DOWN + ".f32", pF, &shuffleInstructions);
    findInstructionsByName(base + CUDA_SHUFFLE_UP + ".i32", pF, &shuffleInstructions);
    findInstructionsByName(base + CUDA_SHUFFLE_UP + ".f32", pF, &shuffleInstructions);
    findInstructionsByName(base + CUDA_SHUFFLE_IDX + ".i32", pF, &shuffleInstructions);
    findInstructionsByName(base + CUDA_SHUFFLE_IDX + ".f32", pF, &shuffleInstructions);
    findInstructionsByName(base + CUDA_SHUFFLE_BFLY + ".i32", pF, &shuffleInstructions);
    findInstructionsByName(base + CUDA_SHUFFLE_BFLY + ".f32", pF, &shuffleInstructions);
}

void GridAnalysisPass::findInstructionsByVar(std::string var, Function *pF)
{
    for (unsigned int i = 0; i < CUDA_MAX_DIM; ++i) {
        varInstructions_t& varInstructions = gridInstructions[i];

        findInstructionsByVar(var, pF, i, &varInstructions[var]);
    }
}

void GridAnalysisPass::findInstructionsByVar(std::string   var,
                                             Function     *pF,
                                             unsigned int  dimension,
                                             InstVector   *out)
{
    // CUDA variables (like threadIdx) are accessed by invoking calls to read
    // special registers.
    std::string calleeName = LLVM_PREFIX;
    calleeName.append(".");
    calleeName.append(CUDA_READ_SPECIAL_REG);
    calleeName.append(".");
    calleeName.append(Util::cudaVarToRegister(var));
    calleeName.append(".");
    calleeName.append(Util::dimensionToString(dimension));

    findInstructionsByName(calleeName, pF, out);
}

void GridAnalysisPass::findInstructionsByName(std::string  name,
                                              Function    *pF,
                                              InstVector  *out)
{
    Function *pCalleeF = pF->getParent()->getFunction(name);
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