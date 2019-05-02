// ============================================================================
// Copyright (c) Richard Rohac, 2019, All rights reserved.
// ============================================================================
// CUDA Coarsening Transformation pass
// -> Based on Alberto's Magni OpenCL coarsening pass algorithm
//    available at https://github.com/HariSeldon/coarsening_pass
// ============================================================================

#include <llvm/Pass.h>
#include <llvm/IR/Instructions.h>

#include "Common.h"
#include "CUDACoarsening.h"
#include "Util.h"
#include "DivergenceAnalysisPass.h"
#include "GridAnalysisPass.h"

// Support functions.
Instruction *getMulInst(Value *value, unsigned int factor);
Instruction *getAddInst(Value *value, unsigned int addend);
Instruction *getAddInst(Value *V1, Value *V2);
Instruction *getShiftInst(Value *value, unsigned int shift);
Instruction *getAndInst(Value *value, unsigned int factor);
Instruction *getDivInst(Value *value, unsigned int divisor);
Instruction *getModuloInst(Value *value, unsigned int modulo);

void CUDACoarseningPass::scaleKernelGrid()
{
    if (m_dimX) {
        scaleKernelGridSizes(0);
        scaleKernelGridIDs(0);
    }

    if (m_dimY) {
        scaleKernelGridSizes(1);
        scaleKernelGridIDs(1);
    }

    if (m_dimZ) {
        scaleKernelGridSizes(2);
        scaleKernelGridIDs(2);
    }
}

void CUDACoarseningPass::scaleKernelGridSizes(int direction)
{
    InstVector sizeInsts = 
                m_blockLevel
                ? m_gridAnalysis->getGridSizeDependentInstructions(direction)
                : m_gridAnalysis->getBlockSizeDependentInstructions(direction);

    for (InstVector::iterator iter = sizeInsts.begin();
         iter != sizeInsts.end();
         ++iter) {
        // Scale size.
        Instruction *inst = *iter;
        Instruction *mul = getMulInst(inst, m_factor);
        mul->insertAfter(inst);
        // Replace uses of the old size with the scaled one.
        Util::replaceUses(inst, mul);
    }
}

void CUDACoarseningPass::scaleKernelGridIDs(int direction)
{
    // origTid = [newTid / st] * cf * st + newTid % st + subid * st
    unsigned int cfst = m_factor * m_stride;

    InstVector tids = 
                m_blockLevel
                ? m_gridAnalysis->getBlockIDDependentInstructions(direction)
                : m_gridAnalysis->getThreadIDDependentInstructions(direction);
    for (InstVector::iterator instIter = tids.begin();
        instIter != tids.end();
        ++instIter) {
        Instruction *inst = *instIter;

        // Compute base of new tid.
        Instruction *div = getDivInst(inst, m_stride); 
        div->insertAfter(inst);
        Instruction *mul = getMulInst(div, cfst);
        mul->insertAfter(div);
        Instruction *modulo = getModuloInst(inst, m_stride);
        modulo->insertAfter(mul);
        Instruction *base = getAddInst(mul, modulo);
        base->insertAfter(modulo);

        // Replace uses of the threadId with the new base.
        Util::replaceUses(inst, base);
        modulo->setOperand(0, inst);
        div->setOperand(0, inst);

        errs() << "Or. adding ";
        inst->dump();
        errs() << "\n";

        // Compute the remaining thread ids.
        m_coarseningMap.insert(
                      std::pair<Instruction *, InstVector>(inst, InstVector()));

        errs() << "Or. adding ";
        base->dump();
        errs() << "\n";
        InstVector &current = m_coarseningMap[base]; // BUG this inserts diff. inst.
        current.reserve(m_factor - 1);

        Instruction *bookmark = base;
        for (unsigned int index = 2; index <= m_factor; ++index) {
            Instruction *add = getAddInst(base, (index - 1) * m_stride);
            add->insertAfter(bookmark);
            current.push_back(add);
            bookmark = add;
        }
    }
}

// Support functions.
//-----------------------------------------------------------------------------
unsigned int getIntWidth(Value *value) {
    Type *type = value->getType();
    IntegerType *intType = dyn_cast<IntegerType>(type);
    assert(intType && "Value type is not integer");
    return intType->getBitWidth();
}

ConstantInt *getConstantInt(unsigned int value, unsigned int width,
                                LLVMContext &context) {
    IntegerType *integer = IntegerType::get(context, width);
    return ConstantInt::get(integer, value);
}

Instruction *getMulInst(Value *value, unsigned int factor) {
    unsigned int width = getIntWidth(value);
    ConstantInt *factorValue = getConstantInt(factor, width, value->getContext());
    Instruction *mul =
        BinaryOperator::Create(Instruction::Mul, value, factorValue);
    mul->setName(value->getName() + ".." + Twine(factor));
    return mul;
}

Instruction *getAddInst(Value *value, unsigned int addend) {
    unsigned int width = getIntWidth(value);
    ConstantInt *addendValue = getConstantInt(addend, width, value->getContext());
    Instruction *add =
        BinaryOperator::Create(Instruction::Add, value, addendValue);
    add->setName(value->getName() + ".." + Twine(addend));
    return add;
}

Instruction *getAddInst(Value *firstValue, Value *secondValue) {
    Instruction *add =
        BinaryOperator::Create(Instruction::Add, firstValue, secondValue);
    add->setName(firstValue->getName() + "..Add");
    return add;
}

Instruction *getShiftInst(Value *value, unsigned int shift) {
    unsigned int width = getIntWidth(value);
    ConstantInt *intValue = getConstantInt(shift, width, value->getContext());
    Instruction *shiftInst =
        BinaryOperator::Create(Instruction::LShr, value, intValue);
    shiftInst->setName(Twine(value->getName()) + "..Shift");
    return shiftInst;
}

Instruction *getAndInst(Value *value, unsigned int factor) {
    unsigned int width = getIntWidth(value);
    ConstantInt *intValue = getConstantInt(factor, width, value->getContext());
    Instruction *andInst =
        BinaryOperator::Create(Instruction::And, value, intValue);
    andInst->setName(Twine(value->getName()) + "..And");
    return andInst;
}

Instruction *getDivInst(Value *value, unsigned int divisor) {
    unsigned int width = getIntWidth(value);
    ConstantInt *intValue = getConstantInt(divisor, width, value->getContext());
    Instruction *divInst = 
        BinaryOperator::Create(Instruction::UDiv, value, intValue);
    divInst->setName(Twine(value->getName()) + "..Div");
    return divInst;
}

Instruction *getModuloInst(Value *value, unsigned int modulo) {
    unsigned int width = getIntWidth(value);
    ConstantInt *intValue = getConstantInt(modulo, width, value->getContext());
    Instruction *moduloInst = 
        BinaryOperator::Create(Instruction::URem, value, intValue);
    moduloInst->setName(Twine(value->getName()) + "..Rem");
    return moduloInst;
}