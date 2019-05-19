// ============================================================================
// Copyright (c) Richard Rohac, 2019, All rights reserved.
// ============================================================================
// CUDA Grid Analysis Pass
// ============================================================================

#ifndef LLVM_LIB_TRANSFORMS_CUDA_COARSENING_GRIDANALYSISPASS_H
#define LLVM_LIB_TRANSFORMS_CUDA_COARSENING_GRIDANALYSISPASS_H

using namespace llvm;

namespace llvm {
}

class GridAnalysisPass : public FunctionPass {
public:
    // CREATORS
    GridAnalysisPass();

    // ACCESSORS
    InstVector getThreadIDDependentInstructions() const;
    InstVector getThreadIDDependentInstructions(unsigned int dimension) const;
    InstVector getBlockSizeDependentInstructions() const;
    InstVector getBlockSizeDependentInstructions(unsigned int dimension) const;

    InstVector getBlockIDDependentInstructions() const;
    InstVector getBlockIDDependentInstructions(unsigned int dimension) const;
    InstVector getGridSizeDependentInstructions() const;
    InstVector getGridSizeDependentInstructions(unsigned int dimension) const;

    InstVector getShuffleInstructions() const;

    // MANIPULATORS
    void getAnalysisUsage(AnalysisUsage& AU) const override;
    bool runOnFunction(Function& F) override;

    // DATA
    static char ID;

private:
    // PRIVATE TYPES
    typedef std::unordered_map<std::string, InstVector> varInstructions_t;

    // PRIVATE MANIPULATORS
    void init();
    void analyse(Function *pF);
    void findInstructionsByVar(std::string var, Function *pF);
    void findInstructionsByVar(std::string   var,
                               Function     *pF,
                               unsigned int  dimension,
                               InstVector   *out);
    void findInstructionsByName(std::string  name,
                                Function    *pF,
                                InstVector  *out);

    // DATA
    std::vector<varInstructions_t> gridInstructions;
    InstVector shuffleInstructions;
};

#endif
