// ============================================================================
// Copyright (c) Richard Rohac, 2019, All rights reserved.
// ============================================================================
// Divergence information
// -> Based on Alberto's Magni OpenCL coarsening pass algorithm
//    available at https://github.com/HariSeldon/coarsening_pass
// ============================================================================

#include "DivergenceInfo.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PostDominators.h"

#include "Common.h"
#include "CUDACoarsening.h"

bool isKernelFunction(const Function &F);

// CREATORS
DivergenceInfo::DivergenceInfo()
: FunctionPass(ID) {
}

// PUBLIC MANIPULATORS
void DivergenceInfo::getAnalysisUsage(AnalysisUsage &Info) const {
    Info.addRequired<LoopInfoWrapperPass>();
    Info.addPreserved<LoopInfoWrapperPass>();
    Info.addRequired<PostDominatorTreeWrapperPass>();
    Info.addRequired<DominatorTreeWrapperPass>();
    Info.setPreservesAll();
}

bool DivergenceInfo::runOnFunction(Function &F) {
    const llvm::NamedMDNode *kernels = 
                F.getParent()->getNamedMetadata("nvvm.annotations");

    if (!kernels) {
        return false;
    }

    if (!isKernelFunction(F)) {
        return false; // TODO should be main pass based!
    }

    errs() << "Divergence analysis invoked at kernel: ";
    errs().write_escaped(F.getName()) << '\n';

    clear();

    m_postDomTree = &getAnalysis<PostDominatorTreeWrapperPass>().getPostDomTree();
    m_domTree = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();

    analyse();

    return true;
}

// PRIVATE MANIPULATORS
void DivergenceInfo::clear() {
    m_divergent.clear();
}

void DivergenceInfo::analyse() {
    InstVector seeds = thread_id_based_instructions();
    InstSet worklist(seeds.begin(), seeds.end());

    while (!worklist.empty()) {
        auto iter = worklist.begin();
        Instruction *inst = *iter;
        worklist.erase(iter);
        m_divergent.push_back(inst);

        InstSet users;

        // Manage branches.
        if (isa<BranchInst>(inst)) {
            BasicBlock *block = findImmediatePostDom(inst->getParent(), m_postDomTree);
            for (auto inst = block->begin(); isa<PHINode>(inst); ++inst) {
                users.insert(inst);
            }
        }

        findUsesOf(inst, users);
        // Add users of the current instruction to the work list.
        for (InstSet::iterator iter = users.begin(), iterEnd = users.end();
            iter != iterEnd; ++iter) {
            if (!isPresent(*iter, m_divergent)) {
                worklist.insert(*iter);
            }
        }
  }
}

// DATA
char DivergenceInfo::ID = 0;

static RegisterPass<DivergenceInfo> X("cuda-divergence-info",
                                      "CUDA Divergence Info Pass",
                                      false /* Only looks at CFG */,
                                      true /* Analysis Pass */);

//static void registerDivergenceInfoPass(const PassManagerBuilder &,
//                                      legacy::PassManagerBase  &PM) {
//    PM.add(new DivergenceInfo());
//}
//static RegisterStandardPasses RegisterMyPass(
//                   PassManagerBuilder::EP_EarlyAsPossible,
//                   registerDivergenceInfoPass);