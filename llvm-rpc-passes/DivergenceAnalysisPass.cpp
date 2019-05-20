// ============================================================================
// Copyright (c) Richard Rohac, 2019, All rights reserved.
// ============================================================================
// Divergence analysis pass
// -> Based on Alberto's Magni OpenCL coarsening pass algorithm
//    available at https://github.com/HariSeldon/coarsening_pass
// ============================================================================

#include "llvm/Pass.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PostDominators.h"

#include "Common.h"
#include "Util.h"
#include "GridAnalysisPass.h"
#include "RegionBounds.h"
#include "DivergentRegion.h"
#include "DivergenceAnalysisPass.h"

extern cl::opt<std::string> CLCoarseningDimension;

using namespace llvm;

char DivergenceAnchorPass::ID = 0;

DivergenceAnchorPass::DivergenceAnchorPass()
: FunctionPass(ID)
{
}

bool DivergenceAnchorPass::runOnFunction(Function& F)
{
    for (BasicBlock &B : F) {
        if (B.getName().find("rpc-anchor_entry") != std::string::npos) {
            return false;
        }
    }

    BasicBlock *entry = &F.getEntryBlock();
    entry->splitBasicBlock(&entry->front(), "..rpc-anchor_entry");

    std::vector<BasicBlock*> toSplit;
    for (BasicBlock &B : F) {
        for (Instruction &I : B) {
            if (isa<ReturnInst>(&I)) {
               toSplit.push_back(&B);
            }
        }
    }

    for (BasicBlock *pB : toSplit) {
        pB->splitBasicBlock(&pB->front(), "..rpc-anchor_end");
    }

    return true;
}

static RegisterPass<DivergenceAnchorPass> Z("cuda-divergence-anchor-pass",
                                            "CUDA Divergence Anchor Pass",
                                            false, // Only looks at CFG
                                            false // Analysis pass
                                            );

// DATA
char DivergenceAnalysisPassTL::ID = 0;
char DivergenceAnalysisPassBL::ID = 0;

// CREATORS
DivergenceAnalysisPass::DivergenceAnalysisPass()
{
}

// PUBLIC ACCESSORS
RegionVector& DivergenceAnalysisPass::getOutermostRegions()
{
    // Use memoization.
    if (m_outermostRegions.empty()) {
        findOutermostRegions();
    }
    return m_outermostRegions;
}

InstVector& DivergenceAnalysisPass::getOutermostInstructions()
{
    // Use memoization.
    if (m_outermostDivergent.empty()) {
        findOutermost(m_divergent, m_regions, m_outermostDivergent);
    }
    return m_outermostDivergent;
}

bool DivergenceAnalysisPass::isDivergent(Instruction *inst)
{
    return isPresent(inst, m_divergent);
}

// PRIVATE MANIPULATORS
void DivergenceAnalysisPass::clear()
{
    m_divergent.clear();
    m_outermostDivergent.clear();
    m_divergentBranches.clear();
    m_regions.clear();
    m_outermostRegions.clear();
}

void DivergenceAnalysisPass::analyse(Function& F)
{
    m_dimension = Util::numeralDimension(CLCoarseningDimension);

    InstVector seeds =
        m_blockLevel
        ? m_grid->getBlockIDDependentInstructions(m_dimension)
        : m_grid->getThreadIDDependentInstructions(m_dimension);

    findUsers(seeds, &m_divergent, false);
}

void DivergenceAnalysisPass::findOutermost(InstVector&   insts,
                                           RegionVector& regions,
                                           InstVector&   result)
{
    // This is called only when the outermost instructions are acutally
    // requested, ie. during coarsening. This is done to be sure that this
    // instructions are computed after the extraction of divergent regions
    // from the CFG.
    result.clear();
    for (auto inst : insts) {
        if (Util::isOutermost(inst, regions)) {
            result.push_back(inst);
        }
    }

    // Remove from result all the calls to builtin functions.
    InstVector builtin =
        m_blockLevel
        ? m_grid->getBlockIDDependentInstructions()
        : m_grid->getThreadIDDependentInstructions();
    InstVector tmp;

    size_t oldSize = result.size();

    std::sort(result.begin(), result.end());
    std::sort(builtin.begin(), builtin.end());
    std::set_difference(result.begin(), result.end(), builtin.begin(),
                        builtin.end(), std::back_inserter(tmp));
    result.swap(tmp);

    assert(result.size() <= oldSize && "Wrong set difference");
}

void DivergenceAnalysisPass::findDivergentBranches()
{
    std::copy_if(m_divergent.begin(),
                 m_divergent.end(),
                 std::back_inserter(m_divergentBranches),
                 [](Instruction *pI) {
                        return isa<BranchInst>(pI);
                 });
}

void DivergenceAnalysisPass::findRegions()
{
    for (Instruction *divBranch : m_divergentBranches) {
        BasicBlock *header = divBranch->getParent();
        BasicBlock *exiting = Util::findImmediatePostDom(header, m_postDomT);

        if (m_loopInfo->isLoopHeader(header)) {
            Loop *loop = m_loopInfo->getLoopFor(header);
            if (loop == m_loopInfo->getLoopFor(exiting))
                exiting = loop->getExitBlock();
        }

        m_regions.push_back(new DivergentRegion(header, exiting));
    }

    // Remove redundant regions. The ones coming from loops.
    m_regions = cleanUpRegions(m_regions, m_domT);
}

void DivergenceAnalysisPass::findOutermostRegions()
{
    m_outermostRegions.clear();
    for (auto region : m_regions) {
        if (Util::isOutermost(region, m_regions)) {
            m_outermostRegions.push_back(region);
        }
    }
}

void DivergenceAnalysisPass::findUsers(InstVector&  seeds,
                                       InstVector  *out,
                                       bool         skipBranches)
{
    InstSet worklist(seeds.begin(), seeds.end());

    while (!worklist.empty()) {
        auto iter = worklist.begin();
        Instruction *inst = *iter;
        worklist.erase(iter);
        out->push_back(inst);

        InstSet users;

        // Manage branches.
        if (isa<BranchInst>(inst)) {
            if (!skipBranches) {
                BasicBlock *block = Util::findImmediatePostDom(inst->getParent(),
                                                            m_postDomT);
                for (auto it = block->begin(); isa<PHINode>(it); ++it) {
                    users.insert(&*it);
                }
            }
        }

        Util::findUsesOf(inst, users, skipBranches);
        // Add users of the current instruction to the work list.
        for (InstSet::iterator iter = users.begin();
             iter != users.end();
             ++iter) {
             if (!isPresent(*iter, *out)) {
                worklist.insert(*iter);
            }
        }
    }
}

RegionVector DivergenceAnalysisPass::cleanUpRegions(RegionVector&        regions,
                                                    const DominatorTree *dt)
{
    RegionVector result;

    for (size_t index1 = 0; index1 < regions.size(); ++index1) {
        DivergentRegion *region1 = regions[index1];
        BlockVector &blocks1 = region1->getBlocks();
  
        bool toAdd = true;

        for (size_t index2 = 0; index2 < regions.size(); ++index2) {
            if(index2 == index1) 
                break;

            DivergentRegion *region2 = regions[index2];
            BlockVector &blocks2 = region2->getBlocks();

            if (is_permutation(blocks1.begin(),
                               blocks1.end(),
                               blocks2.begin()) &&
                dt->dominates(region2->getHeader(), region1->getHeader())) {
                toAdd = false;
                break; 
            }
        }

        if(toAdd) 
        result.push_back(region1); 
    }

    return result;
}

DivergenceAnalysisPassTL::DivergenceAnalysisPassTL()
: FunctionPass(ID) 
{
}

// PUBLIC MANIPULATORS
void DivergenceAnalysisPassTL::getAnalysisUsage(AnalysisUsage& AU) const
{
    AU.addRequired<DivergenceAnchorPass>();
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequired<PostDominatorTreeWrapperPass>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<GridAnalysisPass>();
    AU.setPreservesAll();
}

bool DivergenceAnalysisPassTL::runOnFunction(Function& F)
{
    clear();

    m_loopInfo = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    m_postDomT = &getAnalysis<PostDominatorTreeWrapperPass>().getPostDomTree();
    m_domT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    m_grid = &getAnalysis<GridAnalysisPass>();
    m_blockLevel = false;

    analyse(F);
    findDivergentBranches();
    findRegions();

    return false;
}

static RegisterPass<DivergenceAnalysisPassTL> X("cuda-divergence-analysis-pass-tl",
                                                "CUDA Divergence Analysis Pass TL",
                                                false, // Only looks at CFG
                                                true // Analysis pass
                                                );

DivergenceAnalysisPassBL::DivergenceAnalysisPassBL()
: FunctionPass(ID) 
{
}

// PUBLIC MANIPULATORS
void DivergenceAnalysisPassBL::getAnalysisUsage(AnalysisUsage& AU) const
{
    AU.addRequired<DivergenceAnchorPass>();
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequired<PostDominatorTreeWrapperPass>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<GridAnalysisPass>();
    AU.setPreservesAll();
}

bool DivergenceAnalysisPassBL::runOnFunction(Function& F)
{
    clear();

    m_loopInfo = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    m_postDomT = &getAnalysis<PostDominatorTreeWrapperPass>().getPostDomTree();
    m_domT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    m_grid = &getAnalysis<GridAnalysisPass>();
    m_blockLevel = true;

    analyse(F);
    findDivergentBranches();
    findRegions();

    return false;
}

static RegisterPass<DivergenceAnalysisPassBL> Y("cuda-divergence-analysis-pass-bl",
                                                "CUDA Divergence Analysis Pass BL",
                                                false, // Only looks at CFG
                                                true // Analysis pass
                                                );