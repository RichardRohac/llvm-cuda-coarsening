// ============================================================================
// Branch extraction pass
// -> Based on Alberto's Magni OpenCL coarsening pass algorithm
//    available at https://github.com/HariSeldon/coarsening_pass
// ============================================================================

#include <llvm/Pass.h>

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PostDominators.h"

#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"

#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#include <llvm/IR/Function.h>

#include "Common.h"
#include "Util.h"
#include "GridAnalysisPass.h"
#include "DivergenceAnalysisPass.h"
#include "BenefitAnalysisPass.h"
#include "RegionBounds.h"
#include "DivergentRegion.h"
#include "BranchExtractionPass.h"

extern cl::opt<std::string> CLCoarseningDimension;
extern cl::opt<std::string> CLCoarseningMode;
extern cl::opt<std::string> CLKernelName;

using namespace llvm;

char BranchExtractionPass::ID = 0;

BranchExtractionPass::BranchExtractionPass()
: FunctionPass(ID)
{
}

void BranchExtractionPass::getAnalysisUsage(AnalysisUsage& AU) const
{
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequired<DivergenceAnalysisPassBL>();
    AU.addRequired<DivergenceAnalysisPassTL>();
    AU.addRequired<PostDominatorTreeWrapperPass>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addPreserved<DivergenceAnalysisPassBL>();
    AU.addPreserved<DivergenceAnalysisPassTL>();
}

bool BranchExtractionPass::runOnFunction(Function& F)
{
    // Apply the pass to kernels only.
    if (!Util::isKernelFunction(F))
        return false;

    std::string FunctionName = F.getName();
    FunctionName = Util::demangle(FunctionName);
    FunctionName = Util::nameFromDemangled(FunctionName);
    if (CLKernelName != "" && FunctionName != CLKernelName)
        return false;

    errs() << "BC RUNNING ON " << CLKernelName << "!\n";


    // Perform analyses.
    loopInfo = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    dt = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    pdt = &getAnalysis<PostDominatorTreeWrapperPass>().getPostDomTree();
    divTL = &getAnalysis<DivergenceAnalysisPassTL>();
    divBL = &getAnalysis<DivergenceAnalysisPassBL>();
    RegionVector &regions = CLCoarseningMode == "block" ? divBL->getRegions() :
                    divTL->getRegions();

    // This is terribly inefficient.
    for (auto region : regions) {
        BasicBlock *newExiting = Util::findImmediatePostDom(region->getHeader(), pdt);
        region->setExiting(newExiting);
        region->fillRegion();
        extractBranches(region);
        region->fillRegion();
        isolateRegion(region);
        region->fillRegion();
        region->findAliveValues();
        dt->recalculate(F);
        pdt->recalculate(F);
    }

    std::for_each(
      regions.begin(), regions.end(),
      [this](DivergentRegion *region) { region->fillRegion(); });

    return regions.size() != 0;
}

//------------------------------------------------------------------------------
// Isolate the exiting block from the rest of the graph.
// If it has incoming edges coming from outside the current region
// create a new exiting block for the region.
void BranchExtractionPass::extractBranches(DivergentRegion *region)
{
    BasicBlock *header = region->getHeader();
    BasicBlock *exiting = region->getExiting();
    BasicBlock *newHeader = nullptr;

    if (!loopInfo->isLoopHeader(header))
        newHeader = SplitBlock(header, header->getTerminator(), dt, loopInfo);
    else {
        newHeader = header;
        Loop *loop = loopInfo->getLoopFor(header);
        if (loop == loopInfo->getLoopFor(exiting)) {
        exiting = loop->getExitBlock();
        region->setExiting(exiting);
        }
    }

    Instruction *firstNonPHI = exiting->getFirstNonPHI();
    BasicBlock *newExiting = SplitBlock(exiting, firstNonPHI, dt, loopInfo);
    region->setHeader(newHeader);

    // Check is a region in the has as header exiting.
    // If so replace it with new exiting.
    RegionVector &regions = CLCoarseningMode == "block" ? divBL->getRegions() :
                    divTL->getRegions();

    std::for_each(regions.begin(), regions.end(),
                    [exiting, newExiting](DivergentRegion *region) {
        if (region->getHeader() == exiting)
        region->setHeader(newExiting);
    });
}

// -----------------------------------------------------------------------------
void BranchExtractionPass::isolateRegion(DivergentRegion *region)
{
    BasicBlock *exiting = region->getExiting();

    // If the header dominates the exiting bail out.
    if (dt->dominates(region->getHeader(), region->getExiting()))
        return;

    // TODO.
    // Verify that the incoming branch from outside is pointing to the exiting
    // block.

    // Create a new exiting block.
    BasicBlock *newExiting = BasicBlock::Create(
        exiting->getContext(), exiting->getName() + Twine(".extracted"),
        exiting->getParent(), exiting);
    BranchInst::Create(exiting, newExiting);

    // All the blocks in the region pointing to the exiting are redirected to the
    // new exiting.
    for (auto iter = region->begin(), iterEnd = region->end(); iter != iterEnd;
        ++iter) {
        Instruction *terminator = (*iter)->getTerminator();
        for (unsigned int index = 0; index < terminator->getNumSuccessors();
            ++index) {
            if (terminator->getSuccessor(index) == exiting) {
                terminator->setSuccessor(index, newExiting);
            }
        }
    }

    // 'newExiting' will contain the phi working on the values from the blocks
    // in the region.
    // 'Exiting' will contain the phi working on the values from the blocks
    // outside and in the region.
    PhiVector oldPhis = Util::getPHIs(exiting);

    PhiVector newPhis;
    PhiVector exitPhis;

    InstVector &divInsts = CLCoarseningMode == "block" ? divBL->getInstructions() :
                    divTL->getInstructions();

    for (auto phi: oldPhis) {
        PHINode *newPhi = PHINode::Create(phi->getType(), 0,
                                        phi->getName() + Twine(".new_exiting"),
                                        &*newExiting->begin());
        PHINode *exitPhi = PHINode::Create(phi->getType(), 0,
                                        phi->getName() + Twine(".old_exiting"),
                                        &*exiting->begin());
        for (unsigned int index = 0; index < phi->getNumIncomingValues(); ++index) {
        BasicBlock *BB = phi->getIncomingBlock(index);
        if (contains(*region, BB))
            newPhi->addIncoming(phi->getIncomingValue(index), BB);
        else
            exitPhi->addIncoming(phi->getIncomingValue(index), BB);
        }
        newPhis.push_back(newPhi);
        exitPhis.push_back(exitPhi);

        // Update divInsts.
        if (std::find(divInsts.begin(), divInsts.end(),
                    static_cast<Instruction *>(phi)) !=
            divInsts.end()) {
                divInsts.push_back(newPhi);
                divInsts.push_back(exitPhi);
        }
    }

    unsigned int phiNumber = newPhis.size();
    for (unsigned int phiIndex = 0; phiIndex < phiNumber; ++phiIndex) {
        // Add the edge coming from the 'newExiting' block to the phi nodes in
        // Exiting.
        PHINode *exitPhi = exitPhis[phiIndex];
        PHINode *newPhi = newPhis[phiIndex];
        exitPhi->addIncoming(newPhi, newExiting);

        // Update all the references to the old Phis to the new ones.
        oldPhis[phiIndex]->replaceAllUsesWith(exitPhi);
    }

    // Delete the old phi nodes.
    for (auto toDelete: oldPhis) {
        // Update divInsts.
        InstVector::iterator iter = std::find(divInsts.begin(), divInsts.end(),
                                            static_cast<Instruction *>(toDelete));
        if (iter != divInsts.end()) {
        divInsts.erase(iter);
        }

        toDelete->eraseFromParent();
    }

    region->setExiting(newExiting);
}

static RegisterPass<BranchExtractionPass> Z("be",
                                            "Extract divergent regions",
                                            false, // Only looks at CFG
                                            false // Analysis pass
                                            );