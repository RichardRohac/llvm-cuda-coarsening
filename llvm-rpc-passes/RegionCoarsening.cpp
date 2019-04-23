// ============================================================================
// Copyright (c) Richard Rohac, 2019, All rights reserved.
// ============================================================================
// CUDA Coarsening Transformation pass
// -> Based on Alberto's Magni OpenCL coarsening pass algorithm
//    available at https://github.com/HariSeldon/coarsening_pass
// ============================================================================

#include <llvm/Pass.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Analysis/PostDominators.h>

#include "Common.h"
#include "CUDACoarsening.h"
#include "Util.h"
#include "RegionBounds.h"
#include "DivergentRegion.h"
#include "DivergenceAnalysisPass.h"
#include "GridAnalysisPass.h"

void CUDACoarseningPass::replicateRegion(DivergentRegion *region)
{
    assert(m_domT->dominates(region->getHeader(), region->getExiting()) &&
           "Header does not dominates Exiting");
    assert(m_postDomT->dominates(region->getExiting(), region->getHeader()) &&
         "Exiting does not post dominate Header");

    replicateRegionClassic(region);
}

void CUDACoarseningPass::replicateRegionClassic(DivergentRegion *region)
{
    CoarseningMap aliveMap;
    initAliveMap(region, aliveMap);
    replicateRegionImpl(region, aliveMap);
    updatePlaceholdersWithAlive(aliveMap);
}

void CUDACoarseningPass::initAliveMap(DivergentRegion *region,
                                      CoarseningMap&   aliveMap)
{
    InstVector &aliveInsts = region->getAlive();
    for (auto &inst : aliveInsts) {
        aliveMap.insert(std::pair<Instruction *, InstVector>(inst,
                                                             InstVector()));
    }
}

void CUDACoarseningPass::updateAliveMap(CoarseningMap& aliveMap,
                                        Map&           regionMap)
{
  for (auto &mapIter : aliveMap) {
    InstVector &coarsenedInsts = mapIter.second;
    Value *value = regionMap[mapIter.first];
    assert(value != nullptr && "Missing alive value in region map");
    coarsenedInsts.push_back(dyn_cast<Instruction>(value));
  }
}

void CUDACoarseningPass::updatePlaceholdersWithAlive(CoarseningMap& aliveMap) {
  // Force the addition of the alive values to the coarsening map. 
  for (auto mapIter : aliveMap) {
    Instruction *alive = mapIter.first;
    InstVector &coarsenedInsts = mapIter.second;

    auto cIter = m_coarseningMap.find(alive); 
    if(cIter == m_coarseningMap.end()) {
      m_coarseningMap.insert(std::pair<Instruction *, InstVector>(alive, coarsenedInsts)); 
    }
  }
  
  for (auto &mapIter : aliveMap) {
    Instruction *alive = mapIter.first;
    InstVector &coarsenedInsts = mapIter.second;

    updatePlaceholderMap(alive, coarsenedInsts);
  }
}

void CUDACoarseningPass::replicateRegionImpl(DivergentRegion *region,
                                           CoarseningMap&   aliveMap)
{
    BasicBlock *pred = getPredecessor(region, m_loopInfo);
    BasicBlock *topInsertionPoint = region->getExiting();
    BasicBlock *bottomInsertionPoint = getExit(*region);

    // Replicate the region.
    for (unsigned int index = 0; index < m_factor - 1; ++index) {
        Map valueMap;
        DivergentRegion *newRegion =
            region->clone(".cf" + Twine(index + 2), m_domT, valueMap);
        applyCoarseningMap(*newRegion, index);

        // Connect the region to the CFG.
        Util::changeBlockTarget(topInsertionPoint, newRegion->getHeader());
        Util::changeBlockTarget(newRegion->getExiting(), bottomInsertionPoint);

        // Update the phi nodes of the newly inserted header.
        Util::remapBlocksInPHIs(newRegion->getHeader(), pred, topInsertionPoint);
        // Update the phi nodes in the exit block.
        Util::remapBlocksInPHIs(bottomInsertionPoint, topInsertionPoint,
                        newRegion->getExiting());

        topInsertionPoint = newRegion->getExiting();
        bottomInsertionPoint = getExit(*newRegion);

        delete newRegion;
        updateAliveMap(aliveMap, valueMap);
    }
}