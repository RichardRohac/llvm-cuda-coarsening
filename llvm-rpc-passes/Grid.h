/* #ifndef NDRANGE_H
#define NDRANGE_H

#include "llvm/Pass.h"
#include "CUDACoarsening.h"

using namespace llvm;

namespace llvm {
class Function;
}

class GridAnalysis : public FunctionPass {
  void operator=(const GridAnalysis &);
  GridAnalysis(const GridAnalysis &);

public:
  static char ID;
  GridAnalysis();

  virtual bool runOnFunction(Function &function);
  virtual void getAnalysisUsage(AnalysisUsage &au) const;

public:
  //InstVector getTids();
  //InstVector getSizes();
  //InstVector getTids(int direction);
  //InstVector getSizes(int direction);

  //bool isTid(Instruction *inst);
  //bool isTidInDirection(Instruction *inst, int direction);
  //std::string getType(Instruction *inst) const;
  //bool isCoordinate(Instruction *inst) const;
  //bool isSize(Instruction *inst) const;

  //int getDirection(Instruction *inst) const;

  //bool isGlobal(Instruction *inst) const;
  //bool isLocal(Instruction *inst) const;
  //bool isGlobalSize(Instruction *inst) const;
  //bool isLocalSize(Instruction *inst) const;
  //bool isGroupId(Instruction *inst) const;
  //bool isGroupsNum(Instruction *inst) const;

  //bool isGlobal(Instruction *inst, int direction) const;
  //bool isLocal(Instruction *inst, int direction) const;
  //bool isGlobalSize(Instruction *inst, int direction) const;
  //bool isLocalSize(Instruction *inst, int direction) const;
  //bool isGroupId(Instruction *inst, int direction) const;
 // bool isGroupsNum(Instruction *inst, int dimension) const;

  //void dump();

public:
  static std::string THREAD_INDEX;
  static std::string BLOCK_INDEX;
  static std::string BLOCK_DIM;
  static std::string GRID_DIM;
  static int DIRECTION_NUMBER;

private:
  void init();
  bool isPresentInDirection(Instruction *inst, const std::string &functionName,
                            int direction) const;
  void findKernelIndexByNameAllDirs(std::string calleeName,
                                            Function *caller);

private:
  std::vector<std::map<std::string, InstVector>> oclInsts;
};

// Non-member functions.
//void findOpenCLFunctionCallsByName(std::string calleeName, Function *caller,
//                                   int dimension, InstVector &target);
//void findOpenCLFunctionCalls(Function *callee, Function *caller, int dimension,
//                             InstVector &target);

#endif
 */