#pragma once

#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include "llvm/IR/Instruction.h"
#include "llvm/IR/GlobalValue.h"

#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"

#include "llvm/IR/Intrinsics.h"

#include "llvm/Transforms/Utils.h"
#include <cstdio>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Use.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/Debug.h>
#include <llvm/Transforms/Utils/ValueMapper.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/IRReader/IRReader.h>

#include "llvm/Transforms/IPO/AlwaysInliner.h"

#include <cstdint>
#include <cstdlib>

#include <random>
#include <utility>
#include <vector>

#include "config.hpp"

using namespace llvm;

class FuzztructionSourcePreprocesssingPass : public PassInfoMixin<FuzztructionSourcePreprocesssingPass>
{
public:
    PreservedAnalyses run(Module &M, ModuleAnalysisManager &);

    bool replaceMemFunctions(Module &M);
};
