#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include "llvm/IR/Instruction.h"
#include "llvm/IR/AbstractCallSite.h"
#include "llvm/IR/GlobalValue.h"

#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"

#include "llvm/IR/Intrinsics.h"

#include "llvm/Transforms/Utils.h"
#include <cstdio>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Use.h>
#include <llvm/IR/Value.h>
#include <llvm/IR/Attributes.h>

#include <llvm/Transforms/Utils/ValueMapper.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/IRReader/IRReader.h>

#include <cstdint>
#include <cstdlib>

#include <random>
#include <utility>
#include <vector>
#include <fstream>
#include <set>
#include <map>
#include <filesystem>
#include <string>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

#include "config.hpp"

#include "fuzztruction-preprocessing-pass.hpp"

#define DEBUG_TYPE "ft-patchpoint"

using namespace llvm;
namespace fs = std::filesystem;

// static cl::opt<std::string> PatchingVariable("patching-variable", cl::desc("Specify the variable name and the source file name to be patched"), cl::value_desc("pvar"));

/*
We need the following capabilites:
    - The ability to mutate the values loaded/stored by load and store instructions.
    - Some way to trace which store/load instructions where executed in which order.
        - via. INT3 tracing?
        - via. patch point and custom stub that is called?
        - ?We need some RT to transfer the traces to the parent
*/

enum InsTy
{
    Random = 0,
    Load = 1,
    Store = 2,
    Add = 3,
    Sub = 4,
    Icmp = 5,
    Select = 6,
    Branch = 7,
    Switch = 8
};

const char *insTyStrings[] = {"RANDOM", "LOAD", "STORE", "ADD", "SUB", "ICMP", "SELECT", "BRANCH", "SWITCH"};

typedef struct
{
    std::string insTy;
    std::string func;
    unsigned int line;
} PatchPointInfo;

#define SHM_NAME "/pingu_pass_patchpoint_id_atomic"
#define SHM_SIZE sizeof(std::atomic<int>)

class FuzztructionSourcePass : public PassInfoMixin<FuzztructionSourcePass>
{
public:
    static bool allow_ptr_ty;
    static bool allow_vec_ty;
    static std::string insTyNames[9];

    FuzztructionSourcePass();
    ~FuzztructionSourcePass();

    PreservedAnalyses run(Module &M, ModuleAnalysisManager &);
    bool initializeFuzzingStub(Module &M);
    bool injectPatchPoints(Module &M);
    std::vector<Value *> getPatchpointArgs(Module &M, uint64_t id);
    bool instrumentInsArg(Module &M, Function *stackmap_intr, Instruction *ins, uint8_t op_idx);
    bool instrumentInsOutput(Module &M, Function *stackmap_intr, Instruction *ins);
    bool maybeDeleteFunctionCall(Module &M, CallInst *call_ins, std::set<std::string> &target_functions);
    bool filterInvalidPatchPoints(Module &M);
    bool replaceMemFunctions(Module &M);
    void recordPatchPointInfo(Instruction &I);

private:
    std::vector<std::tuple<std::string, std::string>> patchingVariables;
    std::vector<PatchPointInfo> PatchPointInfoList;

    int shmFD;
    void *shmPtr;
    unsigned int *ppIdAtomic;
};

/*
Specify instruction types, which we want to instrument with probability p
*/
struct InsHook
{
    InsTy type;
    uint8_t probability;

    std::string to_string()
    {
        return "InsHook{ ins_ty=" + FuzztructionSourcePass::insTyNames[type] +
               ", probability=" + std::to_string(probability) + "% }";
    }
};

/*
Split a string containing multiple comma-separated keywords
and return the set of these keywords
*/
std::vector<std::string> split_string(std::string s, char delim)
{
    size_t pos_start = 0, pos_end;
    std::string token;
    std::vector<std::string> res;

    while ((pos_end = s.find(delim, pos_start)) != std::string::npos)
    {
        token = s.substr(pos_start, pos_end - pos_start);
        pos_start = pos_end + 1;
        res.push_back(token);
    }

    res.push_back(s.substr(pos_start));
    return res;
}

/*
Check if an environment variable is set.
*/
bool env_var_set(const char *env_var)
{
    const char *envp = std::getenv(env_var);
    if (envp)
        return true;
    return false;
}

/*
Convert environment variable content to a set.
Expects comma-separated list of values in the env var.
*/
std::vector<std::string> parse_env_var_list(const char *env_var)
{
    const char *envp = std::getenv(env_var);
    if (!envp)
        return std::vector<std::string>();
    return split_string(std::string(envp), /* delim = */ ',');
}

std::string addPrefixToFilename(const std::string &filePath, const std::string &prefix)
{
    fs::path pathObj(filePath);

    // 获取文件名和扩展名
    std::string filename = pathObj.filename().string();
    std::string newFilename = prefix + filename;

    // 构造新的路径
    fs::path newPath = pathObj.parent_path() / newFilename;

    return newPath.string();
}

inline bool operator<(const InsHook &lhs, const InsHook &rhs)
{
    return lhs.type < rhs.type;
}

FuzztructionSourcePass::FuzztructionSourcePass()
{
    shmFD = shm_open(SHM_NAME, O_RDWR | O_CREAT, 0666);
    if (shmFD == -1)
    {
        perror("shm_open");
        exit(1);
    }

    int ft = ftruncate(shmFD, SHM_SIZE);
    if (ft < 0)
    {
        perror("ftruncate");
        close(shmFD);
        exit(1);
    }

    shmPtr = mmap(NULL, SHM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, shmFD, 0);
    if (shmPtr == MAP_FAILED)
    {
        perror("mmap");
        close(shmFD);
        exit(1);
    }

    // Access the atomic integer in shared memory
    ppIdAtomic = reinterpret_cast<unsigned int *>(shmPtr);
    if (ppIdAtomic == nullptr)
    {
        perror("reinterpret_cast");
        close(shmFD);
        exit(1);
    }
}

FuzztructionSourcePass::~FuzztructionSourcePass()
{
    close(shmFD);
}

PreservedAnalyses FuzztructionSourcePass::run(Module &M, ModuleAnalysisManager &MAM)
{
    dbgs() << "FT: FuzztructionSourcePass run on file: " << M.getSourceFileName() << "\n";
    if (env_var_set("PATCHING_VARIABLES"))
    {
        for (std::string s : parse_env_var_list("PATCHING_VARIABLES"))
        {
            dbgs() << "FT: Parsed patching variable: " << s << "\n";
            int pos = s.find_first_of(':');
            if (pos == std::string::npos)
                continue;
            std::string file = s.substr(0, pos);
            std::string var = s.substr(pos + 1);
            patchingVariables.push_back(std::make_tuple(file, var));
        }
    }

    bool ModuleModified = false;

    ModuleModified |= initializeFuzzingStub(M);
    ModuleModified |= injectPatchPoints(M);
    ModuleModified |= filterInvalidPatchPoints(M);

    if (PatchPointInfoList.size() > 0)
    {
        // Dump the PatchPointInfos into the file: .SROUCE_FILE_NAME.pgpp.csv
        auto fileName = addPrefixToFilename(M.getSourceFileName() + ".pgpp.csv", ".");
        std::ofstream file(fileName, std::ios::trunc);
        file << "insTy, func, line" << std::endl;
        if (!file)
        {
            errs() << "Failed to open file " << fileName << " for writing\n";
            exit(1);
        }
        for (auto &info : PatchPointInfoList)
        {
            file << info.insTy << ", " << info.func << ", " << info.line << std::endl;
        }
        file.close();
    }

    dbgs() << "FT: Current global patchpoint id: " << __atomic_load_n(ppIdAtomic, __ATOMIC_SEQ_CST) << "\n";

    std::error_code ErrorCode;
    std::string ModuleFileName = M.getSourceFileName();
    raw_fd_ostream OutputFile(ModuleFileName.append(".ll"), ErrorCode);
    M.print(OutputFile, NULL);
    OutputFile.close();

    if (ModuleModified)
    {
        return PreservedAnalyses::none();
    }
    else
    {
        return PreservedAnalyses::all();
    }
}

void FuzztructionSourcePass::recordPatchPointInfo(Instruction &I)
{
    PatchPointInfo info;

    if (auto *Loc = I.getDebugLoc().get())
    {
        info.line = Loc->getLine();
    }

    if (auto *Func = I.getFunction())
    {
        info.func = Func->getName().str();
    }

    if (auto *load_op = dyn_cast<LoadInst>(&I))
    {
        info.insTy = "LOAD";
    }
    else if (auto *store_op = dyn_cast<StoreInst>(&I))
    {
        info.insTy = "STORE";
    }
    else if (I.getOpcode() == Instruction::Add)
    {
        info.insTy = "ADD";
    }
    else if (I.getOpcode() == Instruction::Sub)
    {
        info.insTy = "SUB";
    }
    else if (I.getOpcode() == Instruction::ICmp)
    {
        info.insTy = "ICMP";
    }
    else if (I.getOpcode() == Instruction::Select)
    {
        info.insTy = "SELECT";
    }
    else if (I.getOpcode() == Instruction::Br)
    {
        info.insTy = "BRANCH";
    }
    else if (I.getOpcode() == Instruction::Switch)
    {
        info.insTy = "SWITCH";
    }
    else
    {
        info.insTy = "RANDOM";
    }

    PatchPointInfoList.push_back(info);
}

/*
Extract integer specified in environment variable.
*/
uint32_t parse_env_var_int(const char *env_var, uint32_t default_val)
{
    const char *envp = std::getenv(env_var);
    if (!envp)
        return default_val;
    uint32_t val = (uint32_t)std::stol(envp);
    return val;
}

/*
Convert set of strings to known instruction types. Ignores unknown elements.
*/
InsTy to_InsTy(std::string input)
{
    // dbgs() << "val=" << val << "\n";
    if (input == "random")
        return InsTy::Random;
    if (input == "load")
        return InsTy::Load;
    if (input == "store")
        return InsTy::Store;
    if (input == "add")
        return InsTy::Add;
    if (input == "sub")
        return InsTy::Sub;
    if (input == "icmp")
        return InsTy::Icmp;
    if (input == "select")
        return InsTy::Select;
    if (input == "branch")
        return InsTy::Branch;
    if (input == "switch")
        return InsTy::Switch;

    errs() << "Unsupported instruction string received: " << input << "\n";
    exit(1);
}

/*
Convert a string of format "name:probability" to InsHook struct.
*/
InsHook to_InsHook(std::string s)
{
    int pos = s.find_first_of(':');
    if (pos == std::string::npos)
        return {to_InsTy(s), 100};
    std::string name = s.substr(0, pos);
    uint32_t prob = std::stol(s.substr(pos + 1));
    assert(prob <= 100 && "Probability must be in range [0, 100]");
    return {to_InsTy(name), (uint8_t)prob};
}

bool FuzztructionSourcePass::initializeFuzzingStub(Module &M)
{
    /*
    Used to initialize our fuzzing stub. We can not use the llvm constructor attribute because
    our stub relies on keystone which has static constructors that are executed after functions
    marked by the constructor attribute. Hence, we can not use keystone at that point in time.
    */
    auto hook_fn = M.getOrInsertFunction("__ft_auto_init", FunctionType::getVoidTy(M.getContext()));
    auto main_fn = M.getFunction("main");
    if (main_fn)
    {
        IRBuilder<> ins_builder(main_fn->getEntryBlock().getFirstNonPHI());
        ins_builder.CreateCall(hook_fn);
    }

    return true;
}

/*
Delete call if one of the functions specified by name is called
*/
bool FuzztructionSourcePass::maybeDeleteFunctionCall(Module &M, CallInst *call_ins, std::set<std::string> &target_functions)
{
    Function *callee = call_ins->getCalledFunction();
    // skip indirect calls
    if (!callee)
    {
        return false;
    }
    // if called function should be deleted, erase it from IR
    if (target_functions.count(callee->getName().str()))
    {
        // if the callee expects a ret value, we cannot simply replace the function
        // TODO: we could determine type and replace Inst with Value
        if (!call_ins->getCalledFunction()->getReturnType()->isVoidTy())
        {
            errs() << "Cannot delete " << callee->getName() << " as it returns\n";
            return false;
        }
        dbgs() << "FT: deleteFunctionCalls(): Deleting call to " << callee->getName() << "\n";
        call_ins->eraseFromParent();
        return true;
    }
    return false;
}

/*
Get vector of default patchpoint arguments we need for every patchpoint.
ID is set depending on which type of instruction is instrumented.
*/
std::vector<Value *> FuzztructionSourcePass::getPatchpointArgs(Module &M, uint64_t id)
{
    IntegerType *i64_type = IntegerType::getInt64Ty(M.getContext());
    IntegerType *i32_type = IntegerType::getInt32Ty(M.getContext());
    IntegerType *i8_type = IntegerType::getInt8Ty(M.getContext());

    std::vector<Value *> patchpoint_args;

    /* The ID of this patch point */
    Constant *c = ConstantInt::get(i64_type, id);
    // Constant *id = ConstantInt::get(i64_type, 0xcafebabe);
    patchpoint_args.push_back(c);

    /* Set the shadown length in bytes */
    Constant *shadow_len = ConstantInt::get(i32_type, FT_PATCH_POINT_SIZE);
    patchpoint_args.push_back(shadow_len);

    /*The function we are calling */
    auto null_ptr = ConstantPointerNull::get(PointerType::get(i8_type, 0));
    // Constant *fnptr = ConstantInt::get(i32_type, 1);
    // auto null_ptr = ConstantExpr::getIntToPtr(fnptr, PointerType::get(i8_type, 0));
    patchpoint_args.push_back(null_ptr);

    /*
    The number of args that should be considered as function arguments.
    Reaming arguments are the live values for which the location will be
    recorded.
     */
    Constant *argcnt = ConstantInt::get(i32_type, 0);
    patchpoint_args.push_back(argcnt);

    return patchpoint_args;
}

/*
Instrument the output value of the instruction. In other words, the value produced by the instruction
is the live value fed into the patchpoint.
*/
bool FuzztructionSourcePass::instrumentInsOutput(Module &M, Function *stackmap_intr, Instruction *ins)
{
    // dbgs() << "instrumentInsOutput called\n";
    Instruction *next_ins = ins;
    /* In case of a load the patchpoint is inserted after the load was executed */
    if (ins)
        next_ins = ins->getNextNode();
    if (!next_ins)
        return false;

    IRBuilder<> ins_builder(next_ins);

    /*
        declare void
        @llvm.experimental.patchpoint.void(i64 <id>, i32 <numBytes>,
                                            i8* <target>, i32 <numArgs>, ...)
    */
    uint64_t id = __atomic_fetch_add(ppIdAtomic, 1, __ATOMIC_SEQ_CST);
    dbgs() << "FT: Inserting patchpoint with id: " << id << "\n";
    // Higher 32 bit is id
    // Lower 32 bit is ins type
    // uint64_t id_ins = (id << 32) | ins->getOpcode();
    std::vector<Value *> patchpoint_args = getPatchpointArgs(M, id);
    patchpoint_args.push_back(ins);
    ins_builder.CreateCall(stackmap_intr, patchpoint_args);

    return true;
}

/*
Instrument (one of) the input value(s) to the instruction (as specified by operand index).
This input value is the live value connected to the patchpoint, where it can be modified before being
processed by the instruction.
*/
bool FuzztructionSourcePass::instrumentInsArg(Module &M, Function *stackmap_intr, Instruction *ins, uint8_t op_idx)
{
    // dbgs() << "instrumentInsArg called\n";
    if (!ins)
        return false;

    IRBuilder<> ins_builder(ins);

    /*
        declare void
        @llvm.experimental.patchpoint.void(i64 <id>, i32 <numBytes>,
                                            i8* <target>, i32 <numArgs>, ...)
    */
    uint64_t id = __atomic_fetch_add(ppIdAtomic, 1, __ATOMIC_SEQ_CST);
    dbgs() << "FT: Inserting patchpoint with id " << id << ": " << M.getSourceFileName() << "\n";
    // Higher 32 bit is id
    // Lower 32 bit is ins type
    // uint64_t id_ins = (id << 32) | ins->getOpcode();
    std::vector<Value *> patchpoint_args = getPatchpointArgs(M, id);

    /* We want to modify argument at op_idx (e.g., 0 for stores) */
    patchpoint_args.push_back(ins->getOperand(op_idx));
    ins_builder.CreateCall(stackmap_intr, patchpoint_args);

    return true;
}

bool isValidTy(Type *ty)
{
    if (ty->isIntegerTy())
        return true;
    if (FuzztructionSourcePass::allow_ptr_ty && ty->isPointerTy())
        return true;
    if (FuzztructionSourcePass::allow_vec_ty && ty->isVectorTy())
        return true;
    return false;
}

/*
Check whether it is reasonable to instrument the given instruction.
Ensure that
1) at least one user exists (else the value will never be used)
2) we support the type (integer, vec, and ptr types currently)
3) we exclude "weird" instructions (e.g., debug instructions, phi nodes etc)
*/
bool canBeInstrumented(Instruction *ins)
{
    // ignore instructions that are never used
    if (ins->users().begin() == ins->users().end())
        return false;
    // ignore non-integer type instructions
    if (!isValidTy(ins->getType()))
        return false;
    if (ins->isKnownSentinel())
        return false;
    if (ins->isCast())
        return false;
    // if (ins->isDebugOrPseudoInst())
    //     return false;
    if (ins->isExceptionalTerminator())
        return false;
    if (ins->isLifetimeStartOrEnd())
        return false;
    if (ins->isEHPad())
        return false;
    if (ins->isFenceLike())
        return false;
    if (ins->isSwiftError())
        return false;
    if (ins->getOpcode() == Instruction::PHI)
        return false;
    return true;
}

/*
Instrument all instructions and delete function calls specified by the user via environment variables.

User can specify instruction types ("load", "store"), for which we want to insert a patchpoint
as well as function names ("abort"), for which we erase any call to (if possible).
Function names are specified in FT_NOP_FN=abort,_bfd_abort.

Instruction types are specified in FT_HOOK_INS=store:50,load,add
Format is 'instruction_name':'probability of selecting a specific instance'.
Instruction name must be one of the following: add, sub, store, load, random

The value random is special in the sense that each instruction we can instrument, is actually instrumented.
We recommend to set a probability, at least for random (to avoid instrumenting too many instructions).
*/
bool FuzztructionSourcePass::injectPatchPoints(Module &M)
{
    std::vector<std::string> patchingVariableNames;
    if (!patchingVariables.empty())
    {
        // Check whether the module file name is in the patchingVariable list
        std::string fileName = fs::path(M.getName().str()).filename().string();
        for (auto &pv : patchingVariables)
        {
            if (fileName == std::get<0>(pv))
            {
                dbgs() << "FT: File " << fileName << " is in the specified patchingVariable list\n";
                patchingVariableNames.push_back(std::get<1>(pv));
            }
        }

        if (patchingVariableNames.empty())
        {
            dbgs() << "FT: File " << fileName << " is not in the specified patchingVariable list\n";
            return false;
        }
    }

    /* Get the patchpoint intrinsic */
    Function *stackmap_intr = Intrinsic::getDeclaration(&M,
                                                        Intrinsic::experimental_patchpoint_void);
    stackmap_intr->setCallingConv(CallingConv::AnyReg);

    FuzztructionSourcePass::allow_ptr_ty = !env_var_set("FT_NO_PTR_TY");
    FuzztructionSourcePass::allow_vec_ty = !env_var_set("FT_NO_VEC_TY");

    // Get functions which should not be called (i.e., for which we delete calls to)
    auto fn_del_vec = parse_env_var_list("FT_NOP_FN");
    std::set<std::string> fn_del(fn_del_vec.begin(), fn_del_vec.end());
    dbgs() << "FT: Deleting function calls to " << fn_del.size() << " functions\n";

    // Get instruction types we want to instrument
    std::set<InsHook> hook_ins = {};
    for (std::string e : parse_env_var_list("FT_HOOK_INS"))
    {
        dbgs() << "FT DEBUG: parsed ins_hook: " << to_InsHook(e).to_string() << "\n";
        hook_ins.insert(to_InsHook(e));
    }
    dbgs() << "FT: Instrumenting " << hook_ins.size() << " types of instructions\n";
    if (!hook_ins.size())
    {
        errs() << "FT: FT_HOOK_INS is not set\n";
    }

    // use random number from hardware to seed mersenne twister prng
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distr(0, 100); // inclusive [0, 100]

    // Track whether we modified the module
    bool modified = false;
    uint64_t num_patchpoints = 0;
    for (auto &F : M)
    {
        for (auto &B : F)
        {
            for (BasicBlock::iterator DI = B.begin(); DI != B.end();)
            {
                // ensure that iterator points to next instruction
                // in case we need to delete the instruction
                Instruction &I = *DI++;

                if (auto *call_ins = dyn_cast<CallInst>(&I))
                {
                    bool deleted = maybeDeleteFunctionCall(M, call_ins, fn_del);
                    modified |= deleted;
                    // No point to continue if we just deleted the instruction
                    if (deleted)
                        continue;
                }

                // Check if the current instruction is hooked.
                for (const auto &ins_hook : hook_ins)
                {
                    bool ins_modified = false;
                    switch (ins_hook.type)
                    {
                    case InsTy::Load:
                        if (auto *load_op = dyn_cast<LoadInst>(&I))
                        {
                            if (distr(gen) <= ins_hook.probability)
                            {
                                auto ty = load_op->getType();
                                if (ty->isPointerTy() || ty->isVectorTy() || ty->isPtrOrPtrVectorTy() || ty->isFunctionTy())
                                {
                                    // Skip codes like:
                                    // %wide.vec = load <4 x i64>, ptr %3, align 8, !tbaa !378
                                    // call void (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.void(i64 32, i32 32, ptr null, i32 0, <4 x i64> %wide.vec)
                                    // TODO:
                                    // Currently, the type information of the operand will not be preserved.
                                    // So whether an operand/variable is a integer scalar, float or vector type
                                    // is unknown.
                                    // Attaching the type information to the operand would be promising.
                                    continue;
                                }
                                if (env_var_set("PATCHING_VARIABLES"))
                                {
                                    auto IValueName = load_op->getPointerOperand()->getName();
                                    for (auto &pvn : patchingVariableNames)
                                    {
                                        auto lowerIValueName = IValueName.lower();
                                        if (lowerIValueName.find(pvn) != std::string::npos)
                                        {
                                            ins_modified = instrumentInsOutput(M, stackmap_intr, &I);
                                            recordPatchPointInfo(I);
                                            break;
                                        }
                                    }
                                }
                                else
                                {
                                    ins_modified = instrumentInsOutput(M, stackmap_intr, &I);
                                    recordPatchPointInfo(I);
                                }
                            }
                        }
                        break;
                    case InsTy::Store:
                        if (auto *store_op = dyn_cast<StoreInst>(&I))
                        {
                            if (distr(gen) <= ins_hook.probability)
                            {
                                auto ty = store_op->getValueOperand()->getType();
                                if (ty->isPointerTy() || ty->isVectorTy() || ty->isPtrOrPtrVectorTy() || ty->isFunctionTy())
                                {
                                    continue;
                                }
                                ins_modified = instrumentInsArg(M, stackmap_intr, &I, /* op_idx = */ 0);
                                recordPatchPointInfo(I);
                            }
                        }
                        break;
                    case InsTy::Add:
                        if (I.getOpcode() == Instruction::Add)
                        {
                            if (distr(gen) <= ins_hook.probability)
                            {
                                ins_modified = instrumentInsArg(M, stackmap_intr, &I, /* op_idx = */ 0);
                                recordPatchPointInfo(I);
                            }
                        }
                        break;
                    case InsTy::Sub:
                        if (I.getOpcode() == Instruction::Sub)
                        {
                            if (distr(gen) <= ins_hook.probability)
                            {
                                ins_modified = instrumentInsArg(M, stackmap_intr, &I, /* op_idx = */ 1);
                                recordPatchPointInfo(I);
                            }
                        }
                        break;
                    case InsTy::Icmp:
                        if (I.getOpcode() == Instruction::ICmp)
                        {
                            if (distr(gen) <= ins_hook.probability)
                            {
                                ins_modified = instrumentInsOutput(M, stackmap_intr, &I);
                                recordPatchPointInfo(I);
                            }
                        }
                        break;
                    case InsTy::Select:
                        if (I.getOpcode() == Instruction::Select)
                        {
                            if (distr(gen) <= ins_hook.probability)
                            {
                                // Arg 0 is the selection mask
                                ins_modified = instrumentInsArg(M, stackmap_intr, &I, /* op_idx = */ 0);
                                recordPatchPointInfo(I);
                            }
                        }
                        break;
                    case InsTy::Branch:
                        // FIXME: Fails to compile.
                        // if (I.getOpcode() == Instruction::Br && distr(gen) <= ins_hook.probability) {
                        //     // Arg 0 is the branch condition (i1)
                        //     ins_modified = instrumentInsArg(M, stackmap_intr, &I, /* op_idx = */ 0);
                        // }
                        break;
                    case InsTy::Switch:
                        if (I.getOpcode() == Instruction::Switch && distr(gen) <= ins_hook.probability)
                        {
                            // Arg 0 is the switch condition (i1)
                            ins_modified = instrumentInsArg(M, stackmap_intr, &I, /* op_idx = */ 0);
                            recordPatchPointInfo(I);
                        }
                        break;
                    case InsTy::Random:
                        if (!canBeInstrumented(&I))
                            break;
                        if (distr(gen) <= ins_hook.probability)
                        {
                            ins_modified = instrumentInsOutput(M, stackmap_intr, &I);
                            recordPatchPointInfo(I);
                        }
                    }
                    if (ins_modified)
                    {
                        modified = true;
                        num_patchpoints++;
                        // instruction cannot have multiple types
                        // no point in trying other types if we just matched
                        break;
                    }
                }
            }
        }
        // llvm::errs() << "dump-start\n";
        // F.dump();
    }
    dbgs() << "FT: Inserted " << num_patchpoints << " patchpoints\n";

    return modified;
}

/*
Filter & delete patchpoints if the live value is already used
by another patchpoint.
*/
bool FuzztructionSourcePass::filterInvalidPatchPoints(Module &M)
{
    bool modified = false;
    Function *stackmap_intr = Intrinsic::getDeclaration(&M,
                                                        Intrinsic::experimental_patchpoint_void);
    stackmap_intr->setCallingConv(CallingConv::AnyReg);

    int num_users = 0;
    dbgs() << "FT: Filtering invalid patch points\n";
    std::set<Value *> used_values = {};
    std::set<Instruction *> pending_deletions = {};
    for (const auto &user : stackmap_intr->users())
    {
        num_users++;
        if (CallBase *call_ins = dyn_cast<CallBase>(user))
        {
            // errs() << "is sen: " << call_ins->isKnownSentinel() << "\n";
            for (unsigned i = 4; i < call_ins->arg_size(); ++i)
            {
                // errs() << "call ins\n";
                // dbgs() << "call ins on dbg\n";
                // call_ins->dump();
                // errs().flush();
                Value *val = call_ins->getArgOperand(i);
                // errs() << "val\n";
                // val->dump();
                if (used_values.count(val) > 0)
                {
                    pending_deletions.insert(call_ins);
                    break;
                }
                else
                {
                    used_values.insert(val);
                }
            }
        }
    }
    for (auto &ins : pending_deletions)
    {
        // assert(ins->isSafeToRemove() && "Instruction is not safe to remove!");
        assert((ins->users().end() == ins->users().begin()) && "Cannot delete call instruction as it has uses");
        modified = true;
        ins->eraseFromParent();
    }
    dbgs() << "FT: Deleted " << pending_deletions.size() << "/" << num_users;
    dbgs() << " patchpoints as live values were already recorded\n";
    return modified;
}

bool FuzztructionSourcePass::allow_ptr_ty = false;
bool FuzztructionSourcePass::allow_vec_ty = false;
std::string FuzztructionSourcePass::insTyNames[] = {"random", "load", "store", "add", "sub", "icmp", "select", "br", "switch"};

void registerCallbacks(PassBuilder &PB)
{
    PB.registerOptimizerLastEPCallback(
        [&](ModulePassManager &MPM, OptimizationLevel)
        {
            if (!env_var_set("FT_DISABLE_INLINEING"))
            {
                MPM.addPass(FuzztructionSourcePreprocesssingPass());
                MPM.addPass(AlwaysInlinerPass());
            }
            MPM.addPass(FuzztructionSourcePass());
        });
}

extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo()
{
    return {
        LLVM_PLUGIN_API_VERSION, "FuzztructionSourcePass", "v0.1",
        registerCallbacks};
}

// static RegisterStandardPasses RegisterSourcePass(
//     PassManagerBuilder::EP_OptimizerLast, registerSourcePass);

// static RegisterStandardPasses RegisterSourcePass0(
//     PassManagerBuilder::EP_EnabledOnOptLevel0, registerSourcePass);

// static RegisterPass<FuzztructionSourcePreprocesssingPass> preprocessingPass("preprocessing", "Fuzztruction Source Preprocessing Pass", false, false);
// static RegisterPass<FuzztructionSourcePass> sourcePass("source", "Fuzztruction Source Pass", false, false);