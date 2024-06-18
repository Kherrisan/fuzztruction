#We only support clang, so we fix this variable!
BASE = /data/exp/dkzou/llvm-17.0.6-ft/build
CC = $(BASE)/bin/clang
CXX = $(BASE)/bin/clang++
AR = $(BASE)/bin/llvm-ar

CFLAGS      ?= -O0 -funroll-loops -g3 -Wall -Wno-pointer-sign
CXXFLAGS    ?= -O0 -funroll-loops -g3 -Wall

# Used for llvm pass compilation
LLVM_CONFIG ?= $(BASE)/bin/llvm-config
CLANG_CFL    = `$(LLVM_CONFIG) --cxxflags` -Wl,-znodelete -fno-rtti -fpic $(CXXFLAGS) -DLLVM_DISABLE_ABI_BREAKING_CHECKS_ENFORCING
CLANG_LFL    = `$(LLVM_CONFIG) --ldflags` $(LDFLAGS)

TARGETS = fuzztruction-source-llvm-pass.so fuzztruction-source-clang-fast fuzztruction-source-clang-fast++ mem_functions.ll

all: $(TARGETS) config.hpp debug.h

clean:
	-rm -f $(TARGETS)
	-find -iname "*.o" -exec rm -v {} \;

# The llvm pass that injects patchpoints during compilation.
fuzztruction-source-llvm-pass.so: fuzztruction-source-llvm-pass.cc fuzztruction-preprocessing-pass.cc
	$(CXX) $(CLANG_CFL) -fPIC -shared $^ -o $@ $(CLANG_LFL)

# Wrapper around clang that injects our custom runtime and llvm pass during compilitation
# of an arbitrary target.
fuzztruction-source-clang-fast: fuzztruction-source-clang-fast.c fuzztruction-source-llvm-pass.so
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

# Link of the c++ compiler to out wrapper.
fuzztruction-source-clang-fast++: fuzztruction-source-clang-fast
	ln -sf fuzztruction-source-clang-fast fuzztruction-source-clang-fast++

# Replacement for common mem related function like memcpy, memmove,...
mem_functions.ll: mem_functions.c fuzztruction-source-clang-fast
	$(CC) -fno-builtin-memcpy -fno-builtin-memcpy -fno-builtin-memmove -fno-discard-value-names -O0 -S -emit-llvm $<

install: all
	install -m 755 fuzztruction-source-clang-fast fuzztruction-source-clang-fast++ /usr/bin
	install -m 644 fuzztruction-source-llvm-pass.so /usr/lib
