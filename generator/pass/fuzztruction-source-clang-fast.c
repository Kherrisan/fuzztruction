/*
This is a wrapper for clang that allows to build targets with our custom compiler
pass.
*/

#define _GNU_SOURCE
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <libgen.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include "debug.h"
#include <assert.h>
#include <sys/wait.h>

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef struct
{
    bool is_cxx;
    bool is_64bit;
    bool x_set;
    bool o_set;
    const char *input_file;
} arg_settings_t;

typedef struct
{
    char const **argv;
    int argc;
} args_t;

const char *PASS_SO_NAME = "fuzztruction-source-llvm-pass.so";
char *pass_path;

void find_pass()
{
    char *guess;
    char *cwd;

    cwd = getcwd(NULL, 0);
    if (!cwd)
    {
        PFATAL("Failed to get CWD");
    }

    /* Test if we find it in the cwd  */
    if (asprintf(&guess, "%s/%s", cwd, PASS_SO_NAME) < 0)
    {
        free(cwd);
        PFATAL("Failed to allocate");
    }
    if (!access(guess, R_OK))
        pass_path = guess;

    free(cwd);

    if (!pass_path)
    {
        free(pass_path);
        pass_path = NULL;
    }
    else
    {
        goto done;
    }

    // FIXME: this path should not be absolute.
    if (asprintf(&guess, "/home/ubuntu/pingu/fuzztruction/generator/pass/%s", PASS_SO_NAME) < 0)
    {
        PFATAL("Failed to allocate");
    }
    if (!access(guess, R_OK))
        pass_path = guess;

done:

    if (!pass_path)
    {
        free(pass_path);
        FATAL("Failed to find %s\n", PASS_SO_NAME);
    }
}

arg_settings_t *parse_argv(char const *argv[], int argc)
{
    arg_settings_t *self = malloc(sizeof(*self));
    if (!self)
        PFATAL("Error during malloc");

    memset(self, 0x00, sizeof(*self));

    char *argv0 = strdup(argv[0]);
    if (!argv0)
        PFATAL("Error durring alloc");

    /* name points into argv0 */
    char *name = basename(argv0);
    if (!strcmp(name, "fuzztruction-source-clang-fast++"))
    {
        // printf("#fuzztruction-source-clang-fast++ was called\n");
        self->is_cxx = true;
    }
    free(argv0);

    bool lastIsFlag = true;
    while (argc--)
    {
        const char *cur = *(argv++);

        if (!strcmp(cur, "-m32"))
            self->is_64bit = false;
        if (!strcmp(cur, "-m64"))
            self->is_64bit = true;
        if (!strcmp(cur, "-x"))
            self->x_set = true;
        if (cur[0] == '-' && cur[1] != 'c')
        {
            // ignore the -c flag
            lastIsFlag = true;
        }
        else
        {
            if (!lastIsFlag)
            {
                self->input_file = cur;
            }
            lastIsFlag = false;
        }
    }

    // printf("input file is %s\n", self->input_file);

    return self;
}

char *change_extension_to_ll(const char *filename)
{
    // Find the last occurrence of '.'
    const char *dot = strrchr(filename, '.');
    size_t length = strlen(filename);
    char *new_filename = (char *)malloc(length + 4); // +3 for .ll and +1 for the null terminator
    if (!new_filename)
    {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    if (dot)
    {
        // Copy the part before the dot
        strncpy(new_filename, filename, dot - filename);
        new_filename[dot - filename] = '\0';

        // Append .ll
        strcat(new_filename, ".ll");
    }
    else
    {
        // If there's no dot, just append .ll to the original filename
        strcpy(new_filename, filename);
        strcat(new_filename, ".ll");
    }

    return new_filename;
}

void compile(int argc, const char *argv[], arg_settings_t *arg_settings, bool to_llvm_ir)
{
    const char *llvm_ir_file;
    const int max_args = argc + 64;
    args_t *self = malloc(sizeof(*self));
    self->argc = 0;
    self->argv = malloc(sizeof(*self->argv) * max_args);

    /* Inject/Replace arguments */
    self->argv[self->argc++] = arg_settings->is_cxx ? "/data/exp/dkzou/llvm-17.0.6-ft/build/bin/clang++" : "/data/exp/dkzou/llvm-17.0.6-ft/build/bin/clang";
    // self->argv[self->argc++] = arg_settings->is_cxx ? "clang++" : "clang";
    // Ignore unkown args
    self->argv[self->argc++] = "-Qunused-arguments";

    if (to_llvm_ir)
    {
        // Compile to llvm IR code
        self->argv[self->argc++] = "-S";
        self->argv[self->argc++] = "-emit-llvm";

        // Run our pass
        char *pass_plugin_arg = malloc(strlen(pass_path) + 64);
        sprintf(pass_plugin_arg, "-fpass-plugin=%s", pass_path);
        self->argv[self->argc++] = "-Xclang";
        self->argv[self->argc++] = pass_plugin_arg;
    }

    // Make sure llvm does not use builtins, since we want to
    // replace all calls with out custom instrumented implementations.
    self->argv[self->argc++] = "-fno-builtin-memcpy";
    self->argv[self->argc++] = "-fno-builtin-memmove";
    self->argv[self->argc++] = "-fno-slp-vectorize";
    self->argv[self->argc++] = "-fno-vectorize";

    // self->argv[self->argc++] = "-mno-sse2";
    self->argv[self->argc++] = "-mno-avx";

    self->argv[self->argc++] = "-fno-discard-value-names";

    /* Process initially passed arguments and potentially drop some of these */
    const char **current = &argv[1];
    while (*current)
    {
        if (!strcmp(*current, "-Wl,-z,defs") || !strcmp(*current, "-Wl,--no-undefined"))
        {
            current++;
            continue;
        }
        if (to_llvm_ir)
        {
            // Compile src/source.c or cc or cpp to src/source.ll
            if (!strcmp(*current, "-emit-obj") || !strcmp(*current, "-c"))
            {
                current++;
                continue;
            }
            if (!strcmp(*current, "-o"))
            {
                // printf("found -o\n");
                llvm_ir_file = change_extension_to_ll(arg_settings->input_file);
                // printf("llvm_ir_file: %s\n", llvm_ir_file);
                self->argv[self->argc++] = "-o";
                self->argv[self->argc++] = llvm_ir_file;
                current += 2;
                continue;
            }
            self->argv[self->argc++] = *current;
            current++;
        }
        else
        {
            // Compile source.ll to source.o or obj
            if (!strcmp(*current, arg_settings->input_file))
            {
                llvm_ir_file = change_extension_to_ll(*current);
                self->argv[self->argc++] = llvm_ir_file;
                current++;
                continue;
            }
            self->argv[self->argc++] = *current;
            current++;
        }
    }

    // Link against our agent that is called by a call our pass injected into main().
    // FIXME: this path should not be absolute.
    self->argv[self->argc++] = "-L/home/ubuntu/pingu/fuzztruction/target/debug";
    self->argv[self->argc++] = "-lgenerator_agent";

    // Enable debug output.
    self->argv[self->argc] = NULL;

    int pid = fork();
    if (pid == 0)
    {
        int ret = execvp(self->argv[0], (char **)self->argv);
        if (ret != 0)
        {
            if (to_llvm_ir)
            {
                fprintf(stderr, "Error in compiling to %s", llvm_ir_file);
                PFATAL("Failed to execute %s\n", self->argv[0]);
            }
            else
            {
                fprintf(stderr, "Error in compiling from %s", llvm_ir_file);
                PFATAL("Failed to execute %s\n", self->argv[0]);
            }
        }
    }
    else if (pid > 0)
    {
        int status;
        waitpid(pid, &status, 0);
        if (status != 0)
        {
            fprintf(stderr, "Error in executing in the child process\n");
            fprintf(stderr, "Child exited with %d\n", status);
            exit(1);
        }
    }
    else
    {
        fprintf(stderr, "Error in executing in the child process\n");
        exit(1);
    }
}

int main(int argc, char const *argv[])
{
    // printf("raw arguments:\n");
    // printf("#argc=%d\n", argc);
    // for (int i = 0; i < argc; i++)
    // {
    //     printf("#[%d]=%s\n", i, argv[i]);
    // }
    // fflush(NULL);

    arg_settings_t *arg_settings;

    if (argc < 2)
    {
        FATAL("Not enough arguments");
    }

    /*
    Get the path to the runtime object file and the pass library.
    Sets pass_path.
    */
    find_pass();

    /* Parse the flags intended for clang and deduce information we might need */
    arg_settings = parse_argv(argv, argc);

    compile(argc, argv, arg_settings, true);
    compile(argc, argv, arg_settings, false);

    // new_args = rewrite_argv(argv, argc, arg_settings);
    free(arg_settings);

    // printf("rewritten call:\n");
    // printf("#argc=%d\n", new_args->argc);
    // for (int i = 0; i < new_args->argc; i++) {
    //     printf("#[%d]=%s\n", i, new_args->argv[i]);
    // }
    // fflush(NULL);

    // execvp(new_args->argv[0], (char **)new_args->argv);

    // PFATAL("Failed to execute %s\n", new_args->argv[0]);

    return 0;
}
