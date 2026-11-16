/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * \file kernel_ut_runner.h
 * \brief Kernel UT runner wrapper that detects child process failures via stdout capture.
 *
 * CANN SDK's RunKernelFunctionOnCpu (kern_fwk.h) forks child processes per core.
 * Each child runs the kernel and calls exit(0) on success or exit(1) on error
 * (via signal handler). The parent's waitpid loop prints "[SUCCESS]" only when
 * status == 0, but silently ignores non-zero status. The function returns void,
 * so ICPU_RUN_KF cannot propagate failures to the test framework.
 *
 * This wrapper exploits an observable side effect: child processes write
 * "[SUCCESS][CORE_N]..." or "[ERROR]..." messages to stdout before exiting.
 * By temporarily redirecting stdout to a capture file before ICPU_RUN_KF,
 * we collect all child output and parse it for failure indicators after
 * ICPU_RUN_KF returns.
 *
 * This approach does NOT rewrite or replace RunKernelFunctionOnCpu.
 * It uses ICPU_RUN_KF from CANN SDK as-is, only adding stdout capture
 * and output parsing around the existing call.
 *
 * Usage:
 *   ASSERT_TRUE(KERNEL_RUN_KF(kernelFunc, blockNum, aGM, bGM, ...))
 *       << "Kernel execution failed";
 */

#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <unistd.h>
#include "kern_fwk.h"

namespace KernelUT {

/**
 * Run kernel via ICPU_RUN_KF with stdout capture to detect child failures.
 *
 * Mechanism:
 * 1. fflush all stdio to empty buffers (prevents inherited buffer duplication in children)
 * 2. dup2 stdout (fd 1) to a temp file
 * 3. Call ICPU_RUN_KF - children inherit redirected fd 1
 *    - On success: child exit(0) flushes "[SUCCESS][CORE_N]..." to capture file
 *    - On error: child signal handler prints "[ERROR]..." then exit(1) to capture file
 * 4. After ICPU_RUN_KF returns (all children reaped by SDK's waitpid):
 *    - Flush parent's stdout to capture file
 *    - Restore original stdout
 *    - Read and parse capture file for [SUCCESS]/[FAILED]/[ERROR] markers
 * 5. Return true only if [SUCCESS] present AND no [FAILED]/[ERROR] found
 */
template <typename Func, typename... Args>
bool RunAndCheck(Func func, unsigned numBlocks, Args... args)
{
    fflush(stdout);
    fflush(stderr);
    std::cout.flush();
    std::cerr.flush();

    int savedFd = dup(STDOUT_FILENO);

    char tmpPath[] = "/tmp/kernel_ut_XXXXXX";
    int captureFd = mkstemp(tmpPath);
    if (captureFd < 0) {
        // fail-closed：捕获失败即无法判定子进程成败，禁止降级为直接执行
        std::cerr << "[KERNEL_UT] mkstemp failed, cannot verify kernel result" << std::endl;
        return false;
    }

    dup2(captureFd, STDOUT_FILENO);

    ICPU_RUN_KF(func, numBlocks, args...);

    fflush(stdout);
    std::cout.flush();

    dup2(savedFd, STDOUT_FILENO);
    close(savedFd);
    std::cout.clear();

    lseek(captureFd, 0, SEEK_SET);
    std::string captured;
    char buf[4096];
    ssize_t n;
    while ((n = read(captureFd, buf, sizeof(buf))) > 0) {
        captured.append(buf, static_cast<size_t>(n));
    }
    close(captureFd);
    unlink(tmpPath);

    if (!captured.empty()) {
        printf("%s", captured.c_str());
        if (captured.back() != '\n') {
            printf("\n");
        }
        fflush(stdout);
    }

    bool hasSuccess = captured.find("[SUCCESS]") != std::string::npos;
    bool hasError = captured.find("[ERROR]") != std::string::npos;
    bool hasFailed = captured.find("[FAILED]") != std::string::npos;

    return hasSuccess && !hasError && !hasFailed;
}

} // namespace KernelUT

/**
 * KERNEL_RUN_KF: Wrapper around ICPU_RUN_KF that returns bool via stdout capture.
 *
 * Usage:
 *   ASSERT_TRUE(KERNEL_RUN_KF(kernelFunc, blockNum, arg1, arg2, ...))
 *       << "Kernel execution failed: one or more cores exited with errors";
 *
 * Returns true  if captured output contains [SUCCESS] and no [ERROR]/[FAILED].
 * Returns false if captured output contains [ERROR] or [FAILED], or no [SUCCESS].
 */
#define KERNEL_RUN_KF(func, numBlocks, ...) \
    ([&]() -> bool { return ::KernelUT::RunAndCheck(func, numBlocks, ##__VA_ARGS__); }())
