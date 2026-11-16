/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AICPU_CUST_CPU_UTILS_H
#define AICPU_CUST_CPU_UTILS_H
#include "cpu_context.h"
namespace aicpu {
#define CUST_KERNEL_LOG_DEBUG(ctx, fmt, ...)                                                                 \
    CustCpuKernelUtils::CustLogDebug(ctx, "[%s:%d][%s:%d]:" fmt, __FILE__, __LINE__, __FUNCTION__, __LINE__, \
                                     ##__VA_ARGS__)
#define CUST_KERNEL_LOG_INFO(ctx, fmt, ...)                                                                 \
    CustCpuKernelUtils::CustLogInfo(ctx, "[%s:%d][%s:%d]:" fmt, __FILE__, __LINE__, __FUNCTION__, __LINE__, \
                                    ##__VA_ARGS__)
#define CUST_KERNEL_LOG_WARNING(ctx, fmt, ...)                                                                 \
    CustCpuKernelUtils::CustLogWarning(ctx, "[%s:%d][%s:%d]:" fmt, __FILE__, __LINE__, __FUNCTION__, __LINE__, \
                                       ##__VA_ARGS__)
#define CUST_KERNEL_LOG_ERROR(ctx, fmt, ...)                                                                 \
    CustCpuKernelUtils::CustLogError(ctx, "[%s:%d][%s:%d]:" fmt, __FILE__, __LINE__, __FUNCTION__, __LINE__, \
                                     ##__VA_ARGS__)

class CustCpuKernelUtils {
public:
    static void CustLogDebug(aicpu::CpuKernelContext& ctx, const char* fmt, ...);
    static void CustLogWarning(aicpu::CpuKernelContext& ctx, const char* fmt, ...);
    static void CustLogInfo(aicpu::CpuKernelContext& ctx, const char* fmt, ...);
    static void CustLogError(aicpu::CpuKernelContext& ctx, const char* fmt, ...);
    static void DumpCustomLog(int32_t module_id, int32_t level, const char* fmt, ...);

private:
    static void WriteCustLog(aicpu::CpuKernelContext& ctx, uint32_t level, const char* fmt, va_list v);
    static void SafeWrite(aicpu::CpuKernelContext& ctx, char* msg, size_t len);
};
} // namespace aicpu
#endif