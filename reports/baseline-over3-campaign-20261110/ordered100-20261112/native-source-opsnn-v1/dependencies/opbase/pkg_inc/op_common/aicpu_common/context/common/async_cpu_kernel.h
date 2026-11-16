/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASYNC_CPU_KERNEL_H
#define ASYNC_CPU_KERNEL_H

#include "cpu_kernel.h"

namespace aicpu {
class AICPU_VISIBILITY AsyncCpuKernel : public CpuKernel {
public:
    using CpuKernel::CpuKernel;

    using DoneCallback = std::function<void(uint32_t status)>;

    virtual uint32_t ComputeAsync(CpuKernelContext& ctx, DoneCallback done) = 0;

    uint32_t Compute(CpuKernelContext& ctx) override;
};
} // namespace aicpu
#endif // ASYNC_CPU_KERNEL_H