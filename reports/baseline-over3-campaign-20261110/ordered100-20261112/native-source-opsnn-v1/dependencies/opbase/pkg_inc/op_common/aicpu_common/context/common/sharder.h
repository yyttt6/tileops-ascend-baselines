/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AICPU_CONTEXT_COMMON_SHARDER_H
#define AICPU_CONTEXT_COMMON_SHARDER_H
#include <functional>
#include <cstdint>
#include "cpu_types.h"

namespace aicpu {
class Sharder {
public:
    explicit Sharder(DeviceType device) : device_(device) {}

    virtual ~Sharder() = default;

    /*
     * ParallelFor shards the "total" units of work.
     * @param total: size of total work
     * @param perUnitSize: expect size of per unit work
     * @param work: process of per unit work
     */
    virtual void ParallelFor(int64_t total, int64_t per_unit_size,
                             const std::function<void(int64_t, int64_t)>& work) const = 0;

    /*
     * Get CPU number
     * @return CPU number
     */
    virtual uint32_t GetCPUNum() const = 0;

private:
    Sharder(const Sharder&) = delete;
    Sharder(Sharder&&) = delete;
    Sharder& operator=(const Sharder&) = delete;
    Sharder& operator=(Sharder&&) = delete;

    DeviceType device_; // device type, HOST/DEVICE
};
} // namespace aicpu
#endif // AICPU_CONTEXT_COMMON_SHARDER_H
