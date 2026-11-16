/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AICPU_CONTEXT_COMMON_HOST_SHARDER_H
#define AICPU_CONTEXT_COMMON_HOST_SHARDER_H
#include "sharder.h"

namespace aicpu {
class HostSharder : public Sharder {
public:
    explicit HostSharder(DeviceType device) : Sharder(device) {};

    ~HostSharder() = default;

    /*
     * ParallelFor shards the "total" units of work.
     * @param total: size of total work
     * @param perUnitSize: expect size of per unit work
     * @param work: process of per unit work
     */
    void ParallelFor(int64_t total, int64_t per_unit_size,
                     const std::function<void(int64_t, int64_t)>& work) const override;

    /*
     * Get CPU number
     * @return CPU number
     */
    uint32_t GetCPUNum() const override;

private:
    HostSharder(const HostSharder&) = delete;
    HostSharder(HostSharder&&) = delete;
    HostSharder& operator=(const HostSharder&) = delete;
    HostSharder& operator=(HostSharder&&) = delete;
};
} // namespace aicpu
#endif // AICPU_CONTEXT_COMMON_HOST_SHARDER_H
