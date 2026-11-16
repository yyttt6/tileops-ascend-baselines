/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AICPU_CONTEXT_COMMON_DEVICE_SHARDER_H
#define AICPU_CONTEXT_COMMON_DEVICE_SHARDER_H
#include "sharder.h"

namespace aicpu {
using ParallelForFunc = void (*)(int64_t total, int64_t per_unit_size,
                                 const std::function<void(int64_t, int64_t)>& work);
using GetCPUNumFunc = uint32_t (*)();
class DeviceSharder : public Sharder {
public:
    explicit DeviceSharder(DeviceType device);

    ~DeviceSharder() override;

    /*
     * ParallelFor shards the "total" units of work.
     * @param total: size of total work
     * @param per_unit_size: expect size of per unit work
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
    DeviceSharder(const DeviceSharder&) = delete;
    DeviceSharder(DeviceSharder&&) = delete;
    DeviceSharder& operator=(const DeviceSharder&) = delete;
    DeviceSharder& operator=(DeviceSharder&&) = delete;

    void* sharder_;
    ParallelForFunc parallel_for_;
    GetCPUNumFunc get_cpu_num_;
};
} // namespace aicpu
#endif // AICPU_CONTEXT_COMMON_DEVICE_SHARDER_H
