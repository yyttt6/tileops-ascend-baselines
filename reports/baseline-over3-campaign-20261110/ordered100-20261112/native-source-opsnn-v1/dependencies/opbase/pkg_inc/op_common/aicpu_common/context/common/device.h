/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AICPU_CONTEXT_COMMON_DEVICE_H
#define AICPU_CONTEXT_COMMON_DEVICE_H

#include "sharder.h"

namespace aicpu {
class Device {
public:
    explicit Device(DeviceType device);

    ~Device();

    /*
     * get device type.
     * @return DeviceType: HOST/DEVICE
     */
    DeviceType GetDeviceType() const;

    /*
     * get sharder.
     * @return Sharder *: host or device sharder
     */
    const Sharder* GetSharder() const;

private:
    Device(const Device&) = delete;
    Device(Device&&) = delete;
    Device& operator=(const Device&) = delete;
    Device& operator=(Device&&) = delete;

    /*
     * init sharder.
     * param device_type: type of device
     * @return Sharder *: not null->success, null->success
     */
    Sharder* InitSharder(DeviceType device_type) const;

    DeviceType device_; // type of device
    Sharder* sharder_;
};
} // namespace aicpu
#endif // AICPU_CONTEXT_COMMON_DEVICE_H
