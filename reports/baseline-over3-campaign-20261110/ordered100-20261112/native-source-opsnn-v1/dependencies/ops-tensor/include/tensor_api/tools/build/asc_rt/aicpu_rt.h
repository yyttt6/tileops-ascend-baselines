/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file aicpu_rt.h
 * \brief
 */

#pragma once

#include <atomic>
#include <thread>

struct AicpuDumpThreadMutex {
    std::atomic<bool> stop{false};
};

class AicpuDumpThreadRes {
public:
    explicit AicpuDumpThreadRes(const void* dumpAddr, const size_t dumpSize, const int32_t deviceId);
    ~AicpuDumpThreadRes();
private:
    struct AicpuDumpThreadMutex mutex_;
    std::thread thread_;
};

extern "C" {
size_t* AicpuSetDumpConfig(const unsigned long *aicpuFileBuf, size_t fileSize);
int AicpuGetDumpConfig(void **addr, size_t *size);
void AicpuDumpPrintBuffer(const void *dumpBuffer, const size_t bufSize);
}
