/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>
#include "blaze_kernel_stub.h"
#include "kernel_ut_runner.h"
#include "tikicpulib.h"
#include "kernel_operator.h"
#include "tensor_api/tensor.h"

namespace WeightQuantBatchMatmulUT {
class GmBuffer {
public:
    explicit GmBuffer(size_t bytes) : bytes_(bytes), address_(static_cast<GM_ADDR>(AscendC::GmAlloc(bytes)))
    {
        if (!address_)
            throw std::bad_alloc();
        std::memset(address_, 0, bytes_);
    }
    ~GmBuffer() { AscendC::GmFree(address_); }
    GmBuffer(const GmBuffer&) = delete;
    GmBuffer& operator=(const GmBuffer&) = delete;
    GM_ADDR Get() const { return address_; }
    template <typename T>
    void Set(const std::vector<T>& values)
    {
        if (values.size() * sizeof(T) > bytes_)
            throw std::out_of_range("GM allocation");
        std::memcpy(address_, values.data(), values.size() * sizeof(T));
    }

private:
    size_t bytes_;
    GM_ADDR address_;
};
inline int64_t Align16(int64_t value) { return (value + 15) / 16 * 16; }
} // namespace WeightQuantBatchMatmulUT
