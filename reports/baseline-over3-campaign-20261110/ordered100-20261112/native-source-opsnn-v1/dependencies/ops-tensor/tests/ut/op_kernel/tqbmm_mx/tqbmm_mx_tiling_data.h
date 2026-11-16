/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * \file tqbmm_mx_tiling_data.h
 * \brief TQBMM MX Kernel UT Tiling数据结构定义
 */

#pragma once

#include "blaze_kernel_stub.h"

#include <cstdint>

#pragma pack(push, 8)
struct TqbmmMxTilingData {
    uint32_t usedCoreNum = 0;
    uint32_t m = 0;
    uint32_t n = 0;
    uint32_t k = 0;
    uint32_t batch = 1;
    uint32_t batchSplitFactor = 1;
    uint32_t kL1 = 0;
    uint32_t baseM = 0;
    uint32_t baseN = 0;
    uint32_t baseK = 0;
    uint8_t l1BufferNum = 2;
    uint8_t l0cDB = 1;
    uint8_t bMustHitL2 = 1;
    uint8_t bias = 0;
};
#pragma pack(pop)

namespace TqbmmUT {

inline void FillTqbmmMxTilingDataDefault(TqbmmMxTilingData& t, uint32_t m, uint32_t n, uint32_t k, uint32_t batch,
                                         uint32_t blockNum)
{
    t.usedCoreNum = blockNum;
    t.m = m;
    t.n = n;
    t.k = k;
    t.batch = batch;
    t.batchSplitFactor = 1;
    t.kL1 = k;
    t.baseM = m;
    t.baseN = n;
    t.baseK = k;
    t.l1BufferNum = 2;
    t.l0cDB = 1;
    t.bMustHitL2 = 1;
    t.bias = 0;
}

} // namespace TqbmmUT
