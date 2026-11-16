/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file weight_quant_batch_matmul_mx_tiling_data.h
 * \brief CPU-debug tiling data for the Weight Quant MX Blaze component UT.
 */

#pragma once

#include <cstdint>

#pragma pack(push, 8)
struct WeightQuantBatchMatmulMxTilingData {
    int64_t m{0};
    int64_t n{0};
    int64_t k{0};
    uint64_t baseM{0};
    uint64_t baseN{0};
    uint64_t baseK{0};
    uint64_t tileShapeKL1{0};
    uint64_t tileShapeScaleKL1{0};
    uint64_t kBubSize{0};
    uint64_t nBubSize{0};
    uint64_t l1BufferNum{0};
    uint64_t hasBias{0};
    uint64_t mTailTile{1};
    uint64_t nTailTile{1};
    uint64_t mBaseTailSplitCnt{1};
    uint64_t nBaseTailSplitCnt{1};
    uint64_t mTailMain{0};
    uint64_t nTailMain{0};
};
#pragma pack(pop)
