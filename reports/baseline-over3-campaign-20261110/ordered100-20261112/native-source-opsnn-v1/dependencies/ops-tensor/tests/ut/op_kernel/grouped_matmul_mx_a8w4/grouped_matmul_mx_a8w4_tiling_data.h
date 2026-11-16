/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul_mx_a8w4_tiling_data.h
 * \brief CPU-debug launch tiling for grouped MX A8W4 kernel smoke tests.
 */

#pragma once

#include <cstdint>

struct GroupedMatmulMxA8W4TilingData {
    uint32_t groupNum{0U};
    uint32_t coreNum{0U};
    uint64_t kSize{0U};
    uint64_t nSize{0U};
    uint8_t cubeNumBlocksN{0U};
    uint32_t mainBlockSize{0U};
    uint64_t mainBlockCount{0U};
    uint16_t firstTailBlockSize{0U};
    uint16_t secondTailBlockSize{0U};
    uint16_t firstTailBlockCount{0U};
    uint16_t secondTailBlockCount{0U};
    uint16_t baseM{0U};
    uint32_t groupListType{0U};
    uint32_t hasBias{0U};
};
