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
 * \file fused_mat_mul.cpp
 * \brief FusedMatMul Kernel UT统一入口
 */

#pragma once

#include <cstring>
#include "fused_mat_mul_with_scale_add.h"

template <typename ElementType>
__global__ __aicore__ void fused_mat_mul_kernel_entry(GM_ADDR x1GM, GM_ADDR x2GM, GM_ADDR x3GM, GM_ADDR yGM,
                                                      GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    FusedMatMulUT::FusedMatMulTilingData tilingData;
    memcpy(&tilingData, tilingGM, sizeof(tilingData));
    FusedMatMulUT::FusedMatMulWithScaleAddWrapper<ElementType>(x1GM, x2GM, x3GM, yGM, workspaceGM, tilingData);
}
