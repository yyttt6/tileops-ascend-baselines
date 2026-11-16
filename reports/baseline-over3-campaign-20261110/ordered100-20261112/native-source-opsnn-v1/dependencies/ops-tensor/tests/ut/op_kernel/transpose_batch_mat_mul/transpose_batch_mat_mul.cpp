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
 * \file transpose_batch_mat_mul.cpp
 * \brief TransposeBatchMatMul Kernel UT统一入口
 */

#pragma once
#include "transpose_batch_mat_mul_basic.h"
#include "transpose_batch_mat_mul_tiling_data.h"

enum TbmmOpType : int8_t {
    OP_TYPE_TBMM_BASIC = 0,
    OP_TYPE_TBMM_TRANS_BATCH_A = 1,
};

template <int8_t OP_TYPE, typename DTYPE_X1, typename DTYPE_X2, typename DTYPE_Y, typename DTYPE_BIAS,
          uint64_t NON_CONTIGUOUS_TYPE = 0>
__global__ __aicore__ void transpose_batch_mat_mul_kernel_entry(GM_ADDR x1GM, GM_ADDR x2GM, GM_ADDR biasGM, GM_ADDR yGM,
                                                                GM_ADDR workspaceGM, GM_ADDR tilingGM, GM_ADDR scaleGM)
{
    const auto* tilingData = reinterpret_cast<const TbmmBasicTilingData*>(tilingGM);

    if constexpr (OP_TYPE == OP_TYPE_TBMM_BASIC || OP_TYPE == OP_TYPE_TBMM_TRANS_BATCH_A) {
        TbmmUT::TbmmBasicWrapper<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, NON_CONTIGUOUS_TYPE>(
            x1GM, x2GM, biasGM, yGM, workspaceGM, *tilingData, scaleGM);
    } else {
        static_assert((OP_TYPE == OP_TYPE_TBMM_BASIC || OP_TYPE == OP_TYPE_TBMM_TRANS_BATCH_A),
                      "Unsupported OP_TYPE value for transpose_batch_mat_mul_kernel_entry");
    }
}
