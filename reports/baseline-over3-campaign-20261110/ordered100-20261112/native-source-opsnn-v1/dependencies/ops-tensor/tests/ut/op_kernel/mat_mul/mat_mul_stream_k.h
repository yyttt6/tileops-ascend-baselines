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
 * \file mat_mul_stream_k.h
 * \brief MatMul StreamK Kernel UT Wrapper
 */

#pragma once

#include "blaze_kernel_stub.h"
#include "kernel_operator.h"
#include "tensor_api/tensor.h"

#include "blaze/gemm/kernel/kernel_universal.h"
#include "blaze/gemm/kernel/kernel_matmul_streamk.h"
#include "blaze/gemm/block/block_mmad_matmul_streamk.h"
#include "blaze/gemm/block/block_scheduler_matmul_streamk.h"
#include "blaze/epilogue/block/block_epilogue_matmul_streamk.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "mat_mul_tiling_data.h"

namespace MatMulV3UT {

template <typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE,
          Blaze::Gemm::MatMulL0C2Out L0C2OUT_MODE = Blaze::Gemm::MatMulL0C2Out::ON_THE_FLY, uint64_t FUSED_OP_TYPE = 0>
__aicore__ inline void MatMulStreamKWrapper(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR cGM, GM_ADDR workspaceGM,
                                            const MatMulV3BasicTilingData& tilingData, int64_t batch = 0)
{
    using AType = A_TYPE;
    using BType = B_TYPE;
    using OutType = C_TYPE;
    using BiasType = BIAS_TYPE;

    using LayoutA = asc::te::nd_ext_layout_ptn;
    using LayoutB = asc::te::nd_ext_layout_ptn;
    using LayoutC = asc::te::nd_ext_layout_ptn;

    if (batch > 1) {
        return;
    }

    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;

    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerMatmulStreamK<ProblemShape>;

    using BlockMmad = Blaze::Gemm::Block::BlockMmad<
        Blaze::Gemm::MatmulMultiBlockWithStreamK<L0C2OUT_MODE, FUSED_OP_TYPE>, AType, LayoutA, BType, LayoutB, OutType,
        LayoutC, BiasType, LayoutC>;

    using FusionOp = Blaze::Gemm::Block::DefaultFusion<OutType, OutType>;

    using BlockEpilogue = Blaze::Gemm::Block::BlockEpilogueMatmulStreamK<
        float, OutType, Blaze::Gemm::MatmulMultiBlockWithStreamK<L0C2OUT_MODE, FUSED_OP_TYPE>>;

    using MatmulKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;

    using Params = typename MatmulKernel::Params;
    Params params = {
        {static_cast<int64_t>(tilingData.m), static_cast<int64_t>(tilingData.n), static_cast<int64_t>(tilingData.k),
         batch},
        {aGM, bGM, cGM, biasGM, nullptr, workspaceGM, tilingData.mL1, tilingData.nL1, tilingData.kL1, tilingData.baseM,
         tilingData.baseN, tilingData.baseK, tilingData.l1BufferNum, tilingData.l0cDB},
        {cGM, workspaceGM},
        {tilingData.usedCoreNum, tilingData.baseM, tilingData.baseN, tilingData.baseK, tilingData.skSingleCoreK,
         tilingData.kL1, tilingData.isHf32, static_cast<uint32_t>(tilingData.l2CacheDisable)}};

    MatmulKernel kernel;
    kernel(params);
}

} // namespace MatMulV3UT
