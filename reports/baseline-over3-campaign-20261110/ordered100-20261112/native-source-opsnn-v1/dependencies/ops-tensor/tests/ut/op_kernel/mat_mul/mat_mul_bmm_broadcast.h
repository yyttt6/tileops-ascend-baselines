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
 * \file mat_mul_bmm_broadcast.h
 * \brief MatMul BmmBroadCast Kernel UT Wrapper
 */

#pragma once

#include "blaze_kernel_stub.h"
#include "kernel_operator.h"
#include "tensor_api/tensor.h"

#include "blaze/gemm/kernel/kernel_universal.h"
#include "blaze/gemm/kernel/kernel_batch_matmul_broadcast.h"
#include "blaze/gemm/block/block_mmad_matmul_basic.h"
#include "blaze/gemm/block/block_scheduler_matmul_basic.h"
#include "blaze/epilogue/block/block_epilogue_empty.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "mat_mul_tiling_data.h"

namespace MatMulV3UT {

template <typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
__aicore__ inline void MatMulBmmBroadCastWrapper(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR cGM,
                                                 GM_ADDR workspaceGM, const MatMulV3BmmBroadcastTilingData& tilingData)
{
    using AType = A_TYPE;
    using BType = B_TYPE;
    using OutType = C_TYPE;
    using BiasType = BIAS_TYPE;

    using LayoutA = asc::te::nd_ext_layout_ptn;
    using LayoutB = asc::te::nd_ext_layout_ptn;
    using LayoutC = asc::te::nd_ext_layout_ptn;
    using LayoutBias = asc::te::nd_ext_layout_ptn;

    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;

    using DispatchPolicy = Blaze::Gemm::MatmulMultiBlockBasic<0, 0, Blaze::Gemm::KernelMmadMultiBlockBmmBroadcast>;

    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerMatmulBasic<ProblemShape>;

    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType, LayoutA, BType, LayoutB, OutType, LayoutC,
                                                    BiasType, LayoutBias>;

    using BlockEpilogue = Blaze::Gemm::Block::BlockEpilogueEmpty;

    using MatmulKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;

    uint64_t totalBatch = static_cast<uint64_t>(tilingData.cBatchDim0) * static_cast<uint64_t>(tilingData.cBatchDim1) *
                          static_cast<uint64_t>(tilingData.cBatchDim2) * static_cast<uint64_t>(tilingData.cBatchDim3);

    using BatchInfo = typename MatmulKernel::BatchInfo;
    using Params = typename MatmulKernel::Params;
    Params params = {
        {static_cast<int64_t>(tilingData.m), static_cast<int64_t>(tilingData.n), static_cast<int64_t>(tilingData.k),
         static_cast<int64_t>(totalBatch)},
        {aGM, bGM, cGM, biasGM, nullptr, workspaceGM, tilingData.mL1, tilingData.nL1, tilingData.kL1, tilingData.baseM,
         tilingData.baseN, tilingData.baseK, tilingData.l1BufferNum, tilingData.l0cDB, nullptr},
        {},
        {static_cast<uint32_t>(tilingData.mL1), static_cast<uint32_t>(tilingData.nL1),
         static_cast<uint32_t>(tilingData.kL1), static_cast<uint32_t>(tilingData.baseM),
         static_cast<uint32_t>(tilingData.baseN), static_cast<uint32_t>(tilingData.baseK),
         static_cast<uint32_t>(tilingData.mTailCnt), static_cast<uint32_t>(tilingData.nTailCnt),
         static_cast<uint32_t>(tilingData.mBaseTailSplitCnt), static_cast<uint32_t>(tilingData.nBaseTailSplitCnt),
         static_cast<uint32_t>(tilingData.mTailMain), static_cast<uint32_t>(tilingData.nTailMain),
         static_cast<uint8_t>(tilingData.isHf32), static_cast<uint32_t>(tilingData.l2CacheDisable),
         static_cast<uint32_t>(tilingData.sliceM), static_cast<uint32_t>(tilingData.srcNdStride),
         static_cast<uint32_t>(tilingData.innerBatch)},
        BatchInfo{static_cast<uint32_t>(tilingData.aBatchDim0), static_cast<uint32_t>(tilingData.bBatchDim0),
                  static_cast<uint32_t>(tilingData.aBatchDim1), static_cast<uint32_t>(tilingData.bBatchDim1),
                  static_cast<uint32_t>(tilingData.cBatchDim1), static_cast<uint32_t>(tilingData.aBatchDim2),
                  static_cast<uint32_t>(tilingData.bBatchDim2), static_cast<uint32_t>(tilingData.cBatchDim2),
                  static_cast<uint32_t>(tilingData.aBatchDim3), static_cast<uint32_t>(tilingData.bBatchDim3),
                  static_cast<uint32_t>(tilingData.cBatchDim3), static_cast<uint32_t>(tilingData.biasBatchDimAll)}};

    MatmulKernel kernel;
    kernel(params);
}

} // namespace MatMulV3UT
