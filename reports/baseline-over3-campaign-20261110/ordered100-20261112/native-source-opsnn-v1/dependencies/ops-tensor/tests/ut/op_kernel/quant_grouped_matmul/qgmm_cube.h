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
 * \file qgmm_cube.h
 * \brief QGMM Cube kernel (kernel_qgmm_cube.h) UT wrapper for CPU smoke tests.
 */

#pragma once

#include "blaze_kernel_stub.h"
#include "kernel_operator.h"
#include "tensor_api/tensor.h"

#include "quant_grouped_matmul_cpu_debug_stub.h"

#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/kernel/kernel_universal.h"
#include "blaze/gemm/block/block_mmad_a8w8_fixpipe_quant.h"
#include "blaze/gemm/block/block_scheduler_gmm_swat_with_tail_split.h"
#include "blaze/epilogue/block/block_epilogue_empty.h"

namespace QGMMCubeUT {

#pragma pack(push, 8)
struct QgmmCubeTilingData {
    uint32_t groupNum;
    int64_t m;
    int64_t n;
    int64_t k;
    uint32_t baseM;
    uint32_t baseN;
    uint32_t baseK;
    uint32_t kAL1;
    uint32_t kBL1;
    uint32_t x1QuantMode;
    uint32_t x2QuantMode;
    uint8_t isBias;
    uint8_t dbL0C;
    int8_t groupType;
    uint8_t groupListType;
    uint8_t singleW;
    uint8_t singleX;
    uint8_t singleY;
};
#pragma pack(pop)

template <typename AType, typename BType, typename CType, typename BiasType, typename X2ScaleType,
          typename LayoutA = asc::te::nd_ext_layout_ptn, typename LayoutB = asc::te::nd_ext_layout_ptn,
          typename LayoutC = asc::te::nd_ext_layout_ptn>
__aicore__ inline void RunQgmmCube(GM_ADDR aDesc, GM_ADDR bDesc, GM_ADDR cDesc, GM_ADDR biasDesc, GM_ADDR scaleA,
                                   GM_ADDR scaleB, GM_ADDR groupList, GM_ADDR gmmArray, const QgmmCubeTilingData& t)
{
    using LayoutBias = asc::te::nd_ext_layout_ptn;
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using Policy = Blaze::Gemm::MatmulWithScaleFixpipeQuant<0UL, false, Blaze::Gemm::KernelGroupedMmadFixpipeQuant>;
    using BTypeTuple = AscendC::Std::tuple<BType, X2ScaleType>;
    using Mmad = Blaze::Gemm::Block::BlockMmad<Policy, AType, LayoutA, BTypeTuple, LayoutB, CType, LayoutC, BiasType,
                                               LayoutBias>;
    using Epilogue = Blaze::Gemm::Block::BlockEpilogueEmpty;
    using Scheduler = Blaze::Gemm::Block::BlockSchedulerGmmSwatWithTailSplit;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, Mmad, Epilogue, Scheduler>;

    typename Kernel::Params p{};
    p.problemShape = {t.m, t.n, t.k, 0};
    p.mmadParams.aGmAddr = aDesc;
    p.mmadParams.bGmAddr = bDesc;
    p.mmadParams.cGmAddr = cDesc;
    p.mmadParams.biasGmAddr = biasDesc;
    p.mmadParams.scaleAGmAddr = scaleA;
    p.mmadParams.scaleBGmAddr = scaleB;
    p.groupListGmAddr = groupList;
    p.gmmArrayGmAddr = reinterpret_cast<__gm__ int32_t*>(gmmArray);
    p.gmmParams.groupNum = t.groupNum;
    p.gmmParams.m = t.m;
    p.gmmParams.n = t.n;
    p.gmmParams.k = t.k;
    p.gmmParams.baseM = t.baseM;
    p.gmmParams.baseN = t.baseN;
    p.gmmParams.baseK = t.baseK;
    p.gmmParams.kAL1 = t.kAL1;
    p.gmmParams.kBL1 = t.kBL1;
    p.gmmParams.x1QuantMode = t.x1QuantMode;
    p.gmmParams.x2QuantMode = t.x2QuantMode;
    p.gmmParams.isBias = t.isBias;
    p.gmmParams.dbL0C = t.dbL0C;
    p.gmmParams.groupType = t.groupType;
    p.gmmParams.groupListType = t.groupListType;
    p.gmmParams.singleW = t.singleW;
    p.gmmParams.singleX = t.singleX;
    p.gmmParams.singleY = t.singleY;

    Kernel kernel;
    kernel(p);
}

} // namespace QGMMCubeUT

template <typename AType, typename BType, typename CType, typename BiasType, typename X2ScaleType,
          typename LayoutA = asc::te::nd_ext_layout_ptn, typename LayoutB = asc::te::nd_ext_layout_ptn>
__global__ __aicore__ void qgmm_cube_kernel_entry(GM_ADDR aDesc, GM_ADDR bDesc, GM_ADDR cDesc, GM_ADDR biasDesc,
                                                  GM_ADDR scaleA, GM_ADDR scaleB, GM_ADDR groupList, GM_ADDR gmmArray,
                                                  GM_ADDR tiling)
{
    QGMMCubeUT::RunQgmmCube<AType, BType, CType, BiasType, X2ScaleType, LayoutA, LayoutB>(
        aDesc, bDesc, cDesc, biasDesc, scaleA, scaleB, groupList, gmmArray,
        *reinterpret_cast<const QGMMCubeUT::QgmmCubeTilingData*>(tiling));
}
