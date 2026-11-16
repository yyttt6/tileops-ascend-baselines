/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or
 * modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 *
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS
 * SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT
 * NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of
 * the software repository for the full text of the License.
 */

/**
 * \file quant_grouped_matmul_mx.h
 * \brief QGMM MX kernel-UT wrapper.
 */
#pragma once

#include <cstdint>

#include "blaze_kernel_stub.h"
#include "kernel_operator.h"
#include "tensor_api/tensor.h"

#include "quant_grouped_matmul_cpu_debug_stub.h"

#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/kernel/kernel_universal.h"
#include "blaze/gemm/kernel/kernel_qgmm_mx.h"

#include "blaze/gemm/block/block_mmad_qgmm_mx.h"

#if defined(ASCENDC_CPU_DEBUG)
#undef half
#endif

#include "blaze/gemm/block/block_scheduler_gmm_swat_with_tail_split.h"
#include "blaze/epilogue/block/block_epilogue_empty.h"

namespace QGMMUT {
#pragma pack(push, 8)
struct QgmmTilingData {
    uint32_t groupNum;
    int64_t m;
    int64_t n;
    int64_t k;
    uint32_t baseM;
    uint32_t baseN;
    uint32_t baseK;
    uint32_t kAL1;
    uint32_t kBL1;
    uint32_t scaleKAL1;
    uint32_t scaleKBL1;
    uint8_t isBias;
    uint8_t dbL0C;
    uint8_t l1BufferStage;
    int8_t groupType;
    uint8_t groupListType;
    uint8_t singleW;
};
#pragma pack(pop)

template <typename AType, typename BType, typename CType, typename BiasType,
          typename LayoutA = asc::te::nd_ext_layout_ptn, typename LayoutB = asc::te::nd_ext_layout_ptn>
__aicore__ inline void RunQgmmMx(GM_ADDR a, GM_ADDR b, GM_ADDR scaleA, GM_ADDR scaleB, GM_ADDR bias, GM_ADDR c,
                                 GM_ADDR groupList, const QgmmTilingData& t)
{
    using Layout = asc::te::nd_ext_layout_ptn;
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using Policy = Blaze::Gemm::GroupedMatmulWithScaleMx<0>;
    using Mmad = Blaze::Gemm::Block::BlockMmad<Policy, AType, LayoutA, BType, LayoutB, CType, Layout, BiasType, Layout>;
    using Epilogue = Blaze::Gemm::Block::BlockEpilogueEmpty;
    using Scheduler = Blaze::Gemm::Block::BlockSchedulerGmmSwatWithTailSplit;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, Mmad, Epilogue, Scheduler>;
    typename Kernel::Params p{};
    p.problemShape = {t.m, t.n, t.k, 0};
    p.mmadParams.aGmAddr = a;
    p.mmadParams.bGmAddr = b;
    p.mmadParams.cGmAddr = c;
    p.mmadParams.biasGmAddr = bias;
    p.mmadParams.scaleAGmAddr = scaleA;
    p.mmadParams.scaleBGmAddr = scaleB;
    p.groupListGmAddr = groupList;
    p.gmmParams = {
        t.groupNum,  t.m,         t.n,      t.k,     t.baseM,         t.baseN,     t.baseK,         t.kAL1,   t.kBL1,
        t.scaleKAL1, t.scaleKBL1, t.isBias, t.dbL0C, t.l1BufferStage, t.groupType, t.groupListType, t.singleW};
    Kernel kernel;
    kernel(p);
}
} // namespace QGMMUT

template <typename AType, typename BType, typename CType, typename BiasType,
          typename LayoutA = asc::te::nd_ext_layout_ptn, typename LayoutB = asc::te::nd_ext_layout_ptn>
__global__ __aicore__ void qgmm_mx_kernel_entry(GM_ADDR a, GM_ADDR b, GM_ADDR scaleA, GM_ADDR scaleB, GM_ADDR bias,
                                                GM_ADDR c, GM_ADDR groupList, GM_ADDR tiling)
{
    QGMMUT::RunQgmmMx<AType, BType, CType, BiasType, LayoutA, LayoutB>(
        a, b, scaleA, scaleB, bias, c, groupList, *reinterpret_cast<const QGMMUT::QgmmTilingData*>(tiling));
}
