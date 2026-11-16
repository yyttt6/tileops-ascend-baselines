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
 * \file gmm_fixpipe_quant.h
 * \brief Kernel-UT wrapper for the grouped S8S4/S4S4 FixPipe Tensor API stack.
 */
#pragma once

#include "blaze_kernel_stub.h"
#include "kernel_operator.h"
#include "tensor_api/tensor.h"

#if defined(ASCENDC_CPU_DEBUG)
#include "lib/matmul_intf.h"
#undef half
#endif

#include "blaze/epilogue/block/block_epilogue_per_token_scale.h"
#include "blaze/gemm/block/block_mmad_a8w8_fixpipe_quant.h"
#include "blaze/gemm/block/block_scheduler_gmm_swat_with_tail_split.h"
#include "blaze/gemm/kernel/kernel_qgmm_mix_fixpipe_quant.h"
#include "blaze/gemm/policy/dispatch_policy.h"

namespace GMMFixpipeUT {

#pragma pack(push, 8)
struct TilingData {
    uint32_t groupNum;
    int64_t m;
    int64_t n;
    int64_t k;
    uint32_t baseM;
    uint32_t baseN;
    uint32_t baseK;
    uint32_t quantGroupSize;
    uint32_t quantMode;
    uint32_t kAL1;
    uint32_t kBL1;
    uint32_t nBufferNum;
    uint8_t dbL0C;
    uint8_t groupListType;
    uint8_t withOffset;
};
#pragma pack(pop)

template <typename OutType, typename LayoutB>
__aicore__ inline void Run(GM_ADDR a, GM_ADDR b, GM_ADDR scale, GM_ADDR perTokenScale, GM_ADDR offset, GM_ADDR rowSum,
                           GM_ADDR workspace, GM_ADDR out, GM_ADDR groupList, const TilingData& t)
{
    using Layout = asc::te::nd_ext_layout_ptn;
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using Policy = Blaze::Gemm::MatmulWithScaleFixpipeQuant<0UL, false,
                                                            Blaze::Gemm::KernelGroupedMmadWithScaleFixpipeQuant>;
    using BTypeTuple = AscendC::Std::tuple<int8_t, uint64_t>;
    using Mmad = Blaze::Gemm::Block::BlockMmad<Policy, int8_t, Layout, BTypeTuple, LayoutB, half, Layout, int32_t,
                                               Layout>;
    using Epilogue = Blaze::Epilogue::Block::BlockEpiloguePerTokenScale<OutType, half>;
    using Scheduler = Blaze::Gemm::Block::BlockSchedulerGmmSwatWithTailSplit;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, Mmad, Epilogue, Scheduler>;

    typename Kernel::Params params{};
    params.mmadParams.aGmAddr = a;
    params.mmadParams.bGmAddr = b;
    params.mmadParams.cGmAddr = workspace;
    params.mmadParams.scaleBGmAddr = scale;
    params.epilogueParams.workspaceGmAddr = workspace;
    params.epilogueParams.perTokenScaleGmAddr = perTokenScale;
    params.epilogueParams.offsetGmAddr = t.withOffset != 0U ? offset : nullptr;
    params.epilogueParams.xRowSumGmAddr = t.withOffset != 0U ? rowSum : nullptr;
    params.epilogueParams.outGmAddr = out;
    params.epilogueParams.n = t.n;
    params.epilogueParams.baseM = t.baseM;
    params.epilogueParams.baseN = t.baseN;
    params.epilogueParams.withOffset = t.withOffset != 0U;
    params.groupListGmAddr = groupList;
    params.gmmParams.groupNum = t.groupNum;
    params.gmmParams.m = t.m;
    params.gmmParams.n = t.n;
    params.gmmParams.k = t.k;
    params.gmmParams.baseM = t.baseM;
    params.gmmParams.baseN = t.baseN;
    params.gmmParams.baseK = t.baseK;
    params.gmmParams.quantGroupSize = t.quantGroupSize;
    params.gmmParams.quantMode = t.quantMode;
    params.gmmParams.kAL1 = t.kAL1;
    params.gmmParams.kBL1 = t.kBL1;
    params.gmmParams.nBufferNum = t.nBufferNum;
    params.gmmParams.dbL0C = t.dbL0C;
    params.gmmParams.groupListType = t.groupListType;

    Kernel kernel;
    kernel(params);
}

} // namespace GMMFixpipeUT

template <typename OutType, typename LayoutB>
__global__ __aicore__ void gmm_fixpipe_quant_kernel_entry(GM_ADDR a, GM_ADDR b, GM_ADDR scale, GM_ADDR perTokenScale,
                                                          GM_ADDR offset, GM_ADDR rowSum, GM_ADDR workspace,
                                                          GM_ADDR out, GM_ADDR groupList, GM_ADDR tiling)
{
    GMMFixpipeUT::Run<OutType, LayoutB>(a, b, scale, perTokenScale, offset, rowSum, workspace, out, groupList,
                                        *reinterpret_cast<const GMMFixpipeUT::TilingData*>(tiling));
}
