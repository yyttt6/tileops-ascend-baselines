/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * Licensed under the CANN Open Software License Agreement Version 2.0 (the "License");
 * you may not use this file except in compliance with the License. You may obtain a copy of the
 * License at
 *
 * https://www.hiascend.com/software/ascend-cann-license
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the
 * License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 * express or implied. See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * \file grouped_matmul_kernel_ut.h
 * \brief Non-quant grouped matmul kernel-UT wrapper (mirrors the shipped arch35 adapter shape).
 */
#pragma once

#include <cstdint>

#include "blaze_kernel_stub.h"
#include "kernel_operator.h"
#include "tensor_api/tensor.h"

#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/kernel/kernel_universal.h"
#include "blaze/gemm/kernel/kernel_grouped_matmul.h"

#include "blaze/gemm/block/block_mmad_matmul_basic.h"
#include "blaze/gemm/block/block_scheduler_grouped_matmul.h"
#include "blaze/epilogue/block/block_epilogue_empty.h"

namespace GMMUT {
#pragma pack(push, 8)
struct GmmTilingData {
    uint32_t groupNum;
    int32_t groupType;
    uint32_t groupListType;
    uint64_t singleX;
    uint64_t singleWeight;
    uint64_t singleY;
    uint32_t hasBias;
    uint32_t weightNoL2Cache;
    uint64_t mTailCnt;
    uint64_t nTailCnt;
    int64_t m;
    int64_t n;
    int64_t k;
    uint32_t baseM;
    uint32_t baseN;
    uint32_t baseK;
    uint32_t stepKa;
    uint32_t stepKb;
    uint16_t dbL0C;
};
#pragma pack(pop)

template <typename AType, typename BType, typename CType, typename BiasType,
          typename LayoutA = AscendC::Te::NDExtLayoutPtn, typename LayoutB = AscendC::Te::NDExtLayoutPtn,
          Blaze::Gemm::MatmulOutputMode OutputMode = Blaze::Gemm::MatmulOutputMode::OVERWRITE>
__aicore__ inline void RunGmm(GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR groupList, GM_ADDR c, GM_ADDR tilingAddr)
{
    const auto& t = *reinterpret_cast<const GmmTilingData*>(tilingAddr);
    using LayoutC = AscendC::Te::NDExtLayoutPtn;
    using LayoutBias = AscendC::Te::NDExtLayoutPtn;
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using DispatchPolicy = Blaze::Gemm::MatmulMultiBlockBasic<0, 0, Blaze::Gemm::KernelGroupedMmadNoQuant, 0,
                                                              OutputMode>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType, LayoutA, BType, LayoutB, CType, LayoutC,
                                                    BiasType, LayoutBias>;
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueEmpty;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerGmmNoQuant;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename Kernel::Params;

    const uint64_t baseM = static_cast<uint64_t>(t.baseM);
    const uint64_t baseN = static_cast<uint64_t>(t.baseN);
    const uint64_t baseK = static_cast<uint64_t>(t.baseK);
    const uint64_t sharedKStep = Blaze::Gemm::Min(static_cast<uint64_t>(t.stepKa), static_cast<uint64_t>(t.stepKb));
    const uint64_t kL1 = baseK * Blaze::Gemm::Max(sharedKStep, static_cast<uint64_t>(1));

    typename Kernel::GMMTiling gmmParams{t.groupNum,     t.groupType, t.groupListType, t.singleX,
                                         t.singleWeight, t.singleY,   t.hasBias,       t.weightNoL2Cache};
    constexpr uint32_t nTailAlign = BlockMmad::WEIGHT_NZ_FORMAT ?
                                        static_cast<uint32_t>(AscendC::Te::C0_ELEMENT<BType>) :
                                        1U;
    typename BlockScheduler::Params schedulerParams{static_cast<int32_t>(baseM),
                                                    static_cast<int32_t>(baseN),
                                                    t.mTailCnt,
                                                    t.nTailCnt,
                                                    1U, // mTailAlign
                                                    nTailAlign,
                                                    t.groupType,
                                                    t.groupNum,
                                                    t.m,
                                                    t.singleX == 1,
                                                    t.singleWeight == 1,
                                                    t.singleY == 1,
                                                    BlockMmad::TRANS_B,
                                                    BlockMmad::WEIGHT_NZ_FORMAT,
                                                    static_cast<uint32_t>(sizeof(BType)),
                                                    t.groupListType};
    typename BlockMmad::Params mmParams{a,         b,       c,   t.hasBias == 0 ? nullptr : bias,
                                        groupList,
                                        nullptr, // workspaceGmAddr
                                        baseM,     baseN,   kL1, t.baseM,
                                        t.baseN,   t.baseK,
                                        2U,                  // l1Stages
                                        t.dbL0C,   nullptr}; // scaleGmAddr
    ProblemShape problemShape{t.m, t.n, t.k, 1};
    Params params{problemShape, mmParams, {}, schedulerParams, gmmParams};
    Kernel kernel;
    kernel(params);
}
} // namespace GMMUT

template <typename AType, typename BType, typename CType, typename BiasType,
          typename LayoutA = AscendC::Te::NDExtLayoutPtn, typename LayoutB = AscendC::Te::NDExtLayoutPtn,
          Blaze::Gemm::MatmulOutputMode OutputMode = Blaze::Gemm::MatmulOutputMode::OVERWRITE>
__global__ __aicore__ void gmm_kernel_entry(GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR groupList, GM_ADDR c,
                                            GM_ADDR tiling)
{
    GMMUT::RunGmm<AType, BType, CType, BiasType, LayoutA, LayoutB, OutputMode>(a, b, bias, groupList, c, tiling);
}
