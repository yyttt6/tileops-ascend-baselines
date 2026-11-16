/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdint>
#include <utility>
#include "gtest/gtest.h"
#include "kernel_ut_runner.h"
#include "tikicpulib.h"
#include "qbmm_mix.h"

namespace {
class MixGmBuffer {
public:
    explicit MixGmBuffer(size_t size) : addr_(reinterpret_cast<GM_ADDR>(AscendC::GmAlloc(size))) {}
    ~MixGmBuffer()
    {
        if (addr_ != nullptr) {
            AscendC::GmFree(addr_);
        }
    }
    MixGmBuffer(const MixGmBuffer&) = delete;
    MixGmBuffer& operator=(const MixGmBuffer&) = delete;
    GM_ADDR Get() const { return addr_; }

private:
    GM_ADDR addr_;
};
} // namespace

// Independent migration wrapper: old QBMMMixTypes fixes NN and INT32 accumulators.
template <class AType_, class BType_, class OutType_, bool TransA_, bool TransB_, bool WithoutBatch_,
          uint64_t FullLoadMode_, bool WeightNz_>
__global__ __aicore__ void QbmmMixMigrationEntry(GM_ADDR a, GM_ADDR b, GM_ADDR s1, GM_ADDR s2, GM_ADDR bias,
                                                 GM_ADDR out, GM_ADDR tiling)
{
    using LayoutA = AscendC::Std::conditional_t<TransA_, asc::te::dn_ext_layout_ptn, asc::te::nd_ext_layout_ptn>;
    using NdLayoutB = AscendC::Std::conditional_t<TransB_, asc::te::dn_ext_layout_ptn, asc::te::nd_ext_layout_ptn>;
    using NzLayoutB = AscendC::Std::conditional_t<TransB_, asc::te::zn_layout_ptn, asc::te::nz_layout_ptn>;
    using LayoutB = AscendC::Std::conditional_t<WeightNz_, NzLayoutB, NdLayoutB>;
    using LayoutC = asc::te::nd_ext_layout_ptn;
    using Shape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using Schedule = AscendC::Std::conditional_t<WithoutBatch_, Blaze::Gemm::KernelMmadWithScaleMixWithoutBatch,
                                                 Blaze::Gemm::KernelMmadWithScaleMix>;
    using Policy = Blaze::Gemm::MatmulWithScaleMix<FullLoadMode_, false, Schedule>;
    using Mmad = Blaze::Gemm::Block::BlockMmad<Policy, AType_, LayoutA, AscendC::Std::tuple<BType_, float>, LayoutB,
                                               OutType_, LayoutC, int32_t, LayoutC>;
    using Epilogue = Blaze::Epilogue::Block::BlockEpilogueDequant<OutType_, int32_t, float, float,
                                                                  typename Mmad::L0CType>;
    using Scheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<Shape, FullLoadMode_, LayoutA, LayoutB,
                                                                           AType_>;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<Shape, Mmad, Epilogue, Scheduler>;
    const auto& td = *reinterpret_cast<const QBMMV3TilingData*>(tiling);
    typename Kernel::Params params{};
    params.problemShape = {td.m, td.n, td.k, td.b};
    if constexpr (WithoutBatch_) {
        QBMMUT::FillMixMmadParams(params.mmParams, a, b, bias, td);
    } else {
        QBMMUT::FillMixMmadParams(params.mmadParams, a, b, bias, td);
        QBMMUT::FillQbmmBatchParams(params.qbmmParams, td);
        QBMMUT::FillQbmmTileParams(params.qbmmParams, td);
    }
    QBMMUT::FillQbmmSchParams(params.schParams, td);
    QBMMUT::FillEpilogueParams(params.epilogueParams, s1, s2, bias, out, td);
    Kernel kernel;
    kernel(params);
}

namespace {
template <class AType_, class BType_, class OutType_, bool TransA_, bool TransB_, bool WithoutBatch_,
          uint64_t FullLoadMode_ = Blaze::Gemm::NONE_FULL_LOAD_MODE, bool WeightNz_ = false>
void RunMix(uint32_t x1Mode, uint32_t x2Mode, uint32_t biasDtype, uint32_t kAL1 = 64, uint32_t kBL1 = 64,
            uint32_t buffers = 2)
{
    constexpr int64_t M = 16, N = 32, K = 64;
    constexpr int64_t BATCH = WithoutBatch_ ? 1 : 2;
    MixGmBuffer a(M * K * sizeof(AType_)), b(BATCH * K * N * sizeof(BType_));
    MixGmBuffer s1(M * sizeof(float)), s2(N * sizeof(float));
    MixGmBuffer bias(BATCH * N * sizeof(int32_t)), out(BATCH * M * N * sizeof(OutType_));
    MixGmBuffer tiling(sizeof(QBMMV3TilingData));
    ASSERT_NE(a.Get(), nullptr);
    ASSERT_NE(b.Get(), nullptr);
    ASSERT_NE(s1.Get(), nullptr);
    ASSERT_NE(s2.Get(), nullptr);
    ASSERT_NE(bias.Get(), nullptr);
    ASSERT_NE(out.Get(), nullptr);
    ASSERT_NE(tiling.Get(), nullptr);
    std::fill_n(a.Get(), M * K * sizeof(AType_), 0);
    std::fill_n(b.Get(), BATCH * K * N * sizeof(BType_), 0);
    std::fill_n(reinterpret_cast<float*>(s1.Get()), M, 0.5F);
    std::fill_n(reinterpret_cast<float*>(s2.Get()), N, 2.0F);
    std::fill_n(bias.Get(), BATCH * N * sizeof(int32_t), 0);
    auto& td = *reinterpret_cast<QBMMV3TilingData*>(tiling.Get());
    td = {};
    td.m = M;
    td.n = N;
    td.k = K;
    td.b = BATCH;
    // Match host tiling: transposed A uses a C0-aligned tile even when logical M is smaller.
    td.baseM = td.baseM_qbmm = TransA_ ? 32 : M;
    td.baseN = td.baseN_qbmm = N;
    td.baseK_qbmm = 32;
    td.kAL1 = kAL1;
    td.kBL1 = kBL1;
    td.nBufferNum = buffers;
    td.dbL0C = 1;
    td.mTailTile = td.nTailTile = td.mBaseTailSplitCnt = td.nBaseTailSplitCnt = 1;
    td.batchA1 = td.batchA2 = td.batchA3 = td.batchA4 = 1;
    td.batchB1 = td.batchB2 = td.batchB3 = 1;
    td.batchB4 = BATCH;
    td.batchC1 = td.batchC2 = td.batchC3 = 1;
    td.batchC4 = BATCH;
    td.biasThreeDim = WithoutBatch_ ? 0 : 1;
    td.isBias = 1;
    td.biasDtype = biasDtype;
    td.x1QuantMode = x1Mode;
    td.x2QuantMode = x2Mode;
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    auto kernel = QbmmMixMigrationEntry<AType_, BType_, OutType_, TransA_, TransB_, WithoutBatch_, FullLoadMode_,
                                        WeightNz_>;
    EXPECT_TRUE(KERNEL_RUN_KF(kernel, 1, a.Get(), b.Get(), s1.Get(), s2.Get(), bias.Get(), out.Get(), tiling.Get()));
    // tikicpulib does not model RegTensor numerics; this is an instantiation/synchronization test.
    // Numerical correctness is verified separately with nonzero device inputs.
}
} // namespace

TEST(QbmmMixNdMigration, Int8FourTransposesAndIntegerBias)
{
    for (uint32_t x1 : {0U, 4U}) {
        for (uint32_t x2 : {1U, 2U}) {
            RunMix<int8_t, int8_t, half, false, false, true>(x1, x2, DT_INT32);
            RunMix<int8_t, int8_t, bfloat16_t, false, true, true>(x1, x2, DT_INT32);
            RunMix<int8_t, int8_t, half, true, false, false>(x1, x2, DT_INT32);
            RunMix<int8_t, int8_t, bfloat16_t, true, true, false>(x1, x2, DT_INT32);
        }
    }
}

TEST(QbmmMixNdMigration, Fp8PairsAndHif8ScaleModes)
{
    for (uint32_t x1 : {1U, 4U}) {
        for (uint32_t x2 : {1U, 2U}) {
            RunMix<fp8_e4m3fn_t, fp8_e4m3fn_t, half, false, false, true>(x1, x2, DT_FLOAT);
            RunMix<fp8_e4m3fn_t, fp8_e5m2_t, bfloat16_t, false, true, false>(x1, x2, DT_FLOAT);
            RunMix<fp8_e5m2_t, fp8_e4m3fn_t, float, true, false, false>(x1, x2, DT_FLOAT);
            RunMix<fp8_e5m2_t, fp8_e5m2_t, half, true, true, true>(x1, x2, DT_FLOAT);
            RunMix<hifloat8_t, hifloat8_t, float, true, true, false>(x1, x2, DT_FLOAT);
        }
    }
}

TEST(QbmmMixNdMigration, SingleBatchAl1)
{
    RunMix<int8_t, int8_t, bfloat16_t, true, false, true, Blaze::Gemm::A_FULL_LOAD_MODE>(4, 2, DT_INT32);
    RunMix<fp8_e5m2_t, fp8_e4m3fn_t, float, false, true, true, Blaze::Gemm::A_FULL_LOAD_MODE>(1, 1, DT_FLOAT);
}

TEST(QbmmMixNdMigration, Int8NzTransposedA)
{
    for (uint32_t x1 : {0U, 4U}) {
        for (uint32_t x2 : {1U, 2U}) {
            RunMix<int8_t, int8_t, half, true, false, true, 0, true>(x1, x2, DT_INT32);
            RunMix<int8_t, int8_t, half, true, true, true, 0, true>(x1, x2, DT_INT32);
            RunMix<int8_t, int8_t, bfloat16_t, true, false, false, 0, true>(x1, x2, DT_INT32);
            RunMix<int8_t, int8_t, bfloat16_t, true, true, false, 0, true>(x1, x2, DT_INT32);
            RunMix<int8_t, int8_t, bfloat16_t, true, false, true, Blaze::Gemm::A_FULL_LOAD_MODE, true>(x1, x2,
                                                                                                       DT_INT32);
            RunMix<int8_t, int8_t, bfloat16_t, true, true, true, Blaze::Gemm::A_FULL_LOAD_MODE, true>(x1, x2, DT_INT32);
        }
    }
}

TEST(QbmmMixNdMigration, IntegerBiasSplitKAndFourBuffers)
{
    // All K-loop nesting orders must initialize bias once per output tile.
    for (auto depths : {std::pair{32U, 64U}, std::pair{64U, 32U}, std::pair{32U, 32U}}) {
        RunMix<int8_t, int8_t, half, true, false, false>(4, 2, DT_INT32, depths.first, depths.second);
        RunMix<int8_t, int8_t, half, true, true, false, 0, true>(4, 2, DT_INT32, depths.first, depths.second);
    }
    RunMix<int8_t, int8_t, half, false, false, false>(4, 2, DT_INT32, 32, 32, 4);
    RunMix<int8_t, int8_t, half, true, true, false, 0, true>(4, 2, DT_INT32, 32, 32, 4);
    RunMix<int8_t, int8_t, half, true, false, true, Blaze::Gemm::A_FULL_LOAD_MODE>(4, 2, DT_INT32, 64, 32, 4);
}
