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
 * \file test_qgmm_cube.cpp
 * \brief CPU smoke tests for kernel_qgmm_cube.h (QGMM Cube, AIC-only quant GMM).
 *
 * These tests are intentionally smoke-only: tikicpulib does not model the
 * quantized Fixpipe/MMAD numerics, so we verify template instantiation, address
 * descriptor handling, GMMArray/groupList parsing and multi-group pipeline
 * execution without comparing output values.
 */

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "gtest/gtest.h"
#include "kernel_ut_runner.h"
#include "tikicpulib.h"
#include "kernel_operator.h"

#include "qgmm_cube.h"

namespace {

constexpr int64_t M0 = 16;
constexpr int64_t M1 = 24;
constexpr int64_t N = 64;
constexpr int64_t K = 64;
constexpr uint32_t GROUP_NUM = 2U;
constexpr uint32_t BLOCK_NUM = 1U;
constexpr uint32_t GMM_ARRAY_LEN = 128U;
constexpr uint32_t QM_DEFAULT = 0U;
constexpr uint32_t QM_PERTENSOR = 1U;
constexpr uint32_t QM_PERCHANNEL = 2U;
constexpr uint32_t GROUP_TYPE_SPLIT_M = 0U;
constexpr uint64_t FIXPIPE_SCALE_ONE = 0x3F80000000000000ULL;

class GmBuffer {
public:
    explicit GmBuffer(size_t bytes) : addr_(reinterpret_cast<GM_ADDR>(AscendC::GmAlloc(bytes))) {}
    ~GmBuffer()
    {
        if (addr_ != nullptr) {
            AscendC::GmFree(reinterpret_cast<void*>(addr_));
        }
    }
    GmBuffer(const GmBuffer&) = delete;
    GmBuffer& operator=(const GmBuffer&) = delete;
    GM_ADDR Get() const { return addr_; }

private:
    GM_ADDR addr_{nullptr};
};

template <typename T>
void FillValues(GM_ADDR addr, size_t count, T value)
{
    std::fill_n(reinterpret_cast<T*>(addr), count, value);
}

void FillBytes(GM_ADDR addr, size_t size, uint8_t value) { std::fill_n(reinterpret_cast<uint8_t*>(addr), size, value); }

uint64_t AsUint64(GM_ADDR addr) { return reinterpret_cast<uint64_t>(addr); }

void FillGroupList(GM_ADDR addr, uint8_t groupListType)
{
    auto* data = reinterpret_cast<int64_t*>(addr);
    if (groupListType == 2U) {
        data[0] = 0;
        data[1] = M0;
        data[2] = 1;
        data[3] = M1;
    } else if (groupListType == 0U) {
        data[0] = M0;
        data[1] = M0 + M1;
    } else {
        data[0] = M0;
        data[1] = M1;
    }
}

void FillGmmArray(GM_ADDR addr)
{
    auto* data = reinterpret_cast<int32_t*>(addr);
    std::fill_n(data, 3U * GMM_ARRAY_LEN, 0);
    // mList is unused for SPLIT_M; keep it as -1 to mirror the host contract.
    for (uint32_t i = 0; i < GMM_ARRAY_LEN; ++i) {
        data[i] = -1;
    }
    for (uint32_t i = 0; i < GMM_ARRAY_LEN; ++i) {
        data[GMM_ARRAY_LEN + i] = static_cast<int32_t>(N);
        data[2U * GMM_ARRAY_LEN + i] = static_cast<int32_t>(K);
    }
}

void FillSingleDescriptor(GM_ADDR desc, GM_ADDR data)
{
    auto* list = reinterpret_cast<uint64_t*>(desc);
    list[0] = 24UL;
    list[1] = 0UL; // dim=0, offset=0 for one entry
    list[2] = 0xffffffffULL;
    list[3] = AsUint64(data);
}

template <typename AType, typename BType, typename CType, typename BiasType, typename X2ScaleType,
          typename LayoutB = asc::te::nd_ext_layout_ptn>
void RunInt8Smoke(uint32_t x2QuantMode, bool withBias, uint8_t groupListType, bool noSplit = false)
{
    const bool perChannel = x2QuantMode == QM_PERCHANNEL;
    constexpr bool needsScale = !std::is_same_v<CType, int32_t>;
    constexpr size_t totalM = M0 + M1;
    const size_t aElements = totalM * K;
    const size_t bElements = static_cast<size_t>(GROUP_NUM) * K * N;
    const size_t cElements = totalM * N;
    const size_t biasElements = static_cast<size_t>(GROUP_NUM) * N;

    GmBuffer aData(aElements * sizeof(AType));
    GmBuffer bData(bElements * sizeof(BType));
    GmBuffer cData(cElements * sizeof(CType));
    GmBuffer biasData(withBias ? biasElements * sizeof(BiasType) : 0U);
    GmBuffer aDesc(4U * sizeof(uint64_t));
    GmBuffer bDesc(4U * sizeof(uint64_t));
    GmBuffer cDesc(4U * sizeof(uint64_t));
    GmBuffer biasDesc(withBias ? 4U * sizeof(uint64_t) : 0U);
    GmBuffer groupList((groupListType == 2U ? GROUP_NUM * 2U : GROUP_NUM) * sizeof(int64_t));
    GmBuffer gmmArray(3U * GMM_ARRAY_LEN * sizeof(int32_t));
    GmBuffer scaleA(0U);
    GmBuffer scaleB(perChannel ? biasElements * sizeof(uint64_t) : GROUP_NUM * sizeof(float));
    GmBuffer tiling(sizeof(QGMMCubeUT::QgmmCubeTilingData));

    ASSERT_NE(aData.Get(), nullptr);
    ASSERT_NE(bData.Get(), nullptr);
    ASSERT_NE(cData.Get(), nullptr);
    ASSERT_NE(aDesc.Get(), nullptr);
    ASSERT_NE(bDesc.Get(), nullptr);
    ASSERT_NE(cDesc.Get(), nullptr);
    ASSERT_NE(groupList.Get(), nullptr);
    ASSERT_NE(gmmArray.Get(), nullptr);
    ASSERT_NE(scaleB.Get(), nullptr);
    ASSERT_NE(tiling.Get(), nullptr);
    if (withBias) {
        ASSERT_NE(biasData.Get(), nullptr);
        ASSERT_NE(biasDesc.Get(), nullptr);
    }

    FillValues<AType>(aData.Get(), aElements, static_cast<AType>(1));
    FillValues<BType>(bData.Get(), bElements, static_cast<BType>(1));
    FillBytes(cData.Get(), cElements * sizeof(CType), 0U);
    if (withBias) {
        FillValues<BiasType>(biasData.Get(), biasElements, static_cast<BiasType>(1));
    }
    if (perChannel) {
        FillValues<uint64_t>(scaleB.Get(), biasElements, FIXPIPE_SCALE_ONE);
    } else {
        // x2QuantMode=PERTENSOR and X2ScaleType=float reads per-group FP32 scalar.
        FillValues<float>(scaleB.Get(), GROUP_NUM, 1.0F);
    }
    FillGroupList(groupList.Get(), groupListType);
    FillGmmArray(gmmArray.Get());
    if (noSplit) {
        auto* shapes = reinterpret_cast<int32_t*>(gmmArray.Get());
        shapes[0] = M0;
        shapes[1] = M1;
    }

    FillSingleDescriptor(aDesc.Get(), aData.Get());
    FillSingleDescriptor(bDesc.Get(), bData.Get());
    FillSingleDescriptor(cDesc.Get(), cData.Get());
    if (withBias) {
        FillSingleDescriptor(biasDesc.Get(), biasData.Get());
    }

    auto* t = reinterpret_cast<QGMMCubeUT::QgmmCubeTilingData*>(tiling.Get());
    *t = {GROUP_NUM,
          static_cast<int64_t>(totalM),
          N,
          K,
          16U,
          64U,
          64U,
          64U,
          64U,
          QM_DEFAULT,
          x2QuantMode,
          static_cast<uint8_t>(withBias ? 1U : 0U),
          1U,
          static_cast<int8_t>(noSplit ? -1 : GROUP_TYPE_SPLIT_M),
          groupListType,
          1U,
          1U,
          1U};

    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    auto fn = qgmm_cube_kernel_entry<AType, BType, CType, BiasType, X2ScaleType, asc::te::nd_ext_layout_ptn, LayoutB>;
    const bool ok = KERNEL_RUN_KF(fn, BLOCK_NUM, aDesc.Get(), bDesc.Get(), cDesc.Get(),
                                  withBias ? biasDesc.Get() : nullptr, nullptr, needsScale ? scaleB.Get() : nullptr,
                                  noSplit ? nullptr : groupList.Get(), gmmArray.Get(), tiling.Get());
    ASSERT_TRUE(ok) << "QGMM Cube kernel execution failed";
}

template <typename AType, typename BType, typename CType, typename BiasType, typename ScaleType = float>
void RunFp8PertensorSmoke(bool doubleScale)
{
    constexpr size_t totalM = M0 + M1;
    const size_t aElements = totalM * K;
    const size_t bElements = static_cast<size_t>(GROUP_NUM) * K * N;
    const size_t cElements = totalM * N;
    const size_t scaleElements = GROUP_NUM;

    GmBuffer aData(aElements * sizeof(AType));
    GmBuffer bData(bElements * sizeof(BType));
    GmBuffer cData(cElements * sizeof(CType));
    GmBuffer aDesc(4U * sizeof(uint64_t));
    GmBuffer bDesc(4U * sizeof(uint64_t));
    GmBuffer cDesc(4U * sizeof(uint64_t));
    GmBuffer groupList(GROUP_NUM * sizeof(int64_t));
    GmBuffer gmmArray(3U * GMM_ARRAY_LEN * sizeof(int32_t));
    GmBuffer scaleA(doubleScale ? scaleElements * sizeof(float) : 0U);
    GmBuffer scaleB(scaleElements * sizeof(ScaleType));
    GmBuffer tiling(sizeof(QGMMCubeUT::QgmmCubeTilingData));

    ASSERT_NE(aData.Get(), nullptr);
    ASSERT_NE(bData.Get(), nullptr);
    ASSERT_NE(cData.Get(), nullptr);
    ASSERT_NE(aDesc.Get(), nullptr);
    ASSERT_NE(bDesc.Get(), nullptr);
    ASSERT_NE(cDesc.Get(), nullptr);
    ASSERT_NE(groupList.Get(), nullptr);
    ASSERT_NE(gmmArray.Get(), nullptr);
    ASSERT_NE(scaleB.Get(), nullptr);
    ASSERT_NE(tiling.Get(), nullptr);
    if (doubleScale) {
        ASSERT_NE(scaleA.Get(), nullptr);
    }

    FillBytes(aData.Get(), aElements * sizeof(AType), 0x38U);
    FillBytes(bData.Get(), bElements * sizeof(BType), 0x38U);
    FillBytes(cData.Get(), cElements * sizeof(CType), 0U);
    if constexpr (std::is_same_v<ScaleType, bfloat16_t>) {
        FillValues<uint16_t>(scaleB.Get(), scaleElements, 0x3F80U); // bf16 1.0
    } else {
        FillValues<float>(scaleB.Get(), scaleElements, 1.0F);
    }
    if (doubleScale) {
        FillValues<float>(scaleA.Get(), scaleElements, 1.0F);
    }
    FillGroupList(groupList.Get(), 1U);
    FillGmmArray(gmmArray.Get());
    FillSingleDescriptor(aDesc.Get(), aData.Get());
    FillSingleDescriptor(bDesc.Get(), bData.Get());
    FillSingleDescriptor(cDesc.Get(), cData.Get());

    auto* t = reinterpret_cast<QGMMCubeUT::QgmmCubeTilingData*>(tiling.Get());
    *t = {GROUP_NUM,
          static_cast<int64_t>(totalM),
          N,
          K,
          16U,
          64U,
          64U,
          64U,
          64U,
          doubleScale ? QM_PERTENSOR : QM_DEFAULT,
          QM_PERTENSOR,
          0U,
          1U,
          GROUP_TYPE_SPLIT_M,
          1U,
          1U,
          1U,
          1U};

    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    auto fn = qgmm_cube_kernel_entry<AType, BType, CType, BiasType, ScaleType, asc::te::nd_ext_layout_ptn,
                                     asc::te::nd_ext_layout_ptn>;
    const bool ok = KERNEL_RUN_KF(fn, BLOCK_NUM, aDesc.Get(), bDesc.Get(), cDesc.Get(), nullptr,
                                  doubleScale ? scaleA.Get() : nullptr, scaleB.Get(), groupList.Get(), gmmArray.Get(),
                                  tiling.Get());
    ASSERT_TRUE(ok) << "QGMM Cube FP8 kernel execution failed";
}

} // namespace

class QgmmCubeKernelTest : public testing::Test {};

TEST_F(QgmmCubeKernelTest, TemplateContracts)
{
    using Layout = asc::te::nd_ext_layout_ptn;
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using Policy = Blaze::Gemm::MatmulWithScaleFixpipeQuant<0UL, false, Blaze::Gemm::KernelGroupedMmadFixpipeQuant>;
    using BTypeTuple = AscendC::Std::tuple<int8_t, float>;
    using Mmad = Blaze::Gemm::Block::BlockMmad<Policy, int8_t, Layout, BTypeTuple, Layout, half, Layout, int32_t,
                                               Layout>;
    using Scheduler = Blaze::Gemm::Block::BlockSchedulerGmmSwatWithTailSplit;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, Mmad, Blaze::Gemm::Block::BlockEpilogueEmpty,
                                                      Scheduler>;
    static_assert(std::is_same_v<typename Kernel::AType, int8_t>);
    static_assert(std::is_same_v<typename Kernel::BType, int8_t>);
    static_assert(std::is_same_v<typename Kernel::CType, half>);
    static_assert(std::is_same_v<typename Kernel::BiasType, int32_t>);
    SUCCEED();
}

TEST_F(QgmmCubeKernelTest, Int8PertensorNdSingleTensorLength)
{
    RunInt8Smoke<int8_t, int8_t, half, int32_t, float>(QM_PERTENSOR, false, 1U);
}

TEST_F(QgmmCubeKernelTest, Int8PertensorNdSingleTensorOffsetWithBias)
{
    RunInt8Smoke<int8_t, int8_t, half, int32_t, float>(QM_PERTENSOR, true, 0U);
}

TEST_F(QgmmCubeKernelTest, Int8PerchannelNdSingleTensorSparse)
{
    RunInt8Smoke<int8_t, int8_t, half, int32_t, uint64_t>(QM_PERCHANNEL, false, 2U);
}

TEST_F(QgmmCubeKernelTest, Int8PerchannelNdSingleTensorLength)
{
    RunInt8Smoke<int8_t, int8_t, half, int32_t, uint64_t>(QM_PERCHANNEL, false, 1U);
}

TEST_F(QgmmCubeKernelTest, Int8PerchannelNdSingleTensorBias)
{
    RunInt8Smoke<int8_t, int8_t, half, int32_t, uint64_t>(QM_PERCHANNEL, true, 1U);
}

TEST_F(QgmmCubeKernelTest, Fp8PertensorNdSingleTensor)
{
    RunFp8PertensorSmoke<fp8_e4m3fn_t, fp8_e4m3fn_t, float, float>(false);
}

TEST_F(QgmmCubeKernelTest, Fp8DoubleScaleNdSingleTensor)
{
    RunFp8PertensorSmoke<fp8_e4m3fn_t, fp8_e4m3fn_t, float, float>(true);
}

TEST_F(QgmmCubeKernelTest, Int8Int32WithoutScale)
{
    RunInt8Smoke<int8_t, int8_t, int32_t, int32_t, uint64_t>(QM_PERCHANNEL, false, 1U);
}

TEST_F(QgmmCubeKernelTest, Int8PertensorTransposedWeight)
{
    RunInt8Smoke<int8_t, int8_t, half, int32_t, float, asc::te::dn_ext_layout_ptn>(QM_PERTENSOR, true, 1U);
}

TEST_F(QgmmCubeKernelTest, Fp8DoubleScaleBf16NdSingleTensor)
{
    RunFp8PertensorSmoke<fp8_e4m3fn_t, fp8_e4m3fn_t, half, float, bfloat16_t>(true);
}

TEST_F(QgmmCubeKernelTest, NoSplitNullGroupList)
{
    for (uint8_t mode = 0; mode < 3; ++mode) {
        RunInt8Smoke<int8_t, int8_t, half, int32_t, float>(QM_PERTENSOR, false, mode, true);
    }
}
