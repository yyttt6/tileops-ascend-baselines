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
 * \file test_gmm_fixpipe_quant.cpp
 * \brief Template and pipeline regression tests for grouped S8S4/S4S4 FixPipe.
 */

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "gtest/gtest.h"
#include "kernel_ut_runner.h"
#include "tikicpulib.h"

#include "gmm_fixpipe_quant.h"

namespace {

constexpr uint32_t GROUP_NUM = 2U;
constexpr int64_t TOTAL_M = 17;
constexpr int64_t N = 16;
constexpr int64_t K = 512;
constexpr int64_t GROUP0_M = 7;
constexpr uint32_t BASE_M = 16U;
constexpr uint32_t BASE_N = 16U;
constexpr uint32_t BASE_K = 128U;
constexpr uint32_t QUANT_GROUP_SIZE = 256U;
constexpr uint32_t BLOCK_NUM = 2U;
constexpr uint64_t FIXPIPE_SCALE_ONE = 0x000040003F800000ULL;
constexpr uint16_t FP16_ZERO = 0x0000U;
constexpr uint16_t FP16_NEGATIVE_ONE = 0xBC00U;

constexpr size_t AlignUp(size_t value, size_t alignment) { return (value + alignment - 1U) / alignment * alignment; }

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
void Fill(GM_ADDR addr, size_t count, T value)
{
    std::fill_n(reinterpret_cast<T*>(addr), count, value);
}

template <typename LayoutB>
constexpr size_t WeightElements()
{
    if constexpr (std::is_same_v<LayoutB, asc::te::nz_layout_ptn>) {
        return GROUP_NUM * AlignUp(K, 16U) * AlignUp(N, 32U);
    }
    if constexpr (std::is_same_v<LayoutB, asc::te::zn_layout_ptn>) {
        return GROUP_NUM * AlignUp(K, 32U) * AlignUp(N, 16U);
    }
    return GROUP_NUM * static_cast<size_t>(K) * N;
}

template <typename LayoutB>
void RunKernelSmoke(bool perGroup, bool withOffset)
{
    const uint32_t quantGroupNum = perGroup ? static_cast<uint32_t>(K) / QUANT_GROUP_SIZE : 1U;
    const size_t outputElements = static_cast<size_t>(TOTAL_M * N);
    const size_t workspaceElements = BLOCK_NUM * static_cast<size_t>(BASE_M) * BASE_N;
    GmBuffer a(static_cast<size_t>(TOTAL_M * K) * sizeof(int8_t));
    GmBuffer b(WeightElements<LayoutB>() * sizeof(int8_t));
    GmBuffer scale(static_cast<size_t>(GROUP_NUM) * quantGroupNum * N * sizeof(uint64_t));
    GmBuffer perTokenScale(static_cast<size_t>(TOTAL_M) * sizeof(float));
    GmBuffer offset(static_cast<size_t>(GROUP_NUM * N) * sizeof(float));
    GmBuffer rowSum(static_cast<size_t>(TOTAL_M) * sizeof(float));
    GmBuffer workspace(workspaceElements * sizeof(half));
    GmBuffer out(outputElements * sizeof(half));
    GmBuffer groupList(GROUP_NUM * sizeof(int64_t));
    GmBuffer tiling(sizeof(GMMFixpipeUT::TilingData));

    ASSERT_NE(a.Get(), nullptr);
    ASSERT_NE(b.Get(), nullptr);
    ASSERT_NE(scale.Get(), nullptr);
    ASSERT_NE(perTokenScale.Get(), nullptr);
    ASSERT_NE(offset.Get(), nullptr);
    ASSERT_NE(rowSum.Get(), nullptr);
    ASSERT_NE(workspace.Get(), nullptr);
    ASSERT_NE(out.Get(), nullptr);
    ASSERT_NE(groupList.Get(), nullptr);
    ASSERT_NE(tiling.Get(), nullptr);

    Fill<int8_t>(a.Get(), static_cast<size_t>(TOTAL_M * K), 1);
    Fill<int8_t>(b.Get(), WeightElements<LayoutB>(), 1);
    Fill<uint64_t>(scale.Get(), static_cast<size_t>(GROUP_NUM) * quantGroupNum * N, FIXPIPE_SCALE_ONE);
    Fill<float>(perTokenScale.Get(), static_cast<size_t>(TOTAL_M), 0.25F);
    Fill<float>(offset.Get(), static_cast<size_t>(GROUP_NUM * N), 0.5F);
    Fill<float>(rowSum.Get(), static_cast<size_t>(TOTAL_M), static_cast<float>(K));
    Fill<uint16_t>(workspace.Get(), workspaceElements, FP16_ZERO);
    Fill<uint16_t>(out.Get(), outputElements, FP16_NEGATIVE_ONE);
    auto* groupListData = reinterpret_cast<int64_t*>(groupList.Get());
    groupListData[0] = GROUP0_M;
    groupListData[1] = TOTAL_M - GROUP0_M;

    auto* t = reinterpret_cast<GMMFixpipeUT::TilingData*>(tiling.Get());
    *t = {GROUP_NUM,
          TOTAL_M,
          N,
          K,
          BASE_M,
          BASE_N,
          BASE_K,
          perGroup ? QUANT_GROUP_SIZE : static_cast<uint32_t>(K),
          perGroup ? static_cast<uint32_t>(Blaze::Gemm::QuantMode::PERGROUP_MODE) :
                     static_cast<uint32_t>(Blaze::Gemm::QuantMode::PERCHANNEL_MODE),
          QUANT_GROUP_SIZE,
          QUANT_GROUP_SIZE,
          2U,
          1U,
          1U,
          static_cast<uint8_t>(withOffset)};

    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    auto fn = gmm_fixpipe_quant_kernel_entry<half, LayoutB>;
    ASSERT_TRUE(KERNEL_RUN_KF(fn, BLOCK_NUM, a.Get(), b.Get(), scale.Get(), perTokenScale.Get(), offset.Get(),
                              rowSum.Get(), workspace.Get(), out.Get(), groupList.Get(), tiling.Get()));
}

} // namespace

class GmmFixpipeQuantTest : public testing::Test {};

TEST_F(GmmFixpipeQuantTest, TemplateContracts)
{
    using Layout = asc::te::nd_ext_layout_ptn;
    using Policy = Blaze::Gemm::MatmulWithScaleFixpipeQuant<0UL, false,
                                                            Blaze::Gemm::KernelGroupedMmadWithScaleFixpipeQuant>;
    using BTypeTuple = AscendC::Std::tuple<int8_t, uint64_t>;
    using Mmad = Blaze::Gemm::Block::BlockMmad<Policy, int8_t, Layout, BTypeTuple, Layout, half, Layout, int32_t,
                                               Layout>;
    using Epilogue = Blaze::Epilogue::Block::BlockEpiloguePerTokenScale<half, half>;
    static_assert(std::is_same_v<typename Policy::ScheduleType, Blaze::Gemm::KernelGroupedMmadWithScaleFixpipeQuant>);
    static_assert(std::is_same_v<typename Mmad::AType, int8_t>);
    static_assert(std::is_same_v<typename Mmad::BType, int8_t>);
    static_assert(std::is_same_v<typename Mmad::X2ScaleType, uint64_t>);
    static_assert(std::is_same_v<typename Epilogue::FixpipeType, half>);
    SUCCEED();
}

TEST_F(GmmFixpipeQuantTest, PerChannelNdWithOffsetSmoke) { RunKernelSmoke<asc::te::nd_ext_layout_ptn>(false, true); }

TEST_F(GmmFixpipeQuantTest, PerChannelNzN16Smoke) { RunKernelSmoke<asc::te::nz_layout_ptn>(false, false); }

TEST_F(GmmFixpipeQuantTest, PerGroupNdSmoke) { RunKernelSmoke<asc::te::nd_ext_layout_ptn>(true, false); }

TEST_F(GmmFixpipeQuantTest, PerGroupNzN16Smoke) { RunKernelSmoke<asc::te::nz_layout_ptn>(true, false); }

TEST_F(GmmFixpipeQuantTest, PerChannelTransposedNzSmoke) { RunKernelSmoke<asc::te::zn_layout_ptn>(false, false); }
