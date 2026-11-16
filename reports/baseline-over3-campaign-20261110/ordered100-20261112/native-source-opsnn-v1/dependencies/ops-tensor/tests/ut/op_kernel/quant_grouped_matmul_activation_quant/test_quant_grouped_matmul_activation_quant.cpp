/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_quant_grouped_matmul_activation_quant.cpp
 * \brief QGMM MX activation-quantization public template contract tests.
 */

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "gtest/gtest.h"
#include "kernel_ut_runner.h"
#include "qgmm_mx_activation_quant.h"
#include "tikicpulib.h"

namespace {
constexpr int64_t M = 16;
constexpr int64_t N = 64;
constexpr int64_t K = 64;
constexpr uint32_t BLOCK_NUM = 1;
constexpr uint32_t GROUP_NUM = 1;
constexpr size_t SCALE_K = 2;

template <typename T>
inline constexpr bool IS_FP4_TYPE = std::is_same_v<T, fp4x2_e2m1_t> || std::is_same_v<T, fp4x2_e1m2_t>;

class GmBuffer {
public:
    explicit GmBuffer(size_t bytes) : ptr_(static_cast<GM_ADDR>(AscendC::GmAlloc(bytes))) {}
    ~GmBuffer()
    {
        if (ptr_ != nullptr) {
            AscendC::GmFree(ptr_);
        }
    }

    GmBuffer(const GmBuffer&) = delete;
    GmBuffer& operator=(const GmBuffer&) = delete;

    GM_ADDR Get() const { return ptr_; }

private:
    GM_ADDR ptr_{};
};

template <typename T>
size_t MxDataSize(size_t elementCount)
{
    if constexpr (IS_FP4_TYPE<T>) {
        return (elementCount + 1UL) / 2UL;
    }
    return elementCount * sizeof(T);
}

template <typename T>
constexpr uint8_t MxOneByte()
{
    if constexpr (IS_FP4_TYPE<T>) {
        return 0x22U;
    }
    if constexpr (std::is_same_v<T, fp8_e5m2_t>) {
        return 0x3CU;
    }
    return 0x38U;
}

constexpr size_t AlignUp(size_t value, size_t alignment) { return (value + alignment - 1UL) / alignment * alignment; }

template <typename AType, typename BType, typename OutputType>
void CheckPublicAssembly()
{
    using Types = QgmmMxActivationQuantTypes<AType, BType, OutputType>;
    using Kernel = typename Types::Kernel;
    using Epilogue = typename Types::BlockEpilogue;

    static_assert(Kernel::HAS_ACTIVATION_QUANT);
    static_assert(std::is_same_v<typename Types::DispatchPolicy::ScheduleType,
                                 Blaze::Gemm::KernelGroupedMmadWithScaleMxActivationQuant>);
    static_assert(std::is_same_v<typename Types::BlockMmad::CType, float>);
    static_assert(std::is_same_v<typename Epilogue::DataTypeIn, float>);
    static_assert(std::is_same_v<typename Epilogue::DataTypeOut, OutputType>);

    typename Kernel::GMMTiling tiling{2, 4, 128, 64, 128, 256, 64, 64, 64, 64, 64, 0, 2, 2, 0, 1, 1};
    EXPECT_EQ(tiling.groupNum, 2U);
    EXPECT_EQ(tiling.l1BufferStage, 2U);
    EXPECT_EQ(tiling.groupType, 0);
    EXPECT_EQ(tiling.groupListType, 1U);
    EXPECT_EQ(tiling.singleW, 1U);

    typename Epilogue::Params params{nullptr, nullptr, 128, 256, 2, 7.0f};
    EXPECT_EQ(params.baseM, 128U);
    EXPECT_EQ(params.baseN, 256U);
    EXPECT_EQ(params.scaleAlg, 2U);
    EXPECT_FLOAT_EQ(params.dstTypeMax, 7.0f);
}

template <typename AType, typename BType, typename OutputType, typename LayoutB>
void RunKernelSmoke(uint8_t groupListType, uint32_t scaleAlg)
{
    constexpr bool transB = std::is_same_v<LayoutB, asc::te::zn_layout_ptn>;
    constexpr size_t c0 = IS_FP4_TYPE<BType> ? 64UL : 32UL;
    const size_t storedK = transB ? AlignUp(K, c0) : AlignUp(K, 16UL);
    const size_t storedN = transB ? AlignUp(N, 16UL) : AlignUp(N, c0);
    const size_t xSize = MxDataSize<AType>(M * K);
    const size_t weightSize = MxDataSize<BType>(storedK * storedN);
    const size_t outputSize = MxDataSize<OutputType>(M * N);
    const size_t xScaleSize = M * SCALE_K * sizeof(fp8_e8m0_t);
    const size_t weightScaleSize = N * SCALE_K * sizeof(fp8_e8m0_t);
    const size_t outputScaleSize = M * SCALE_K * sizeof(fp8_e8m0_t);

    GmBuffer x(xSize);
    GmBuffer weight(weightSize);
    GmBuffer weightScale(weightScaleSize);
    GmBuffer xScale(xScaleSize);
    GmBuffer groupList(sizeof(int64_t));
    GmBuffer y(outputSize);
    GmBuffer yScale(outputScaleSize);
    GmBuffer tiling(sizeof(GMMAQUT::GmmaqTilingData));
    ASSERT_NE(x.Get(), nullptr);
    ASSERT_NE(weight.Get(), nullptr);
    ASSERT_NE(weightScale.Get(), nullptr);
    ASSERT_NE(xScale.Get(), nullptr);
    ASSERT_NE(groupList.Get(), nullptr);
    ASSERT_NE(y.Get(), nullptr);
    ASSERT_NE(yScale.Get(), nullptr);
    ASSERT_NE(tiling.Get(), nullptr);

    std::fill_n(reinterpret_cast<uint8_t*>(x.Get()), xSize, MxOneByte<AType>());
    std::fill_n(reinterpret_cast<uint8_t*>(weight.Get()), weightSize, MxOneByte<BType>());
    std::fill_n(reinterpret_cast<uint8_t*>(weightScale.Get()), weightScaleSize, 0x7fU);
    std::fill_n(reinterpret_cast<uint8_t*>(xScale.Get()), xScaleSize, 0x7fU);
    std::fill_n(reinterpret_cast<uint8_t*>(y.Get()), outputSize, 0U);
    std::fill_n(reinterpret_cast<uint8_t*>(yScale.Get()), outputScaleSize, 0U);
    *reinterpret_cast<int64_t*>(groupList.Get()) = M;
    *reinterpret_cast<GMMAQUT::GmmaqTilingData*>(tiling.Get()) = {
        GROUP_NUM, M, N, K, 16, 64, 64, 64, 64, 64, 64, 0, 1, 2, 0, groupListType, 1, scaleAlg, 0.0f};

    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    auto fn = GmmaqMxKernelEntry<AType, BType, OutputType, LayoutB>;
    ASSERT_TRUE(KERNEL_RUN_KF(fn, BLOCK_NUM, x.Get(), weight.Get(), weightScale.Get(), xScale.Get(), groupList.Get(),
                              y.Get(), yScale.Get(), tiling.Get()))
        << "GMMAQ MX Blaze kernel execution failed";
}
} // namespace

TEST(QgmmMxActivationQuantTest, AssembleE4M3) { CheckPublicAssembly<fp8_e4m3fn_t, fp8_e4m3fn_t, fp8_e4m3fn_t>(); }

TEST(QgmmMxActivationQuantTest, AssembleE5M2) { CheckPublicAssembly<fp8_e5m2_t, fp8_e4m3fn_t, fp8_e5m2_t>(); }

TEST(QgmmMxActivationQuantTest, AssembleE2M1E2M1) { CheckPublicAssembly<fp4x2_e2m1_t, fp4x2_e2m1_t, fp8_e4m3fn_t>(); }

TEST(QgmmMxActivationQuantTest, AssembleE2M1E1M2) { CheckPublicAssembly<fp4x2_e2m1_t, fp4x2_e1m2_t, fp8_e4m3fn_t>(); }

TEST(QgmmMxActivationQuantTest, AssembleE1M2E2M1) { CheckPublicAssembly<fp4x2_e1m2_t, fp4x2_e2m1_t, fp8_e4m3fn_t>(); }

TEST(QgmmMxActivationQuantTest, AssembleE1M2E1M2) { CheckPublicAssembly<fp4x2_e1m2_t, fp4x2_e1m2_t, fp8_e5m2_t>(); }

TEST(QgmmMxActivationQuantTest, AssembleFp4E2M1Output)
{
    CheckPublicAssembly<fp4x2_e2m1_t, fp4x2_e1m2_t, fp4x2_e2m1_t>();
}

TEST(QgmmMxActivationQuantTest, AssembleFp4E1M2Output)
{
    CheckPublicAssembly<fp4x2_e1m2_t, fp4x2_e2m1_t, fp4x2_e1m2_t>();
}

TEST(QgmmMxActivationQuantTest, KernelMxFp8E4M3NzOffsetOcp)
{
    RunKernelSmoke<fp8_e4m3fn_t, fp8_e4m3fn_t, fp8_e4m3fn_t, asc::te::nz_layout_ptn>(0, 0);
}

TEST(QgmmMxActivationQuantTest, KernelMxFp8E5M2ZnLengthCublas)
{
    RunKernelSmoke<fp8_e5m2_t, fp8_e4m3fn_t, fp8_e5m2_t, asc::te::zn_layout_ptn>(1, 1);
}

TEST(QgmmMxActivationQuantTest, KernelMxFp4E1M2E1M2NzOffset)
{
    RunKernelSmoke<fp4x2_e1m2_t, fp4x2_e1m2_t, fp4x2_e1m2_t, asc::te::nz_layout_ptn>(0, 0);
}

TEST(QgmmMxActivationQuantTest, KernelMxFp4E1M2E2M1ZnLength)
{
    RunKernelSmoke<fp4x2_e1m2_t, fp4x2_e2m1_t, fp4x2_e1m2_t, asc::te::zn_layout_ptn>(1, 0);
}

TEST(QgmmMxActivationQuantTest, KernelMxFp4E2M1E1M2NzLengthDynamicDtypeRange)
{
    RunKernelSmoke<fp4x2_e2m1_t, fp4x2_e1m2_t, fp4x2_e2m1_t, asc::te::nz_layout_ptn>(1, 2);
}

TEST(QgmmMxActivationQuantTest, KernelMxFp4E2M1E2M1ZnOffset)
{
    RunKernelSmoke<fp4x2_e2m1_t, fp4x2_e2m1_t, fp4x2_e2m1_t, asc::te::zn_layout_ptn>(0, 0);
}
