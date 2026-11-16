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
 * @brief Float4 E2M1 标杆值测试
 *
 * 格式: 1 sign + 2 exp (bias=1) + 1 man
 * 参考: OCP Microscaling Formats (MX) v1.0
 *
 * 测试方向:
 * 1. float -> 位模式: 验证 float 输入转换后的位表示
 * 2. 位模式 -> float: 验证位模式的语义解释是否符合 OCP 规范
 */

#include <cmath>
#include <limits>
#include "gtest/gtest.h"

#include "float_benchmark_data.h"
#include "opdev/float4_e2m1.h"

namespace op {
namespace benchmark {

// ============================================================================
// 辅助函数
// ============================================================================

static Float4E2M1 FromBits(uint8_t bits) { return Float4E2M1(bits, Float4E2M1::FromBits()); }

// ============================================================================
// Float4 E2M1 标杆值定义 (来源: OCP MX MXFP4)
// E2M1: 1 sign + 2 exp (bias=1) + 1 mantissa, max=6.0
// 位布局: bit3=sign, bits2-1=exp(2bits), bit0=man(1bit), bias=1
// ============================================================================

// 所有可表示的正数值测试用例
static constexpr BenchmarkCase<Float4E2M1> kE2M1_AllPositiveCases[] = {
    {0.0f, 0x0, 0.0f, 0.0f, "zero", "OCP"},
    {0.5f, 0x1, 0.5f, 0.0f, "denorm 0.5 (man=1)", "OCP"},
    {1.0f, 0x2, 1.0f, 0.0f, "1.0 (exp=1, man=0)", "OCP"},
    {1.5f, 0x3, 1.5f, 0.0f, "1.5 (exp=1, man=1)", "OCP"},
    {2.0f, 0x4, 2.0f, 0.0f, "2.0 (exp=2, man=0)", "OCP"},
    {3.0f, 0x5, 3.0f, 0.0f, "3.0 (exp=2, man=1)", "OCP"},
    {4.0f, 0x6, 4.0f, 0.0f, "4.0 (exp=3, man=0)", "OCP"},
    {6.0f, 0x7, 6.0f, 0.0f, "max 6.0 (exp=3, man=1)", "OCP"},
};

// ============================================================================
// 测试类
// ============================================================================

class Float4E2M1Benchmark : public testing::Test {
protected:
    void RunBenchmark(const BenchmarkCase<Float4E2M1>& tc)
    {
        Float4E2M1 result(tc.input);

        // 验证位表示
        EXPECT_EQ(result.value, tc.expected_bits) << "Input: " << tc.input << ", Desc: " << tc.description;

        // 验证转换值
        float actual = static_cast<float>(result);
        EXPECT_NEAR(actual, tc.expected_value, tc.tolerance) << "Desc: " << tc.description;
    }
};

// ============================================================================
// 测试用例: float -> 位模式 (验证转换正确性)
// ============================================================================

TEST_F(Float4E2M1Benchmark, AllPositiveValues)
{
    for (const auto& tc : kE2M1_AllPositiveCases) {
        RunBenchmark(tc);
    }
}

TEST_F(Float4E2M1Benchmark, NegativeValues)
{
    // 负值：最高位为符号位
    Float4E2M1 neg_half(-0.5f);
    EXPECT_EQ(neg_half.value, 0x9); // 1 00 1
    EXPECT_FLOAT_EQ(static_cast<float>(neg_half), -0.5f);

    Float4E2M1 neg_one(-1.0f);
    EXPECT_EQ(neg_one.value, 0xA); // 1 01 0
    EXPECT_FLOAT_EQ(static_cast<float>(neg_one), -1.0f);

    Float4E2M1 neg_max(-6.0f);
    EXPECT_EQ(neg_max.value, 0xF); // 1 11 1
    EXPECT_FLOAT_EQ(static_cast<float>(neg_max), -6.0f);
}

TEST_F(Float4E2M1Benchmark, MaxValue)
{
    Float4E2M1 max_val(6.0f);
    float result = static_cast<float>(max_val);
    EXPECT_FLOAT_EQ(result, 6.0f);
    EXPECT_EQ(max_val.value, 0x7);
}

TEST_F(Float4E2M1Benchmark, OverflowClamp)
{
    Float4E2M1 overflow(100.0f);
    float result = static_cast<float>(overflow);
    // 溢出应该 clamp 到最大值 6.0
    EXPECT_FLOAT_EQ(result, 6.0f);
    EXPECT_EQ(overflow.value, 0x7);
}

TEST_F(Float4E2M1Benchmark, UnderflowToZero)
{
    Float4E2M1 tiny(0.1f);
    float result = static_cast<float>(tiny);
    // 小于最小 denorm 值应舍入到 0 或 0.5
    EXPECT_TRUE(result == 0.0f || result == 0.5f);
}

TEST_F(Float4E2M1Benchmark, NaNClampToMax)
{
    // E2M1 没有 NaN，输入 NaN 应 clamp 到最大值
    Float4E2M1 nan_val(std::nanf(""));
    float result = static_cast<float>(nan_val);
    EXPECT_FLOAT_EQ(result, 6.0f); // 正 NaN -> +max
}

TEST_F(Float4E2M1Benchmark, InfinityClampToMax)
{
    // E2M1 没有无穷大，输入 Inf 应 clamp 到最大值
    Float4E2M1 inf_val(std::numeric_limits<float>::infinity());
    float result = static_cast<float>(inf_val);
    EXPECT_FLOAT_EQ(result, 6.0f);

    Float4E2M1 neg_inf_val(-std::numeric_limits<float>::infinity());
    result = static_cast<float>(neg_inf_val);
    EXPECT_FLOAT_EQ(result, -6.0f);
}

TEST_F(Float4E2M1Benchmark, RoundingBehavior)
{
    // 测试舍入行为 (round-to-nearest-even)
    // 1.25 应该舍入到 1.0 或 1.5
    Float4E2M1 val1(1.25f);
    float r1 = static_cast<float>(val1);
    EXPECT_TRUE(r1 == 1.0f || r1 == 1.5f);

    // 2.5 应该舍入到 2.0 或 3.0
    Float4E2M1 val2(2.5f);
    float r2 = static_cast<float>(val2);
    EXPECT_TRUE(r2 == 2.0f || r2 == 3.0f);
}

// ============================================================================
// 测试用例: 位模式 -> float (OCP 规范符合性验证)
// ============================================================================

TEST_F(Float4E2M1Benchmark, OcpSpec_Zeros)
{
    // OCP: S.00.0 = ±0
    Float4E2M1 pos_zero = FromBits(0x0);
    EXPECT_EQ(pos_zero.value, 0x0);
    EXPECT_FLOAT_EQ(static_cast<float>(pos_zero), 0.0f);

    Float4E2M1 neg_zero = FromBits(0x8);
    EXPECT_EQ(neg_zero.value, 0x8);
    EXPECT_FLOAT_EQ(static_cast<float>(neg_zero), -0.0f);
}

TEST_F(Float4E2M1Benchmark, OcpSpec_Subnormal)
{
    // OCP: S.00.1 = ±2^0 × 0.5 = ±0.5 (唯一的 denorm 值)
    Float4E2M1 sub = FromBits(0x1);
    EXPECT_EQ(sub.value, 0x1);
    EXPECT_FLOAT_EQ(static_cast<float>(sub), 0.5f);

    Float4E2M1 neg_sub = FromBits(0x9);
    EXPECT_EQ(neg_sub.value, 0x9);
    EXPECT_FLOAT_EQ(static_cast<float>(neg_sub), -0.5f);
}

TEST_F(Float4E2M1Benchmark, OcpSpec_Normals)
{
    // OCP: S.01.0 = ±2^0 × 1.0 = ±1.0
    Float4E2M1 min_norm = FromBits(0x2);
    EXPECT_EQ(min_norm.value, 0x2);
    EXPECT_FLOAT_EQ(static_cast<float>(min_norm), 1.0f);

    // OCP: S.11.1 = ±2^2 × 1.5 = ±6.0
    Float4E2M1 max_norm = FromBits(0x7);
    EXPECT_EQ(max_norm.value, 0x7);
    EXPECT_FLOAT_EQ(static_cast<float>(max_norm), 6.0f);

    Float4E2M1 neg_max = FromBits(0xF);
    EXPECT_EQ(neg_max.value, 0xF);
    EXPECT_FLOAT_EQ(static_cast<float>(neg_max), -6.0f);
}

TEST_F(Float4E2M1Benchmark, OcpSpec_AllPositiveValues)
{
    // 验证所有正数值位模式的语义解释
    // 0x0 = 0.0
    EXPECT_FLOAT_EQ(static_cast<float>(FromBits(0x0)), 0.0f);
    // 0x1 = 0.5 (denorm)
    EXPECT_FLOAT_EQ(static_cast<float>(FromBits(0x1)), 0.5f);
    // 0x2 = 1.0
    EXPECT_FLOAT_EQ(static_cast<float>(FromBits(0x2)), 1.0f);
    // 0x3 = 1.5
    EXPECT_FLOAT_EQ(static_cast<float>(FromBits(0x3)), 1.5f);
    // 0x4 = 2.0
    EXPECT_FLOAT_EQ(static_cast<float>(FromBits(0x4)), 2.0f);
    // 0x5 = 3.0
    EXPECT_FLOAT_EQ(static_cast<float>(FromBits(0x5)), 3.0f);
    // 0x6 = 4.0
    EXPECT_FLOAT_EQ(static_cast<float>(FromBits(0x6)), 4.0f);
    // 0x7 = 6.0
    EXPECT_FLOAT_EQ(static_cast<float>(FromBits(0x7)), 6.0f);
}

} // namespace benchmark
} // namespace op
