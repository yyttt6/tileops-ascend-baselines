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
 * @brief Float6 E3M2 标杆值测试
 *
 * 格式: 1 sign + 3 exp (bias=3) + 2 man
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
#include "opdev/float6_e3m2.h"

namespace op {
namespace benchmark {

// ============================================================================
// 辅助函数
// ============================================================================

static Float6E3M2 FromBits(uint8_t bits) { return Float6E3M2(bits, Float6E3M2::FromBits()); }

// ============================================================================
// Float6 E3M2 标杆值定义 (来源: OCP MX/NVIDIA)
// E3M2: 1 sign + 3 exp (bias=3) + 2 mantissa, max=28
// ============================================================================

// 位布局: bit5=sign, bits4-2=exp(3bits), bits1-0=man(2bits), bias=3
static constexpr BenchmarkCase<Float6E3M2> kE3M2_TypicalCases[] = {
    {0.0f, 0x00, 0.0f, 0.0f, "zero", "OCP"},      {1.0f, 0x0C, 1.0f, 0.0f, "one", "OCP"},
    {2.0f, 0x10, 2.0f, 0.0f, "two", "OCP"},       {4.0f, 0x14, 4.0f, 0.0f, "four", "OCP"},
    {8.0f, 0x18, 8.0f, 0.0f, "eight", "OCP"},     {0.5f, 0x08, 0.5f, 0.0f, "half", "OCP"},
    {0.25f, 0x04, 0.25f, 0.0f, "quarter", "OCP"}, {1.75f, 0x0F, 1.75f, 0.0f, "1.75", "OCP"},
};

// ============================================================================
// 测试类
// ============================================================================

class Float6E3M2Benchmark : public testing::Test {
protected:
    void RunBenchmark(const BenchmarkCase<Float6E3M2>& tc)
    {
        Float6E3M2 result(tc.input);

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

TEST_F(Float6E3M2Benchmark, TypicalValues)
{
    for (const auto& tc : kE3M2_TypicalCases) {
        RunBenchmark(tc);
    }
}

TEST_F(Float6E3M2Benchmark, NegativeValues)
{
    Float6E3M2 neg_one(-1.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(neg_one), -1.0f);

    Float6E3M2 neg_half(-0.5f);
    EXPECT_FLOAT_EQ(static_cast<float>(neg_half), -0.5f);
}

TEST_F(Float6E3M2Benchmark, MaxValue)
{
    // E3M2 max = 28.0
    Float6E3M2 max_val(28.0f);
    float result = static_cast<float>(max_val);
    EXPECT_NEAR(result, 28.0f, 0.5f);
}

TEST_F(Float6E3M2Benchmark, OverflowClamp)
{
    Float6E3M2 overflow(100.0f);
    float result = static_cast<float>(overflow);
    // 溢出应该 clamp 到最大值
    EXPECT_LE(result, 28.0f);
}

// ============================================================================
// 测试用例: 位模式 -> float (OCP 规范符合性验证)
// ============================================================================

TEST_F(Float6E3M2Benchmark, OcpSpec_Zeros)
{
    // OCP: S.000.00 = ±0
    Float6E3M2 pos_zero = FromBits(0x00);
    EXPECT_EQ(pos_zero.value, 0x00);
    EXPECT_FLOAT_EQ(static_cast<float>(pos_zero), 0.0f);

    Float6E3M2 neg_zero = FromBits(0x20);
    EXPECT_EQ(neg_zero.value, 0x20);
    EXPECT_FLOAT_EQ(static_cast<float>(neg_zero), -0.0f);
}

TEST_F(Float6E3M2Benchmark, OcpSpec_Subnormals)
{
    // OCP: S.000.01 = ±2^(-2) × 0.25 = ±0.0625
    Float6E3M2 min_sub = FromBits(0x01);
    EXPECT_EQ(min_sub.value, 0x01);
    EXPECT_FLOAT_EQ(static_cast<float>(min_sub), 0.0625f);

    // OCP: S.000.11 = ±2^(-2) × 0.75 = ±0.1875
    Float6E3M2 max_sub = FromBits(0x03);
    EXPECT_EQ(max_sub.value, 0x03);
    EXPECT_FLOAT_EQ(static_cast<float>(max_sub), 0.1875f);
}

TEST_F(Float6E3M2Benchmark, OcpSpec_Normals)
{
    // OCP: S.001.00 = ±2^(-2) × 1.0 = ±0.25
    Float6E3M2 min_norm = FromBits(0x04);
    EXPECT_EQ(min_norm.value, 0x04);
    EXPECT_FLOAT_EQ(static_cast<float>(min_norm), 0.25f);

    // OCP: S.111.11 = ±2^4 × 1.75 = ±28.0
    Float6E3M2 max_norm = FromBits(0x1F);
    EXPECT_EQ(max_norm.value, 0x1F);
    EXPECT_FLOAT_EQ(static_cast<float>(max_norm), 28.0f);

    Float6E3M2 neg_max = FromBits(0x3F);
    EXPECT_EQ(neg_max.value, 0x3F);
    EXPECT_FLOAT_EQ(static_cast<float>(neg_max), -28.0f);
}

TEST_F(Float6E3M2Benchmark, OcpSpec_TypicalBitPatterns)
{
    // 验证典型位模式的语义解释
    // 1.0 = 0.011.00 = 0x0C (exp=3, man=0 -> 2^0 * 1.0)
    EXPECT_FLOAT_EQ(static_cast<float>(FromBits(0x0C)), 1.0f);

    // 2.0 = 0.100.00 = 0x10 (exp=4, man=0 -> 2^1 * 1.0)
    EXPECT_FLOAT_EQ(static_cast<float>(FromBits(0x10)), 2.0f);

    // 0.5 = 0.010.00 = 0x08 (exp=2, man=0 -> 2^(-1) * 1.0)
    EXPECT_FLOAT_EQ(static_cast<float>(FromBits(0x08)), 0.5f);

    // 1.75 = 0.011.11 = 0x0F (exp=3, man=3 -> 2^0 * 1.75)
    EXPECT_FLOAT_EQ(static_cast<float>(FromBits(0x0F)), 1.75f);

    // 4.0 = 0.101.00 = 0x14 (exp=5, man=0 -> 2^2 * 1.0)
    EXPECT_FLOAT_EQ(static_cast<float>(FromBits(0x14)), 4.0f);

    // 8.0 = 0.110.00 = 0x18 (exp=6, man=0 -> 2^3 * 1.0)
    EXPECT_FLOAT_EQ(static_cast<float>(FromBits(0x18)), 8.0f);

    // 16.0 = 0.111.00 = 0x1C (exp=7, man=0 -> 2^4 * 1.0)
    EXPECT_FLOAT_EQ(static_cast<float>(FromBits(0x1C)), 16.0f);
}

} // namespace benchmark
} // namespace op
