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
 * @brief Float8 E5M2 标杆值测试
 *
 * 格式: 1 sign + 5 exp (bias=15) + 2 man
 * 参考: OCP Microscaling Formats (MX) v1.0, ONNX Float8
 *
 * 测试方向:
 * 1. float -> 位模式: 验证 float 输入转换后的位表示
 * 2. 位模式 -> float: 验证位模式的语义解释是否符合 OCP 规范
 */

#include <cmath>
#include <limits>
#include "gtest/gtest.h"

#include "float_benchmark_data.h"
#include "opdev/float8_e5m2.h"

namespace op {
namespace benchmark {

// ============================================================================
// 辅助函数
// ============================================================================

static Float8E5M2 FromBits(uint8_t bits) { return Float8E5M2(bits, Float8E5M2::FromBits()); }

// ============================================================================
// Float8 E5M2 标杆值定义 (来源: ONNX/OCP/NVIDIA)
// 位布局: bit7=sign, bits6-2=exp(5bits), bits1-0=man(2bits), bias=15
// ============================================================================

// 边界值测试用例
static constexpr BenchmarkCase<Float8E5M2> kE5M2_BoundaryCases[] = {
    {6.1035e-5f, 0x04, 6.1035e-5f, 1e-9f, "min normal (2^-14)", "OCP"},
    {57344.0f, 0x7B, 57344.0f, 0.0f, "max positive", "OCP"},
    {-57344.0f, 0xFB, -57344.0f, 0.0f, "min negative", "OCP"},
};

// 特殊值测试用例
static constexpr BenchmarkCase<Float8E5M2> kE5M2_SpecialCases[] = {
    {0.0f, 0x00, 0.0f, 0.0f, "+zero", "OCP"},
    {-0.0f, 0x80, -0.0f, 0.0f, "-zero", "OCP"},
};

// 典型值测试用例
static constexpr BenchmarkCase<Float8E5M2> kE5M2_TypicalCases[] = {
    {1.0f, 0x3C, 1.0f, 0.0f, "one", "OCP"},   {2.0f, 0x40, 2.0f, 0.0f, "two", "OCP"},
    {0.5f, 0x38, 0.5f, 0.0f, "half", "OCP"},  {1.5f, 0x3E, 1.5f, 0.0f, "1.5", "OCP"},
    {3.0f, 0x42, 3.0f, 0.0f, "three", "OCP"}, {4.0f, 0x44, 4.0f, 0.0f, "four", "OCP"},
};

// ============================================================================
// 测试类
// ============================================================================

class Float8E5M2Benchmark : public testing::Test {
protected:
    void RunBenchmark(const BenchmarkCase<Float8E5M2>& tc)
    {
        Float8E5M2 result(tc.input);

        // 验证位表示
        EXPECT_EQ(result.value, tc.expected_bits) << "Input: " << tc.input << ", Desc: " << tc.description;

        // 验证转换值
        float actual = static_cast<float>(result);
        if (std::isnan(tc.expected_value)) {
            EXPECT_TRUE(std::isnan(actual)) << "Desc: " << tc.description;
        } else {
            EXPECT_NEAR(actual, tc.expected_value, tc.tolerance) << "Desc: " << tc.description;
        }
    }
};

// ============================================================================
// 测试用例: float -> 位模式 (验证转换正确性)
// ============================================================================

TEST_F(Float8E5M2Benchmark, SpecialValues)
{
    for (const auto& tc : kE5M2_SpecialCases) {
        RunBenchmark(tc);
    }
}

TEST_F(Float8E5M2Benchmark, BoundaryValues)
{
    for (const auto& tc : kE5M2_BoundaryCases) {
        RunBenchmark(tc);
    }
}

TEST_F(Float8E5M2Benchmark, TypicalValues)
{
    for (const auto& tc : kE5M2_TypicalCases) {
        RunBenchmark(tc);
    }
}

TEST_F(Float8E5M2Benchmark, InfinityHandling)
{
    // E5M2 支持 Infinity
    Float8E5M2 inf_val(std::numeric_limits<float>::infinity());
    EXPECT_TRUE(std::isinf(static_cast<float>(inf_val)));
    EXPECT_GT(static_cast<float>(inf_val), 0.0f);
    EXPECT_EQ(inf_val.value, 0x7C);
}

TEST_F(Float8E5M2Benchmark, NaNHandling)
{
    Float8E5M2 nan_val(std::nanf(""));
    EXPECT_TRUE(std::isnan(static_cast<float>(nan_val)));
}

TEST_F(Float8E5M2Benchmark, OverflowToInfinity)
{
    Float8E5M2 overflow_val(100000.0f);
    EXPECT_TRUE(std::isinf(static_cast<float>(overflow_val)));
    EXPECT_GT(static_cast<float>(overflow_val), 0.0f);
}

TEST_F(Float8E5M2Benchmark, NegativeInfinity)
{
    Float8E5M2 neg_inf_val(-std::numeric_limits<float>::infinity());
    EXPECT_TRUE(std::isinf(static_cast<float>(neg_inf_val)));
    EXPECT_LT(static_cast<float>(neg_inf_val), 0.0f);
    EXPECT_EQ(neg_inf_val.value, 0xFC);
}

// ============================================================================
// 测试用例: 位模式 -> float (OCP 规范符合性验证)
// ============================================================================

TEST_F(Float8E5M2Benchmark, OcpSpec_Zeros)
{
    // OCP: S.00000.00 = ±0
    Float8E5M2 pos_zero = FromBits(0x00);
    EXPECT_EQ(pos_zero.value, 0x00);
    EXPECT_FLOAT_EQ(static_cast<float>(pos_zero), 0.0f);

    Float8E5M2 neg_zero = FromBits(0x80);
    EXPECT_EQ(neg_zero.value, 0x80);
    EXPECT_FLOAT_EQ(static_cast<float>(neg_zero), -0.0f);
}

TEST_F(Float8E5M2Benchmark, OcpSpec_Subnormals)
{
    // OCP E5M2 subnormal: exp=0, value = (-1)^s × 2^(1-bias) × (man/4)
    // 位模式 0x01: exp=0, man=01 -> 2^(1-15) × (1/4) = 2^(-14) × 0.25 = 2^(-16)
    Float8E5M2 min_sub = FromBits(0x01);
    EXPECT_EQ(min_sub.value, 0x01);
    float expected = std::ldexp(1.0f, -16); // 2^(-16)
    EXPECT_FLOAT_EQ(static_cast<float>(min_sub), expected);

    // 位模式 0x03: exp=0, man=11 -> 2^(1-15) × (3/4) = 2^(-14) × 0.75
    Float8E5M2 max_sub = FromBits(0x03);
    EXPECT_EQ(max_sub.value, 0x03);
    expected = std::ldexp(0.75f, -14); // 2^(-14) × 0.75
    EXPECT_FLOAT_EQ(static_cast<float>(max_sub), expected);
}

TEST_F(Float8E5M2Benchmark, OcpSpec_Normals)
{
    // OCP: S.00001.00 = ±2^(-14)
    Float8E5M2 min_norm = FromBits(0x04);
    EXPECT_EQ(min_norm.value, 0x04);
    float expected = std::ldexp(1.0f, -14);
    EXPECT_FLOAT_EQ(static_cast<float>(min_norm), expected);

    // OCP: S.11110.11 = ±2^15 × 1.75 = ±57344
    Float8E5M2 max_norm = FromBits(0x7B);
    EXPECT_EQ(max_norm.value, 0x7B);
    EXPECT_FLOAT_EQ(static_cast<float>(max_norm), 57344.0f);

    Float8E5M2 neg_max = FromBits(0xFB);
    EXPECT_EQ(neg_max.value, 0xFB);
    EXPECT_FLOAT_EQ(static_cast<float>(neg_max), -57344.0f);
}

TEST_F(Float8E5M2Benchmark, OcpSpec_Infinities)
{
    // OCP: S.11111.00 = ±Inf
    Float8E5M2 pos_inf = FromBits(0x7C);
    EXPECT_EQ(pos_inf.value, 0x7C);
    EXPECT_TRUE(std::isinf(static_cast<float>(pos_inf)));
    EXPECT_GT(static_cast<float>(pos_inf), 0.0f);

    Float8E5M2 neg_inf = FromBits(0xFC);
    EXPECT_EQ(neg_inf.value, 0xFC);
    EXPECT_TRUE(std::isinf(static_cast<float>(neg_inf)));
    EXPECT_LT(static_cast<float>(neg_inf), 0.0f);
}

TEST_F(Float8E5M2Benchmark, OcpSpec_NaN)
{
    // OCP: S.11111.{01, 10, 11} = NaN
    Float8E5M2 nan1 = FromBits(0x7D);
    EXPECT_TRUE(std::isnan(static_cast<float>(nan1)));

    Float8E5M2 nan2 = FromBits(0x7E);
    EXPECT_TRUE(std::isnan(static_cast<float>(nan2)));

    Float8E5M2 nan3 = FromBits(0x7F);
    EXPECT_TRUE(std::isnan(static_cast<float>(nan3)));
}

TEST_F(Float8E5M2Benchmark, OcpSpec_TypicalBitPatterns)
{
    // 验证典型位模式的语义解释
    // 1.0 = 0.01111.00 = 0x3C (exp=15, man=0 -> 2^0 * 1.0)
    EXPECT_FLOAT_EQ(static_cast<float>(FromBits(0x3C)), 1.0f);

    // 2.0 = 0.10000.00 = 0x40 (exp=16, man=0 -> 2^1 * 1.0)
    EXPECT_FLOAT_EQ(static_cast<float>(FromBits(0x40)), 2.0f);

    // 0.5 = 0.01110.00 = 0x38 (exp=14, man=0 -> 2^(-1) * 1.0)
    EXPECT_FLOAT_EQ(static_cast<float>(FromBits(0x38)), 0.5f);

    // 1.5 = 0.01111.10 = 0x3E (exp=15, man=2 -> 2^0 * 1.5)
    EXPECT_FLOAT_EQ(static_cast<float>(FromBits(0x3E)), 1.5f);

    // 1.75 = 0.01111.11 = 0x3F (exp=15, man=3 -> 2^0 * 1.75)
    EXPECT_FLOAT_EQ(static_cast<float>(FromBits(0x3F)), 1.75f);
}

} // namespace benchmark
} // namespace op
