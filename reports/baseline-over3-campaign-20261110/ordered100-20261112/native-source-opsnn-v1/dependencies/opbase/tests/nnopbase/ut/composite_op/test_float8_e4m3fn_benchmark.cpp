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
 * @brief Float8 E4M3FN 标杆值测试
 *
 * 格式: 1 sign + 4 exp (bias=7) + 3 man
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
#include "opdev/float8_e4m3fn.h"

namespace op {
namespace benchmark {

// ============================================================================
// 辅助函数
// ============================================================================

static Float8E4M3FN FromBits(uint8_t bits) { return Float8E4M3FN(bits, Float8E4M3FN::FromBits()); }

// ============================================================================
// Float8 E4M3FN 标杆值定义 (来源: ONNX/OCP/NVIDIA)
// ============================================================================

// 边界值测试用例
static constexpr BenchmarkCase<Float8E4M3FN> kE4M3FN_BoundaryCases[] = {
    {0.015625f, 0x08, 0.015625f, 0.0f, "min normal (2^-6)", "OCP"},
    {0.001953125f, 0x01, 0.001953125f, 0.0f, "min subnormal", "OCP"},
    {448.0f, 0x7E, 448.0f, 0.0f, "max positive", "OCP"},
    {-448.0f, 0xFE, -448.0f, 0.0f, "min negative", "OCP"},
};

// 特殊值测试用例
static constexpr BenchmarkCase<Float8E4M3FN> kE4M3FN_SpecialCases[] = {
    {0.0f, 0x00, 0.0f, 0.0f, "+zero", "OCP"},
    {-0.0f, 0x80, -0.0f, 0.0f, "-zero", "OCP"},
};

// 典型值测试用例
static constexpr BenchmarkCase<Float8E4M3FN> kE4M3FN_TypicalCases[] = {
    {1.0f, 0x38, 1.0f, 0.0f, "one", "OCP"},    {2.0f, 0x40, 2.0f, 0.0f, "two", "OCP"},
    {0.5f, 0x30, 0.5f, 0.0f, "half", "OCP"},   {1.5f, 0x3C, 1.5f, 0.0f, "1.5", "OCP"},
    {1.25f, 0x3A, 1.25f, 0.0f, "1.25", "OCP"}, {4.0f, 0x48, 4.0f, 0.0f, "four", "OCP"},
    {8.0f, 0x50, 8.0f, 0.0f, "eight", "OCP"},  {0.25f, 0x28, 0.25f, 0.0f, "quarter", "OCP"},
    {3.0f, 0x44, 3.0f, 0.0f, "three", "OCP"},  {6.0f, 0x4C, 6.0f, 0.0f, "six", "OCP"},
    {7.0f, 0x4E, 7.0f, 0.0f, "seven", "OCP"},
};

// 溢出处理测试用例
static constexpr BenchmarkCase<Float8E4M3FN> kE4M3FN_OverflowCases[] = {
    {1000.0f, 0x7E, 448.0f, 0.0f, "positive overflow clamp", "OCP"},
    {-1000.0f, 0xFE, -448.0f, 0.0f, "negative overflow clamp", "OCP"},
};

// ============================================================================
// 测试类
// ============================================================================

class Float8E4M3FNBenchmark : public testing::Test {
protected:
    // 测试方向1: float -> 位模式
    void RunBenchmark(const BenchmarkCase<Float8E4M3FN>& tc)
    {
        Float8E4M3FN result(tc.input);

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

TEST_F(Float8E4M3FNBenchmark, SpecialValues)
{
    for (const auto& tc : kE4M3FN_SpecialCases) {
        RunBenchmark(tc);
    }
}

TEST_F(Float8E4M3FNBenchmark, BoundaryValues)
{
    for (const auto& tc : kE4M3FN_BoundaryCases) {
        RunBenchmark(tc);
    }
}

TEST_F(Float8E4M3FNBenchmark, TypicalValues)
{
    for (const auto& tc : kE4M3FN_TypicalCases) {
        RunBenchmark(tc);
    }
}

TEST_F(Float8E4M3FNBenchmark, OverflowHandling)
{
    for (const auto& tc : kE4M3FN_OverflowCases) {
        RunBenchmark(tc);
    }
}

TEST_F(Float8E4M3FNBenchmark, NaNHandling)
{
    Float8E4M3FN nan_val(std::nanf(""));
    EXPECT_TRUE(nan_val.IsNaN());
    EXPECT_EQ(nan_val.value, 0x7F);
}

TEST_F(Float8E4M3FNBenchmark, InfinityClamp)
{
    Float8E4M3FN inf_val(std::numeric_limits<float>::infinity());
    EXPECT_FLOAT_EQ(static_cast<float>(inf_val), 448.0f);
    EXPECT_EQ(inf_val.value, 0x7E);

    Float8E4M3FN neg_inf_val(-std::numeric_limits<float>::infinity());
    EXPECT_FLOAT_EQ(static_cast<float>(neg_inf_val), -448.0f);
    EXPECT_EQ(neg_inf_val.value, 0xFE);
}

// ============================================================================
// 测试用例: 位模式 -> float (OCP 规范符合性验证)
// ============================================================================

TEST_F(Float8E4M3FNBenchmark, OcpSpec_Zeros)
{
    // OCP: S.0000.000 = ±0
    Float8E4M3FN pos_zero = FromBits(0x00);
    EXPECT_EQ(pos_zero.value, 0x00);
    EXPECT_FLOAT_EQ(static_cast<float>(pos_zero), 0.0f);

    Float8E4M3FN neg_zero = FromBits(0x80);
    EXPECT_EQ(neg_zero.value, 0x80);
    EXPECT_FLOAT_EQ(static_cast<float>(neg_zero), -0.0f);
}

TEST_F(Float8E4M3FNBenchmark, OcpSpec_Subnormals)
{
    // OCP: S.0000.001 = ±2^(-9) = ±0.001953125
    Float8E4M3FN min_sub = FromBits(0x01);
    EXPECT_EQ(min_sub.value, 0x01);
    EXPECT_FLOAT_EQ(static_cast<float>(min_sub), 0.001953125f);

    Float8E4M3FN neg_min_sub = FromBits(0x81);
    EXPECT_EQ(neg_min_sub.value, 0x81);
    EXPECT_FLOAT_EQ(static_cast<float>(neg_min_sub), -0.001953125f);

    // OCP: S.0000.111 = ±2^(-6) × 0.875 = ±0.013671875
    Float8E4M3FN max_sub = FromBits(0x07);
    EXPECT_EQ(max_sub.value, 0x07);
    EXPECT_NEAR(static_cast<float>(max_sub), 0.013671875f, 1e-9f);
}

TEST_F(Float8E4M3FNBenchmark, OcpSpec_Normals)
{
    // OCP: S.0001.000 = ±2^(-6) = ±0.015625
    Float8E4M3FN min_norm = FromBits(0x08);
    EXPECT_EQ(min_norm.value, 0x08);
    EXPECT_FLOAT_EQ(static_cast<float>(min_norm), 0.015625f);

    // OCP: S.1111.110 = ±2^8 × 1.75 = ±448
    Float8E4M3FN max_norm = FromBits(0x7E);
    EXPECT_EQ(max_norm.value, 0x7E);
    EXPECT_FLOAT_EQ(static_cast<float>(max_norm), 448.0f);

    Float8E4M3FN neg_max = FromBits(0xFE);
    EXPECT_EQ(neg_max.value, 0xFE);
    EXPECT_FLOAT_EQ(static_cast<float>(neg_max), -448.0f);
}

TEST_F(Float8E4M3FNBenchmark, OcpSpec_NaN)
{
    // OCP: S.1111.111 = NaN
    Float8E4M3FN nan_val = FromBits(0x7F);
    EXPECT_EQ(nan_val.value, 0x7F);
    EXPECT_TRUE(std::isnan(static_cast<float>(nan_val)));
}

TEST_F(Float8E4M3FNBenchmark, OcpSpec_TypicalBitPatterns)
{
    // 验证典型位模式的语义解释
    // 1.0 = 0.0111.000 = 0x38 (exp=7, man=0 -> 2^0 * 1.0)
    EXPECT_FLOAT_EQ(static_cast<float>(FromBits(0x38)), 1.0f);

    // 2.0 = 0.1000.000 = 0x40 (exp=8, man=0 -> 2^1 * 1.0)
    EXPECT_FLOAT_EQ(static_cast<float>(FromBits(0x40)), 2.0f);

    // 0.5 = 0.0110.000 = 0x30 (exp=6, man=0 -> 2^(-1) * 1.0)
    EXPECT_FLOAT_EQ(static_cast<float>(FromBits(0x30)), 0.5f);

    // 1.5 = 0.0111.100 = 0x3C (exp=7, man=4 -> 2^0 * 1.5)
    EXPECT_FLOAT_EQ(static_cast<float>(FromBits(0x3C)), 1.5f);

    // 1.875 = 0.0111.111 = 0x3F (exp=7, man=7 -> 2^0 * 1.875)
    EXPECT_FLOAT_EQ(static_cast<float>(FromBits(0x3F)), 1.875f);

    // 4.0 = 0.1001.000 = 0x48 (exp=9, man=0 -> 2^2 * 1.0)
    EXPECT_FLOAT_EQ(static_cast<float>(FromBits(0x48)), 4.0f);

    // 8.0 = 0.1010.000 = 0x50 (exp=10, man=0 -> 2^3 * 1.0)
    EXPECT_FLOAT_EQ(static_cast<float>(FromBits(0x50)), 8.0f);

    // 112.0 = 0.1101.110 = 0x6E (exp=13, man=6 -> 2^6 * 1.75)
    EXPECT_FLOAT_EQ(static_cast<float>(FromBits(0x6E)), 112.0f);

    // 224.0 = 0.1110.110 = 0x76 (exp=14, man=6 -> 2^7 * 1.75)
    EXPECT_FLOAT_EQ(static_cast<float>(FromBits(0x76)), 224.0f);
}

} // namespace benchmark
} // namespace op
