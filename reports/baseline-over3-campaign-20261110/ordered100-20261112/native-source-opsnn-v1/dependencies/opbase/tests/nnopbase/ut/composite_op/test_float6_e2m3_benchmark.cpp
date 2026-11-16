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
 * @brief Float6 E2M3 标杆值测试
 *
 * 格式: 1 sign + 2 exp (bias=1) + 3 man
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
#include "opdev/float6_e2m3.h"

namespace op {
namespace benchmark {

// ============================================================================
// 辅助函数
// ============================================================================

static Float6E2M3 FromBits(uint8_t bits) { return Float6E2M3(bits, Float6E2M3::FromBits()); }

// ============================================================================
// Float6 E2M3 标杆值定义 (来源: OCP MX)
// E2M3: 1 sign + 2 exp (bias=1) + 3 mantissa, max=7.5
// 位布局: bit5=sign, bits4-3=exp(2bits), bits2-0=man(3bits), bias=1
// ============================================================================

static constexpr BenchmarkCase<Float6E2M3> kE2M3_TypicalCases[] = {
    {0.0f, 0x00, 0.0f, 0.0f, "zero", "OCP"},      {0.125f, 0x01, 0.125f, 0.0f, "0.125", "OCP"},
    {0.25f, 0x02, 0.25f, 0.0f, "0.25", "OCP"},    {0.5f, 0x04, 0.5f, 0.0f, "half", "OCP"},
    {0.875f, 0x07, 0.875f, 0.0f, "0.875", "OCP"}, {1.0f, 0x08, 1.0f, 0.0f, "one", "OCP"},
    {1.5f, 0x0C, 1.5f, 0.0f, "1.5", "OCP"},       {2.0f, 0x10, 2.0f, 0.0f, "two", "OCP"},
};

// ============================================================================
// 测试类
// ============================================================================

class Float6E2M3Benchmark : public testing::Test {
protected:
    void RunBenchmark(const BenchmarkCase<Float6E2M3>& tc)
    {
        Float6E2M3 result(tc.input);

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

TEST_F(Float6E2M3Benchmark, TypicalValues)
{
    for (const auto& tc : kE2M3_TypicalCases) {
        RunBenchmark(tc);
    }
}

TEST_F(Float6E2M3Benchmark, NegativeValues)
{
    Float6E2M3 neg_one(-1.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(neg_one), -1.0f);

    Float6E2M3 neg_half(-0.5f);
    EXPECT_FLOAT_EQ(static_cast<float>(neg_half), -0.5f);
}

TEST_F(Float6E2M3Benchmark, MaxValue)
{
    // E2M3 max = 7.5
    Float6E2M3 max_val(7.5f);
    float result = static_cast<float>(max_val);
    EXPECT_NEAR(result, 7.5f, 0.1f);
}

TEST_F(Float6E2M3Benchmark, OverflowClamp)
{
    Float6E2M3 overflow(100.0f);
    float result = static_cast<float>(overflow);
    // 溢出应该 clamp 到最大值
    EXPECT_LE(result, 8.0f);
}

// ============================================================================
// 测试用例: 位模式 -> float (OCP 规范符合性验证)
// ============================================================================

TEST_F(Float6E2M3Benchmark, OcpSpec_Zeros)
{
    // OCP: S.00.000 = ±0
    Float6E2M3 pos_zero = FromBits(0x00);
    EXPECT_EQ(pos_zero.value, 0x00);
    EXPECT_FLOAT_EQ(static_cast<float>(pos_zero), 0.0f);

    Float6E2M3 neg_zero = FromBits(0x20);
    EXPECT_EQ(neg_zero.value, 0x20);
    EXPECT_FLOAT_EQ(static_cast<float>(neg_zero), -0.0f);
}

TEST_F(Float6E2M3Benchmark, OcpSpec_Subnormals)
{
    // OCP: S.00.001 = ±2^0 × 0.125 = ±0.125
    Float6E2M3 min_sub = FromBits(0x01);
    EXPECT_EQ(min_sub.value, 0x01);
    EXPECT_FLOAT_EQ(static_cast<float>(min_sub), 0.125f);

    // OCP: S.00.111 = ±2^0 × 0.875 = ±0.875
    Float6E2M3 max_sub = FromBits(0x07);
    EXPECT_EQ(max_sub.value, 0x07);
    EXPECT_FLOAT_EQ(static_cast<float>(max_sub), 0.875f);
}

TEST_F(Float6E2M3Benchmark, OcpSpec_Normals)
{
    // OCP: S.01.000 = ±2^0 × 1.0 = ±1.0
    Float6E2M3 min_norm = FromBits(0x08);
    EXPECT_EQ(min_norm.value, 0x08);
    EXPECT_FLOAT_EQ(static_cast<float>(min_norm), 1.0f);

    // OCP: S.11.111 = ±2^2 × 1.875 = ±7.5
    Float6E2M3 max_norm = FromBits(0x1F);
    EXPECT_EQ(max_norm.value, 0x1F);
    EXPECT_FLOAT_EQ(static_cast<float>(max_norm), 7.5f);
}

TEST_F(Float6E2M3Benchmark, OcpSpec_TypicalBitPatterns)
{
    // 验证典型位模式的语义解释
    // 0.5 = 0.00.100 = 0x04 (denorm: man=4 -> 0.5)
    EXPECT_FLOAT_EQ(static_cast<float>(FromBits(0x04)), 0.5f);

    // 1.5 = 0.01.100 = 0x0C (exp=1, man=4 -> 2^0 * 1.5)
    EXPECT_FLOAT_EQ(static_cast<float>(FromBits(0x0C)), 1.5f);

    // 2.0 = 0.10.000 = 0x10 (exp=2, man=0 -> 2^1 * 1.0)
    EXPECT_FLOAT_EQ(static_cast<float>(FromBits(0x10)), 2.0f);

    // 3.0 = 0.10.100 = 0x14 (exp=2, man=4 -> 2^1 * 1.5)
    EXPECT_FLOAT_EQ(static_cast<float>(FromBits(0x14)), 3.0f);

    // 4.0 = 0.11.000 = 0x18 (exp=3, man=0 -> 2^2 * 1.0)
    EXPECT_FLOAT_EQ(static_cast<float>(FromBits(0x18)), 4.0f);

    // 1.875 = 0.01.111 = 0x0F (exp=1, man=7 -> 2^0 * 1.875)
    EXPECT_FLOAT_EQ(static_cast<float>(FromBits(0x0F)), 1.875f);

    // 7.5 = 0.11.111 = 0x1F (exp=3, man=7 -> 2^2 * 1.875)
    EXPECT_FLOAT_EQ(static_cast<float>(FromBits(0x1F)), 7.5f);
}

} // namespace benchmark
} // namespace op
