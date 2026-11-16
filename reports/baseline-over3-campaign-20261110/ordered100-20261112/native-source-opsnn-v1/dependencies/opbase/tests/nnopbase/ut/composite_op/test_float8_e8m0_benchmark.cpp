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
 * @brief Float8 E8M0 标杆值测试
 *
 * 格式: 8-bit exponent only, no sign/mantissa, bias=127
 * 参考: OCP Microscaling Formats (MX) v1.0 - Scale Factor Format
 *
 * 测试方向:
 * 1. float -> 位模式: 验证 float 输入转换后的位表示
 * 2. 位模式 -> float: 验证位模式的语义解释是否符合 OCP 规范
 */

#include <cmath>
#include <limits>
#include "gtest/gtest.h"

#include "float_benchmark_data.h"
#include "opdev/float8_e8m0.h"

namespace op {
namespace benchmark {

// ============================================================================
// 辅助函数
// ============================================================================

static Float8E8M0 FromBits(uint8_t bits) { return Float8E8M0(bits, Float8E8M0::FromBits()); }

// ============================================================================
// Float8 E8M0 标杆值定义 (来源: OCP MX - Scale Factor Format)
// E8M0: 8-bit exponent only, bias=127, 只能表示 2 的幂次
// ============================================================================

static constexpr BenchmarkCase<Float8E8M0> kE8M0_BoundaryCases[] = {
    {1.0f, 0x7F, 1.0f, 0.0f, "one (2^0)", "OCP"},        {2.0f, 0x80, 2.0f, 0.0f, "two (2^1)", "OCP"},
    {0.5f, 0x7E, 0.5f, 0.0f, "half (2^-1)", "OCP"},      {4.0f, 0x81, 4.0f, 0.0f, "four (2^2)", "OCP"},
    {0.25f, 0x7D, 0.25f, 0.0f, "quarter (2^-2)", "OCP"},
};

// ============================================================================
// 测试类
// ============================================================================

class Float8E8M0Benchmark : public testing::Test {
protected:
    void RunBenchmark(const BenchmarkCase<Float8E8M0>& tc)
    {
        Float8E8M0 result(tc.input);

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

TEST_F(Float8E8M0Benchmark, BoundaryValues)
{
    for (const auto& tc : kE8M0_BoundaryCases) {
        RunBenchmark(tc);
    }
}

TEST_F(Float8E8M0Benchmark, PowerOfTwoOnly)
{
    // E8M0 只能表示 2 的幂次
    // 验证非 2 的幂次值的行为
    Float8E8M0 val(3.0f); // 3 不是 2 的幂
    float result = static_cast<float>(val);
    // 实现应该将其舍入到最近的 2 的幂
    EXPECT_TRUE(result == 2.0f || result == 4.0f);
}

TEST_F(Float8E8M0Benchmark, LargePowerOfTwo)
{
    Float8E8M0 large_val(256.0f); // 2^8
    float result = static_cast<float>(large_val);
    EXPECT_FLOAT_EQ(result, 256.0f);
}

TEST_F(Float8E8M0Benchmark, SmallPowerOfTwo)
{
    Float8E8M0 small_val(0.125f); // 2^-3
    float result = static_cast<float>(small_val);
    EXPECT_FLOAT_EQ(result, 0.125f);
}

TEST_F(Float8E8M0Benchmark, NegativeValueHandling)
{
    // E8M0 是无符号格式，负值应处理为 NaN
    Float8E8M0 neg_val(-1.0f);
    EXPECT_TRUE(neg_val.IsNaN());
}

TEST_F(Float8E8M0Benchmark, NaNHandling)
{
    Float8E8M0 nan_val(std::nanf(""));
    EXPECT_TRUE(nan_val.IsNaN());
    EXPECT_EQ(nan_val.value, 0xFF);
}

// ============================================================================
// 测试用例: 位模式 -> float (OCP 规范符合性验证)
// ============================================================================

TEST_F(Float8E8M0Benchmark, OcpSpec_PowersOfTwo)
{
    // OCP: value = 2^(exp - 127)
    // 0x7F = 2^(127-127) = 2^0 = 1.0
    Float8E8M0 one = FromBits(0x7F);
    EXPECT_EQ(one.value, 0x7F);
    EXPECT_FLOAT_EQ(static_cast<float>(one), 1.0f);

    // 0x80 = 2^(128-127) = 2^1 = 2.0
    Float8E8M0 two = FromBits(0x80);
    EXPECT_EQ(two.value, 0x80);
    EXPECT_FLOAT_EQ(static_cast<float>(two), 2.0f);

    // 0x7E = 2^(126-127) = 2^(-1) = 0.5
    Float8E8M0 half = FromBits(0x7E);
    EXPECT_EQ(half.value, 0x7E);
    EXPECT_FLOAT_EQ(static_cast<float>(half), 0.5f);
}

TEST_F(Float8E8M0Benchmark, OcpSpec_NaN)
{
    // OCP: 0xFF = NaN
    Float8E8M0 nan_val = FromBits(0xFF);
    EXPECT_EQ(nan_val.value, 0xFF);
    EXPECT_TRUE(nan_val.IsNaN());
}

TEST_F(Float8E8M0Benchmark, OcpSpec_MaxMin)
{
    // 0xFE = 2^127 (max value)
    Float8E8M0 max_val = FromBits(0xFE);
    EXPECT_EQ(max_val.value, 0xFE);
    EXPECT_FLOAT_EQ(static_cast<float>(max_val), std::ldexp(1.0f, 127));

    // 0x00 = Zero
    Float8E8M0 zero_val = FromBits(0x00);
    EXPECT_EQ(zero_val.value, 0x00);
    EXPECT_FLOAT_EQ(static_cast<float>(zero_val), 0.0f);

    // 0x01 = 2^(-126) (min positive value)
    Float8E8M0 min_pos = FromBits(0x01);
    EXPECT_EQ(min_pos.value, 0x01);
    EXPECT_FLOAT_EQ(static_cast<float>(min_pos), std::ldexp(1.0f, -126));
}

TEST_F(Float8E8M0Benchmark, OcpSpec_TypicalBitPatterns)
{
    // 验证典型位模式的语义解释
    // 0x7F = 2^0 = 1.0
    EXPECT_FLOAT_EQ(static_cast<float>(FromBits(0x7F)), 1.0f);

    // 0x80 = 2^1 = 2.0
    EXPECT_FLOAT_EQ(static_cast<float>(FromBits(0x80)), 2.0f);

    // 0x81 = 2^2 = 4.0
    EXPECT_FLOAT_EQ(static_cast<float>(FromBits(0x81)), 4.0f);

    // 0x82 = 2^3 = 8.0
    EXPECT_FLOAT_EQ(static_cast<float>(FromBits(0x82)), 8.0f);

    // 0x7D = 2^(-2) = 0.25
    EXPECT_FLOAT_EQ(static_cast<float>(FromBits(0x7D)), 0.25f);

    // 0x7C = 2^(-3) = 0.125
    EXPECT_FLOAT_EQ(static_cast<float>(FromBits(0x7C)), 0.125f);
}

} // namespace benchmark
} // namespace op
