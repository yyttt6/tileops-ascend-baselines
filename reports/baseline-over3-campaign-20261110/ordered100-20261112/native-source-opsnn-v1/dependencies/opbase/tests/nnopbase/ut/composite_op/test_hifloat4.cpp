/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <limits>
#include <sstream>

#include "opdev/hifloat4.h"
#include "opdev/bfloat16.h"
#include "gtest/gtest.h"

// ============================================================================
// HiFloat4 Tests
// ============================================================================

class TestHiFloat4 : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TestHiFloat4, DefaultConstructor)
{
    op::HiFloat4 val;
    EXPECT_EQ(val.value, 0);
    EXPECT_TRUE(val.IsZero());
    EXPECT_FLOAT_EQ(static_cast<float>(val), 0.0f);
}

TEST_F(TestHiFloat4, FromBits)
{
    // 0x0 = +0
    op::HiFloat4 zero(0x0, op::HiFloat4::FromBits());
    EXPECT_TRUE(zero.IsZero());

    // 0x8 = -0
    op::HiFloat4 neg_zero(0x8, op::HiFloat4::FromBits());
    EXPECT_TRUE(neg_zero.IsZero());

    // 0xF = min value (1 1 11) = -1.75
    op::HiFloat4 min_val(0xF, op::HiFloat4::FromBits());
    EXPECT_FLOAT_EQ(static_cast<float>(min_val), -1.75f);
    EXPECT_FALSE(min_val.IsNaN());

    // 0x4 = 1.0 (0 1 00)
    op::HiFloat4 one(0x4, op::HiFloat4::FromBits());
    EXPECT_FLOAT_EQ(static_cast<float>(one), 1.0f);

    // 0x5 = 1.25 (0 1 01)
    op::HiFloat4 one_point_two_five(0x5, op::HiFloat4::FromBits());
    EXPECT_FLOAT_EQ(static_cast<float>(one_point_two_five), 1.25f);

    // 0x7 = max value (0 1 11) = 1.75
    op::HiFloat4 max_val(0x7, op::HiFloat4::FromBits());
    EXPECT_FLOAT_EQ(static_cast<float>(max_val), 1.75f);

    // 0x1 = min positive denorm (0 0 01) = 0.25
    op::HiFloat4 min_denorm(0x1, op::HiFloat4::FromBits());
    EXPECT_FLOAT_EQ(static_cast<float>(min_denorm), 0.25f);
}

TEST_F(TestHiFloat4, FloatConversion)
{
    // Test basic values
    op::HiFloat4 one(1.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(one), 1.0f);

    op::HiFloat4 one_point_five(1.5f);
    EXPECT_NEAR(static_cast<float>(one_point_five), 1.5f, 0.15f);

    // Test negative values
    op::HiFloat4 neg_one(-1.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(neg_one), -1.0f);

    // Test zero
    op::HiFloat4 zero(0.0f);
    EXPECT_TRUE(zero.IsZero());

    // Test NaN input - HiFloat4 has no NaN, input is clamped to max (1.75)
    op::HiFloat4 nan_input(std::nanf(""));
    EXPECT_FALSE(nan_input.IsNaN());
    EXPECT_FLOAT_EQ(static_cast<float>(nan_input), 1.75f);
}

TEST_F(TestHiFloat4, OverflowClamp)
{
    // HiFloat4 has no infinity, overflow should clamp to max value (1.75)
    op::HiFloat4 large(100.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(large), 1.75f);

    op::HiFloat4 inf(std::numeric_limits<float>::infinity());
    EXPECT_FLOAT_EQ(static_cast<float>(inf), 1.75f);

    // Negative overflow also clamps to min value (-1.75)
    op::HiFloat4 neg_inf(-std::numeric_limits<float>::infinity());
    EXPECT_FLOAT_EQ(static_cast<float>(neg_inf), -1.75f);
    EXPECT_FALSE(neg_inf.IsNaN());
}

TEST_F(TestHiFloat4, ArithmeticOperations)
{
    op::HiFloat4 a(1.0f);
    op::HiFloat4 b(0.5f);

    // Addition (may overflow)
    op::HiFloat4 sum(static_cast<float>(a) + static_cast<float>(b));
    EXPECT_NEAR(static_cast<float>(sum), 1.5f, 0.3f);

    // Subtraction
    op::HiFloat4 diff(static_cast<float>(a) - static_cast<float>(b));
    EXPECT_NEAR(static_cast<float>(diff), 0.5f, 0.2f);

    // Multiplication (may underflow to 0.5)
    op::HiFloat4 prod(static_cast<float>(a) * static_cast<float>(b));
    EXPECT_NEAR(static_cast<float>(prod), 0.5f, 0.2f);

    // Unary negation
    op::HiFloat4 neg_a(-static_cast<float>(a));
    EXPECT_FLOAT_EQ(static_cast<float>(neg_a), -1.0f);
}

TEST_F(TestHiFloat4, ComparisonOperations)
{
    op::HiFloat4 a(1.0f);
    op::HiFloat4 b(1.5f);
    op::HiFloat4 c(1.0f);

    EXPECT_TRUE(static_cast<float>(a) < static_cast<float>(b));
    EXPECT_TRUE(static_cast<float>(a) <= static_cast<float>(b));
    EXPECT_TRUE(static_cast<float>(a) <= static_cast<float>(c));
    EXPECT_TRUE(static_cast<float>(a) == static_cast<float>(c));
    EXPECT_TRUE(static_cast<float>(a) != static_cast<float>(b));
    EXPECT_TRUE(static_cast<float>(b) > static_cast<float>(a));
    EXPECT_TRUE(static_cast<float>(b) >= static_cast<float>(a));
}

TEST_F(TestHiFloat4, TypeConversion)
{
    op::HiFloat4 val(1.0f);

    // Test implicit conversion to double
    double d = val;
    EXPECT_DOUBLE_EQ(d, 1.0);

    // Test implicit conversion to float
    float f = val;
    EXPECT_FLOAT_EQ(f, 1.0f);

    // Test non-zero value
    EXPECT_TRUE(static_cast<float>(val) != 0.0f);

    op::HiFloat4 zero(0.0f);
    EXPECT_TRUE(static_cast<float>(zero) == 0.0f);
}

TEST_F(TestHiFloat4, DoubleConversion)
{
    // Test double constructor
    op::HiFloat4 val_from_double(1.25);
    EXPECT_DOUBLE_EQ(static_cast<double>(val_from_double), 1.25);

    // Test double assignment operator
    op::HiFloat4 val_assign;
    val_assign = 1.5;
    EXPECT_DOUBLE_EQ(static_cast<double>(val_assign), 1.5);

    // Test implicit conversion to double
    op::HiFloat4 val_todouble(1.75);
    EXPECT_DOUBLE_EQ(static_cast<double>(val_todouble), 1.75);

    double d = val_todouble;
    EXPECT_DOUBLE_EQ(d, 1.75);

    // Test negative double
    op::HiFloat4 neg_val(-1.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(neg_val), -1.0);

    // Test overflow with double (HiFloat4 max is 1.75)
    op::HiFloat4 large(100.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(large), 1.75); // clamped to max
}

TEST_F(TestHiFloat4, StdFunctions)
{
    op::HiFloat4 val(1.0f);
    op::HiFloat4 neg_val(-1.0f);
    // HiFloat4 has no NaN, nan input is clamped to max (1.75)
    op::HiFloat4 nan_input(std::nanf(""));

    EXPECT_FALSE(std::isinf(val));
    EXPECT_FALSE(std::isnan(val));
    EXPECT_TRUE(std::isfinite(val));
    EXPECT_FLOAT_EQ(std::abs(static_cast<float>(neg_val)), 1.0f);
    // nan input is clamped to max value (1.75), not NaN
    EXPECT_FALSE(std::isnan(nan_input));
    EXPECT_FLOAT_EQ(static_cast<float>(nan_input), 1.75f);
}

TEST_F(TestHiFloat4, NumericLimits)
{
    EXPECT_FLOAT_EQ(static_cast<float>(std::numeric_limits<op::HiFloat4>::max()), 1.75f);
    EXPECT_FLOAT_EQ(static_cast<float>(std::numeric_limits<op::HiFloat4>::min()), 1.0f);
    // HiFloat4 has no NaN, quiet_NaN returns zero as fallback
    EXPECT_FALSE(std::isnan(std::numeric_limits<op::HiFloat4>::quiet_NaN()));
}

TEST_F(TestHiFloat4, OutputStream)
{
    op::HiFloat4 val(1.25f);
    std::ostringstream oss;
    oss << val;
    EXPECT_EQ(oss.str(), "1.25");
}

TEST_F(TestHiFloat4, AllValues)
{
    // Test all 16 possible values for HiFloat4
    // Format: S E MM (1 sign bit, 1 exponent bit, 2 mantissa bits)
    // HiFloat4: bias=1, denorm = 2^(1-bias) * (man/4) = man/4
    // No NaN encoding - all bit patterns are valid numbers
    struct TestCase {
        uint8_t bits;
        float expected;
    };

    TestCase cases[] = {
        {0x0, 0.0f},   // 0 0 00 = +0
        {0x1, 0.25f},  // 0 0 01 = denorm: 1/4 = 0.25
        {0x2, 0.5f},   // 0 0 10 = denorm: 2/4 = 0.5
        {0x3, 0.75f},  // 0 0 11 = denorm: 3/4 = 0.75
        {0x4, 1.0f},   // 0 1 00 = 2^0 * 1.0 = 1.0
        {0x5, 1.25f},  // 0 1 01 = 2^0 * 1.25 = 1.25
        {0x6, 1.5f},   // 0 1 10 = 2^0 * 1.5 = 1.5
        {0x7, 1.75f},  // 0 1 11 = 2^0 * 1.75 = 1.75 (max)
        {0x8, -0.0f},  // 1 0 00 = -0
        {0x9, -0.25f}, // 1 0 01 = -denorm: -0.25
        {0xA, -0.5f},  // 1 0 10 = -denorm: -0.5
        {0xB, -0.75f}, // 1 0 11 = -denorm: -0.75
        {0xC, -1.0f},  // 1 1 00 = -1.0
        {0xD, -1.25f}, // 1 1 01 = -1.25
        {0xE, -1.5f},  // 1 1 10 = -1.5
        {0xF, -1.75f}, // 1 1 11 = -1.75 (min)
    };

    for (const auto& tc : cases) {
        op::HiFloat4 val(tc.bits, op::HiFloat4::FromBits());
        EXPECT_FLOAT_EQ(static_cast<float>(val), tc.expected) << "bits: 0x" << std::hex << (int)tc.bits;
        // HiFloat4 has no NaN - all bit patterns are valid numbers
        EXPECT_FALSE(val.IsNaN()) << "bits: 0x" << std::hex << (int)tc.bits;
    }
}

// ============================================================================
// FP16 and BFloat16 Conversion Tests
// ============================================================================

TEST(HiFloat4Fp16Conversion, FromFp16)
{
    op::fp16_t fp16_val(1.5f);
    op::HiFloat4 hifp4_from_fp16(fp16_val);
    EXPECT_NEAR(static_cast<float>(hifp4_from_fp16), 1.5f, 0.1f);
}

TEST(HiFloat4Fp16Conversion, ToFp16)
{
    op::HiFloat4 hifp4(1.25f);
    op::fp16_t fp16_result(static_cast<float>(hifp4));
    EXPECT_NEAR(static_cast<float>(fp16_result), 1.25f, 0.1f);
}

TEST(HiFloat4BFloat16Conversion, FromBFloat16)
{
    op::bfloat16 bf16_val(1.0f);
    op::HiFloat4 hifp4_from_bf16(bf16_val);
    EXPECT_NEAR(static_cast<float>(hifp4_from_bf16), 1.0f, 0.1f);
}

TEST(HiFloat4BFloat16Conversion, ToBFloat16)
{
    op::HiFloat4 hifp4(1.5f);
    op::bfloat16 bf16_result(static_cast<float>(hifp4));
    EXPECT_NEAR(static_cast<float>(bf16_result), 1.5f, 0.1f);
}

// ============================================================================
// Division by Zero Tests (HiFloat4 has no NaN or Infinity)
// ============================================================================

TEST(HiFloat4DivisionByZero, PositiveDivZero)
{
    // Per HiFloat4 specification (same as E1M2 extension format):
    // NO NaN and NO Infinity encodings - all bit patterns are valid numbers
    // Division by zero (x/0 where x != 0) results in infinity in float,
    // which is clamped to max value (1.75) preserving sign

    op::HiFloat4 positive(1.0f);
    op::HiFloat4 zero(0.0f);

    // Positive / zero = +max (1.75) - HiFloat4 has no infinity
    op::HiFloat4 result_pos(static_cast<float>(positive) / static_cast<float>(zero));
    EXPECT_FLOAT_EQ(static_cast<float>(result_pos), 1.75f) << "Positive/zero should clamp to max (1.75)";
    EXPECT_FALSE(result_pos.IsNaN());
}

TEST(HiFloat4DivisionByZero, NegativeDivZero)
{
    op::HiFloat4 negative(-1.0f);
    op::HiFloat4 zero(0.0f);

    // Negative / zero = -max (-1.75) - sign preserved
    op::HiFloat4 result_neg(static_cast<float>(negative) / static_cast<float>(zero));
    EXPECT_FLOAT_EQ(static_cast<float>(result_neg), -1.75f) << "Negative/zero should clamp to min (-1.75)";
    EXPECT_FALSE(result_neg.IsNaN());
}

TEST(HiFloat4DivisionByZero, ZeroDivZero)
{
    op::HiFloat4 zero(0.0f);

    // Zero / zero = NaN in float, clamped to max (1.75) in HiFloat4
    op::HiFloat4 zero_div_zero(static_cast<float>(zero) / static_cast<float>(zero));
    EXPECT_FLOAT_EQ(static_cast<float>(zero_div_zero), 1.75f) << "Zero/zero (NaN) should clamp to max";
    EXPECT_FALSE(zero_div_zero.IsNaN());
}

TEST(HiFloat4DivisionByZero, CompoundAssignment)
{
    op::HiFloat4 a(1.0f);
    op::HiFloat4 zero(0.0f);

    // Compound assignment division by zero
    a = op::HiFloat4(static_cast<float>(a) / static_cast<float>(zero));
    EXPECT_FLOAT_EQ(static_cast<float>(a), 1.75f) << "Compound division by zero should clamp to max";
}
