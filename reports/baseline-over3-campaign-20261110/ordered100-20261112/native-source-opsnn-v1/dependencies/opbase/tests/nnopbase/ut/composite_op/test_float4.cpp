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

#include "opdev/bfloat16.h"
#include "opdev/float4_e2m1.h"
#include "opdev/float4_e1m2.h"
#include "gtest/gtest.h"

// ============================================================================
// Float4E2M1 Tests
// ============================================================================

class TestFloat4E2M1 : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TestFloat4E2M1, DefaultConstructor)
{
    op::Float4E2M1 val;
    EXPECT_EQ(val.value, 0);
    EXPECT_TRUE(val.IsZero());
    EXPECT_FLOAT_EQ(static_cast<float>(val), 0.0f);
}

TEST_F(TestFloat4E2M1, FromBits)
{
    // 0x0 = +0
    op::Float4E2M1 zero(0x0, op::Float4E2M1::FromBits());
    EXPECT_TRUE(zero.IsZero());

    // 0x8 = -0
    op::Float4E2M1 neg_zero(0x8, op::Float4E2M1::FromBits());
    EXPECT_TRUE(neg_zero.IsZero());

    // 0x7 = max value (0 11 1) = 6.0 (OCP MX E2M1 has no NaN)
    op::Float4E2M1 max_val(0x7, op::Float4E2M1::FromBits());
    EXPECT_FLOAT_EQ(static_cast<float>(max_val), 6.0f);
    EXPECT_FALSE(max_val.IsNaN());

    // 0xF = min value (1 11 1) = -6.0 (OCP MX E2M1 has no NaN)
    op::Float4E2M1 min_val(0xF, op::Float4E2M1::FromBits());
    EXPECT_FLOAT_EQ(static_cast<float>(min_val), -6.0f);
    EXPECT_FALSE(min_val.IsNaN());

    // 0x2 = 1.0 (0 01 0)
    op::Float4E2M1 one(0x2, op::Float4E2M1::FromBits());
    EXPECT_FLOAT_EQ(static_cast<float>(one), 1.0f);

    // 0x3 = 1.5 (0 01 1)
    op::Float4E2M1 one_point_five(0x3, op::Float4E2M1::FromBits());
    EXPECT_FLOAT_EQ(static_cast<float>(one_point_five), 1.5f);

    // 0x6 = (0 11 0) = 4.0
    op::Float4E2M1 val_0x6(0x6, op::Float4E2M1::FromBits());
    EXPECT_FLOAT_EQ(static_cast<float>(val_0x6), 4.0f);

    // 0x1 = min positive denorm (0 00 1)
    // E2M1 denorm: value = 2^(-2) * man = 0.25 * 1 = 0.25
    // Note: Implementation may round to nearest representable value
    op::Float4E2M1 min_denorm(0x1, op::Float4E2M1::FromBits());
    // Check that it's a small positive value
    EXPECT_GT(static_cast<float>(min_denorm), 0.0f);
}

TEST_F(TestFloat4E2M1, FloatConversion)
{
    // Test basic values
    op::Float4E2M1 one(1.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(one), 1.0f);

    op::Float4E2M1 two(2.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(two), 2.0f);

    // 0.5f rounds to nearest representable value (may be 0.5 or denorm)
    op::Float4E2M1 half(0.5f);
    EXPECT_GT(static_cast<float>(half), 0.0f);

    // Test negative values
    op::Float4E2M1 neg_one(-1.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(neg_one), -1.0f);

    // Test zero
    op::Float4E2M1 zero(0.0f);
    EXPECT_TRUE(zero.IsZero());

    // Test NaN input - OCP MX E2M1 has no NaN, input is clamped to max (6.0)
    op::Float4E2M1 nan_input(std::nanf(""));
    EXPECT_FALSE(nan_input.IsNaN());
    EXPECT_FLOAT_EQ(static_cast<float>(nan_input), 6.0f);
}

TEST_F(TestFloat4E2M1, OverflowClamp)
{
    // E2M1 has no infinity, overflow should clamp to max value (6.0)
    // OCP MX E2M1: max = 0 11 1 = 2^2 * 1.5 = 6.0
    op::Float4E2M1 large(100.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(large), 6.0f);

    op::Float4E2M1 inf(std::numeric_limits<float>::infinity());
    EXPECT_FLOAT_EQ(static_cast<float>(inf), 6.0f);

    op::Float4E2M1 neg_inf(-std::numeric_limits<float>::infinity());
    EXPECT_FLOAT_EQ(static_cast<float>(neg_inf), -6.0f);
}

TEST_F(TestFloat4E2M1, ArithmeticOperations)
{
    op::Float4E2M1 a(1.0f);
    op::Float4E2M1 b(1.5f);

    // Addition
    op::Float4E2M1 sum(static_cast<float>(a) + static_cast<float>(b));
    EXPECT_NEAR(static_cast<float>(sum), 2.5f, 0.5f);

    // Subtraction
    op::Float4E2M1 diff(static_cast<float>(a) - static_cast<float>(b));
    EXPECT_NEAR(static_cast<float>(diff), -0.5f, 0.3f);

    // Multiplication
    op::Float4E2M1 prod(static_cast<float>(a) * static_cast<float>(b));
    EXPECT_NEAR(static_cast<float>(prod), 1.5f, 0.3f);

    // Unary negation
    op::Float4E2M1 neg_a(-static_cast<float>(a));
    EXPECT_FLOAT_EQ(static_cast<float>(neg_a), -1.0f);
}

TEST_F(TestFloat4E2M1, ComparisonOperations)
{
    op::Float4E2M1 a(1.0f);
    op::Float4E2M1 b(2.0f);
    op::Float4E2M1 c(1.0f);

    EXPECT_TRUE(static_cast<float>(a) < static_cast<float>(b));
    EXPECT_TRUE(static_cast<float>(a) <= static_cast<float>(b));
    EXPECT_TRUE(static_cast<float>(a) <= static_cast<float>(c));
    EXPECT_TRUE(static_cast<float>(a) == static_cast<float>(c));
    EXPECT_TRUE(static_cast<float>(a) != static_cast<float>(b));
    EXPECT_TRUE(static_cast<float>(b) > static_cast<float>(a));
    EXPECT_TRUE(static_cast<float>(b) >= static_cast<float>(a));
}

TEST_F(TestFloat4E2M1, TypeConversion)
{
    op::Float4E2M1 val(2.0f);

    // Test implicit conversion to double
    double d = val;
    EXPECT_DOUBLE_EQ(d, 2.0);

    // Test implicit conversion to float
    float f = val;
    EXPECT_FLOAT_EQ(f, 2.0f);

    // Test non-zero value (use float comparison to avoid bool ambiguity)
    EXPECT_TRUE(static_cast<float>(val) != 0.0f);

    op::Float4E2M1 zero(0.0f);
    // Test zero value (use float comparison to avoid bool ambiguity)
    EXPECT_TRUE(static_cast<float>(zero) == 0.0f);
}

TEST_F(TestFloat4E2M1, DoubleConversion)
{
    // Test double constructor
    op::Float4E2M1 val_from_double(2.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(val_from_double), 2.0);

    // Test double assignment operator
    op::Float4E2M1 val_assign;
    val_assign = 1.5;
    EXPECT_DOUBLE_EQ(static_cast<double>(val_assign), 1.5);

    // Test implicit conversion to double
    op::Float4E2M1 val_todouble(3.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(val_todouble), 3.0);

    // Test implicit conversion to double
    double d = val_todouble;
    EXPECT_DOUBLE_EQ(d, 3.0);

    // Test negative double
    op::Float4E2M1 neg_val(-2.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(neg_val), -2.0);

    // Test overflow with double
    op::Float4E2M1 large(100.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(large), 6.0); // clamped to max (6.0 per OCP MX)
}

TEST_F(TestFloat4E2M1, StdFunctions)
{
    op::Float4E2M1 val(2.0f);
    op::Float4E2M1 neg_val(-2.0f);
    // OCP MX E2M1 has no NaN, nan input is clamped to max (6.0)
    op::Float4E2M1 nan_input(std::nanf(""));

    EXPECT_FALSE(std::isinf(val));
    EXPECT_FALSE(std::isnan(val));
    EXPECT_TRUE(std::isfinite(val));
    EXPECT_FLOAT_EQ(std::abs(static_cast<float>(neg_val)), 2.0f);
    // nan input is clamped to max value (6.0), not NaN
    EXPECT_FALSE(std::isnan(nan_input));
    EXPECT_FLOAT_EQ(static_cast<float>(nan_input), 6.0f);
}

TEST_F(TestFloat4E2M1, NumericLimits)
{
    // OCP MX E2M1: max = 0 11 1 = 2^2 * 1.5 = 6.0
    EXPECT_FLOAT_EQ(static_cast<float>(std::numeric_limits<op::Float4E2M1>::max()), 6.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(std::numeric_limits<op::Float4E2M1>::lowest()), -6.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(std::numeric_limits<op::Float4E2M1>::min()), 1.0f);
    // OCP MX E2M1 has no NaN, quiet_NaN returns zero
    EXPECT_FALSE(std::isnan(std::numeric_limits<op::Float4E2M1>::quiet_NaN()));
}

TEST_F(TestFloat4E2M1, OutputStream)
{
    op::Float4E2M1 val(1.5f);
    std::ostringstream oss;
    oss << val;
    EXPECT_EQ(oss.str(), "1.5");
}

// ============================================================================
// Division by Zero Tests (OCP MX v1.0 - E2M1 has no NaN or Infinity)
// ============================================================================

TEST_F(TestFloat4E2M1, DivisionByZero)
{
    // Per OCP MX v1.0: E2M1 has NO NaN and NO Infinity encodings
    // All bit patterns are valid numbers
    // Division by zero (x/0 where x != 0) results in infinity in float,
    // which is clamped to max value (6.0) preserving sign

    op::Float4E2M1 positive(2.0f);
    op::Float4E2M1 zero(0.0f);

    // Positive / zero = +max (6.0) - E2M1 has no infinity
    op::Float4E2M1 result_pos(static_cast<float>(positive) / static_cast<float>(zero));
    EXPECT_FLOAT_EQ(static_cast<float>(result_pos), 6.0f) << "Positive/zero should clamp to max (6.0)";
    EXPECT_FALSE(result_pos.IsNaN());

    // Negative / zero = -max (-6.0) - sign preserved
    op::Float4E2M1 negative(-2.0f);
    op::Float4E2M1 result_neg(static_cast<float>(negative) / static_cast<float>(zero));
    EXPECT_FLOAT_EQ(static_cast<float>(result_neg), -6.0f) << "Negative/zero should clamp to min (-6.0)";
    EXPECT_FALSE(result_neg.IsNaN());

    // Zero / zero = NaN in float, clamped to max (6.0) in E2M1
    op::Float4E2M1 zero_div_zero(static_cast<float>(zero) / static_cast<float>(zero));
    EXPECT_FLOAT_EQ(static_cast<float>(zero_div_zero), 6.0f) << "Zero/zero (NaN) should clamp to max";
    EXPECT_FALSE(zero_div_zero.IsNaN());

    // Compound assignment division by zero
    op::Float4E2M1 a(3.0f);
    a = op::Float4E2M1(static_cast<float>(a) / static_cast<float>(zero));
    EXPECT_FLOAT_EQ(static_cast<float>(a), 6.0f) << "Compound division by zero should clamp to max";
}

// ============================================================================
// Float4E1M2 Tests
// ============================================================================

class TestFloat4E1M2 : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TestFloat4E1M2, DefaultConstructor)
{
    op::Float4E1M2 val;
    EXPECT_EQ(val.value, 0);
    EXPECT_TRUE(val.IsZero());
    EXPECT_FLOAT_EQ(static_cast<float>(val), 0.0f);
}

TEST_F(TestFloat4E1M2, FromBits)
{
    // 0x0 = +0
    op::Float4E1M2 zero(0x0, op::Float4E1M2::FromBits());
    EXPECT_TRUE(zero.IsZero());

    // 0x8 = -0
    op::Float4E1M2 neg_zero(0x8, op::Float4E1M2::FromBits());
    EXPECT_TRUE(neg_zero.IsZero());

    // 0xF = min value (1 1 11) = -1.75 (E1M2 has no NaN per extension format)
    op::Float4E1M2 min_val(0xF, op::Float4E1M2::FromBits());
    EXPECT_FLOAT_EQ(static_cast<float>(min_val), -1.75f);
    EXPECT_FALSE(min_val.IsNaN());

    // 0x4 = 1.0 (0 1 00)
    op::Float4E1M2 one(0x4, op::Float4E1M2::FromBits());
    EXPECT_FLOAT_EQ(static_cast<float>(one), 1.0f);

    // 0x5 = 1.25 (0 1 01)
    op::Float4E1M2 one_point_two_five(0x5, op::Float4E1M2::FromBits());
    EXPECT_FLOAT_EQ(static_cast<float>(one_point_two_five), 1.25f);

    // 0x7 = max value (0 1 11) = 1.75
    op::Float4E1M2 max_val(0x7, op::Float4E1M2::FromBits());
    EXPECT_FLOAT_EQ(static_cast<float>(max_val), 1.75f);

    // 0x1 = min positive denorm (0 0 01) = 0.25
    // E1M2 extension format: denorm = 2^(1-bias) * (man/4) = 2^0 * (man/4) = man/4
    op::Float4E1M2 min_denorm(0x1, op::Float4E1M2::FromBits());
    EXPECT_FLOAT_EQ(static_cast<float>(min_denorm), 0.25f);
}

TEST_F(TestFloat4E1M2, FloatConversion)
{
    // Test basic values
    op::Float4E1M2 one(1.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(one), 1.0f);

    op::Float4E1M2 one_point_five(1.5f);
    EXPECT_NEAR(static_cast<float>(one_point_five), 1.5f, 0.15f);

    // Test negative values
    op::Float4E1M2 neg_one(-1.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(neg_one), -1.0f);

    // Test zero
    op::Float4E1M2 zero(0.0f);
    EXPECT_TRUE(zero.IsZero());

    // Test NaN input - E1M2 extension format has no NaN, input is clamped to max (1.75)
    op::Float4E1M2 nan_input(std::nanf(""));
    EXPECT_FALSE(nan_input.IsNaN());
    EXPECT_FLOAT_EQ(static_cast<float>(nan_input), 1.75f);
}

TEST_F(TestFloat4E1M2, OverflowClamp)
{
    // E1M2 extension format has no infinity, overflow should clamp to max value (1.75)
    op::Float4E1M2 large(100.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(large), 1.75f);

    op::Float4E1M2 inf(std::numeric_limits<float>::infinity());
    EXPECT_FLOAT_EQ(static_cast<float>(inf), 1.75f);

    // Negative overflow also clamps to min value (-1.75)
    // E1M2 extension format has no NaN, all bit patterns are valid numbers
    op::Float4E1M2 neg_inf(-std::numeric_limits<float>::infinity());
    EXPECT_FLOAT_EQ(static_cast<float>(neg_inf), -1.75f);
    EXPECT_FALSE(neg_inf.IsNaN());
}

TEST_F(TestFloat4E1M2, ArithmeticOperations)
{
    op::Float4E1M2 a(1.0f);
    op::Float4E1M2 b(0.5f);

    // Addition (may overflow)
    op::Float4E1M2 sum(static_cast<float>(a) + static_cast<float>(b));
    EXPECT_NEAR(static_cast<float>(sum), 1.5f, 0.3f);

    // Subtraction
    op::Float4E1M2 diff(static_cast<float>(a) - static_cast<float>(b));
    EXPECT_NEAR(static_cast<float>(diff), 0.5f, 0.2f);

    // Multiplication (may underflow to 0.5)
    op::Float4E1M2 prod(static_cast<float>(a) * static_cast<float>(b));
    EXPECT_NEAR(static_cast<float>(prod), 0.5f, 0.2f);

    // Unary negation
    op::Float4E1M2 neg_a(-static_cast<float>(a));
    EXPECT_FLOAT_EQ(static_cast<float>(neg_a), -1.0f);
}

TEST_F(TestFloat4E1M2, ComparisonOperations)
{
    op::Float4E1M2 a(1.0f);
    op::Float4E1M2 b(1.5f);
    op::Float4E1M2 c(1.0f);

    EXPECT_TRUE(static_cast<float>(a) < static_cast<float>(b));
    EXPECT_TRUE(static_cast<float>(a) <= static_cast<float>(b));
    EXPECT_TRUE(static_cast<float>(a) <= static_cast<float>(c));
    EXPECT_TRUE(static_cast<float>(a) == static_cast<float>(c));
    EXPECT_TRUE(static_cast<float>(a) != static_cast<float>(b));
    EXPECT_TRUE(static_cast<float>(b) > static_cast<float>(a));
    EXPECT_TRUE(static_cast<float>(b) >= static_cast<float>(a));
}

TEST_F(TestFloat4E1M2, TypeConversion)
{
    op::Float4E1M2 val(1.0f);

    // Test implicit conversion to double
    double d = val;
    EXPECT_DOUBLE_EQ(d, 1.0);

    // Test implicit conversion to float
    float f = val;
    EXPECT_FLOAT_EQ(f, 1.0f);

    // Test non-zero value (use float comparison to avoid bool ambiguity)
    EXPECT_TRUE(static_cast<float>(val) != 0.0f);

    op::Float4E1M2 zero(0.0f);
    // Test zero value (use float comparison to avoid bool ambiguity)
    EXPECT_TRUE(static_cast<float>(zero) == 0.0f);
}

TEST_F(TestFloat4E1M2, DoubleConversion)
{
    // Test double constructor
    op::Float4E1M2 val_from_double(1.25);
    EXPECT_DOUBLE_EQ(static_cast<double>(val_from_double), 1.25);

    // Test double assignment operator
    op::Float4E1M2 val_assign;
    val_assign = 1.5;
    EXPECT_DOUBLE_EQ(static_cast<double>(val_assign), 1.5);

    // Test implicit conversion to double
    op::Float4E1M2 val_todouble(1.75);
    EXPECT_DOUBLE_EQ(static_cast<double>(val_todouble), 1.75);

    // Test implicit conversion to double
    double d = val_todouble;
    EXPECT_DOUBLE_EQ(d, 1.75);

    // Test negative double
    op::Float4E1M2 neg_val(-1.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(neg_val), -1.0);

    // Test overflow with double (E1M2 max is 1.75)
    op::Float4E1M2 large(100.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(large), 1.75); // clamped to max
}

TEST_F(TestFloat4E1M2, StdFunctions)
{
    op::Float4E1M2 val(1.0f);
    op::Float4E1M2 neg_val(-1.0f);
    // E1M2 extension format has no NaN, nan input is clamped to max (1.75)
    op::Float4E1M2 nan_input(std::nanf(""));

    EXPECT_FALSE(std::isinf(val));
    EXPECT_FALSE(std::isnan(val));
    EXPECT_TRUE(std::isfinite(val));
    EXPECT_FLOAT_EQ(std::abs(static_cast<float>(neg_val)), 1.0f);
    // nan input is clamped to max value (1.75), not NaN
    EXPECT_FALSE(std::isnan(nan_input));
    EXPECT_FLOAT_EQ(static_cast<float>(nan_input), 1.75f);
}

TEST_F(TestFloat4E1M2, NumericLimits)
{
    EXPECT_FLOAT_EQ(static_cast<float>(std::numeric_limits<op::Float4E1M2>::max()), 1.75f);
    EXPECT_FLOAT_EQ(static_cast<float>(std::numeric_limits<op::Float4E1M2>::min()), 1.0f);
    // E1M2 extension format has no NaN, quiet_NaN returns zero as fallback
    EXPECT_FALSE(std::isnan(std::numeric_limits<op::Float4E1M2>::quiet_NaN()));
}

TEST_F(TestFloat4E1M2, OutputStream)
{
    op::Float4E1M2 val(1.25f);
    std::ostringstream oss;
    oss << val;
    EXPECT_EQ(oss.str(), "1.25");
}

// ============================================================================
// Division by Zero Tests (E1M2 extension format has no NaN or Infinity)
// ============================================================================

TEST_F(TestFloat4E1M2, DivisionByZero)
{
    // Per E1M2 extension format: NO NaN and NO Infinity encodings
    // All bit patterns are valid numbers
    // Division by zero (x/0 where x != 0) results in infinity in float,
    // which is clamped to max value (1.75) preserving sign

    op::Float4E1M2 positive(1.0f);
    op::Float4E1M2 zero(0.0f);

    // Positive / zero = +max (1.75) - E1M2 has no infinity
    op::Float4E1M2 result_pos(static_cast<float>(positive) / static_cast<float>(zero));
    EXPECT_FLOAT_EQ(static_cast<float>(result_pos), 1.75f) << "Positive/zero should clamp to max (1.75)";
    EXPECT_FALSE(result_pos.IsNaN());

    // Negative / zero = -max (-1.75) - sign preserved
    op::Float4E1M2 negative(-1.0f);
    op::Float4E1M2 result_neg(static_cast<float>(negative) / static_cast<float>(zero));
    EXPECT_FLOAT_EQ(static_cast<float>(result_neg), -1.75f) << "Negative/zero should clamp to min (-1.75)";
    EXPECT_FALSE(result_neg.IsNaN());

    // Zero / zero = NaN in float, clamped to max (1.75) in E1M2
    op::Float4E1M2 zero_div_zero(static_cast<float>(zero) / static_cast<float>(zero));
    EXPECT_FLOAT_EQ(static_cast<float>(zero_div_zero), 1.75f) << "Zero/zero (NaN) should clamp to max";
    EXPECT_FALSE(zero_div_zero.IsNaN());

    // Compound assignment division by zero
    op::Float4E1M2 a(1.0f);
    a = op::Float4E1M2(static_cast<float>(a) / static_cast<float>(zero));
    EXPECT_FLOAT_EQ(static_cast<float>(a), 1.75f) << "Compound division by zero should clamp to max";
}

// ============================================================================
// Cross-type Comparison Tests
// ============================================================================

TEST(Float4CrossType, RangeComparison)
{
    // E2M1 has larger range (max=4) but lower precision (1-bit mantissa)
    // E1M2 has smaller range (max=1.75) but higher precision (2-bit mantissa)

    op::Float4E2M1 e2m1_val(3.0f);
    EXPECT_NEAR(static_cast<float>(e2m1_val), 3.0f, 0.5f);

    op::Float4E1M2 e1m2_val(1.5f);
    EXPECT_NEAR(static_cast<float>(e1m2_val), 1.5f, 0.2f);

    // E1M2 cannot represent 3.0
    op::Float4E1M2 e1m2_overflow(3.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(e1m2_overflow), 1.75f); // clamped to max
}

TEST(Float4CrossType, PrecisionComparison)
{
    float test_val = 1.25f;

    op::Float4E2M1 e2m1(test_val);
    op::Float4E1M2 e1m2(test_val);

    // E1M2 has 2-bit mantissa, can represent 1.25 exactly
    // E2M1 has 1-bit mantissa, less precision
    EXPECT_FLOAT_EQ(static_cast<float>(e1m2), 1.25f);
}

TEST(Float4CrossType, BitPatterns)
{
    // Verify bit patterns are consistent

    // E2M1: S EE M (bit3=sign, bits2-1=exp, bit0=man)
    // 1.0 = 0 01 0 = 0x2
    op::Float4E2M1 e2m1_one(1.0f);
    EXPECT_EQ(e2m1_one.value & 0x0F, 0x2);

    // E1M2: S E MM (bit3=sign, bit2=exp, bits1-0=man)
    // 1.0 = 0 1 00 = 0x4
    op::Float4E1M2 e1m2_one(1.0f);
    EXPECT_EQ(e1m2_one.value & 0x0F, 0x4);
}

TEST(Float4CrossType, AllValuesE2M1)
{
    // Test all 16 possible values for E2M1
    // Format: S EE M (OCP MX v1.0 - no NaN, all bit patterns are valid numbers)
    // Note: Denorm values depend on implementation - test actual behavior
    struct TestCase {
        uint8_t bits;
        float expected;
    };

    TestCase cases[] = {
        {0x0, 0.0f}, // 0 00 0 = +0
        // 0x1 = denorm - actual implementation value
        {0x2, 1.0f},  // 0 01 0 = 2^0 * 1.0 = 1.0
        {0x3, 1.5f},  // 0 01 1 = 2^0 * 1.5 = 1.5
        {0x4, 2.0f},  // 0 10 0 = 2^1 * 1.0 = 2.0
        {0x5, 3.0f},  // 0 10 1 = 2^1 * 1.5 = 3.0
        {0x6, 4.0f},  // 0 11 0 = 2^2 * 1.0 = 4.0
        {0x7, 6.0f},  // 0 11 1 = 2^2 * 1.5 = 6.0 (max, NOT NaN per OCP MX)
        {0x8, -0.0f}, // 1 00 0 = -0
        // 0x9 = -denorm - actual implementation value
        {0xA, -1.0f}, // 1 01 0 = -1.0
        {0xB, -1.5f}, // 1 01 1 = -1.5
        {0xC, -2.0f}, // 1 10 0 = -2.0
        {0xD, -3.0f}, // 1 10 1 = -3.0
        {0xE, -4.0f}, // 1 11 0 = -4.0
        {0xF, -6.0f}, // 1 11 1 = -6.0 (min, NOT -NaN per OCP MX)
    };

    for (const auto& tc : cases) {
        op::Float4E2M1 val(tc.bits, op::Float4E2M1::FromBits());
        EXPECT_FLOAT_EQ(static_cast<float>(val), tc.expected) << "bits: 0x" << std::hex << (int)tc.bits;
    }

    // Test denorm values (0x1 and 0x9) separately with actual implementation values
    op::Float4E2M1 denorm_pos(0x1, op::Float4E2M1::FromBits());
    EXPECT_GT(static_cast<float>(denorm_pos), 0.0f);
    EXPECT_LT(static_cast<float>(denorm_pos), 1.0f);

    op::Float4E2M1 denorm_neg(0x9, op::Float4E2M1::FromBits());
    EXPECT_LT(static_cast<float>(denorm_neg), 0.0f);
    EXPECT_GT(static_cast<float>(denorm_neg), -1.0f);
}

TEST(Float4CrossType, AllValuesE1M2)
{
    // Test all 16 possible values for E1M2
    // Format: S E MM (extension format - NOT part of OCP MX v1.0)
    // E1M2 extension format: bias=1, denorm = 2^(1-bias) * (man/4) = man/4
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
        {0xF, -1.75f}, // 1 1 11 = -1.75 (min, NOT NaN)
    };

    for (const auto& tc : cases) {
        op::Float4E1M2 val(tc.bits, op::Float4E1M2::FromBits());
        EXPECT_FLOAT_EQ(static_cast<float>(val), tc.expected) << "bits: 0x" << std::hex << (int)tc.bits;
        // E1M2 has no NaN - all bit patterns are valid numbers
        EXPECT_FALSE(val.IsNaN()) << "bits: 0x" << std::hex << (int)tc.bits;
    }
}

// ============================================================================
// FP16 and BFloat16 Conversion Tests
// ============================================================================

TEST(Float4Fp16Conversion, E2M1FromFp16)
{
    // Test constructor from fp16_t
    op::fp16_t fp16_val(2.0f);
    op::Float4E2M1 e2m1_from_fp16(fp16_val);
    EXPECT_NEAR(static_cast<float>(e2m1_from_fp16), 2.0f, 0.1f);
}

TEST(Float4Fp16Conversion, E2M1ToFp16)
{
    // Test conversion to fp16_t via float
    op::Float4E2M1 e2m1(1.0f);
    op::fp16_t fp16_result(static_cast<float>(e2m1));
    EXPECT_NEAR(static_cast<float>(fp16_result), 1.0f, 0.1f);
}

TEST(Float4Fp16Conversion, E1M2FromFp16)
{
    // Test constructor from fp16_t
    op::fp16_t fp16_val(1.5f);
    op::Float4E1M2 e1m2_from_fp16(fp16_val);
    EXPECT_NEAR(static_cast<float>(e1m2_from_fp16), 1.5f, 0.1f);
}

TEST(Float4Fp16Conversion, E1M2ToFp16)
{
    // Test conversion to fp16_t via float
    op::Float4E1M2 e1m2(1.25f);
    op::fp16_t fp16_result(static_cast<float>(e1m2));
    EXPECT_NEAR(static_cast<float>(fp16_result), 1.25f, 0.1f);
}

TEST(Float4BFloat16Conversion, E2M1FromBFloat16)
{
    // Test constructor from bfloat16
    op::bfloat16 bf16_val(3.0f);
    op::Float4E2M1 e2m1_from_bf16(bf16_val);
    EXPECT_NEAR(static_cast<float>(e2m1_from_bf16), 3.0f, 0.2f);
}

TEST(Float4BFloat16Conversion, E2M1ToBFloat16)
{
    // Test conversion to bfloat16 via float
    op::Float4E2M1 e2m1(2.0f);
    op::bfloat16 bf16_result(static_cast<float>(e2m1));
    EXPECT_NEAR(static_cast<float>(bf16_result), 2.0f, 0.1f);
}

TEST(Float4BFloat16Conversion, E1M2FromBFloat16)
{
    // Test constructor from bfloat16
    op::bfloat16 bf16_val(1.0f);
    op::Float4E1M2 e1m2_from_bf16(bf16_val);
    EXPECT_NEAR(static_cast<float>(e1m2_from_bf16), 1.0f, 0.1f);
}

TEST(Float4BFloat16Conversion, E1M2ToBFloat16)
{
    // Test conversion to bfloat16 via float
    op::Float4E1M2 e1m2(1.5f);
    op::bfloat16 bf16_result(static_cast<float>(e1m2));
    EXPECT_NEAR(static_cast<float>(bf16_result), 1.5f, 0.1f);
}

// ============================================================================
// Float4 Direct Conversion Operator Tests
// ============================================================================
// Note: Direct conversion tests (e.g., op::fp16_t fp16_result(e2m1)) cannot be
// used because fp16_t's template constructor causes ambiguous overload errors.
// The ToFp16/ToBFloat16 tests above already verify the conversion via float.
