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
#include "opdev/float6_e3m2.h"
#include "opdev/float6_e2m3.h"
#include "gtest/gtest.h"

// ============================================================================
// Float6E3M2 Tests
// ============================================================================

class TestFloat6E3M2 : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TestFloat6E3M2, DefaultConstructor)
{
    op::Float6E3M2 val;
    EXPECT_EQ(val.value, 0);
    EXPECT_TRUE(val.IsZero());
    EXPECT_FLOAT_EQ(static_cast<float>(val), 0.0f);
}

TEST_F(TestFloat6E3M2, FromBits)
{
    // 0x00 = +0
    op::Float6E3M2 zero(0x00, op::Float6E3M2::FromBits());
    EXPECT_TRUE(zero.IsZero());

    // 0x20 = -0
    op::Float6E3M2 neg_zero(0x20, op::Float6E3M2::FromBits());
    EXPECT_TRUE(neg_zero.IsZero());

    // 0x1F = max value (0 111 11) = 28.0 (OCP MX E3M2 has no NaN)
    op::Float6E3M2 max_val(0x1F, op::Float6E3M2::FromBits());
    EXPECT_FLOAT_EQ(static_cast<float>(max_val), 28.0f);
    EXPECT_FALSE(max_val.IsNaN());

    // 0x3F = min value (1 111 11) = -28.0 (OCP MX E3M2 has no NaN)
    op::Float6E3M2 min_val(0x3F, op::Float6E3M2::FromBits());
    EXPECT_FLOAT_EQ(static_cast<float>(min_val), -28.0f);
    EXPECT_FALSE(min_val.IsNaN());

    // 0x0C = 1.0 (0 011 00)
    op::Float6E3M2 one(0x0C, op::Float6E3M2::FromBits());
    EXPECT_FLOAT_EQ(static_cast<float>(one), 1.0f);

    // 0x04 = min positive normal = 2^-2 = 0.25 (0 001 00)
    op::Float6E3M2 min_normal(0x04, op::Float6E3M2::FromBits());
    EXPECT_FLOAT_EQ(static_cast<float>(min_normal), 0.25f);

    // 0x1E = (0 111 10) = 2^4 * 1.5 = 24
    op::Float6E3M2 val_24(0x1E, op::Float6E3M2::FromBits());
    EXPECT_FLOAT_EQ(static_cast<float>(val_24), 24.0f);
}

TEST_F(TestFloat6E3M2, FloatConversion)
{
    // Test basic values
    op::Float6E3M2 one(1.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(one), 1.0f);

    op::Float6E3M2 two(2.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(two), 2.0f);

    op::Float6E3M2 half(0.5f);
    EXPECT_FLOAT_EQ(static_cast<float>(half), 0.5f);

    // Test negative values
    op::Float6E3M2 neg_one(-1.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(neg_one), -1.0f);

    // Test zero
    op::Float6E3M2 zero(0.0f);
    EXPECT_TRUE(zero.IsZero());

    // Test NaN input - OCP MX E3M2 has no NaN, clamped to max (28.0)
    op::Float6E3M2 nan_input(std::nanf(""));
    EXPECT_FALSE(nan_input.IsNaN());
    EXPECT_FLOAT_EQ(static_cast<float>(nan_input), 28.0f);
}

TEST_F(TestFloat6E3M2, OverflowClamp)
{
    // E3M2 has no infinity, overflow should clamp to max value (28.0)
    // OCP MX E3M2: max = 0 111 11 = 2^4 * 1.75 = 28.0
    op::Float6E3M2 large(100.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(large), 28.0f);

    op::Float6E3M2 inf(std::numeric_limits<float>::infinity());
    EXPECT_FLOAT_EQ(static_cast<float>(inf), 28.0f);

    op::Float6E3M2 neg_inf(-std::numeric_limits<float>::infinity());
    EXPECT_FLOAT_EQ(static_cast<float>(neg_inf), -28.0f);
}

TEST_F(TestFloat6E3M2, ArithmeticOperations)
{
    op::Float6E3M2 a(2.0f);
    op::Float6E3M2 b(3.0f);

    // Addition
    op::Float6E3M2 sum(static_cast<float>(a) + static_cast<float>(b));
    EXPECT_FLOAT_EQ(static_cast<float>(sum), 5.0f);

    // Subtraction
    op::Float6E3M2 diff(static_cast<float>(a) - static_cast<float>(b));
    EXPECT_FLOAT_EQ(static_cast<float>(diff), -1.0f);

    // Multiplication
    op::Float6E3M2 prod(static_cast<float>(a) * static_cast<float>(b));
    EXPECT_FLOAT_EQ(static_cast<float>(prod), 6.0f);

    // Division
    op::Float6E3M2 quot(static_cast<float>(a) / static_cast<float>(b));
    EXPECT_NEAR(static_cast<float>(quot), 0.6666667f, 0.1f);

    // Unary negation
    op::Float6E3M2 neg_a(-static_cast<float>(a));
    EXPECT_FLOAT_EQ(static_cast<float>(neg_a), -2.0f);
}

TEST_F(TestFloat6E3M2, ComparisonOperations)
{
    op::Float6E3M2 a(2.0f);
    op::Float6E3M2 b(4.0f);
    op::Float6E3M2 c(2.0f);

    EXPECT_TRUE(static_cast<float>(a) < static_cast<float>(b));
    EXPECT_TRUE(static_cast<float>(a) <= static_cast<float>(b));
    EXPECT_TRUE(static_cast<float>(a) <= static_cast<float>(c));
    EXPECT_TRUE(static_cast<float>(a) == static_cast<float>(c));
    EXPECT_TRUE(static_cast<float>(a) != static_cast<float>(b));
    EXPECT_TRUE(static_cast<float>(b) > static_cast<float>(a));
    EXPECT_TRUE(static_cast<float>(b) >= static_cast<float>(a));
}

TEST_F(TestFloat6E3M2, CompoundAssignment)
{
    op::Float6E3M2 a(2.0f);

    a = op::Float6E3M2(static_cast<float>(a) + 3.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(a), 5.0f);

    a = op::Float6E3M2(static_cast<float>(a) - 1.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(a), 4.0f);

    a = op::Float6E3M2(static_cast<float>(a) * 2.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(a), 8.0f);

    a = op::Float6E3M2(static_cast<float>(a) / 2.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(a), 4.0f);
}

TEST_F(TestFloat6E3M2, TypeConversion)
{
    op::Float6E3M2 val(3.0f);

    // Test implicit conversion to double
    double d = val;
    EXPECT_DOUBLE_EQ(d, 3.0);

    // Test implicit conversion to float
    float f = val;
    EXPECT_FLOAT_EQ(f, 3.0f);

    // Test non-zero value (use float comparison to avoid bool ambiguity)
    EXPECT_TRUE(static_cast<float>(val) != 0.0f);

    op::Float6E3M2 zero(0.0f);
    // Test zero value (use float comparison to avoid bool ambiguity)
    EXPECT_TRUE(static_cast<float>(zero) == 0.0f);
}

TEST_F(TestFloat6E3M2, DoubleConversion)
{
    // Test double constructor
    op::Float6E3M2 val_from_double(3.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(val_from_double), 3.0);

    // Test double assignment operator
    op::Float6E3M2 val_assign;
    val_assign = 2.0;
    EXPECT_DOUBLE_EQ(static_cast<double>(val_assign), 2.0);

    // Test implicit conversion to double
    op::Float6E3M2 val_todouble(4.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(val_todouble), 4.0);

    // Test implicit conversion to double
    double d = val_todouble;
    EXPECT_DOUBLE_EQ(d, 4.0);

    // Test negative double
    op::Float6E3M2 neg_val(-2.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(neg_val), -2.0);

    // Test overflow with double
    // OCP MX E3M2: max = 0 111 11 = 2^4 * 1.75 = 28.0
    op::Float6E3M2 large(100.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(large), 28.0); // clamped to max (28.0 per OCP MX)
}

TEST_F(TestFloat6E3M2, StdFunctions)
{
    op::Float6E3M2 val(4.0f);
    op::Float6E3M2 neg_val(-4.0f);
    // OCP MX E3M2 has no NaN, nan input is clamped to max (28.0)
    op::Float6E3M2 nan_input(std::nanf(""));

    EXPECT_FALSE(std::isinf(val));
    EXPECT_FALSE(std::isnan(val));
    EXPECT_TRUE(std::isfinite(val));
    EXPECT_FLOAT_EQ(std::abs(static_cast<float>(neg_val)), 4.0f);
    // nan input is clamped to max value (28.0), not NaN
    EXPECT_FALSE(std::isnan(nan_input));
    EXPECT_FLOAT_EQ(static_cast<float>(nan_input), 28.0f);
}

TEST_F(TestFloat6E3M2, NumericLimits)
{
    // OCP MX E3M2: max = 0 111 11 = 2^4 * 1.75 = 28.0
    EXPECT_FLOAT_EQ(static_cast<float>(std::numeric_limits<op::Float6E3M2>::max()), 28.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(std::numeric_limits<op::Float6E3M2>::lowest()), -28.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(std::numeric_limits<op::Float6E3M2>::min()), 0.25f);
    // OCP MX E3M2 has no NaN, quiet_NaN returns zero
    EXPECT_FALSE(std::isnan(std::numeric_limits<op::Float6E3M2>::quiet_NaN()));
}

TEST_F(TestFloat6E3M2, OutputStream)
{
    op::Float6E3M2 val(3.0f);
    std::ostringstream oss;
    oss << val;
    EXPECT_EQ(oss.str(), "3");
}

// ============================================================================
// Division by Zero Tests (OCP MX v1.0 - E3M2 has no NaN or Infinity)
// ============================================================================

TEST_F(TestFloat6E3M2, DivisionByZero)
{
    // Per OCP MX v1.0: E3M2 has NO NaN and NO Infinity encodings
    // All bit patterns are valid numbers
    // Division by zero (x/0 where x != 0) results in infinity in float,
    // which is clamped to max value (28.0) preserving sign

    op::Float6E3M2 positive(4.0f);
    op::Float6E3M2 zero(0.0f);

    // Positive / zero = +max (28.0) - E3M2 has no infinity
    op::Float6E3M2 result_pos(static_cast<float>(positive) / static_cast<float>(zero));
    EXPECT_FLOAT_EQ(static_cast<float>(result_pos), 28.0f) << "Positive/zero should clamp to max (28.0)";
    EXPECT_FALSE(result_pos.IsNaN());

    // Negative / zero = -max (-28.0) - sign preserved
    op::Float6E3M2 negative(-4.0f);
    op::Float6E3M2 result_neg(static_cast<float>(negative) / static_cast<float>(zero));
    EXPECT_FLOAT_EQ(static_cast<float>(result_neg), -28.0f) << "Negative/zero should clamp to min (-28.0)";
    EXPECT_FALSE(result_neg.IsNaN());

    // Zero / zero = NaN in float, clamped to max (28.0) in E3M2
    op::Float6E3M2 zero_div_zero(static_cast<float>(zero) / static_cast<float>(zero));
    EXPECT_FLOAT_EQ(static_cast<float>(zero_div_zero), 28.0f) << "Zero/zero (NaN) should clamp to max";
    EXPECT_FALSE(zero_div_zero.IsNaN());

    // Compound assignment division by zero
    op::Float6E3M2 a(8.0f);
    a = op::Float6E3M2(static_cast<float>(a) / static_cast<float>(zero));
    EXPECT_FLOAT_EQ(static_cast<float>(a), 28.0f) << "Compound division by zero should clamp to max";
}

// ============================================================================
// Float6E2M3 Tests
// ============================================================================

class TestFloat6E2M3 : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TestFloat6E2M3, DefaultConstructor)
{
    op::Float6E2M3 val;
    EXPECT_EQ(val.value, 0);
    EXPECT_TRUE(val.IsZero());
    EXPECT_FLOAT_EQ(static_cast<float>(val), 0.0f);
}

TEST_F(TestFloat6E2M3, FromBits)
{
    // 0x00 = +0
    op::Float6E2M3 zero(0x00, op::Float6E2M3::FromBits());
    EXPECT_TRUE(zero.IsZero());

    // 0x20 = -0
    op::Float6E2M3 neg_zero(0x20, op::Float6E2M3::FromBits());
    EXPECT_TRUE(neg_zero.IsZero());

    // OCP MX E2M3 has NO NaN encoding - all bit patterns are valid numbers
    // 0x1F = max value (0 11 111) = 2^2 * 1.875 = 7.5
    op::Float6E2M3 max_val(0x1F, op::Float6E2M3::FromBits());
    EXPECT_FLOAT_EQ(static_cast<float>(max_val), 7.5f);
    EXPECT_FALSE(max_val.IsNaN());

    // 0x3F = min value (1 11 111) = -7.5 (NOT NaN per OCP MX)
    op::Float6E2M3 min_val(0x3F, op::Float6E2M3::FromBits());
    EXPECT_FLOAT_EQ(static_cast<float>(min_val), -7.5f);
    EXPECT_FALSE(min_val.IsNaN());

    // E2M3: bit layout is bit5=sign, bits4-3=exp, bits2-0=man
    // 0x08 = 0 01 000 = exp=1, man=0 => 2^(1-1) * 1.0 = 1.0
    op::Float6E2M3 one(0x08, op::Float6E2M3::FromBits());
    EXPECT_FLOAT_EQ(static_cast<float>(one), 1.0f);

    // 0x10 = 0 10 000 = exp=2, man=0 => 2^(2-1) * 1.0 = 2.0
    op::Float6E2M3 two(0x10, op::Float6E2M3::FromBits());
    EXPECT_FLOAT_EQ(static_cast<float>(two), 2.0f);

    // 0x04 = 0 00 100 = denorm: exp=0, man=4 => 2^0 * 0.5 = 0.5
    op::Float6E2M3 half(0x04, op::Float6E2M3::FromBits());
    EXPECT_FLOAT_EQ(static_cast<float>(half), 0.5f);
}

TEST_F(TestFloat6E2M3, FloatConversion)
{
    // Test basic values
    op::Float6E2M3 one(1.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(one), 1.0f);

    op::Float6E2M3 two(2.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(two), 2.0f);

    op::Float6E2M3 half(0.5f);
    EXPECT_FLOAT_EQ(static_cast<float>(half), 0.5f);

    // Test negative values
    op::Float6E2M3 neg_one(-1.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(neg_one), -1.0f);

    // Test zero
    op::Float6E2M3 zero(0.0f);
    EXPECT_TRUE(zero.IsZero());

    // Test NaN input - OCP MX E2M3 has no NaN, clamped to max (7.5)
    op::Float6E2M3 nan_input(std::nanf(""));
    EXPECT_FALSE(nan_input.IsNaN());
    EXPECT_FLOAT_EQ(static_cast<float>(nan_input), 7.5f);
}

TEST_F(TestFloat6E2M3, OverflowClamp)
{
    // E2M3 has no infinity, overflow should clamp to max value (7.5)
    // OCP MX E2M3: max = 0 11 111 = 2^2 * 1.875 = 7.5
    op::Float6E2M3 large(100.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(large), 7.5f);

    op::Float6E2M3 inf(std::numeric_limits<float>::infinity());
    EXPECT_FLOAT_EQ(static_cast<float>(inf), 7.5f);

    op::Float6E2M3 neg_inf(-std::numeric_limits<float>::infinity());
    EXPECT_FLOAT_EQ(static_cast<float>(neg_inf), -7.5f);
}

TEST_F(TestFloat6E2M3, ArithmeticOperations)
{
    op::Float6E2M3 a(2.0f);
    op::Float6E2M3 b(3.0f);

    // Addition
    op::Float6E2M3 sum(static_cast<float>(a) + static_cast<float>(b));
    EXPECT_NEAR(static_cast<float>(sum), 5.0f, 0.2f);

    // Subtraction
    op::Float6E2M3 diff(static_cast<float>(a) - static_cast<float>(b));
    EXPECT_FLOAT_EQ(static_cast<float>(diff), -1.0f);

    // Multiplication
    op::Float6E2M3 prod(static_cast<float>(a) * static_cast<float>(b));
    EXPECT_NEAR(static_cast<float>(prod), 6.0f, 0.2f);

    // Division
    op::Float6E2M3 quot(static_cast<float>(a) / static_cast<float>(b));
    EXPECT_NEAR(static_cast<float>(quot), 0.6666667f, 0.1f);

    // Unary negation
    op::Float6E2M3 neg_a(-static_cast<float>(a));
    EXPECT_FLOAT_EQ(static_cast<float>(neg_a), -2.0f);
}

TEST_F(TestFloat6E2M3, ComparisonOperations)
{
    op::Float6E2M3 a(2.0f);
    op::Float6E2M3 b(4.0f);
    op::Float6E2M3 c(2.0f);

    EXPECT_TRUE(static_cast<float>(a) < static_cast<float>(b));
    EXPECT_TRUE(static_cast<float>(a) <= static_cast<float>(b));
    EXPECT_TRUE(static_cast<float>(a) <= static_cast<float>(c));
    EXPECT_TRUE(static_cast<float>(a) == static_cast<float>(c));
    EXPECT_TRUE(static_cast<float>(a) != static_cast<float>(b));
    EXPECT_TRUE(static_cast<float>(b) > static_cast<float>(a));
    EXPECT_TRUE(static_cast<float>(b) >= static_cast<float>(a));
}

TEST_F(TestFloat6E2M3, CompoundAssignment)
{
    op::Float6E2M3 a(2.0f);

    a = op::Float6E2M3(static_cast<float>(a) + 3.0f);
    EXPECT_NEAR(static_cast<float>(a), 5.0f, 0.2f);

    a = op::Float6E2M3(static_cast<float>(a) - 1.0f);
    EXPECT_NEAR(static_cast<float>(a), 4.0f, 0.2f);

    a = op::Float6E2M3(static_cast<float>(a) * 1.5f);
    EXPECT_NEAR(static_cast<float>(a), 6.0f, 0.3f);

    a = op::Float6E2M3(static_cast<float>(a) / 2.0f);
    EXPECT_NEAR(static_cast<float>(a), 3.0f, 0.2f);
}

TEST_F(TestFloat6E2M3, TypeConversion)
{
    op::Float6E2M3 val(3.0f);

    // Test implicit conversion to double
    double d = val;
    EXPECT_DOUBLE_EQ(d, 3.0);

    // Test implicit conversion to float
    float f = val;
    EXPECT_FLOAT_EQ(f, 3.0f);

    // Test non-zero value is truthy (use explicit float comparison)
    EXPECT_TRUE(static_cast<float>(val) != 0.0f);

    op::Float6E2M3 zero(0.0f);
    // Test zero value is falsy (use explicit float comparison)
    EXPECT_TRUE(static_cast<float>(zero) == 0.0f);
}

TEST_F(TestFloat6E2M3, DoubleConversion)
{
    // Test double constructor
    op::Float6E2M3 val_from_double(3.0);
    EXPECT_NEAR(static_cast<double>(val_from_double), 3.0, 0.2);

    // Test double assignment operator
    op::Float6E2M3 val_assign;
    val_assign = 2.0;
    EXPECT_DOUBLE_EQ(static_cast<double>(val_assign), 2.0);

    // Test implicit conversion to double
    op::Float6E2M3 val_todouble(4.0);
    EXPECT_NEAR(static_cast<double>(val_todouble), 4.0, 0.2);

    // Test implicit conversion to double
    double d = val_todouble;
    EXPECT_NEAR(d, 4.0, 0.2);

    // Test negative double
    op::Float6E2M3 neg_val(-2.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(neg_val), -2.0);

    // Test overflow with double
    // OCP MX E2M3: max = 0 11 111 = 2^2 * 1.875 = 7.5
    op::Float6E2M3 large(100.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(large), 7.5); // clamped to max (7.5 per OCP MX)
}

TEST_F(TestFloat6E2M3, StdFunctions)
{
    op::Float6E2M3 val(4.0f);
    op::Float6E2M3 neg_val(-4.0f);
    // OCP MX E2M3 has no NaN, nan input is clamped to max (7.5)
    op::Float6E2M3 nan_input(std::nanf(""));

    EXPECT_FALSE(std::isinf(val));
    EXPECT_FALSE(std::isnan(val));
    EXPECT_TRUE(std::isfinite(val));
    EXPECT_FLOAT_EQ(std::abs(static_cast<float>(neg_val)), 4.0f);
    // nan input is clamped to max value (7.5), not NaN
    EXPECT_FALSE(std::isnan(nan_input));
    EXPECT_FLOAT_EQ(static_cast<float>(nan_input), 7.5f);
}

TEST_F(TestFloat6E2M3, NumericLimits)
{
    // OCP MX E2M3: max = 0 11 111 = 2^2 * 1.875 = 7.5
    EXPECT_FLOAT_EQ(static_cast<float>(std::numeric_limits<op::Float6E2M3>::max()), 7.5f);
    EXPECT_FLOAT_EQ(static_cast<float>(std::numeric_limits<op::Float6E2M3>::lowest()), -7.5f);
    EXPECT_FLOAT_EQ(static_cast<float>(std::numeric_limits<op::Float6E2M3>::min()), 1.0f);
    // OCP MX E2M3 has no NaN, quiet_NaN returns zero
    EXPECT_FALSE(std::isnan(std::numeric_limits<op::Float6E2M3>::quiet_NaN()));
}

TEST_F(TestFloat6E2M3, OutputStream)
{
    op::Float6E2M3 val(3.0f);
    std::ostringstream oss;
    oss << val;
    EXPECT_EQ(oss.str(), "3");
}

// ============================================================================
// Division by Zero Tests (OCP MX v1.0 - E2M3 has no NaN or Infinity)
// ============================================================================

TEST_F(TestFloat6E2M3, DivisionByZero)
{
    // Per OCP MX v1.0: E2M3 has NO NaN and NO Infinity encodings
    // All bit patterns are valid numbers
    // Division by zero (x/0 where x != 0) results in infinity in float,
    // which is clamped to max value (7.5) preserving sign

    op::Float6E2M3 positive(4.0f);
    op::Float6E2M3 zero(0.0f);

    // Positive / zero = +max (7.5) - E2M3 has no infinity
    op::Float6E2M3 result_pos(static_cast<float>(positive) / static_cast<float>(zero));
    EXPECT_FLOAT_EQ(static_cast<float>(result_pos), 7.5f) << "Positive/zero should clamp to max (7.5)";
    EXPECT_FALSE(result_pos.IsNaN());

    // Negative / zero = -max (-7.5) - sign preserved
    op::Float6E2M3 negative(-4.0f);
    op::Float6E2M3 result_neg(static_cast<float>(negative) / static_cast<float>(zero));
    EXPECT_FLOAT_EQ(static_cast<float>(result_neg), -7.5f) << "Negative/zero should clamp to min (-7.5)";
    EXPECT_FALSE(result_neg.IsNaN());

    // Zero / zero = NaN in float, clamped to max (7.5) in E2M3
    op::Float6E2M3 zero_div_zero(static_cast<float>(zero) / static_cast<float>(zero));
    EXPECT_FLOAT_EQ(static_cast<float>(zero_div_zero), 7.5f) << "Zero/zero (NaN) should clamp to max";
    EXPECT_FALSE(zero_div_zero.IsNaN());

    // Compound assignment division by zero
    op::Float6E2M3 a(4.0f);
    a = op::Float6E2M3(static_cast<float>(a) / static_cast<float>(zero));
    EXPECT_FLOAT_EQ(static_cast<float>(a), 7.5f) << "Compound division by zero should clamp to max";
}

// ============================================================================
// Cross-type Comparison Tests
// ============================================================================

TEST(Float6CrossType, RangeComparison)
{
    // E3M2 has larger range (max=24) but lower precision (2-bit mantissa)
    // E2M3 has smaller range (max=7) but higher precision (3-bit mantissa)

    op::Float6E3M2 e3m2_large(20.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(e3m2_large), 20.0f);

    op::Float6E2M3 e2m3_large(6.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(e2m3_large), 6.0f);

    // E2M3 cannot represent 20
    // OCP MX E2M3: max = 7.5
    op::Float6E2M3 e2m3_overflow(20.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(e2m3_overflow), 7.5f); // clamped to max (7.5)
}

TEST(Float6CrossType, PrecisionComparison)
{
    float test_val = 1.125f; // 1 + 1/8

    op::Float6E3M2 e3m2(test_val);
    op::Float6E2M3 e2m3(test_val);

    // E2M3 has 3-bit mantissa, can represent 1.125 exactly
    // E3M2 has 2-bit mantissa, less precision
    EXPECT_NEAR(static_cast<float>(e2m3), 1.125f, 0.01f);
}

TEST(Float6CrossType, BitPatterns)
{
    // Verify bit patterns are consistent

    // E3M2: S EEE MM (bit5=sign, bits4-2=exp, bits1-0=man)
    // 1.0 = 0 011 00 = 0x0C
    op::Float6E3M2 e3m2_one(1.0f);
    EXPECT_EQ(e3m2_one.value & 0x3F, 0x0C);

    // E2M3: S EE MMM (bit5=sign, bits4-3=exp, bits2-0=man)
    // 1.0 = 0 01 000 = 0x08 (exp=1, man=0)
    op::Float6E2M3 e2m3_one(1.0f);
    EXPECT_EQ(e2m3_one.value & 0x3F, 0x08);
}

// ============================================================================
// FP16 and BFloat16 Conversion Tests
// ============================================================================

TEST(Float6Fp16Conversion, E3M2FromFp16)
{
    // Test constructor from fp16_t
    op::fp16_t fp16_val(4.0f);
    op::Float6E3M2 e3m2_from_fp16(fp16_val);
    EXPECT_NEAR(static_cast<float>(e3m2_from_fp16), 4.0f, 0.2f);
}

TEST(Float6Fp16Conversion, E3M2ToFp16)
{
    // Test conversion to fp16_t via float
    op::Float6E3M2 e3m2(8.0f);
    op::fp16_t fp16_result(static_cast<float>(e3m2));
    EXPECT_NEAR(static_cast<float>(fp16_result), 8.0f, 0.5f);
}

TEST(Float6Fp16Conversion, E2M3FromFp16)
{
    // Test constructor from fp16_t
    op::fp16_t fp16_val(3.0f);
    op::Float6E2M3 e2m3_from_fp16(fp16_val);
    EXPECT_NEAR(static_cast<float>(e2m3_from_fp16), 3.0f, 0.2f);
}

TEST(Float6Fp16Conversion, E2M3ToFp16)
{
    // Test conversion to fp16_t via float
    // E2M3 has limited precision, 5.0 might round to 5.5 or similar
    op::Float6E2M3 e2m3(5.0f);
    op::fp16_t fp16_result(static_cast<float>(e2m3));
    EXPECT_NEAR(static_cast<float>(fp16_result), static_cast<float>(e2m3), 0.5f);
}

TEST(Float6BFloat16Conversion, E3M2FromBFloat16)
{
    // Test constructor from bfloat16
    op::bfloat16 bf16_val(6.0f);
    op::Float6E3M2 e3m2_from_bf16(bf16_val);
    EXPECT_NEAR(static_cast<float>(e3m2_from_bf16), 6.0f, 0.3f);
}

TEST(Float6BFloat16Conversion, E3M2ToBFloat16)
{
    // Test conversion to bfloat16 via float
    op::Float6E3M2 e3m2(12.0f);
    op::bfloat16 bf16_result(static_cast<float>(e3m2));
    EXPECT_NEAR(static_cast<float>(bf16_result), 12.0f, 0.5f);
}

TEST(Float6BFloat16Conversion, E2M3FromBFloat16)
{
    // Test constructor from bfloat16
    op::bfloat16 bf16_val(2.5f);
    op::Float6E2M3 e2m3_from_bf16(bf16_val);
    EXPECT_NEAR(static_cast<float>(e2m3_from_bf16), 2.5f, 0.1f);
}

TEST(Float6BFloat16Conversion, E2M3ToBFloat16)
{
    // Test conversion to bfloat16 via float
    op::Float6E2M3 e2m3(4.0f);
    op::bfloat16 bf16_result(static_cast<float>(e2m3));
    EXPECT_NEAR(static_cast<float>(bf16_result), static_cast<float>(e2m3), 0.5f);
}

// ============================================================================
// Float6 Direct Conversion Operator Tests
// ============================================================================
// Note: Direct conversion tests (e.g., op::fp16_t fp16_result(e3m2)) cannot be
// used because fp16_t's template constructor causes ambiguous overload errors.
// The ToFp16/ToBFloat16 tests above already verify the conversion via float.
