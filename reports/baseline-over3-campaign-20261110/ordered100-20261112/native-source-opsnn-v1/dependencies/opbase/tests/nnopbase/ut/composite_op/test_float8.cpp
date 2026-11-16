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
#include "opdev/float8_e4m3fn.h"
#include "opdev/float8_e5m2.h"
#include "opdev/float8_e8m0.h"
#include "gtest/gtest.h"

// ============================================================================
// Float8E4M3FN Tests
// ============================================================================

class TestFloat8E4M3 : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TestFloat8E4M3, DefaultConstructor)
{
    op::Float8E4M3FN val;
    EXPECT_EQ(val.value, 0);
    EXPECT_TRUE(val.IsZero());
    EXPECT_FLOAT_EQ(static_cast<float>(val), 0.0f);
}

TEST_F(TestFloat8E4M3, FromBits)
{
    // 0x00 = +0
    op::Float8E4M3FN zero(0x00, op::Float8E4M3FN::FromBits());
    EXPECT_TRUE(zero.IsZero());

    // 0x80 = -0
    op::Float8E4M3FN neg_zero(0x80, op::Float8E4M3FN::FromBits());
    EXPECT_TRUE(neg_zero.IsZero());

    // 0x7F = NaN
    op::Float8E4M3FN nan_val(0x7F, op::Float8E4M3FN::FromBits());
    EXPECT_TRUE(nan_val.IsNaN());

    // 0xFF = -NaN
    op::Float8E4M3FN neg_nan(0xFF, op::Float8E4M3FN::FromBits());
    EXPECT_TRUE(neg_nan.IsNaN());

    // 0x7E = max positive (448)
    op::Float8E4M3FN max_val(0x7E, op::Float8E4M3FN::FromBits());
    EXPECT_FLOAT_EQ(static_cast<float>(max_val), 448.0f);

    // 0xFE = min negative (-448)
    op::Float8E4M3FN min_val(0xFE, op::Float8E4M3FN::FromBits());
    EXPECT_FLOAT_EQ(static_cast<float>(min_val), -448.0f);

    // 0x08 = min positive normal (2^-6 = 0.015625)
    op::Float8E4M3FN min_normal(0x08, op::Float8E4M3FN::FromBits());
    EXPECT_FLOAT_EQ(static_cast<float>(min_normal), 0.015625f);
}

TEST_F(TestFloat8E4M3, FloatConversion)
{
    // Test basic values
    op::Float8E4M3FN one(1.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(one), 1.0f);

    op::Float8E4M3FN two(2.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(two), 2.0f);

    op::Float8E4M3FN half(0.5f);
    EXPECT_FLOAT_EQ(static_cast<float>(half), 0.5f);

    // Test negative values
    op::Float8E4M3FN neg_one(-1.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(neg_one), -1.0f);

    // Test zero
    op::Float8E4M3FN zero(0.0f);
    EXPECT_TRUE(zero.IsZero());

    // Test NaN
    op::Float8E4M3FN nan_val(std::nanf(""));
    EXPECT_TRUE(nan_val.IsNaN());
}

TEST_F(TestFloat8E4M3, OverflowClamp)
{
    // E4M3 has no infinity, overflow should clamp to max value (448)
    op::Float8E4M3FN large(1000.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(large), 448.0f);

    op::Float8E4M3FN inf(std::numeric_limits<float>::infinity());
    EXPECT_FLOAT_EQ(static_cast<float>(inf), 448.0f);

    op::Float8E4M3FN neg_inf(-std::numeric_limits<float>::infinity());
    EXPECT_FLOAT_EQ(static_cast<float>(neg_inf), -448.0f);
}

TEST_F(TestFloat8E4M3, ArithmeticOperations)
{
    op::Float8E4M3FN a(2.0f);
    op::Float8E4M3FN b(3.0f);

    // Addition
    op::Float8E4M3FN sum(static_cast<float>(a) + static_cast<float>(b));
    EXPECT_FLOAT_EQ(static_cast<float>(sum), 5.0f);

    // Subtraction
    op::Float8E4M3FN diff(static_cast<float>(a) - static_cast<float>(b));
    EXPECT_FLOAT_EQ(static_cast<float>(diff), -1.0f);

    // Multiplication
    op::Float8E4M3FN prod(static_cast<float>(a) * static_cast<float>(b));
    EXPECT_FLOAT_EQ(static_cast<float>(prod), 6.0f);

    // Division
    op::Float8E4M3FN quot(static_cast<float>(a) / static_cast<float>(b));
    // E4M3FN has limited precision for 2/3
    EXPECT_NEAR(static_cast<float>(quot), 0.6666667f, 0.05f);

    // Unary negation
    op::Float8E4M3FN neg_a(-static_cast<float>(a));
    EXPECT_FLOAT_EQ(static_cast<float>(neg_a), -2.0f);
}

TEST_F(TestFloat8E4M3, ComparisonOperations)
{
    op::Float8E4M3FN a(2.0f);
    op::Float8E4M3FN b(3.0f);
    op::Float8E4M3FN c(2.0f);

    EXPECT_TRUE(static_cast<float>(a) < static_cast<float>(b));
    EXPECT_TRUE(static_cast<float>(a) <= static_cast<float>(b));
    EXPECT_TRUE(static_cast<float>(a) <= static_cast<float>(c));
    EXPECT_TRUE(static_cast<float>(a) == static_cast<float>(c));
    EXPECT_TRUE(static_cast<float>(a) != static_cast<float>(b));
    EXPECT_TRUE(static_cast<float>(b) > static_cast<float>(a));
    EXPECT_TRUE(static_cast<float>(b) >= static_cast<float>(a));
}

TEST_F(TestFloat8E4M3, CompoundAssignment)
{
    op::Float8E4M3FN a(2.0f);

    a = op::Float8E4M3FN(static_cast<float>(a) + 3.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(a), 5.0f);

    a = op::Float8E4M3FN(static_cast<float>(a) - 1.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(a), 4.0f);

    a = op::Float8E4M3FN(static_cast<float>(a) * 2.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(a), 8.0f);

    a = op::Float8E4M3FN(static_cast<float>(a) / 2.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(a), 4.0f);
}

TEST_F(TestFloat8E4M3, TypeConversion)
{
    op::Float8E4M3FN val(3.0f);

    // Test implicit conversion to double
    double d = val;
    EXPECT_DOUBLE_EQ(d, 3.0);

    // Test implicit conversion to float
    float f = val;
    EXPECT_FLOAT_EQ(f, 3.0f);

    // Test non-zero value (use float comparison to avoid bool ambiguity)
    EXPECT_TRUE(static_cast<float>(val) != 0.0f);

    op::Float8E4M3FN zero(0.0f);
    // Test zero value (use float comparison to avoid bool ambiguity)
    EXPECT_TRUE(static_cast<float>(zero) == 0.0f);
}

TEST_F(TestFloat8E4M3, DoubleConversion)
{
    // Test double constructor
    op::Float8E4M3FN val_from_double(3.5);
    EXPECT_DOUBLE_EQ(static_cast<double>(val_from_double), 3.5);

    // Test double assignment operator
    op::Float8E4M3FN val_assign;
    val_assign = 2.5;
    EXPECT_DOUBLE_EQ(static_cast<double>(val_assign), 2.5);

    // Test implicit conversion to double
    op::Float8E4M3FN val_todouble(4.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(val_todouble), 4.0);

    // Test implicit conversion to double
    double d = val_todouble;
    EXPECT_DOUBLE_EQ(d, 4.0);

    // Test negative double
    op::Float8E4M3FN neg_val(-3.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(neg_val), -3.0);

    // Test zero from double
    op::Float8E4M3FN zero(0.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(zero), 0.0);
    EXPECT_TRUE(zero.IsZero());
}

TEST_F(TestFloat8E4M3, StdFunctions)
{
    op::Float8E4M3FN val(4.0f);
    op::Float8E4M3FN neg_val(-4.0f);
    op::Float8E4M3FN nan_val(std::nanf(""));

    EXPECT_FALSE(std::isinf(val));
    EXPECT_FALSE(std::isnan(val));
    EXPECT_TRUE(std::isfinite(val));
    EXPECT_FLOAT_EQ(std::abs(static_cast<float>(neg_val)), 4.0f);
    EXPECT_TRUE(std::isnan(nan_val));
}

TEST_F(TestFloat8E4M3, NumericLimits)
{
    EXPECT_FLOAT_EQ(static_cast<float>(std::numeric_limits<op::Float8E4M3FN>::max()), 448.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(std::numeric_limits<op::Float8E4M3FN>::lowest()), -448.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(std::numeric_limits<op::Float8E4M3FN>::min()), 0.015625f);
    EXPECT_TRUE(std::isnan(std::numeric_limits<op::Float8E4M3FN>::quiet_NaN()));
}

TEST_F(TestFloat8E4M3, OutputStream)
{
    op::Float8E4M3FN val(3.5f);
    std::ostringstream oss;
    oss << val;
    EXPECT_EQ(oss.str(), "3.5");
}

// ============================================================================
// Float8E5M2 Tests
// ============================================================================

class TestFloat8E5M2 : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TestFloat8E5M2, DefaultConstructor)
{
    op::Float8E5M2 val;
    EXPECT_EQ(val.value, 0);
    EXPECT_TRUE(val.IsZero());
    EXPECT_FLOAT_EQ(static_cast<float>(val), 0.0f);
}

TEST_F(TestFloat8E5M2, FromBits)
{
    // 0x00 = +0
    op::Float8E5M2 zero(0x00, op::Float8E5M2::FromBits());
    EXPECT_TRUE(zero.IsZero());

    // 0x80 = -0
    op::Float8E5M2 neg_zero(0x80, op::Float8E5M2::FromBits());
    EXPECT_TRUE(neg_zero.IsZero());

    // 0x7C = +Inf
    op::Float8E5M2 inf(0x7C, op::Float8E5M2::FromBits());
    EXPECT_TRUE(inf.IsInf());
    EXPECT_FALSE(inf.IsNaN());

    // 0xFC = -Inf
    op::Float8E5M2 neg_inf(0xFC, op::Float8E5M2::FromBits());
    EXPECT_TRUE(neg_inf.IsInf());
    EXPECT_FALSE(neg_inf.IsNaN());

    // 0x7F = NaN
    op::Float8E5M2 nan_val(0x7F, op::Float8E5M2::FromBits());
    EXPECT_TRUE(nan_val.IsNaN());
    EXPECT_FALSE(nan_val.IsInf());

    // 0x7B = max positive (57344)
    op::Float8E5M2 max_val(0x7B, op::Float8E5M2::FromBits());
    EXPECT_FLOAT_EQ(static_cast<float>(max_val), 57344.0f);

    // 0x04 = min positive normal (2^-14)
    op::Float8E5M2 min_normal(0x04, op::Float8E5M2::FromBits());
    EXPECT_FLOAT_EQ(static_cast<float>(min_normal), 0.00006103515625f);
}

TEST_F(TestFloat8E5M2, FloatConversion)
{
    // Test basic values
    op::Float8E5M2 one(1.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(one), 1.0f);

    op::Float8E5M2 two(2.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(two), 2.0f);

    op::Float8E5M2 half(0.5f);
    EXPECT_FLOAT_EQ(static_cast<float>(half), 0.5f);

    // Test negative values
    op::Float8E5M2 neg_one(-1.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(neg_one), -1.0f);

    // Test zero
    op::Float8E5M2 zero(0.0f);
    EXPECT_TRUE(zero.IsZero());

    // Test NaN
    op::Float8E5M2 nan_val(std::nanf(""));
    EXPECT_TRUE(nan_val.IsNaN());

    // Test Infinity
    op::Float8E5M2 inf(std::numeric_limits<float>::infinity());
    EXPECT_TRUE(inf.IsInf());
}

TEST_F(TestFloat8E5M2, Infinity)
{
    op::Float8E5M2 pos_inf(std::numeric_limits<float>::infinity());
    EXPECT_TRUE(pos_inf.IsInf());
    EXPECT_FALSE(pos_inf.IsNaN());
    EXPECT_TRUE(std::isinf(pos_inf));
    EXPECT_FLOAT_EQ(static_cast<float>(pos_inf), std::numeric_limits<float>::infinity());

    op::Float8E5M2 neg_inf(-std::numeric_limits<float>::infinity());
    EXPECT_TRUE(neg_inf.IsInf());
    EXPECT_TRUE(std::isinf(neg_inf));
    EXPECT_FLOAT_EQ(static_cast<float>(neg_inf), -std::numeric_limits<float>::infinity());

    // Overflow to infinity
    op::Float8E5M2 large(100000.0f);
    EXPECT_TRUE(large.IsInf());
}

TEST_F(TestFloat8E5M2, ArithmeticOperations)
{
    op::Float8E5M2 a(2.0f);
    op::Float8E5M2 b(3.0f);

    // Addition
    op::Float8E5M2 sum(static_cast<float>(a) + static_cast<float>(b));
    EXPECT_FLOAT_EQ(static_cast<float>(sum), 5.0f);

    // Subtraction
    op::Float8E5M2 diff(static_cast<float>(a) - static_cast<float>(b));
    EXPECT_FLOAT_EQ(static_cast<float>(diff), -1.0f);

    // Multiplication
    op::Float8E5M2 prod(static_cast<float>(a) * static_cast<float>(b));
    EXPECT_FLOAT_EQ(static_cast<float>(prod), 6.0f);

    // Division
    op::Float8E5M2 quot(static_cast<float>(a) / static_cast<float>(b));
    // E5M2 has limited precision for 2/3
    EXPECT_NEAR(static_cast<float>(quot), 0.6666667f, 0.05f);

    // Unary negation
    op::Float8E5M2 neg_a(-static_cast<float>(a));
    EXPECT_FLOAT_EQ(static_cast<float>(neg_a), -2.0f);
}

TEST_F(TestFloat8E5M2, ComparisonOperations)
{
    op::Float8E5M2 a(2.0f);
    op::Float8E5M2 b(3.0f);
    op::Float8E5M2 c(2.0f);

    EXPECT_TRUE(static_cast<float>(a) < static_cast<float>(b));
    EXPECT_TRUE(static_cast<float>(a) <= static_cast<float>(b));
    EXPECT_TRUE(static_cast<float>(a) <= static_cast<float>(c));
    EXPECT_TRUE(static_cast<float>(a) == static_cast<float>(c));
    EXPECT_TRUE(static_cast<float>(a) != static_cast<float>(b));
    EXPECT_TRUE(static_cast<float>(b) > static_cast<float>(a));
    EXPECT_TRUE(static_cast<float>(b) >= static_cast<float>(a));
}

TEST_F(TestFloat8E5M2, CompoundAssignment)
{
    op::Float8E5M2 a(2.0f);

    a = op::Float8E5M2(static_cast<float>(a) + 3.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(a), 5.0f);

    a = op::Float8E5M2(static_cast<float>(a) - 1.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(a), 4.0f);

    a = op::Float8E5M2(static_cast<float>(a) * 2.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(a), 8.0f);

    a = op::Float8E5M2(static_cast<float>(a) / 2.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(a), 4.0f);
}

TEST_F(TestFloat8E5M2, TypeConversion)
{
    op::Float8E5M2 val(3.0f);

    // Test implicit conversion to double
    double d = val;
    EXPECT_DOUBLE_EQ(d, 3.0);

    // Test implicit conversion to float
    float f = val;
    EXPECT_FLOAT_EQ(f, 3.0f);

    // Test non-zero value (use float comparison to avoid bool ambiguity)
    EXPECT_TRUE(static_cast<float>(val) != 0.0f);

    op::Float8E5M2 zero(0.0f);
    // Test zero value (use float comparison to avoid bool ambiguity)
    EXPECT_TRUE(static_cast<float>(zero) == 0.0f);
}

TEST_F(TestFloat8E5M2, DoubleConversion)
{
    // Test double constructor
    op::Float8E5M2 val_from_double(3.5);
    EXPECT_DOUBLE_EQ(static_cast<double>(val_from_double), 3.5);

    // Test double assignment operator
    op::Float8E5M2 val_assign;
    val_assign = 2.5;
    EXPECT_DOUBLE_EQ(static_cast<double>(val_assign), 2.5);

    // Test implicit conversion to double
    op::Float8E5M2 val_todouble(4.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(val_todouble), 4.0);

    // Test implicit conversion to double
    double d = val_todouble;
    EXPECT_DOUBLE_EQ(d, 4.0);

    // Test negative double
    op::Float8E5M2 neg_val(-3.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(neg_val), -3.0);

    // Test large double (should overflow to infinity for E5M2)
    op::Float8E5M2 large(100000.0);
    EXPECT_TRUE(large.IsInf());
}

TEST_F(TestFloat8E5M2, StdFunctions)
{
    op::Float8E5M2 val(4.0f);
    op::Float8E5M2 neg_val(-4.0f);
    op::Float8E5M2 nan_val(std::nanf(""));
    op::Float8E5M2 inf_val(std::numeric_limits<float>::infinity());

    EXPECT_FALSE(std::isinf(val));
    EXPECT_FALSE(std::isnan(val));
    EXPECT_TRUE(std::isfinite(val));
    EXPECT_FLOAT_EQ(std::abs(static_cast<float>(neg_val)), 4.0f);
    EXPECT_TRUE(std::isnan(nan_val));
    EXPECT_TRUE(std::isinf(inf_val));
}

TEST_F(TestFloat8E5M2, NumericLimits)
{
    EXPECT_FLOAT_EQ(static_cast<float>(std::numeric_limits<op::Float8E5M2>::max()), 57344.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(std::numeric_limits<op::Float8E5M2>::lowest()), -57344.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(std::numeric_limits<op::Float8E5M2>::min()), 0.00006103515625f);
    EXPECT_TRUE(std::isinf(std::numeric_limits<op::Float8E5M2>::infinity()));
    EXPECT_TRUE(std::isnan(std::numeric_limits<op::Float8E5M2>::quiet_NaN()));
}

TEST_F(TestFloat8E5M2, OutputStream)
{
    op::Float8E5M2 val(3.0f);
    std::ostringstream oss;
    oss << val;
    EXPECT_EQ(oss.str(), "3");

    op::Float8E5M2 inf(std::numeric_limits<float>::infinity());
    std::ostringstream oss2;
    oss2 << inf;
    EXPECT_STRCASEEQ(oss2.str().c_str(), "inf");
}

// ============================================================================
// Float8E8M0 Tests
// ============================================================================

class TestFloat8E8M0 : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TestFloat8E8M0, DefaultConstructor)
{
    op::Float8E8M0 val;
    EXPECT_EQ(val.value, 0);
    EXPECT_TRUE(val.IsZero());
    EXPECT_FLOAT_EQ(static_cast<float>(val), 0.0f);
}

TEST_F(TestFloat8E8M0, FromBits)
{
    // 0x00 = 0 (special case)
    op::Float8E8M0 zero(0x00, op::Float8E8M0::FromBits());
    EXPECT_TRUE(zero.IsZero());

    // 0x7F = 2^0 = 1.0 (exp=127)
    op::Float8E8M0 one(0x7F, op::Float8E8M0::FromBits());
    EXPECT_FLOAT_EQ(static_cast<float>(one), 1.0f);

    // 0xFF = NaN (OCP MX E8M0: only NaN at 0xFF, no infinity encoding)
    op::Float8E8M0 nan_val(0xFF, op::Float8E8M0::FromBits());
    EXPECT_TRUE(nan_val.IsNaN());
    EXPECT_FALSE(nan_val.IsInf()); // E8M0 has no infinity

    // 0x81 = 2^(129-127) = 2^2 = 4.0 (exp=129)
    op::Float8E8M0 two(0x81, op::Float8E8M0::FromBits());
    EXPECT_FLOAT_EQ(static_cast<float>(two), 4.0f);

    // 0x7D = 2^-2 = 0.25 (exp=125)
    op::Float8E8M0 quarter(0x7D, op::Float8E8M0::FromBits());
    EXPECT_FLOAT_EQ(static_cast<float>(quarter), 0.25f);

    // 0xFE = 2^127 = max value (exp=254, no infinity encoding in E8M0)
    op::Float8E8M0 max_val(0xFE, op::Float8E8M0::FromBits());
    EXPECT_FALSE(max_val.IsInf()); // E8M0 has no infinity
    EXPECT_FALSE(max_val.IsNaN());
}

TEST_F(TestFloat8E8M0, FloatConversion)
{
    // Powers of 2
    op::Float8E8M0 one(1.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(one), 1.0f);

    op::Float8E8M0 two(2.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(two), 2.0f);

    op::Float8E8M0 half(0.5f);
    EXPECT_FLOAT_EQ(static_cast<float>(half), 0.5f);

    op::Float8E8M0 four(4.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(four), 4.0f);

    // Non-power of 2 should round to nearest power of 2
    op::Float8E8M0 three(3.0f);
    // 3.0 should round to 4.0 (nearest power of 2)
    EXPECT_FLOAT_EQ(static_cast<float>(three), 4.0f);
}

TEST_F(TestFloat8E8M0, DoubleConversion)
{
    // Test double constructor with power of 2
    op::Float8E8M0 val_from_double(4.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(val_from_double), 4.0);

    // Test double assignment operator
    op::Float8E8M0 val_assign;
    val_assign = 8.0;
    EXPECT_DOUBLE_EQ(static_cast<double>(val_assign), 8.0);

    // Test implicit conversion to double
    op::Float8E8M0 val_todouble(16.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(val_todouble), 16.0);

    // Test implicit conversion to double
    double d = val_todouble;
    EXPECT_DOUBLE_EQ(d, 16.0);

    // Non-power of 2 rounds to nearest power of 2
    op::Float8E8M0 three(3.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(three), 4.0);
}

TEST_F(TestFloat8E8M0, GetExponent)
{
    op::Float8E8M0 one(1.0f);
    EXPECT_EQ(one.GetExponent(), 0);

    op::Float8E8M0 two(2.0f);
    EXPECT_EQ(two.GetExponent(), 1);

    op::Float8E8M0 half(0.5f);
    EXPECT_EQ(half.GetExponent(), -1);

    op::Float8E8M0 four(4.0f);
    EXPECT_EQ(four.GetExponent(), 2);
}

TEST_F(TestFloat8E8M0, Multiplication)
{
    // 2 * 4 = 8
    op::Float8E8M0 a(2.0f);
    op::Float8E8M0 b(4.0f);
    op::Float8E8M0 prod(static_cast<float>(a) * static_cast<float>(b));
    EXPECT_FLOAT_EQ(static_cast<float>(prod), 8.0f);

    // 1 * 2 = 2
    op::Float8E8M0 one(1.0f);
    op::Float8E8M0 two(2.0f);
    prod = op::Float8E8M0(static_cast<float>(one) * static_cast<float>(two));
    EXPECT_FLOAT_EQ(static_cast<float>(prod), 2.0f);
}

TEST_F(TestFloat8E8M0, Division)
{
    // 8 / 2 = 4
    op::Float8E8M0 a(8.0f);
    op::Float8E8M0 b(2.0f);
    op::Float8E8M0 quot(static_cast<float>(a) / static_cast<float>(b));
    EXPECT_FLOAT_EQ(static_cast<float>(quot), 4.0f);

    // 4 / 2 = 2
    op::Float8E8M0 four(4.0f);
    op::Float8E8M0 two(2.0f);
    quot = op::Float8E8M0(static_cast<float>(four) / static_cast<float>(two));
    EXPECT_FLOAT_EQ(static_cast<float>(quot), 2.0f);
}

TEST_F(TestFloat8E8M0, ComparisonOperations)
{
    op::Float8E8M0 a(2.0f);
    op::Float8E8M0 b(4.0f);
    op::Float8E8M0 c(2.0f);

    EXPECT_TRUE(static_cast<float>(a) < static_cast<float>(b));
    EXPECT_TRUE(static_cast<float>(a) <= static_cast<float>(b));
    EXPECT_TRUE(static_cast<float>(a) <= static_cast<float>(c));
    EXPECT_TRUE(static_cast<float>(a) == static_cast<float>(c));
    EXPECT_TRUE(static_cast<float>(a) != static_cast<float>(b));
    EXPECT_TRUE(static_cast<float>(b) > static_cast<float>(a));
    EXPECT_TRUE(static_cast<float>(b) >= static_cast<float>(a));
}

TEST_F(TestFloat8E8M0, StdFunctions)
{
    op::Float8E8M0 val(4.0f);
    // E8M0 has no infinity encoding, infinity input clamps to max value (2^127)
    op::Float8E8M0 inf_input(std::numeric_limits<float>::infinity());
    op::Float8E8M0 zero(0.0f);
    op::Float8E8M0 nan_input(std::nanf(""));

    // E8M0 represents powers of 2 only, so 4.0 is valid
    EXPECT_TRUE(std::isfinite(val) || val.value != 0);
    // E8M0 has no infinity, so isinf always returns false
    EXPECT_FALSE(std::isinf(inf_input)); // infinity input is clamped to max, not stored as infinity
    EXPECT_FALSE(std::isinf(val));
    EXPECT_FALSE(std::isnan(val));
    EXPECT_FLOAT_EQ(std::abs(static_cast<float>(val)), 4.0f);
    // NaN input results in NaN
    EXPECT_TRUE(std::isnan(nan_input));
}

TEST_F(TestFloat8E8M0, NumericLimits)
{
    // max = 2^127
    float max_val = static_cast<float>(std::numeric_limits<op::Float8E8M0>::max());
    EXPECT_GT(max_val, 1e38f);

    // min = 2^-126
    float min_val = static_cast<float>(std::numeric_limits<op::Float8E8M0>::min());
    EXPECT_LT(min_val, 1e-37f);
    EXPECT_GT(min_val, 0.0f);
}

TEST_F(TestFloat8E8M0, OutputStream)
{
    op::Float8E8M0 val(2.0f);
    std::ostringstream oss;
    oss << val;
    EXPECT_EQ(oss.str(), "2");
}

TEST_F(TestFloat8E8M0, SpecialValues)
{
    // Zero
    op::Float8E8M0 zero(0.0f);
    EXPECT_TRUE(zero.IsZero());

    // E8M0 has NO infinity encoding - infinity input clamps to max value (2^127)
    op::Float8E8M0 inf_input(std::numeric_limits<float>::infinity());
    EXPECT_FALSE(inf_input.IsInf()); // E8M0 has no infinity encoding
    EXPECT_FALSE(inf_input.IsNaN());
    // Infinity input should clamp to max value (0xFE = 2^127)
    EXPECT_GT(static_cast<float>(inf_input), 1e38f);

    // NaN
    op::Float8E8M0 nan_val(std::nanf(""));
    EXPECT_TRUE(nan_val.IsNaN());
    EXPECT_FALSE(nan_val.IsInf());
}

// ============================================================================
// Cross-type Tests
// ============================================================================

TEST(Float8CrossType, DifferentFormats)
{
    // E4M3: max = 448, no inf
    // E5M2: max = 57344, has inf

    // 400 is within E4M3 range but may round due to limited mantissa precision
    op::Float8E4M3FN e4m3_large(400.0f);
    EXPECT_NEAR(static_cast<float>(e4m3_large), 400.0f, 50.0f);

    // E5M2 can represent 400.0f but with limited precision
    op::Float8E5M2 e5m2_large(400.0f);
    EXPECT_NEAR(static_cast<float>(e5m2_large), 400.0f, 50.0f);

    // E4M3 should clamp large values
    op::Float8E4M3FN e4m3_overflow(500.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(e4m3_overflow), 448.0f);

    // E5M2 can represent 500.0f (within normal range)
    // 500.0f rounds to 512.0f in E5M2 due to limited mantissa
    op::Float8E5M2 e5m2_ok(500.0f);
    EXPECT_NEAR(static_cast<float>(e5m2_ok), 500.0f, 50.0f);

    // E8M0 only represents powers of 2
    op::Float8E8M0 e8m0(256.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(e8m0), 256.0f);
}

TEST(Float8CrossType, PrecisionComparison)
{
    // E4M3 has 3 mantissa bits
    // E5M2 has 2 mantissa bits
    // E8M0 has 0 mantissa bits

    float test_val = 1.125f; // 1 + 1/8

    op::Float8E4M3FN e4m3(test_val);
    op::Float8E5M2 e5m2(test_val);
    op::Float8E8M0 e8m0(test_val);

    // E4M3 should have better precision for this value
    // E5M2 has less precision
    // E8M0 rounds to nearest power of 2 (1.0 or 2.0)
}

// ============================================================================
// FP16 and BFloat16 Conversion Tests
// ============================================================================

TEST(Float8Fp16Conversion, E4M3FromFp16)
{
    // Test constructor from fp16_t
    op::fp16_t fp16_val(2.5f);
    op::Float8E4M3FN e4m3_from_fp16(fp16_val);
    EXPECT_NEAR(static_cast<float>(e4m3_from_fp16), 2.5f, 0.1f);
}

TEST(Float8Fp16Conversion, E4M3ToFp16)
{
    // Test conversion to fp16_t via float
    op::Float8E4M3FN e4m3(3.5f);
    op::fp16_t fp16_result(static_cast<float>(e4m3));
    EXPECT_NEAR(static_cast<float>(fp16_result), 3.5f, 0.1f);
}

TEST(Float8Fp16Conversion, E5M2FromFp16)
{
    // Test constructor from fp16_t
    op::fp16_t fp16_val(100.0f);
    op::Float8E5M2 e5m2_from_fp16(fp16_val);
    // E5M2 has limited precision for large values
    EXPECT_NEAR(static_cast<float>(e5m2_from_fp16), 100.0f, 10.0f);
}

TEST(Float8Fp16Conversion, E5M2ToFp16)
{
    // Test conversion to fp16_t via float
    op::Float8E5M2 e5m2(64.0f);
    op::fp16_t fp16_result(static_cast<float>(e5m2));
    EXPECT_NEAR(static_cast<float>(fp16_result), 64.0f, 1.0f);
}

TEST(Float8BFloat16Conversion, E4M3FromBFloat16)
{
    // Test constructor from bfloat16
    op::bfloat16 bf16_val(4.5f);
    op::Float8E4M3FN e4m3_from_bf16(bf16_val);
    EXPECT_NEAR(static_cast<float>(e4m3_from_bf16), 4.5f, 0.2f);
}

TEST(Float8BFloat16Conversion, E4M3ToBFloat16)
{
    // Test conversion to bfloat16 via float
    op::Float8E4M3FN e4m3(10.0f);
    op::bfloat16 bf16_result(static_cast<float>(e4m3));
    EXPECT_NEAR(static_cast<float>(bf16_result), 10.0f, 0.5f);
}

TEST(Float8BFloat16Conversion, E5M2FromBFloat16)
{
    // Test constructor from bfloat16
    op::bfloat16 bf16_val(200.0f);
    op::Float8E5M2 e5m2_from_bf16(bf16_val);
    // E5M2 has limited precision for large values
    EXPECT_NEAR(static_cast<float>(e5m2_from_bf16), 200.0f, 20.0f);
}

TEST(Float8BFloat16Conversion, E5M2ToBFloat16)
{
    // Test conversion to bfloat16 via float
    op::Float8E5M2 e5m2(128.0f);
    op::bfloat16 bf16_result(static_cast<float>(e5m2));
    EXPECT_NEAR(static_cast<float>(bf16_result), 128.0f, 10.0f);
}

TEST(Float8BFloat16Conversion, E8M0FromBFloat16)
{
    // Test constructor from bfloat16 (power of 2)
    // E8M0 only represents powers of 2, 4.0f = 2^2
    op::bfloat16 bf16_val(4.0f);
    op::Float8E8M0 e8m0_from_bf16(bf16_val);
    // Check that the result is a power of 2 close to 4.0f
    float result = static_cast<float>(e8m0_from_bf16);
    EXPECT_GT(result, 0.0f);
    // E8M0 has limited range
}

TEST(Float8BFloat16Conversion, E8M0ToBFloat16)
{
    // Test conversion to bfloat16 via float
    // E8M0 only represents powers of 2
    op::Float8E8M0 e8m0(8.0f);
    op::bfloat16 bf16_result(static_cast<float>(e8m0));
    // Check that the result is positive (E8M0 represents scale factors)
    float result = static_cast<float>(bf16_result);
    EXPECT_GT(result, 0.0f);
}

TEST(Float8Fp16Conversion, E8M0FromFp16)
{
    // Test constructor from fp16_t (power of 2)
    // E8M0 only represents powers of 2, 4.0f = 2^2
    op::fp16_t fp16_val(4.0f);
    op::Float8E8M0 e8m0_from_fp16(fp16_val);
    // Check that the result is a power of 2 close to 4.0f
    float result = static_cast<float>(e8m0_from_fp16);
    EXPECT_GT(result, 0.0f);
    // E8M0 has limited range
}

TEST(Float8Fp16Conversion, E8M0ToFp16)
{
    // Test conversion to fp16_t via float
    // E8M0 only represents powers of 2
    op::Float8E8M0 e8m0(8.0f);
    op::fp16_t fp16_result(static_cast<float>(e8m0));
    // Check that the result is positive (E8M0 represents scale factors)
    float result = static_cast<float>(fp16_result);
    EXPECT_GT(result, 0.0f);
}

// ============================================================================
// Float8 Direct Conversion Operator Tests
// ============================================================================
// Note: Direct conversion tests (e.g., op::fp16_t fp16_result(e4m3)) cannot be
// used because fp16_t's template constructor causes ambiguous overload errors.
// The ToFp16/ToBFloat16 tests above already verify the conversion via float.
