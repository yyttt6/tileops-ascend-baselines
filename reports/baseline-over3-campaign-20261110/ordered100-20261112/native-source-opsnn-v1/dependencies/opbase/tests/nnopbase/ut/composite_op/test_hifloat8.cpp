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
#include <cstdint>
#include <sstream>
#include <limits>

#include "opdev/hifloat8.h"
#include "gtest/gtest.h"

class TestHiFP8 : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test special values: NaN
TEST_F(TestHiFP8, NaN)
{
    op::HiFloat8 nanVal(op::HiFloat8::HIF8_NAN_VALUE, op::HiFloat8::FromBits());
    EXPECT_TRUE(nanVal.IsNaN());
    EXPECT_TRUE(std::isnan(static_cast<float>(nanVal)));

    // Test NaN from float conversion
    op::HiFloat8 nanFromFloat(std::numeric_limits<float>::quiet_NaN());
    EXPECT_TRUE(nanFromFloat.IsNaN());
}

// Test special values: Infinity
TEST_F(TestHiFP8, Infinity)
{
    op::HiFloat8 posInf(op::HiFloat8::HIF8_INF_VALUE, op::HiFloat8::FromBits());
    EXPECT_TRUE(posInf.IsInf());
    EXPECT_FALSE(posInf.IsNaN());
    float f = static_cast<float>(posInf);
    EXPECT_TRUE(std::isinf(f));
    EXPECT_GT(f, 0.0f);

    op::HiFloat8 negInf(op::HiFloat8::HIF8_INF_VALUE | 0x80, op::HiFloat8::FromBits()); // Set sign bit
    EXPECT_TRUE(negInf.IsInf());
    f = static_cast<float>(negInf);
    EXPECT_TRUE(std::isinf(f));
    EXPECT_LT(f, 0.0f);

    // Test infinity from float conversion
    op::HiFloat8 infFromFloat(std::numeric_limits<float>::infinity());
    EXPECT_TRUE(infFromFloat.IsInf());

    op::HiFloat8 negInfFromFloat(-std::numeric_limits<float>::infinity());
    EXPECT_TRUE(negInfFromFloat.IsInf());
}

// Test zero
TEST_F(TestHiFP8, Zero)
{
    op::HiFloat8 zero;
    EXPECT_TRUE(zero.IsZero());
    EXPECT_EQ(static_cast<float>(zero), 0.0f);

    op::HiFloat8 zeroFromRaw(0, op::HiFloat8::FromBits());
    EXPECT_TRUE(zeroFromRaw.IsZero());
    EXPECT_EQ(static_cast<float>(zeroFromRaw), 0.0f);

    op::HiFloat8 zeroFromFloat(0.0f);
    EXPECT_TRUE(zeroFromFloat.IsZero());

    op::HiFloat8 negZeroFromFloat(-0.0f);
    EXPECT_TRUE(negZeroFromFloat.IsZero());
}

// Test positive values
TEST_F(TestHiFP8, PositiveValues)
{
    // Test 1.0
    op::HiFloat8 one(1.0f);
    float oneFloat = static_cast<float>(one);
    EXPECT_NEAR(oneFloat, 1.0f, 0.1f);

    // Test 2.0
    op::HiFloat8 two(2.0f);
    float twoFloat = static_cast<float>(two);
    EXPECT_NEAR(twoFloat, 2.0f, 0.2f);

    // Test 0.5
    op::HiFloat8 half(0.5f);
    float halfFloat = static_cast<float>(half);
    EXPECT_NEAR(halfFloat, 0.5f, 0.05f);

    // Test small value
    op::HiFloat8 small(0.125f);
    float smallFloat = static_cast<float>(small);
    EXPECT_NEAR(smallFloat, 0.125f, 0.01f);
}

// Test negative values
TEST_F(TestHiFP8, NegativeValues)
{
    op::HiFloat8 negOne(-1.0f);
    float negOneFloat = static_cast<float>(negOne);
    EXPECT_NEAR(negOneFloat, -1.0f, 0.1f);

    op::HiFloat8 negTwo(-2.0f);
    float negTwoFloat = static_cast<float>(negTwo);
    EXPECT_NEAR(negTwoFloat, -2.0f, 0.2f);

    op::HiFloat8 negHalf(-0.5f);
    float negHalfFloat = static_cast<float>(negHalf);
    EXPECT_NEAR(negHalfFloat, -0.5f, 0.05f);
}

// Test conversion from float and back
TEST_F(TestHiFP8, FloatRoundTrip)
{
    float testValues[] = {0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 2.0f, -2.0f, 0.25f, -0.25f, 4.0f, -4.0f, 0.125f, -0.125f};

    for (float val : testValues) {
        op::HiFloat8 hif8(val);
        float recovered = static_cast<float>(hif8);
        // Allow for quantization error
        float relativeError = std::abs(recovered - val) / std::max(std::abs(val), 1e-6f);
        EXPECT_LT(relativeError, 0.15f) << "Value: " << val << ", Recovered: " << recovered;
    }
}

// Test FromBits constructor
TEST_F(TestHiFP8, FromBitsConstructor)
{
    op::HiFloat8 zeroFromBits(0, op::HiFloat8::FromBits());
    EXPECT_EQ(zeroFromBits.value, 0);
    EXPECT_TRUE(zeroFromBits.IsZero());

    op::HiFloat8 nanFromBits(op::HiFloat8::HIF8_NAN_VALUE, op::HiFloat8::FromBits());
    EXPECT_TRUE(nanFromBits.IsNaN());

    op::HiFloat8 infFromBits(op::HiFloat8::HIF8_INF_VALUE, op::HiFloat8::FromBits());
    EXPECT_TRUE(infFromBits.IsInf());
}

// Test FromBits constructor (replaces FromRawBits)
TEST_F(TestHiFP8, FromBitsConstructorRaw)
{
    op::HiFloat8 val(0x00, op::HiFloat8::FromBits());
    EXPECT_TRUE(val.IsZero());

    val = op::HiFloat8(0x80, op::HiFloat8::FromBits()); // NaN
    EXPECT_TRUE(val.IsNaN());

    val = op::HiFloat8(0x6F, op::HiFloat8::FromBits()); // +Inf
    EXPECT_TRUE(val.IsInf());
}

// Test comparison operators
TEST_F(TestHiFP8, ComparisonOperators)
{
    op::HiFloat8 a(1.0f);
    op::HiFloat8 b(2.0f);
    op::HiFloat8 c(1.0f);

    EXPECT_TRUE(static_cast<float>(a) == static_cast<float>(c));
    EXPECT_TRUE(static_cast<float>(a) != static_cast<float>(b));
    EXPECT_TRUE(static_cast<float>(a) < static_cast<float>(b));
    EXPECT_TRUE(static_cast<float>(a) <= static_cast<float>(b));
    EXPECT_TRUE(static_cast<float>(a) <= static_cast<float>(c));
    EXPECT_TRUE(static_cast<float>(b) > static_cast<float>(a));
    EXPECT_TRUE(static_cast<float>(b) >= static_cast<float>(a));
    EXPECT_TRUE(static_cast<float>(a) >= static_cast<float>(c));
}

// Test arithmetic operators
TEST_F(TestHiFP8, ArithmeticOperators)
{
    op::HiFloat8 a(2.0f);
    op::HiFloat8 b(3.0f);

    // Addition
    op::HiFloat8 sum(static_cast<float>(a) + static_cast<float>(b));
    float sumFloat = static_cast<float>(sum);
    EXPECT_NEAR(sumFloat, 5.0f, 0.5f);

    // Subtraction
    op::HiFloat8 diff(static_cast<float>(b) - static_cast<float>(a));
    float diffFloat = static_cast<float>(diff);
    EXPECT_NEAR(diffFloat, 1.0f, 0.2f);

    // Multiplication
    op::HiFloat8 prod(static_cast<float>(a) * static_cast<float>(b));
    float prodFloat = static_cast<float>(prod);
    EXPECT_NEAR(prodFloat, 6.0f, 0.6f);

    // Division
    op::HiFloat8 quot(static_cast<float>(b) / static_cast<float>(a));
    float quotFloat = static_cast<float>(quot);
    EXPECT_NEAR(quotFloat, 1.5f, 0.2f);

    // Negation
    op::HiFloat8 neg(-static_cast<float>(a));
    float negFloat = static_cast<float>(neg);
    EXPECT_NEAR(negFloat, -2.0f, 0.2f);
}

// Test compound assignment operators
TEST_F(TestHiFP8, CompoundAssignmentOperators)
{
    op::HiFloat8 a(2.0f);
    op::HiFloat8 b(3.0f);

    a = op::HiFloat8(static_cast<float>(a) + static_cast<float>(b));
    EXPECT_NEAR(static_cast<float>(a), 5.0f, 0.5f);

    a = op::HiFloat8(static_cast<float>(a) - static_cast<float>(b));
    EXPECT_NEAR(static_cast<float>(a), 2.0f, 0.3f);

    a = op::HiFloat8(static_cast<float>(a) * static_cast<float>(b));
    EXPECT_NEAR(static_cast<float>(a), 6.0f, 0.6f);

    a = op::HiFloat8(static_cast<float>(a) / static_cast<float>(b));
    EXPECT_NEAR(static_cast<float>(a), 2.0f, 0.3f);
}

// Test stream output operator
TEST_F(TestHiFP8, StreamOutput)
{
    op::HiFloat8 val(1.5f);
    std::stringstream ss;
    ss << val;
    std::string str = ss.str();
    EXPECT_FALSE(str.empty());
    // Should contain a number representation
    EXPECT_TRUE(str.find("1") != std::string::npos || str.find(".") != std::string::npos);
}

// Test std namespace functions
TEST_F(TestHiFP8, StdNamespaceFunctions)
{
    op::HiFloat8 one(1.0f);

    // Test std::isinf
    op::HiFloat8 inf(op::HiFloat8::HIF8_INF_VALUE, op::HiFloat8::FromBits());
    EXPECT_TRUE(std::isinf(inf));

    // Test std::isnan
    op::HiFloat8 nanVal(op::HiFloat8::HIF8_NAN_VALUE, op::HiFloat8::FromBits());
    EXPECT_TRUE(std::isnan(nanVal));

    // Test std::isfinite
    EXPECT_TRUE(std::isfinite(one));
    EXPECT_FALSE(std::isfinite(inf));
    EXPECT_FALSE(std::isfinite(nanVal));
}

// Test numeric_limits specialization
TEST_F(TestHiFP8, NumericLimits)
{
    // Test that numeric_limits is properly specialized
    EXPECT_TRUE(std::numeric_limits<op::HiFloat8>::is_specialized);
    EXPECT_TRUE(std::numeric_limits<op::HiFloat8>::is_signed);
    EXPECT_FALSE(std::numeric_limits<op::HiFloat8>::is_integer);
    EXPECT_EQ(std::numeric_limits<op::HiFloat8>::digits, 3);

    // Test special values via numeric_limits
    op::HiFloat8 nanVal = std::numeric_limits<op::HiFloat8>::quiet_NaN();
    EXPECT_TRUE(nanVal.IsNaN());

    op::HiFloat8 infVal = std::numeric_limits<op::HiFloat8>::infinity();
    EXPECT_TRUE(infVal.IsInf());
}

// Test sizeof
TEST_F(TestHiFP8, SizeOf)
{
    EXPECT_EQ(sizeof(op::HiFloat8), sizeof(uint8_t));
    EXPECT_EQ(sizeof(op::HiFloat8), 1u);
}

// Test denormal range (small values)
TEST_F(TestHiFP8, DenormalRange)
{
    // Test very small positive values
    float smallVal = 1e-6f;
    op::HiFloat8 hif8Small(smallVal);
    float recovered = static_cast<float>(hif8Small);
    // Should either be close to the value or zero (underflow)
    if (!hif8Small.IsZero()) {
        EXPECT_LT(std::abs(recovered - smallVal) / smallVal, 0.2f);
    }
}

// Test large values (overflow to infinity)
TEST_F(TestHiFP8, LargeValuesOverflow)
{
    // Large positive value should become infinity
    op::HiFloat8 large(std::numeric_limits<float>::max());
    EXPECT_TRUE(large.IsInf());

    // Large negative value should become negative infinity
    op::HiFloat8 largeNeg(-std::numeric_limits<float>::max());
    EXPECT_TRUE(largeNeg.IsInf());
}

// Test dynamic range - basic tests for valid range values
TEST_F(TestHiFP8, DynamicRange)
{
    // Test values around 1.0 (well within range)
    op::HiFloat8 one(1.0f);
    float oneRecovered = static_cast<float>(one);
    EXPECT_NEAR(oneRecovered, 1.0f, 0.2f);

    // Test 2.0
    op::HiFloat8 two(2.0f);
    float twoRecovered = static_cast<float>(two);
    EXPECT_NEAR(twoRecovered, 2.0f, 0.3f);

    // Test 0.5
    op::HiFloat8 half(0.5f);
    float halfRecovered = static_cast<float>(half);
    EXPECT_NEAR(halfRecovered, 0.5f, 0.1f);

    // Test 0.25
    op::HiFloat8 quarter(0.25f);
    float quarterRecovered = static_cast<float>(quarter);
    EXPECT_NEAR(quarterRecovered, 0.25f, 0.05f);
}
