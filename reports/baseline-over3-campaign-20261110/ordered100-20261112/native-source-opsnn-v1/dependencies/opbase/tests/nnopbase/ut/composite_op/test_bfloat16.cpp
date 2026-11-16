/**
 * Copyright (c) 2025 - 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <numeric>

#include "opdev/bfloat16.h"
#include "opdev/float8_e4m3fn.h"
#include "opdev/float8_e5m2.h"
#include "opdev/float8_e8m0.h"
#include "opdev/float6_e3m2.h"
#include "opdev/float6_e2m3.h"
#include "opdev/float4_e2m1.h"
#include "opdev/float4_e1m2.h"
#include "gtest/gtest.h"
#include "sstream"

class TestBFloat16 : public testing::Test {};

TEST_F(TestBFloat16, Inf)
{
    op::bfloat16 posInf(std::numeric_limits<float>::infinity());
    EXPECT_FLOAT_EQ(static_cast<float>(posInf), std::numeric_limits<float>::infinity());
    std::stringstream s;
    s << posInf;
    EXPECT_STRCASEEQ("inf", s.str().c_str());

    op::bfloat16 negInf(-std::numeric_limits<float>::infinity());
    EXPECT_FLOAT_EQ(static_cast<float>(negInf), -std::numeric_limits<float>::infinity());
    s.clear();
    s.str("");
    s << negInf;
    EXPECT_STRCASEEQ("-inf", s.str().c_str());

    op::bfloat16 posInf1(std::numeric_limits<float>::max());
    EXPECT_FLOAT_EQ(static_cast<float>(posInf1), std::numeric_limits<float>::infinity());

    op::bfloat16 negInf1(std::numeric_limits<float>::lowest());
    EXPECT_FLOAT_EQ(static_cast<float>(negInf1), -std::numeric_limits<float>::infinity());
}

TEST_F(TestBFloat16, Nan)
{
    op::bfloat16 nan(std::numeric_limits<float>::quiet_NaN());
    EXPECT_TRUE(std::isnan(static_cast<float>(nan)));
    std::stringstream s;
    s << nan;
    EXPECT_STRCASEEQ("nan", s.str().c_str());

    nan = op::bfloat16(std::numeric_limits<float>::signaling_NaN());
    EXPECT_TRUE(std::isnan(static_cast<float>(nan)));
    s.clear();
    s.str("");
    s << nan;
    EXPECT_STRCASEEQ("nan", s.str().c_str());
}

TEST_F(TestBFloat16, normal)
{
    float pi = 3.1415926;
    op::bfloat16 bf_pi(pi);
    op::bfloat16 bf_pi1(pi + std::numeric_limits<float>::epsilon());
    EXPECT_TRUE(static_cast<float>(std::abs(bf_pi1 - bf_pi)) < std::numeric_limits<float>::epsilon());

    op::fp16_t fp = 3;

    op::bfloat16 bf = fp;
    EXPECT_EQ(op::fp16_t(bf), op::fp16_t(bf));

    op::fp16_t fp1 = static_cast<op::tagFp16>(bf);
    EXPECT_EQ(fp1, static_cast<op::tagFp16>(bf));
}

// ============================================================================
// BFloat16 to Float8/Float6/Float4 Conversion Tests
// ============================================================================

TEST_F(TestBFloat16, ConvertToFloat8E4M3)
{
    op::bfloat16 bf16_val(2.5f);
    op::Float8E4M3FN e4m3(static_cast<float>(bf16_val));
    EXPECT_NEAR(static_cast<float>(e4m3), 2.5f, 0.1f);
}

TEST_F(TestBFloat16, ConvertToFloat8E5M2)
{
    op::bfloat16 bf16_val(100.0f);
    op::Float8E5M2 e5m2(static_cast<float>(bf16_val));
    EXPECT_NEAR(static_cast<float>(e5m2), 100.0f, 10.0f);
}

TEST_F(TestBFloat16, ConvertToFloat8E8M0)
{
    op::bfloat16 bf16_val(4.0f);
    op::Float8E8M0 e8m0(static_cast<float>(bf16_val));
    EXPECT_FLOAT_EQ(static_cast<float>(e8m0), 4.0f);
}

TEST_F(TestBFloat16, ConvertToFloat6E3M2)
{
    op::bfloat16 bf16_val(4.0f);
    op::Float6E3M2 e3m2(static_cast<float>(bf16_val));
    EXPECT_NEAR(static_cast<float>(e3m2), 4.0f, 0.2f);
}

TEST_F(TestBFloat16, ConvertToFloat6E2M3)
{
    op::bfloat16 bf16_val(3.0f);
    op::Float6E2M3 e2m3(static_cast<float>(bf16_val));
    EXPECT_NEAR(static_cast<float>(e2m3), 3.0f, 0.2f);
}

TEST_F(TestBFloat16, ConvertToFloat4E2M1)
{
    op::bfloat16 bf16_val(2.0f);
    op::Float4E2M1 e2m1(static_cast<float>(bf16_val));
    EXPECT_NEAR(static_cast<float>(e2m1), 2.0f, 0.1f);
}

TEST_F(TestBFloat16, ConvertToFloat4E1M2)
{
    op::bfloat16 bf16_val(1.5f);
    op::Float4E1M2 e1m2(static_cast<float>(bf16_val));
    EXPECT_NEAR(static_cast<float>(e1m2), 1.5f, 0.1f);
}
