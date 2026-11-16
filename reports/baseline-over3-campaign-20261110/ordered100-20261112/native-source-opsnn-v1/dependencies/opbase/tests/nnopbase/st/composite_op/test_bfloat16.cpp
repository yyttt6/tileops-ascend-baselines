/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <numeric>

#include "opdev/bfloat16.h"
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
