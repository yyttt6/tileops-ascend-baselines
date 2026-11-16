/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "opdev/fp16_t.h"
#include "gtest/gtest.h"

#define CHECK_FP16_TEST_RESULT(fp16Val, expectSign, expectExp, expectMan) \
    op::ExtractFP16(fp16Val.val, &actualSign, &actualExp, &actualMan);    \
    EXPECT_EQ(actualSign, expectSign);                                    \
    EXPECT_EQ(actualExp, expectExp);                                      \
    EXPECT_EQ(actualMan, expectMan);

using namespace op;

class TestFp16 : public testing::Test {};

TEST_F(TestFp16, InfNaN)
{
    uint16_t actualSign = 0;
    int16_t actualExp = 0;
    uint16_t actualMan = 0;
    // double
    fp16_t doubleInfMinus = static_cast<double>(65504);
    CHECK_FP16_TEST_RESULT(doubleInfMinus, 0u, 30, 0x7FF);
    fp16_t doubleInf = static_cast<double>(65520);
    EXPECT_EQ(doubleInf.val, 31744);
    fp16_t doubeInfPlus = static_cast<double>(100000.0);
    EXPECT_EQ(doubeInfPlus.val, 31744);
    fp16_t doubeInfInf = static_cast<double>(INFINITY);
    EXPECT_EQ(doubeInfInf.val, 31744);
    fp16_t doubeNan = static_cast<double>(NAN);
    EXPECT_EQ(doubeNan.val, 32767);

    // float
    fp16_t floatInfMinus = static_cast<double>(65504);
    CHECK_FP16_TEST_RESULT(floatInfMinus, 0u, 30, 0x7FF);
    fp16_t floatInf = static_cast<float>(65520);
    EXPECT_EQ(floatInf.val, 31744);
    fp16_t floatInfPlus = static_cast<float>(100000.0);
    EXPECT_EQ(floatInfPlus.val, 31744);
    fp16_t floatInfInf = static_cast<float>(INFINITY);
    EXPECT_EQ(floatInfInf.val, 31744);
    fp16_t floatNan = static_cast<float>(NAN);
    EXPECT_EQ(floatNan.val, 32767);

    // int32
    fp16_t int32InfMinus = static_cast<double>(65504);
    CHECK_FP16_TEST_RESULT(int32InfMinus, 0u, 30, 0x7FF);
    fp16_t int32Inf = static_cast<int32_t>(65520);
    EXPECT_EQ(int32Inf.val, 31744);
    fp16_t int32InfPlus = static_cast<int32_t>(100000.0);
    EXPECT_EQ(int32InfPlus.val, 31744);

    // int64
    fp16_t int64InfMinus = static_cast<double>(65504);
    CHECK_FP16_TEST_RESULT(int64InfMinus, 0u, 30, 0x7FF);
    fp16_t int64Inf = static_cast<int64_t>(65520);
    EXPECT_EQ(int64Inf.val, 31744);
    fp16_t int64InfPlus = static_cast<int64_t>(100000.0);
    EXPECT_EQ(int64InfPlus.val, 31744);
}

TEST_F(TestFp16, FormatToString)
{
    fp16_t val = 8.1;
    fp16_t val2 = 9.5;
    fp16_t midVal = 35500;
    fp16_t largeVal = 65500.0;
    double doubleVal = val.toDouble();
    int8_t int8Val = val.toInt8();
    int8_t midInt8Val = midVal.toInt8();
    int8_t largeInt8Val = largeVal.toInt8();
    uint8_t uint8Val = val.toUInt8();
    int16_t int16Val = val.toInt16();
    int16_t midInt16Val = midVal.toInt16();
    int16_t largeInt16Val = largeVal.toInt16();
    uint16_t uint16Val = val.toUInt16();
    int32_t int32Val = val.toInt32();
    uint32_t uint32Val = val.toUInt32();
    uint32_t largeUint32Val = largeVal.toUInt32();
    fp16_t addVal = val + val;
    fp16_t mulVal = val * val;
    fp16_t divVal = val / val;
    fp16_t subVal = val - val;
    divVal = val / val2;
    divVal = val2 / val;
    divVal = val2 / fp16_t(0);
    divVal = fp16_t(0) / val;
    divVal = fp16_t(0);
    divVal += val;
    divVal -= val2;
    divVal *= fp16_t(1.0);
    divVal /= fp16_t(1.0);
    fp16_t int8ToFp16 = int8Val;
    fp16_t uin8ToFp16 = uint8Val;
    fp16_t int16ToFp16 = int16Val;
    fp16_t largeInt16ToFp16 = largeInt16Val;
    fp16_t uint16ToFp16 = uint16Val;
    fp16_t int32ToFp16 = int32Val;
    fp16_t uint32ToFp16 = uint32Val;
    fp16_t largeUint32ToFp16 = largeUint32Val;
    fp16_t resVal = val;
    resVal = largeVal;
    uint16_t uValue = 1;
    resVal = uValue;
    uValue = -1;
    resVal = uValue;
    uValue = 0;
    resVal = uValue;

    EXPECT_EQ(uint32Val, 8);
    EXPECT_TRUE(fp16_t(3.2) > fp16_t(3.1));
    EXPECT_TRUE(fp16_t(3.2) < fp16_t(3.3));
    EXPECT_TRUE(fp16_t(3.2) <= fp16_t(3.3));
    EXPECT_TRUE(fp16_t(3.2) != fp16_t(3.3));
}
