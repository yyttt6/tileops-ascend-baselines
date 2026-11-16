/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "opdev/data_type_utils.h"
#include "gtest/gtest.h"

class TestDataTypeUtils : public testing::Test {};

TEST_F(TestDataTypeUtils, TestCanCast)
{
    EXPECT_TRUE(op::CanCast(op::DataType::DT_INT32, op::DataType::DT_FLOAT));
    EXPECT_TRUE(op::CanCast(op::DataType::DT_COMPLEX128, op::DataType::DT_COMPLEX64));
    EXPECT_TRUE(op::CanCast(op::DataType::DT_BOOL, op::DataType::DT_BOOL));
    EXPECT_FALSE(op::CanCast(op::DataType::DT_FLOAT, op::DataType::DT_INT32));
    EXPECT_FALSE(op::CanCast(op::DataType::DT_COMPLEX64, op::DataType::DT_INT32));
    EXPECT_FALSE(op::CanCast(op::DataType::DT_FLOAT, op::DataType::DT_BOOL));
}

#define TEST_PROMOTE(typeA, typeB, TypeExpected) EXPECT_EQ(TypeExpected, op::PromoteType(typeA, typeB))

TEST_F(TestDataTypeUtils, TestPromoteType)
{
    TEST_PROMOTE(op::DataType::DT_UINT8, op::DataType::DT_UINT8, op::DataType::DT_UINT8);
    TEST_PROMOTE(op::DataType::DT_UINT8, op::DataType::DT_INT8, op::DataType::DT_INT16);
    TEST_PROMOTE(op::DataType::DT_UINT8, op::DataType::DT_INT16, op::DataType::DT_INT16);
    TEST_PROMOTE(op::DataType::DT_UINT8, op::DataType::DT_INT32, op::DataType::DT_INT32);
    TEST_PROMOTE(op::DataType::DT_UINT8, op::DataType::DT_INT64, op::DataType::DT_INT64);
    TEST_PROMOTE(op::DataType::DT_UINT8, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT16);
    TEST_PROMOTE(op::DataType::DT_UINT8, op::DataType::DT_FLOAT, op::DataType::DT_FLOAT);
    TEST_PROMOTE(op::DataType::DT_UINT8, op::DataType::DT_DOUBLE, op::DataType::DT_DOUBLE);
    TEST_PROMOTE(op::DataType::DT_UINT8, op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX64);
    TEST_PROMOTE(op::DataType::DT_UINT8, op::DataType::DT_COMPLEX128, op::DataType::DT_COMPLEX128);
    TEST_PROMOTE(op::DataType::DT_UINT8, op::DataType::DT_BOOL, op::DataType::DT_UINT8);
    TEST_PROMOTE(op::DataType::DT_UINT8, op::DataType::DT_QINT8, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_UINT8, op::DataType::DT_QINT16, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_UINT8, op::DataType::DT_QINT32, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_UINT8, op::DataType::DT_BF16, op::DataType::DT_BF16);

    TEST_PROMOTE(op::DataType::DT_INT8, op::DataType::DT_UINT8, op::DataType::DT_INT16);
    TEST_PROMOTE(op::DataType::DT_INT8, op::DataType::DT_INT8, op::DataType::DT_INT8);
    TEST_PROMOTE(op::DataType::DT_INT8, op::DataType::DT_INT16, op::DataType::DT_INT16);
    TEST_PROMOTE(op::DataType::DT_INT8, op::DataType::DT_INT32, op::DataType::DT_INT32);
    TEST_PROMOTE(op::DataType::DT_INT8, op::DataType::DT_INT64, op::DataType::DT_INT64);
    TEST_PROMOTE(op::DataType::DT_INT8, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT16);
    TEST_PROMOTE(op::DataType::DT_INT8, op::DataType::DT_FLOAT, op::DataType::DT_FLOAT);
    TEST_PROMOTE(op::DataType::DT_INT8, op::DataType::DT_DOUBLE, op::DataType::DT_DOUBLE);
    TEST_PROMOTE(op::DataType::DT_INT8, op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX64);
    TEST_PROMOTE(op::DataType::DT_INT8, op::DataType::DT_COMPLEX128, op::DataType::DT_COMPLEX128);
    TEST_PROMOTE(op::DataType::DT_INT8, op::DataType::DT_BOOL, op::DataType::DT_INT8);
    TEST_PROMOTE(op::DataType::DT_INT8, op::DataType::DT_QINT8, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_INT8, op::DataType::DT_QINT16, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_INT8, op::DataType::DT_QINT32, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_INT8, op::DataType::DT_BF16, op::DataType::DT_BF16);

    TEST_PROMOTE(op::DataType::DT_INT16, op::DataType::DT_UINT8, op::DataType::DT_INT16);
    TEST_PROMOTE(op::DataType::DT_INT16, op::DataType::DT_INT8, op::DataType::DT_INT16);
    TEST_PROMOTE(op::DataType::DT_INT16, op::DataType::DT_INT16, op::DataType::DT_INT16);
    TEST_PROMOTE(op::DataType::DT_INT16, op::DataType::DT_INT32, op::DataType::DT_INT32);
    TEST_PROMOTE(op::DataType::DT_INT16, op::DataType::DT_INT64, op::DataType::DT_INT64);
    TEST_PROMOTE(op::DataType::DT_INT16, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT16);
    TEST_PROMOTE(op::DataType::DT_INT16, op::DataType::DT_FLOAT, op::DataType::DT_FLOAT);
    TEST_PROMOTE(op::DataType::DT_INT16, op::DataType::DT_DOUBLE, op::DataType::DT_DOUBLE);
    TEST_PROMOTE(op::DataType::DT_INT16, op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX64);
    TEST_PROMOTE(op::DataType::DT_INT16, op::DataType::DT_COMPLEX128, op::DataType::DT_COMPLEX128);
    TEST_PROMOTE(op::DataType::DT_INT16, op::DataType::DT_BOOL, op::DataType::DT_INT16);
    TEST_PROMOTE(op::DataType::DT_INT16, op::DataType::DT_QINT8, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_INT16, op::DataType::DT_QINT16, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_INT16, op::DataType::DT_QINT32, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_INT16, op::DataType::DT_BF16, op::DataType::DT_BF16);

    TEST_PROMOTE(op::DataType::DT_INT32, op::DataType::DT_UINT8, op::DataType::DT_INT32);
    TEST_PROMOTE(op::DataType::DT_INT32, op::DataType::DT_INT8, op::DataType::DT_INT32);
    TEST_PROMOTE(op::DataType::DT_INT32, op::DataType::DT_INT16, op::DataType::DT_INT32);
    TEST_PROMOTE(op::DataType::DT_INT32, op::DataType::DT_INT32, op::DataType::DT_INT32);
    TEST_PROMOTE(op::DataType::DT_INT32, op::DataType::DT_INT64, op::DataType::DT_INT64);
    TEST_PROMOTE(op::DataType::DT_INT32, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT16);
    TEST_PROMOTE(op::DataType::DT_INT32, op::DataType::DT_FLOAT, op::DataType::DT_FLOAT);
    TEST_PROMOTE(op::DataType::DT_INT32, op::DataType::DT_DOUBLE, op::DataType::DT_DOUBLE);
    TEST_PROMOTE(op::DataType::DT_INT32, op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX64);
    TEST_PROMOTE(op::DataType::DT_INT32, op::DataType::DT_COMPLEX128, op::DataType::DT_COMPLEX128);
    TEST_PROMOTE(op::DataType::DT_INT32, op::DataType::DT_BOOL, op::DataType::DT_INT32);
    TEST_PROMOTE(op::DataType::DT_INT32, op::DataType::DT_QINT8, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_INT32, op::DataType::DT_QINT16, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_INT32, op::DataType::DT_QINT32, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_INT32, op::DataType::DT_BF16, op::DataType::DT_BF16);

    TEST_PROMOTE(op::DataType::DT_INT64, op::DataType::DT_UINT8, op::DataType::DT_INT64);
    TEST_PROMOTE(op::DataType::DT_INT64, op::DataType::DT_INT8, op::DataType::DT_INT64);
    TEST_PROMOTE(op::DataType::DT_INT64, op::DataType::DT_INT16, op::DataType::DT_INT64);
    TEST_PROMOTE(op::DataType::DT_INT64, op::DataType::DT_INT32, op::DataType::DT_INT64);
    TEST_PROMOTE(op::DataType::DT_INT64, op::DataType::DT_INT64, op::DataType::DT_INT64);
    TEST_PROMOTE(op::DataType::DT_INT64, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT16);
    TEST_PROMOTE(op::DataType::DT_INT64, op::DataType::DT_FLOAT, op::DataType::DT_FLOAT);
    TEST_PROMOTE(op::DataType::DT_INT64, op::DataType::DT_DOUBLE, op::DataType::DT_DOUBLE);
    TEST_PROMOTE(op::DataType::DT_INT64, op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX64);
    TEST_PROMOTE(op::DataType::DT_INT64, op::DataType::DT_COMPLEX128, op::DataType::DT_COMPLEX128);
    TEST_PROMOTE(op::DataType::DT_INT64, op::DataType::DT_BOOL, op::DataType::DT_INT64);
    TEST_PROMOTE(op::DataType::DT_INT64, op::DataType::DT_QINT8, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_INT64, op::DataType::DT_QINT16, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_INT64, op::DataType::DT_QINT32, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_INT64, op::DataType::DT_BF16, op::DataType::DT_BF16);

    TEST_PROMOTE(op::DataType::DT_FLOAT16, op::DataType::DT_UINT8, op::DataType::DT_FLOAT16);
    TEST_PROMOTE(op::DataType::DT_FLOAT16, op::DataType::DT_INT8, op::DataType::DT_FLOAT16);
    TEST_PROMOTE(op::DataType::DT_FLOAT16, op::DataType::DT_INT16, op::DataType::DT_FLOAT16);
    TEST_PROMOTE(op::DataType::DT_FLOAT16, op::DataType::DT_INT32, op::DataType::DT_FLOAT16);
    TEST_PROMOTE(op::DataType::DT_FLOAT16, op::DataType::DT_INT64, op::DataType::DT_FLOAT16);
    TEST_PROMOTE(op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT16);
    TEST_PROMOTE(op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_FLOAT);
    TEST_PROMOTE(op::DataType::DT_FLOAT16, op::DataType::DT_DOUBLE, op::DataType::DT_DOUBLE);
    TEST_PROMOTE(op::DataType::DT_FLOAT16, op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX64);
    TEST_PROMOTE(op::DataType::DT_FLOAT16, op::DataType::DT_COMPLEX128, op::DataType::DT_COMPLEX128);
    TEST_PROMOTE(op::DataType::DT_FLOAT16, op::DataType::DT_BOOL, op::DataType::DT_FLOAT16);
    TEST_PROMOTE(op::DataType::DT_FLOAT16, op::DataType::DT_QINT8, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_FLOAT16, op::DataType::DT_QINT16, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_FLOAT16, op::DataType::DT_QINT32, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_FLOAT16, op::DataType::DT_BF16, op::DataType::DT_FLOAT);

    TEST_PROMOTE(op::DataType::DT_FLOAT, op::DataType::DT_UINT8, op::DataType::DT_FLOAT);
    TEST_PROMOTE(op::DataType::DT_FLOAT, op::DataType::DT_INT8, op::DataType::DT_FLOAT);
    TEST_PROMOTE(op::DataType::DT_FLOAT, op::DataType::DT_INT16, op::DataType::DT_FLOAT);
    TEST_PROMOTE(op::DataType::DT_FLOAT, op::DataType::DT_INT32, op::DataType::DT_FLOAT);
    TEST_PROMOTE(op::DataType::DT_FLOAT, op::DataType::DT_INT64, op::DataType::DT_FLOAT);
    TEST_PROMOTE(op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT);
    TEST_PROMOTE(op::DataType::DT_FLOAT, op::DataType::DT_FLOAT, op::DataType::DT_FLOAT);
    TEST_PROMOTE(op::DataType::DT_FLOAT, op::DataType::DT_DOUBLE, op::DataType::DT_DOUBLE);
    TEST_PROMOTE(op::DataType::DT_FLOAT, op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX64);
    TEST_PROMOTE(op::DataType::DT_FLOAT, op::DataType::DT_COMPLEX128, op::DataType::DT_COMPLEX128);
    TEST_PROMOTE(op::DataType::DT_FLOAT, op::DataType::DT_BOOL, op::DataType::DT_FLOAT);
    TEST_PROMOTE(op::DataType::DT_FLOAT, op::DataType::DT_QINT8, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_FLOAT, op::DataType::DT_QINT16, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_FLOAT, op::DataType::DT_QINT32, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_FLOAT, op::DataType::DT_BF16, op::DataType::DT_FLOAT);

    TEST_PROMOTE(op::DataType::DT_DOUBLE, op::DataType::DT_UINT8, op::DataType::DT_DOUBLE);
    TEST_PROMOTE(op::DataType::DT_DOUBLE, op::DataType::DT_INT8, op::DataType::DT_DOUBLE);
    TEST_PROMOTE(op::DataType::DT_DOUBLE, op::DataType::DT_INT16, op::DataType::DT_DOUBLE);
    TEST_PROMOTE(op::DataType::DT_DOUBLE, op::DataType::DT_INT32, op::DataType::DT_DOUBLE);
    TEST_PROMOTE(op::DataType::DT_DOUBLE, op::DataType::DT_INT64, op::DataType::DT_DOUBLE);
    TEST_PROMOTE(op::DataType::DT_DOUBLE, op::DataType::DT_FLOAT16, op::DataType::DT_DOUBLE);
    TEST_PROMOTE(op::DataType::DT_DOUBLE, op::DataType::DT_FLOAT, op::DataType::DT_DOUBLE);
    TEST_PROMOTE(op::DataType::DT_DOUBLE, op::DataType::DT_DOUBLE, op::DataType::DT_DOUBLE);
    TEST_PROMOTE(op::DataType::DT_DOUBLE, op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128);
    TEST_PROMOTE(op::DataType::DT_DOUBLE, op::DataType::DT_COMPLEX128, op::DataType::DT_COMPLEX128);
    TEST_PROMOTE(op::DataType::DT_DOUBLE, op::DataType::DT_BOOL, op::DataType::DT_DOUBLE);
    TEST_PROMOTE(op::DataType::DT_DOUBLE, op::DataType::DT_QINT8, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_DOUBLE, op::DataType::DT_QINT16, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_DOUBLE, op::DataType::DT_QINT32, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_DOUBLE, op::DataType::DT_BF16, op::DataType::DT_DOUBLE);

    TEST_PROMOTE(op::DataType::DT_COMPLEX64, op::DataType::DT_UINT8, op::DataType::DT_COMPLEX64);
    TEST_PROMOTE(op::DataType::DT_COMPLEX64, op::DataType::DT_INT8, op::DataType::DT_COMPLEX64);
    TEST_PROMOTE(op::DataType::DT_COMPLEX64, op::DataType::DT_INT16, op::DataType::DT_COMPLEX64);
    TEST_PROMOTE(op::DataType::DT_COMPLEX64, op::DataType::DT_INT32, op::DataType::DT_COMPLEX64);
    TEST_PROMOTE(op::DataType::DT_COMPLEX64, op::DataType::DT_INT64, op::DataType::DT_COMPLEX64);
    TEST_PROMOTE(op::DataType::DT_COMPLEX64, op::DataType::DT_FLOAT16, op::DataType::DT_COMPLEX64);
    TEST_PROMOTE(op::DataType::DT_COMPLEX64, op::DataType::DT_FLOAT, op::DataType::DT_COMPLEX64);
    TEST_PROMOTE(op::DataType::DT_COMPLEX64, op::DataType::DT_DOUBLE, op::DataType::DT_COMPLEX128);
    TEST_PROMOTE(op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX64);
    TEST_PROMOTE(op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128, op::DataType::DT_COMPLEX128);
    TEST_PROMOTE(op::DataType::DT_COMPLEX64, op::DataType::DT_BOOL, op::DataType::DT_COMPLEX64);
    TEST_PROMOTE(op::DataType::DT_COMPLEX64, op::DataType::DT_QINT8, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_COMPLEX64, op::DataType::DT_QINT16, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_COMPLEX64, op::DataType::DT_QINT32, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_COMPLEX64, op::DataType::DT_BF16, op::DataType::DT_COMPLEX64);

    TEST_PROMOTE(op::DataType::DT_COMPLEX128, op::DataType::DT_UINT8, op::DataType::DT_COMPLEX128);
    TEST_PROMOTE(op::DataType::DT_COMPLEX128, op::DataType::DT_INT8, op::DataType::DT_COMPLEX128);
    TEST_PROMOTE(op::DataType::DT_COMPLEX128, op::DataType::DT_INT16, op::DataType::DT_COMPLEX128);
    TEST_PROMOTE(op::DataType::DT_COMPLEX128, op::DataType::DT_INT32, op::DataType::DT_COMPLEX128);
    TEST_PROMOTE(op::DataType::DT_COMPLEX128, op::DataType::DT_INT64, op::DataType::DT_COMPLEX128);
    TEST_PROMOTE(op::DataType::DT_COMPLEX128, op::DataType::DT_FLOAT16, op::DataType::DT_COMPLEX128);
    TEST_PROMOTE(op::DataType::DT_COMPLEX128, op::DataType::DT_FLOAT, op::DataType::DT_COMPLEX128);
    TEST_PROMOTE(op::DataType::DT_COMPLEX128, op::DataType::DT_DOUBLE, op::DataType::DT_COMPLEX128);
    TEST_PROMOTE(op::DataType::DT_COMPLEX128, op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128);
    TEST_PROMOTE(op::DataType::DT_COMPLEX128, op::DataType::DT_COMPLEX128, op::DataType::DT_COMPLEX128);
    TEST_PROMOTE(op::DataType::DT_COMPLEX128, op::DataType::DT_BOOL, op::DataType::DT_COMPLEX128);
    TEST_PROMOTE(op::DataType::DT_COMPLEX128, op::DataType::DT_QINT8, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_COMPLEX128, op::DataType::DT_QINT16, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_COMPLEX128, op::DataType::DT_QINT32, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_COMPLEX128, op::DataType::DT_BF16, op::DataType::DT_COMPLEX128);

    TEST_PROMOTE(op::DataType::DT_BOOL, op::DataType::DT_UINT8, op::DataType::DT_UINT8);
    TEST_PROMOTE(op::DataType::DT_BOOL, op::DataType::DT_INT8, op::DataType::DT_INT8);
    TEST_PROMOTE(op::DataType::DT_BOOL, op::DataType::DT_INT16, op::DataType::DT_INT16);
    TEST_PROMOTE(op::DataType::DT_BOOL, op::DataType::DT_INT32, op::DataType::DT_INT32);
    TEST_PROMOTE(op::DataType::DT_BOOL, op::DataType::DT_INT64, op::DataType::DT_INT64);
    TEST_PROMOTE(op::DataType::DT_BOOL, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT16);
    TEST_PROMOTE(op::DataType::DT_BOOL, op::DataType::DT_FLOAT, op::DataType::DT_FLOAT);
    TEST_PROMOTE(op::DataType::DT_BOOL, op::DataType::DT_DOUBLE, op::DataType::DT_DOUBLE);
    TEST_PROMOTE(op::DataType::DT_BOOL, op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX64);
    TEST_PROMOTE(op::DataType::DT_BOOL, op::DataType::DT_COMPLEX128, op::DataType::DT_COMPLEX128);
    TEST_PROMOTE(op::DataType::DT_BOOL, op::DataType::DT_BOOL, op::DataType::DT_BOOL);
    TEST_PROMOTE(op::DataType::DT_BOOL, op::DataType::DT_QINT8, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_BOOL, op::DataType::DT_QINT16, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_BOOL, op::DataType::DT_QINT32, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_BOOL, op::DataType::DT_BF16, op::DataType::DT_BF16);

    TEST_PROMOTE(op::DataType::DT_QINT8, op::DataType::DT_UINT8, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT8, op::DataType::DT_INT8, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT8, op::DataType::DT_INT16, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT8, op::DataType::DT_INT32, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT8, op::DataType::DT_INT64, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT8, op::DataType::DT_FLOAT16, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT8, op::DataType::DT_FLOAT, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT8, op::DataType::DT_DOUBLE, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT8, op::DataType::DT_COMPLEX64, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT8, op::DataType::DT_COMPLEX128, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT8, op::DataType::DT_BOOL, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT8, op::DataType::DT_QINT8, op::DataType::DT_QINT8);
    TEST_PROMOTE(op::DataType::DT_QINT8, op::DataType::DT_QINT16, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT8, op::DataType::DT_QINT32, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT8, op::DataType::DT_BF16, op::DataType::DT_UNDEFINED);

    TEST_PROMOTE(op::DataType::DT_QINT16, op::DataType::DT_UINT8, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT16, op::DataType::DT_INT8, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT16, op::DataType::DT_INT16, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT16, op::DataType::DT_INT32, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT16, op::DataType::DT_INT64, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT16, op::DataType::DT_FLOAT16, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT16, op::DataType::DT_FLOAT, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT16, op::DataType::DT_DOUBLE, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT16, op::DataType::DT_COMPLEX64, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT16, op::DataType::DT_COMPLEX128, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT16, op::DataType::DT_BOOL, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT16, op::DataType::DT_QINT8, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT16, op::DataType::DT_QINT16, op::DataType::DT_QINT16);
    TEST_PROMOTE(op::DataType::DT_QINT16, op::DataType::DT_QINT32, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT16, op::DataType::DT_BF16, op::DataType::DT_UNDEFINED);

    TEST_PROMOTE(op::DataType::DT_QINT32, op::DataType::DT_UINT8, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT32, op::DataType::DT_INT8, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT32, op::DataType::DT_INT16, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT32, op::DataType::DT_INT32, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT32, op::DataType::DT_INT64, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT32, op::DataType::DT_FLOAT16, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT32, op::DataType::DT_FLOAT, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT32, op::DataType::DT_DOUBLE, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT32, op::DataType::DT_COMPLEX64, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT32, op::DataType::DT_COMPLEX128, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT32, op::DataType::DT_BOOL, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT32, op::DataType::DT_QINT8, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT32, op::DataType::DT_QINT16, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_QINT32, op::DataType::DT_QINT32, op::DataType::DT_QINT32);
    TEST_PROMOTE(op::DataType::DT_QINT16, op::DataType::DT_BF16, op::DataType::DT_UNDEFINED);

    TEST_PROMOTE(op::DataType::DT_BF16, op::DataType::DT_UINT8, op::DataType::DT_BF16);
    TEST_PROMOTE(op::DataType::DT_BF16, op::DataType::DT_INT8, op::DataType::DT_BF16);
    TEST_PROMOTE(op::DataType::DT_BF16, op::DataType::DT_INT16, op::DataType::DT_BF16);
    TEST_PROMOTE(op::DataType::DT_BF16, op::DataType::DT_INT32, op::DataType::DT_BF16);
    TEST_PROMOTE(op::DataType::DT_BF16, op::DataType::DT_INT64, op::DataType::DT_BF16);
    TEST_PROMOTE(op::DataType::DT_BF16, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT);
    TEST_PROMOTE(op::DataType::DT_BF16, op::DataType::DT_FLOAT, op::DataType::DT_FLOAT);
    TEST_PROMOTE(op::DataType::DT_BF16, op::DataType::DT_DOUBLE, op::DataType::DT_DOUBLE);
    TEST_PROMOTE(op::DataType::DT_BF16, op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX64);
    TEST_PROMOTE(op::DataType::DT_BF16, op::DataType::DT_COMPLEX128, op::DataType::DT_COMPLEX128);
    TEST_PROMOTE(op::DataType::DT_BF16, op::DataType::DT_BOOL, op::DataType::DT_BF16);
    TEST_PROMOTE(op::DataType::DT_BF16, op::DataType::DT_QINT8, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_BF16, op::DataType::DT_QINT16, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_BF16, op::DataType::DT_QINT32, op::DataType::DT_UNDEFINED);
    TEST_PROMOTE(op::DataType::DT_BF16, op::DataType::DT_BF16, op::DataType::DT_BF16);
}

TEST_F(TestDataTypeUtils, ToDataType) { EXPECT_EQ(op::ToDataType("DT_FLOAT16"), op::DataType::DT_FLOAT16); }

TEST_F(TestDataTypeUtils, ToString1)
{
    EXPECT_EQ(op::ToString({op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT}), "[DT_FLOAT16,DT_FLOAT,]");
    EXPECT_EQ(op::ToString({}), "[]");
}

TEST_F(TestDataTypeUtils, TypeSize)
{
    EXPECT_EQ(op::TypeSize(op::DataType::DT_MAX), sizeof(op::complex128));
    EXPECT_EQ(op::TypeSize(op::DataType::DT_UNDEFINED), sizeof(op::complex128));
}

TEST_F(TestDataTypeUtils, ShapeBtyes)
{
    int64_t shape = 17;
    int64_t fp16Btyes = op::CalcShapeBytes(shape, op::DataType::DT_FLOAT16);
    EXPECT_EQ(fp16Btyes, 34);

    int64_t int4Btyes = op::CalcShapeBytes(shape, op::DataType::DT_INT4);
    EXPECT_EQ(int4Btyes, 9);

    int64_t int4BtyesCeil = op::CalcShapeBytes(shape, op::DataType::DT_INT4, true);
    EXPECT_EQ(int4BtyesCeil, 8);

    shape = std::numeric_limits<int64_t>::max();
    fp16Btyes = op::CalcShapeBytes(shape, op::DataType::DT_FLOAT16);
    EXPECT_EQ(fp16Btyes, -2);

    int4Btyes = op::CalcShapeBytes(shape, op::DataType::DT_INT4);
    EXPECT_EQ(int4Btyes, 0);
}
