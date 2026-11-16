/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include <vector>
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_executor.h"

using namespace op;
using namespace std;

class CommonTypesTest : public testing::Test {
public:
    bool ShapeEq(const op::Shape& shape, const vector<int64_t>& value)
    {
        if (shape.GetDimNum() != value.size()) {
            return false;
        }

        for (size_t i = 0; i < value.size(); i++) {
            if (shape.GetDim(i) != value[i]) {
                return false;
            }
        }

        return true;
    }
};

TEST_F(CommonTypesTest, TestAclStorage)
{
    aclStorage storage;
    storage.SetAddr(nullptr);
    EXPECT_EQ(storage.GetAddr(), nullptr);
    char data[] = {1, 2, 3};
    storage.SetAddr(data);
    EXPECT_EQ(storage.GetAddr(), static_cast<void*>(data));
}

#define CHECK_SCALAR(scalarName, value, valueType, dataType)                               \
    valueType scalarName##valueType = value;                                               \
    aclScalar scalarName(&scalarName##valueType, dataType);                                \
    scalarName.ToString();                                                                 \
    ASSERT_NE(scalarName.GetData(), nullptr);                                              \
    ASSERT_EQ(scalarName.Size(), sizeof(valueType));                                       \
    EXPECT_EQ(memcmp(scalarName.GetData(), &scalarName##valueType, sizeof(valueType)), 0); \
    EXPECT_EQ(scalarName.GetDataType(), dataType)

TEST_F(CommonTypesTest, TestAclScalar)
{
    CHECK_SCALAR(fpScalar, 3.2, float, DataType::DT_FLOAT);
    CHECK_SCALAR(fp16Scalar, 3.2, fp16_t, DataType::DT_FLOAT16);
    CHECK_SCALAR(bf16Scalar, 3.2, bfloat16, DataType::DT_BF16);
    CHECK_SCALAR(int8Scalar, 3, int8_t, DataType::DT_INT8);
    CHECK_SCALAR(int16Scalar, 3, int16_t, DataType::DT_INT16);
    CHECK_SCALAR(uint16Scalar, 3, uint16_t, DataType::DT_UINT16);
    CHECK_SCALAR(uint8Scalar, 3, uint8_t, DataType::DT_UINT8);
    CHECK_SCALAR(int32Scalar, 3, int32_t, DataType::DT_INT32);
    CHECK_SCALAR(int64Scalar, 3, int64_t, DataType::DT_INT64);
    CHECK_SCALAR(uint32Scalar, 3, uint32_t, DataType::DT_UINT32);
    CHECK_SCALAR(uint64Scalar, 3, uint64_t, DataType::DT_UINT64);
    CHECK_SCALAR(boolScalar, 1, bool, DataType::DT_BOOL);
    CHECK_SCALAR(doubleScalar, 3, double, DataType::DT_DOUBLE);
    using complex64 = std::complex<float>;
    using complex128 = std::complex<double>;
    CHECK_SCALAR(complex64Scalar, complex64(3.2, 3.2), complex64, DataType::DT_COMPLEX64);
    CHECK_SCALAR(complex128Scalar, complex128(3.2, 3.2), complex128, DataType::DT_COMPLEX128);
}

#define CHECK_SCALAR2TYPES(scalar, value, valueType, dataType)                                   \
    valueType scalar##valueType = value;                                                         \
    aclScalar scalar(&scalar##valueType, dataType);                                              \
    EXPECT_FLOAT_EQ(static_cast<float>(scalar##valueType), scalar.ToFloat());                    \
    EXPECT_DOUBLE_EQ(static_cast<double>(scalar##valueType), scalar.ToDouble());                 \
    EXPECT_EQ(static_cast<bool>(scalar##valueType), scalar.ToBool());                            \
    EXPECT_EQ(static_cast<int8_t>(scalar##valueType), scalar.ToInt8());                          \
    EXPECT_EQ(static_cast<int16_t>(scalar##valueType), scalar.ToInt16());                        \
    EXPECT_EQ(static_cast<int32_t>(scalar##valueType), scalar.ToInt32());                        \
    EXPECT_EQ(static_cast<int64_t>(scalar##valueType), scalar.ToInt64());                        \
    EXPECT_EQ(static_cast<uint8_t>(scalar##valueType), scalar.ToUint8());                        \
    EXPECT_EQ(static_cast<uint16_t>(scalar##valueType), scalar.ToUint16());                      \
    EXPECT_EQ(static_cast<uint32_t>(scalar##valueType), scalar.ToUint32());                      \
    EXPECT_EQ(static_cast<uint64_t>(scalar##valueType), scalar.ToUint64());                      \
    if (dataType == DataType::DT_UINT16) {                                                       \
        EXPECT_EQ(op::fp16_t(static_cast<float>(scalar##valueType)).val, scalar.ToFp16().val);   \
    } else {                                                                                     \
        EXPECT_EQ(op::fp16_t(scalar##valueType).val, scalar.ToFp16().val);                       \
    }                                                                                            \
    EXPECT_EQ(op::bfloat16(scalar##valueType), scalar.ToBf16());                                 \
    EXPECT_EQ(std::complex<float>(static_cast<float>(scalar##valueType)), scalar.ToComplex64()); \
    EXPECT_EQ(std::complex<double>(static_cast<double>(scalar##valueType)), scalar.ToComplex128())

TEST_F(CommonTypesTest, TestAclScalar_ToTypes)
{
    CHECK_SCALAR2TYPES(floatScalar, 3.2, float, DataType::DT_FLOAT);
    CHECK_SCALAR2TYPES(doubleScalar, 3.2, double, DataType::DT_DOUBLE);
    CHECK_SCALAR2TYPES(int8Scalar, -3, int8_t, DataType::DT_INT8);
    CHECK_SCALAR2TYPES(int16Scalar, -3, int16_t, DataType::DT_INT16);
    CHECK_SCALAR2TYPES(int32Scalar, -3, int32_t, DataType::DT_INT32);
    CHECK_SCALAR2TYPES(int64Scalar, -3, int64_t, DataType::DT_INT64);
    CHECK_SCALAR2TYPES(uint8Scalar, 3, uint8_t, DataType::DT_UINT8);
    CHECK_SCALAR2TYPES(uint16Scalar, 3, uint16_t, DataType::DT_UINT16);
    CHECK_SCALAR2TYPES(uint32Scalar, 3, uint32_t, DataType::DT_UINT32);
    CHECK_SCALAR2TYPES(uint64Scalar, 3, uint64_t, DataType::DT_UINT64);
    CHECK_SCALAR2TYPES(float16Scalar, 3.2, fp16_t, DataType::DT_FLOAT16);
    CHECK_SCALAR2TYPES(bf16Scalar, 3.2, bfloat16, DataType::DT_BF16);
    using complex64 = std::complex<float>;
    using complex128 = std::complex<double>;
    complex64 complex64Value = {3.2, 3.2};
    aclScalar complex64Scalar(&complex64Value, DataType::DT_COMPLEX64);
    EXPECT_EQ(complex64Scalar.ToComplex64(), complex64Value);
    EXPECT_EQ(complex64Scalar.ToComplex128(), static_cast<complex128>(complex64Value));

    complex128 complex128Value = {3.2, 3.2};
    aclScalar complex128Scalar(&complex128Value, DataType::DT_COMPLEX128);
    EXPECT_EQ(complex128Scalar.ToComplex64(), static_cast<complex64>(complex128Value));
    EXPECT_EQ(complex128Scalar.ToComplex128(), complex128Value);
}

TEST_F(CommonTypesTest, TestAclScalarCheckOverflowsFloatToOther)
{
    constexpr float floatValueMaxBool = 1;
    aclScalar floatScalarMaxBool(&floatValueMaxBool, op::DataType::DT_FLOAT);
    EXPECT_FALSE(floatScalarMaxBool.CheckOverflows<bool>());
    constexpr float floatValueMinBool = 0;
    aclScalar floatScalarMinBool(&floatValueMinBool, op::DataType::DT_FLOAT);
    EXPECT_FALSE(floatScalarMinBool.CheckOverflows<bool>());
    constexpr float floatValueMaxBoolP = floatValueMaxBool + 1;
    aclScalar floatScalarMaxBoolP(&floatValueMaxBoolP, op::DataType::DT_FLOAT);
    EXPECT_TRUE(floatScalarMaxBoolP.CheckOverflows<bool>());
    constexpr float floatValueMinBoolP = floatValueMinBool - 1;
    aclScalar floatScalarMinBoolP(&floatValueMinBoolP, op::DataType::DT_FLOAT);
    EXPECT_TRUE(floatScalarMinBoolP.CheckOverflows<bool>());

    constexpr float floatValueMaxInt8 = std::numeric_limits<int8_t>::max();
    aclScalar floatScalarMaxInt8(&floatValueMaxInt8, op::DataType::DT_FLOAT);
    EXPECT_FALSE(floatScalarMaxInt8.CheckOverflows<int8_t>());
    constexpr float floatValueMinInt8 = std::numeric_limits<int8_t>::lowest();
    aclScalar floatScalarMinInt8(&floatValueMinInt8, op::DataType::DT_FLOAT);
    EXPECT_FALSE(floatScalarMinInt8.CheckOverflows<int8_t>());
    constexpr float floatValueMaxInt8P = floatValueMaxInt8 + 1;
    aclScalar floatScalarMaxInt8P(&floatValueMaxInt8P, op::DataType::DT_FLOAT);
    EXPECT_TRUE(floatScalarMaxInt8P.CheckOverflows<int8_t>());
    constexpr float floatValueMinInt8P = floatValueMinInt8 - 1;
    aclScalar floatScalarMinInt8P(&floatValueMinInt8P, op::DataType::DT_FLOAT);
    EXPECT_TRUE(floatScalarMinInt8P.CheckOverflows<int8_t>());

    constexpr float floatValueMaxInt16 = std::numeric_limits<int16_t>::max();
    aclScalar floatScalarMaxInt16(&floatValueMaxInt16, op::DataType::DT_FLOAT);
    EXPECT_FALSE(floatScalarMaxInt16.CheckOverflows<int16_t>());
    constexpr float floatValueMinInt16 = std::numeric_limits<int16_t>::lowest();
    aclScalar floatScalarMinInt16(&floatValueMinInt16, op::DataType::DT_FLOAT);
    EXPECT_FALSE(floatScalarMinInt16.CheckOverflows<int16_t>());
    constexpr float floatValueMaxInt16P = floatValueMaxInt16 + 1;
    aclScalar floatScalarMaxInt16P(&floatValueMaxInt16P, op::DataType::DT_FLOAT);
    EXPECT_TRUE(floatScalarMaxInt16P.CheckOverflows<int16_t>());
    constexpr float floatValueMinInt16P = floatValueMinInt16 - 1;
    aclScalar floatScalarMinInt16P(&floatValueMinInt16P, op::DataType::DT_FLOAT);
    EXPECT_TRUE(floatScalarMinInt16P.CheckOverflows<int16_t>());

    constexpr float floatValueMaxInt32 = std::numeric_limits<int32_t>::max();
    aclScalar floatScalarMaxInt32(&floatValueMaxInt32, op::DataType::DT_FLOAT);
    EXPECT_FALSE(floatScalarMaxInt32.CheckOverflows<int32_t>());
    constexpr float floatValueMinInt32 = std::numeric_limits<int32_t>::min();
    aclScalar floatScalarMinInt32(&floatValueMinInt32, op::DataType::DT_FLOAT);
    EXPECT_FALSE(floatScalarMinInt32.CheckOverflows<int32_t>());
    constexpr float floatValueMaxInt32P = std::numeric_limits<float>::max();
    aclScalar floatScalarMaxInt32P(&floatValueMaxInt32P, op::DataType::DT_FLOAT);
    EXPECT_TRUE(floatScalarMaxInt32P.CheckOverflows<int32_t>());
    constexpr float floatValueMinInt32P = std::numeric_limits<float>::lowest();
    aclScalar floatScalarMinInt32P(&floatValueMinInt32P, op::DataType::DT_FLOAT);
    EXPECT_TRUE(floatScalarMinInt32P.CheckOverflows<int32_t>());

    constexpr float floatValueMaxInt64 = std::numeric_limits<int64_t>::max();
    aclScalar floatScalarMaxInt64(&floatValueMaxInt64, op::DataType::DT_FLOAT);
    EXPECT_FALSE(floatScalarMaxInt64.CheckOverflows<int64_t>());
    constexpr float floatValueMinInt64 = std::numeric_limits<int64_t>::min();
    aclScalar floatScalarMinInt64(&floatValueMinInt64, op::DataType::DT_FLOAT);
    EXPECT_FALSE(floatScalarMinInt64.CheckOverflows<int64_t>());
    constexpr float floatValueMaxInt64P = std::numeric_limits<float>::max();
    aclScalar floatScalarMaxInt64P(&floatValueMaxInt64P, op::DataType::DT_FLOAT);
    EXPECT_TRUE(floatScalarMaxInt64P.CheckOverflows<int64_t>());
    constexpr float floatValueMinInt64P = std::numeric_limits<float>::lowest();
    aclScalar floatScalarMinInt64P(&floatValueMinInt64P, op::DataType::DT_FLOAT);
    EXPECT_TRUE(floatScalarMinInt64P.CheckOverflows<int64_t>());

    constexpr float floatValueMaxUint8 = std::numeric_limits<uint8_t>::max();
    aclScalar floatScalarMaxUint8(&floatValueMaxUint8, op::DataType::DT_FLOAT);
    EXPECT_FALSE(floatScalarMaxUint8.CheckOverflows<uint8_t>());
    constexpr float floatValueMinUint8 = std::numeric_limits<uint8_t>::lowest();
    aclScalar floatScalarMinUint8(&floatValueMinUint8, op::DataType::DT_FLOAT);
    EXPECT_FALSE(floatScalarMinUint8.CheckOverflows<uint8_t>());
    constexpr float floatValueMaxUint8P = floatValueMaxUint8 + 1;
    aclScalar floatScalarMaxUint8P(&floatValueMaxUint8P, op::DataType::DT_FLOAT);
    EXPECT_TRUE(floatScalarMaxUint8P.CheckOverflows<uint8_t>());
    constexpr float floatValueMinUint8P = floatValueMinUint8 - 1;
    aclScalar floatScalarMinUint8P(&floatValueMinUint8P, op::DataType::DT_FLOAT);
    EXPECT_TRUE(floatScalarMinUint8P.CheckOverflows<uint8_t>());

    constexpr float floatValueMaxUint16 = std::numeric_limits<uint16_t>::max();
    aclScalar floatScalarMaxUint16(&floatValueMaxUint16, op::DataType::DT_FLOAT);
    EXPECT_FALSE(floatScalarMaxUint16.CheckOverflows<uint16_t>());
    constexpr float floatValueMinUint16 = std::numeric_limits<uint16_t>::lowest();
    aclScalar floatScalarMinUint16(&floatValueMinUint16, op::DataType::DT_FLOAT);
    EXPECT_FALSE(floatScalarMinUint16.CheckOverflows<uint16_t>());
    constexpr float floatValueMaxUint16P = floatValueMaxUint16 + 1;
    aclScalar floatScalarMaxUint16P(&floatValueMaxUint16P, op::DataType::DT_FLOAT);
    EXPECT_TRUE(floatScalarMaxUint16P.CheckOverflows<uint16_t>());
    constexpr float floatValueMinUint16P = floatValueMinUint16 - 1;
    aclScalar floatScalarMinUint16P(&floatValueMinUint16P, op::DataType::DT_FLOAT);
    EXPECT_TRUE(floatScalarMinUint16P.CheckOverflows<uint16_t>());

    constexpr float floatValueMaxUint32 = std::numeric_limits<uint32_t>::max();
    aclScalar floatScalarMaxUint32(&floatValueMaxUint32, op::DataType::DT_FLOAT);
    EXPECT_FALSE(floatScalarMaxUint32.CheckOverflows<uint32_t>());
    constexpr float floatValueMinUint32 = std::numeric_limits<uint32_t>::min();
    aclScalar floatScalarMinUint32(&floatValueMinUint32, op::DataType::DT_FLOAT);
    EXPECT_FALSE(floatScalarMinUint32.CheckOverflows<uint32_t>());
    constexpr float floatValueMaxUint32P = std::numeric_limits<float>::max();
    aclScalar floatScalarMaxUint32P(&floatValueMaxUint32P, op::DataType::DT_FLOAT);
    EXPECT_TRUE(floatScalarMaxUint32P.CheckOverflows<uint32_t>());
    constexpr float floatValueMinUint32P = std::numeric_limits<float>::lowest();
    aclScalar floatScalarMinUint32P(&floatValueMinUint32P, op::DataType::DT_FLOAT);
    EXPECT_TRUE(floatScalarMinUint32P.CheckOverflows<uint32_t>());

    constexpr float floatValueMaxUint64 = std::numeric_limits<uint64_t>::max();
    aclScalar floatScalarMaxUint64(&floatValueMaxUint64, op::DataType::DT_FLOAT);
    EXPECT_FALSE(floatScalarMaxUint64.CheckOverflows<uint64_t>());
    constexpr float floatValueMinUint64 = std::numeric_limits<uint64_t>::min();
    aclScalar floatScalarMinUint64(&floatValueMinUint64, op::DataType::DT_FLOAT);
    EXPECT_FALSE(floatScalarMinUint64.CheckOverflows<uint64_t>());
    constexpr float floatValueMaxUint64P = std::numeric_limits<float>::max();
    aclScalar floatScalarMaxUint64P(&floatValueMaxUint64P, op::DataType::DT_FLOAT);
    EXPECT_TRUE(floatScalarMaxUint64P.CheckOverflows<uint64_t>());
    constexpr float floatValueMinUint64P = std::numeric_limits<float>::lowest();
    aclScalar floatScalarMinUint64P(&floatValueMinUint64P, op::DataType::DT_FLOAT);
    EXPECT_TRUE(floatScalarMinUint64P.CheckOverflows<uint64_t>());

    float floatValueMaxFp16 = std::numeric_limits<fp16_t>::max();
    aclScalar floatScalarMaxFp16(&floatValueMaxFp16, op::DataType::DT_FLOAT);
    EXPECT_FALSE(floatScalarMaxFp16.CheckOverflows<fp16_t>());
    float floatValueMinFp16 = std::numeric_limits<fp16_t>::lowest();
    aclScalar floatScalarMinFp16(&floatValueMinFp16, op::DataType::DT_FLOAT);
    EXPECT_FALSE(floatScalarMinFp16.CheckOverflows<fp16_t>());
    float floatValueMaxFp16P = floatValueMaxFp16 + 1;
    aclScalar floatScalarMaxFp16P(&floatValueMaxFp16P, op::DataType::DT_FLOAT);
    EXPECT_TRUE(floatScalarMaxFp16P.CheckOverflows<fp16_t>());
    float floatValueMinFp16P = floatValueMinFp16 - 1;
    aclScalar floatScalarMinFp16P(&floatValueMinFp16P, op::DataType::DT_FLOAT);
    EXPECT_TRUE(floatScalarMinFp16P.CheckOverflows<fp16_t>());

    float floatValueMaxBf16 = std::numeric_limits<bfloat16>::max();
    aclScalar floatScalarMaxBf16(&floatValueMaxBf16, op::DataType::DT_FLOAT);
    EXPECT_FALSE(floatScalarMaxBf16.CheckOverflows<bfloat16>());
    float floatValueMinBf16 = std::numeric_limits<bfloat16>::lowest();
    aclScalar floatScalarMinBf16(&floatValueMinBf16, op::DataType::DT_FLOAT);
    EXPECT_FALSE(floatScalarMinBf16.CheckOverflows<bfloat16>());
    float floatValueMaxBf16P = floatValueMaxBf16 + 10;
    aclScalar floatScalarMaxBf16P(&floatValueMaxBf16P, op::DataType::DT_FLOAT);
    EXPECT_FALSE(floatScalarMaxBf16P.CheckOverflows<bfloat16>());
    float floatValueMinBf16P = floatValueMinBf16 - 1;
    aclScalar floatScalarMinBf16P(&floatValueMinBf16P, op::DataType::DT_FLOAT);
    EXPECT_FALSE(floatScalarMinBf16P.CheckOverflows<bfloat16>());
    float floatValueMaxBf16P1 = std::numeric_limits<float>::max();
    aclScalar floatScalarMaxBf16P1(&floatValueMaxBf16P1, op::DataType::DT_FLOAT);
    EXPECT_FALSE(floatScalarMaxBf16P.CheckOverflows<bfloat16>());
    float floatValueMinBf16P1 = std::numeric_limits<float>::lowest();
    aclScalar floatScalarMinBf16P1(&floatValueMinBf16P1, op::DataType::DT_FLOAT);
    EXPECT_FALSE(floatScalarMinBf16P.CheckOverflows<bfloat16>());

    float floatValueMaxDouble = std::numeric_limits<float>::max();
    aclScalar floatScalarMaxDouble(&floatValueMaxDouble, op::DataType::DT_FLOAT);
    EXPECT_FALSE(floatScalarMaxDouble.CheckOverflows<double>());
    float floatValueMinDouble = std::numeric_limits<float>::lowest();
    aclScalar floatScalarMinDouble(&floatValueMinDouble, op::DataType::DT_FLOAT);
    EXPECT_FALSE(floatScalarMinDouble.CheckOverflows<double>());
    float floatValueMaxDoubleP = floatValueMaxDouble + 1;
    aclScalar floatScalarMaxDoubleP(&floatValueMaxDoubleP, op::DataType::DT_FLOAT);
    EXPECT_FALSE(floatScalarMaxDoubleP.CheckOverflows<double>());
    float floatValueMinDoubleP = floatValueMinDouble - 1;
    aclScalar floatScalarMinDoubleP(&floatValueMinDoubleP, op::DataType::DT_FLOAT);
    EXPECT_FALSE(floatScalarMinDoubleP.CheckOverflows<double>());

    float floatValueMaxFloat = std::numeric_limits<float>::max();
    aclScalar floatScalarMaxFloat(&floatValueMaxFloat, op::DataType::DT_FLOAT);
    EXPECT_FALSE(floatScalarMaxFloat.CheckOverflows<double>());
    float floatValueMinFloat = std::numeric_limits<float>::lowest();
    aclScalar floatScalarMinFloat(&floatValueMinFloat, op::DataType::DT_FLOAT);
    EXPECT_FALSE(floatScalarMinFloat.CheckOverflows<double>());
    float floatValueMaxFloatP = floatValueMaxFloat + 1;
    aclScalar floatScalarMaxFloatP(&floatValueMaxFloatP, op::DataType::DT_FLOAT);
    EXPECT_FALSE(floatScalarMaxFloatP.CheckOverflows<double>());
    float floatValueMinFloatP = floatValueMinFloat - 1;
    aclScalar floatScalarMinFloatP(&floatValueMinFloatP, op::DataType::DT_FLOAT);
    EXPECT_FALSE(floatScalarMinFloatP.CheckOverflows<double>());
}

TEST_F(CommonTypesTest, TestAclScalarCheckOverflowsDoubleToOther)
{
    constexpr double doubleValueMaxBool = 1;
    aclScalar doubleScalarMaxBool(&doubleValueMaxBool, op::DataType::DT_DOUBLE);
    EXPECT_FALSE(doubleScalarMaxBool.CheckOverflows<bool>());
    constexpr double doubleValueMinBool = 0;
    aclScalar doubleScalarMinBool(&doubleValueMinBool, op::DataType::DT_DOUBLE);
    EXPECT_FALSE(doubleScalarMinBool.CheckOverflows<bool>());
    constexpr double doubleValueMaxBoolP = doubleValueMaxBool + 1;
    aclScalar doubleScalarMaxBoolP(&doubleValueMaxBoolP, op::DataType::DT_DOUBLE);
    EXPECT_TRUE(doubleScalarMaxBoolP.CheckOverflows<bool>());
    constexpr double doubleValueMinBoolP = doubleValueMinBool - 1;
    aclScalar doubleScalarMinBoolP(&doubleValueMinBoolP, op::DataType::DT_DOUBLE);
    EXPECT_TRUE(doubleScalarMinBoolP.CheckOverflows<bool>());

    constexpr double doubleValueMaxInt8 = std::numeric_limits<int8_t>::max();
    aclScalar doubleScalarMaxInt8(&doubleValueMaxInt8, op::DataType::DT_DOUBLE);
    EXPECT_FALSE(doubleScalarMaxInt8.CheckOverflows<int8_t>());
    constexpr double doubleValueMinInt8 = std::numeric_limits<int8_t>::lowest();
    aclScalar doubleScalarMinInt8(&doubleValueMinInt8, op::DataType::DT_DOUBLE);
    EXPECT_FALSE(doubleScalarMinInt8.CheckOverflows<int8_t>());
    constexpr double doubleValueMaxInt8P = doubleValueMaxInt8 + 1;
    aclScalar doubleScalarMaxInt8P(&doubleValueMaxInt8P, op::DataType::DT_DOUBLE);
    EXPECT_TRUE(doubleScalarMaxInt8P.CheckOverflows<int8_t>());
    constexpr double doubleValueMinInt8P = doubleValueMinInt8 - 1;
    aclScalar doubleScalarMinInt8P(&doubleValueMinInt8P, op::DataType::DT_DOUBLE);
    EXPECT_TRUE(doubleScalarMinInt8P.CheckOverflows<int8_t>());

    constexpr double doubleValueMaxInt16 = std::numeric_limits<int16_t>::max();
    aclScalar doubleScalarMaxInt16(&doubleValueMaxInt16, op::DataType::DT_DOUBLE);
    EXPECT_FALSE(doubleScalarMaxInt16.CheckOverflows<int16_t>());
    constexpr double doubleValueMinInt16 = std::numeric_limits<int16_t>::lowest();
    aclScalar doubleScalarMinInt16(&doubleValueMinInt16, op::DataType::DT_DOUBLE);
    EXPECT_FALSE(doubleScalarMinInt16.CheckOverflows<int16_t>());
    constexpr double doubleValueMaxInt16P = doubleValueMaxInt16 + 1;
    aclScalar doubleScalarMaxInt16P(&doubleValueMaxInt16P, op::DataType::DT_DOUBLE);
    EXPECT_TRUE(doubleScalarMaxInt16P.CheckOverflows<int16_t>());
    constexpr double doubleValueMinInt16P = doubleValueMinInt16 - 1;
    aclScalar doubleScalarMinInt16P(&doubleValueMinInt16P, op::DataType::DT_DOUBLE);
    EXPECT_TRUE(doubleScalarMinInt16P.CheckOverflows<int16_t>());

    constexpr double doubleValueMaxInt32 = std::numeric_limits<int32_t>::max();
    aclScalar doubleScalarMaxInt32(&doubleValueMaxInt32, op::DataType::DT_DOUBLE);
    EXPECT_FALSE(doubleScalarMaxInt32.CheckOverflows<int32_t>());
    constexpr double doubleValueMinInt32 = std::numeric_limits<int32_t>::min();
    aclScalar doubleScalarMinInt32(&doubleValueMinInt32, op::DataType::DT_DOUBLE);
    EXPECT_FALSE(doubleScalarMinInt32.CheckOverflows<int32_t>());
    constexpr double doubleValueMaxInt32P = std::numeric_limits<float>::max();
    aclScalar doubleScalarMaxInt32P(&doubleValueMaxInt32P, op::DataType::DT_DOUBLE);
    EXPECT_TRUE(doubleScalarMaxInt32P.CheckOverflows<int32_t>());
    constexpr double doubleValueMinInt32P = std::numeric_limits<float>::lowest();
    aclScalar doubleScalarMinInt32P(&doubleValueMinInt32P, op::DataType::DT_DOUBLE);
    EXPECT_TRUE(doubleScalarMinInt32P.CheckOverflows<int32_t>());

    constexpr double doubleValueMaxInt64 = std::numeric_limits<int64_t>::max();
    aclScalar doubleScalarMaxInt64(&doubleValueMaxInt64, op::DataType::DT_DOUBLE);
    EXPECT_FALSE(doubleScalarMaxInt64.CheckOverflows<int64_t>());
    constexpr double doubleValueMinInt64 = std::numeric_limits<int64_t>::min();
    aclScalar doubleScalarMinInt64(&doubleValueMinInt64, op::DataType::DT_DOUBLE);
    EXPECT_FALSE(doubleScalarMinInt64.CheckOverflows<int64_t>());
    constexpr double doubleValueMaxInt64P = std::numeric_limits<float>::max();
    aclScalar doubleScalarMaxInt64P(&doubleValueMaxInt64P, op::DataType::DT_DOUBLE);
    EXPECT_TRUE(doubleScalarMaxInt64P.CheckOverflows<int64_t>());
    constexpr double doubleValueMinInt64P = std::numeric_limits<float>::lowest();
    aclScalar doubleScalarMinInt64P(&doubleValueMinInt64P, op::DataType::DT_DOUBLE);
    EXPECT_TRUE(doubleScalarMinInt64P.CheckOverflows<int64_t>());

    constexpr double doubleValueMaxUint8 = std::numeric_limits<uint8_t>::max();
    aclScalar doubleScalarMaxUint8(&doubleValueMaxUint8, op::DataType::DT_DOUBLE);
    EXPECT_FALSE(doubleScalarMaxUint8.CheckOverflows<uint8_t>());
    constexpr double doubleValueMinUint8 = std::numeric_limits<uint8_t>::lowest();
    aclScalar doubleScalarMinUint8(&doubleValueMinUint8, op::DataType::DT_DOUBLE);
    EXPECT_FALSE(doubleScalarMinUint8.CheckOverflows<uint8_t>());
    constexpr double doubleValueMaxUint8P = doubleValueMaxUint8 + 1;
    aclScalar doubleScalarMaxUint8P(&doubleValueMaxUint8P, op::DataType::DT_DOUBLE);
    EXPECT_TRUE(doubleScalarMaxUint8P.CheckOverflows<uint8_t>());
    constexpr double doubleValueMinUint8P = doubleValueMinUint8 - 1;
    aclScalar doubleScalarMinUint8P(&doubleValueMinUint8P, op::DataType::DT_DOUBLE);
    EXPECT_TRUE(doubleScalarMinUint8P.CheckOverflows<uint8_t>());

    constexpr double doubleValueMaxUint16 = std::numeric_limits<uint16_t>::max();
    aclScalar doubleScalarMaxUint16(&doubleValueMaxUint16, op::DataType::DT_DOUBLE);
    EXPECT_FALSE(doubleScalarMaxUint16.CheckOverflows<uint16_t>());
    constexpr double doubleValueMinUint16 = std::numeric_limits<uint16_t>::lowest();
    aclScalar doubleScalarMinUint16(&doubleValueMinUint16, op::DataType::DT_DOUBLE);
    EXPECT_FALSE(doubleScalarMinUint16.CheckOverflows<uint16_t>());
    constexpr double doubleValueMaxUint16P = doubleValueMaxUint16 + 1;
    aclScalar doubleScalarMaxUint16P(&doubleValueMaxUint16P, op::DataType::DT_DOUBLE);
    EXPECT_TRUE(doubleScalarMaxUint16P.CheckOverflows<uint16_t>());
    constexpr double doubleValueMinUint16P = doubleValueMinUint16 - 1;
    aclScalar doubleScalarMinUint16P(&doubleValueMinUint16P, op::DataType::DT_DOUBLE);
    EXPECT_TRUE(doubleScalarMinUint16P.CheckOverflows<uint16_t>());

    constexpr double doubleValueMaxUint32 = std::numeric_limits<uint32_t>::max();
    aclScalar doubleScalarMaxUint32(&doubleValueMaxUint32, op::DataType::DT_DOUBLE);
    EXPECT_FALSE(doubleScalarMaxUint32.CheckOverflows<uint32_t>());
    constexpr double doubleValueMinUint32 = std::numeric_limits<uint32_t>::min();
    aclScalar doubleScalarMinUint32(&doubleValueMinUint32, op::DataType::DT_DOUBLE);
    EXPECT_FALSE(doubleScalarMinUint32.CheckOverflows<uint32_t>());
    constexpr double doubleValueMaxUint32P = std::numeric_limits<float>::max();
    aclScalar doubleScalarMaxUint32P(&doubleValueMaxUint32P, op::DataType::DT_DOUBLE);
    EXPECT_TRUE(doubleScalarMaxUint32P.CheckOverflows<uint32_t>());
    constexpr double doubleValueMinUint32P = std::numeric_limits<float>::lowest();
    aclScalar doubleScalarMinUint32P(&doubleValueMinUint32P, op::DataType::DT_DOUBLE);
    EXPECT_TRUE(doubleScalarMinUint32P.CheckOverflows<uint32_t>());

    constexpr double doubleValueMaxUint64 = std::numeric_limits<uint64_t>::max();
    aclScalar doubleScalarMaxUint64(&doubleValueMaxUint64, op::DataType::DT_DOUBLE);
    EXPECT_FALSE(doubleScalarMaxUint64.CheckOverflows<uint64_t>());
    constexpr double doubleValueMinUint64 = std::numeric_limits<uint64_t>::min();
    aclScalar doubleScalarMinUint64(&doubleValueMinUint64, op::DataType::DT_DOUBLE);
    EXPECT_FALSE(doubleScalarMinUint64.CheckOverflows<uint64_t>());
    constexpr double doubleValueMaxUint64P = std::numeric_limits<float>::max();
    aclScalar doubleScalarMaxUint64P(&doubleValueMaxUint64P, op::DataType::DT_DOUBLE);
    EXPECT_TRUE(doubleScalarMaxUint64P.CheckOverflows<uint64_t>());
    constexpr double doubleValueMinUint64P = std::numeric_limits<float>::lowest();
    aclScalar doubleScalarMinUint64P(&doubleValueMinUint64P, op::DataType::DT_DOUBLE);
    EXPECT_TRUE(doubleScalarMinUint64P.CheckOverflows<uint64_t>());

    double doubleValueMaxFp16 = std::numeric_limits<fp16_t>::max();
    aclScalar doubleScalarMaxFp16(&doubleValueMaxFp16, op::DataType::DT_DOUBLE);
    EXPECT_FALSE(doubleScalarMaxFp16.CheckOverflows<fp16_t>());
    double doubleValueMinFp16 = std::numeric_limits<fp16_t>::lowest();
    aclScalar doubleScalarMinFp16(&doubleValueMinFp16, op::DataType::DT_DOUBLE);
    EXPECT_FALSE(doubleScalarMinFp16.CheckOverflows<fp16_t>());
    double doubleValueMaxFp16P = doubleValueMaxFp16 + 1;
    aclScalar doubleScalarMaxFp16P(&doubleValueMaxFp16P, op::DataType::DT_DOUBLE);
    EXPECT_TRUE(doubleScalarMaxFp16P.CheckOverflows<fp16_t>());
    double doubleValueMinFp16P = doubleValueMinFp16 - 1;
    aclScalar doubleScalarMinFp16P(&doubleValueMinFp16P, op::DataType::DT_DOUBLE);
    EXPECT_TRUE(doubleScalarMinFp16P.CheckOverflows<fp16_t>());

    double doubleValueMaxBf16 = std::numeric_limits<bfloat16>::max();
    aclScalar doubleScalarMaxBf16(&doubleValueMaxBf16, op::DataType::DT_DOUBLE);
    EXPECT_FALSE(doubleScalarMaxBf16.CheckOverflows<bfloat16>());
    double doubleValueMinBf16 = std::numeric_limits<bfloat16>::lowest();
    aclScalar doubleScalarMinBf16(&doubleValueMinBf16, op::DataType::DT_DOUBLE);
    EXPECT_FALSE(doubleScalarMinBf16.CheckOverflows<bfloat16>());
    double doubleValueMaxBf16P = doubleValueMaxBf16 + 1;
    aclScalar doubleScalarMaxBf16P(&doubleValueMaxBf16P, op::DataType::DT_DOUBLE);
    EXPECT_FALSE(doubleScalarMaxBf16P.CheckOverflows<bfloat16>());
    double doubleValueMinBf16P = doubleValueMinBf16 - 1;
    aclScalar doubleScalarMinBf16P(&doubleValueMinBf16P, op::DataType::DT_DOUBLE);
    EXPECT_FALSE(doubleScalarMinBf16P.CheckOverflows<bfloat16>());
    double doubleValueMaxBf16P1 = std::numeric_limits<float>::max();
    aclScalar doubleScalarMaxBf16P1(&doubleValueMaxBf16P1, op::DataType::DT_DOUBLE);
    EXPECT_TRUE(doubleScalarMaxBf16P1.CheckOverflows<bfloat16>());
    double doubleValueMinBf16P1 = std::numeric_limits<float>::lowest();
    aclScalar doubleScalarMinBf16P1(&doubleValueMinBf16P1, op::DataType::DT_DOUBLE);
    EXPECT_TRUE(doubleScalarMinBf16P1.CheckOverflows<bfloat16>());

    constexpr double doubleValueMaxFloat = std::numeric_limits<float>::max();
    aclScalar doubleScalarMaxFloat(&doubleValueMaxFloat, op::DataType::DT_DOUBLE);
    EXPECT_FALSE(doubleScalarMaxFloat.CheckOverflows<float>());
    constexpr double doubleValueMinFloat = std::numeric_limits<float>::lowest();
    aclScalar doubleScalarMinFloat(&doubleValueMinFloat, op::DataType::DT_DOUBLE);
    EXPECT_FALSE(doubleScalarMinFloat.CheckOverflows<float>());
    constexpr double doubleValueMaxFloatP = doubleValueMaxFloat * 2;
    aclScalar doubleScalarMaxFloatP(&doubleValueMaxFloatP, op::DataType::DT_DOUBLE);
    EXPECT_TRUE(doubleScalarMaxFloatP.CheckOverflows<float>());
    constexpr double doubleValueMinFloatP = doubleValueMinFloat * 2;
    aclScalar doubleScalarMinFloatP(&doubleValueMinFloatP, op::DataType::DT_DOUBLE);
    EXPECT_TRUE(doubleScalarMinFloatP.CheckOverflows<float>());

    constexpr double doubleValueMaxDouble = std::numeric_limits<float>::max();
    aclScalar doubleScalarMaxDouble(&doubleValueMaxDouble, op::DataType::DT_DOUBLE);
    EXPECT_FALSE(doubleScalarMaxDouble.CheckOverflows<double>());
    constexpr double doubleValueMinDouble = std::numeric_limits<float>::lowest();
    aclScalar doubleScalarMinDouble(&doubleValueMinDouble, op::DataType::DT_DOUBLE);
    EXPECT_FALSE(doubleScalarMinDouble.CheckOverflows<double>());
    constexpr double doubleValueMaxDoubleP = doubleValueMaxDouble * 2;
    aclScalar doubleScalarMaxDoubleP(&doubleValueMaxDoubleP, op::DataType::DT_DOUBLE);
    EXPECT_FALSE(doubleScalarMaxDoubleP.CheckOverflows<double>());
    constexpr double doubleValueMinDoubleP = doubleValueMinDouble * 2;
    aclScalar doubleScalarMinDoubleP(&doubleValueMinDoubleP, op::DataType::DT_DOUBLE);
    EXPECT_FALSE(doubleScalarMinDoubleP.CheckOverflows<double>());
}

TEST_F(CommonTypesTest, TestAclScalarCheckOverflowsBfloat16ToOther)
{
    bfloat16 floatValueMaxUint8(std::numeric_limits<uint8_t>::max());
    aclScalar floatScalarMaxUint8(&floatValueMaxUint8, op::DataType::DT_BF16);
    EXPECT_FALSE(floatScalarMaxUint8.CheckOverflows<uint8_t>());
    bfloat16 floatValueMinUint8 = std::numeric_limits<uint8_t>::lowest();
    aclScalar floatScalarMinUint8(&floatValueMinUint8, op::DataType::DT_BF16);
    EXPECT_FALSE(floatScalarMinUint8.CheckOverflows<uint8_t>());
    bfloat16 floatValueMaxUint8P = floatValueMaxUint8 + 1;
    aclScalar floatScalarMaxUint8P(&floatValueMaxUint8P, op::DataType::DT_BF16);
    EXPECT_TRUE(floatScalarMaxUint8P.CheckOverflows<uint8_t>());
    bfloat16 floatValueMinUint8P = floatValueMinUint8 - static_cast<bfloat16>(1.0);
    aclScalar floatScalarMinUint8P(&floatValueMinUint8P, op::DataType::DT_BF16);
    EXPECT_TRUE(floatScalarMinUint8P.CheckOverflows<uint8_t>());
}

#define CHECK_UNWS_TENSOR(tensorName, viewShape, storageShape, dataType, strides, viewOffset, format, tensorData) \
    vector<int64_t> _viewShape = viewShape;                                                                       \
    vector<int64_t> _storageShape = storageShape;                                                                 \
    aclTensor tensorName(_viewShape.data(), _viewShape.size(), dataType, strides, viewOffset, format,             \
                         _storageShape.data(), _storageShape.size(), tensorData);                                 \
    EXPECT_TRUE(ShapeEq(tensorName.GetViewShape(), viewShape));                                                   \
    EXPECT_TRUE(ShapeEq(tensorName.GetOriginalShape(), storageShape));                                            \
    op::Shape expectStorageShape({1});                                                                            \
    EXPECT_TRUE(ShapeEq(tensorName.GetStorageShape(), storageShape));                                             \
    EXPECT_EQ(tensorName.GetDataType(), op::ToOpDataType(dataType));                                              \
    EXPECT_EQ(tensorName.GetViewFormat(), op::ToOpFormat(format));                                                \
    EXPECT_EQ(tensorName.GetOriginalFormat(), op::Format::FORMAT_ND);                                             \
    EXPECT_EQ(tensorName.GetStorageAddr(), tensorData);                                                           \
    EXPECT_EQ(tensorName.GetData(),                                                                               \
              static_cast<char*>(tensorData) + viewOffset * op::TypeSize(op::ToOpDataType(dataType)));            \
    EXPECT_EQ(tensorName.IsFromWorkspace(), false);                                                               \
    EXPECT_EQ(tensorName.GetViewOffset(), viewOffset);                                                            \
    EXPECT_EQ(tensorName.GetStorageAddr(), tensorData);                                                           \
    EXPECT_EQ(tensorName.GetStorageFormat(), op::Format::FORMAT_ND);                                              \
    EXPECT_EQ(tensorName.GetPlacement(), TensorPlacement::kOnDeviceHbm);                                          \
    EXPECT_NE(tensorName.GetTensor(), nullptr);                                                                   \
    EXPECT_EQ(tensorName.IsView(), false);

TEST_F(CommonTypesTest, CreateEmptyViewUNWSTensor)
{
    std::vector<int64_t> strides = {1};
    CHECK_UNWS_TENSOR(a, {0}, {1}, aclDataType::ACL_FLOAT, strides.data(), 0, aclFormat::ACL_FORMAT_ND, nullptr);
    EXPECT_EQ(a.GetStorageFormat(), op::Format::FORMAT_ND);
    EXPECT_EQ(a.GetViewStrides(), op::Strides({1}));
}

TEST_F(CommonTypesTest, CreateEmptyStorageUNWSTensor)
{
    std::vector<int64_t> strides = {1};
    CHECK_UNWS_TENSOR(a, {0}, {0}, aclDataType::ACL_FLOAT, strides.data(), 0, aclFormat::ACL_FORMAT_ND, nullptr);
    EXPECT_EQ(a.GetViewStrides(), op::Strides({1}));
    EXPECT_TRUE(a.IsEmpty());
}

TEST_F(CommonTypesTest, CreateContiguousUNWSTensor001)
{
    std::vector<int64_t> strides = {2, 1};
    CHECK_UNWS_TENSOR(a, std::vector<int64_t>({4, 2}), std::vector<int64_t>({8}), aclDataType::ACL_FLOAT,
                      strides.data(), 0, aclFormat::ACL_FORMAT_ND, nullptr);
    EXPECT_EQ(a.GetViewStrides(), op::Strides({2, 1}));
}

TEST_F(CommonTypesTest, CreateContiguousUNWSTensor002)
{
    CHECK_UNWS_TENSOR(a, std::vector<int64_t>({1, 4, 2}), std::vector<int64_t>({8}), aclDataType::ACL_FLOAT, nullptr, 0,
                      aclFormat::ACL_FORMAT_ND, nullptr);
    EXPECT_EQ(a.GetViewStrides(), op::Strides({8, 2, 1}));
    a.SetViewOffset(3);
    EXPECT_EQ(a.GetViewOffset(), 3);
    float data[] = {1, 2, 3};
    a.SetStorageAddr(data);
    EXPECT_EQ(a.GetStorageAddr(), data);

    op::Shape shape1({1, 2, 16, 16});
    a.SetStorageShape(shape1);
    EXPECT_EQ(a.GetStorageShape(), shape1);
    a.SetViewShape(shape1);
    EXPECT_EQ(a.GetStorageShape(), shape1);
    a.SetStorageFormat(Format::FORMAT_FRACTAL_NZ);
    EXPECT_EQ(a.GetStorageFormat(), Format::FORMAT_FRACTAL_NZ);
    a.SetOriginalFormat(Format::FORMAT_FRACTAL_NZ);
    EXPECT_EQ(a.GetOriginalFormat(), Format::FORMAT_FRACTAL_NZ);
    a.SetViewFormat(Format::FORMAT_FRACTAL_NZ);
    EXPECT_EQ(a.GetViewFormat(), Format::FORMAT_FRACTAL_NZ);
    op::Strides strides1({2 * 16 * 16, 16 * 16, 16, 1});
    a.SetViewStrides(strides1);
    EXPECT_EQ(a.GetViewStrides(), strides1);
    a.SetWorkspaceOffset(512);
    EXPECT_EQ(a.GetWorkspaceOffset(), 512UL);
    a.SetViewShape(op::Shape({1, 0, 1}));
    EXPECT_TRUE(a.IsEmpty());
}

TEST_F(CommonTypesTest, CreateContiguousUNWSTensor003)
{
    vector<float> values(8, 3.0);
    CHECK_UNWS_TENSOR(a, std::vector<int64_t>({1, 4, 2}), std::vector<int64_t>({16}), aclDataType::ACL_FLOAT, nullptr,
                      3, aclFormat::ACL_FORMAT_ND, static_cast<void*>(values.data()));
    EXPECT_EQ(a.GetViewStrides(), op::Strides({8, 2, 1}));
}

TEST_F(CommonTypesTest, CreateUnContiguousUNWSTensor001)
{
    std::vector<int64_t> strides = {8, 1};
    CHECK_UNWS_TENSOR(a, std::vector<int64_t>({4, 2}), std::vector<int64_t>({32}), aclDataType::ACL_FLOAT,
                      strides.data(), 0, aclFormat::ACL_FORMAT_ND, nullptr);
    EXPECT_EQ(a.GetViewStrides(), op::Strides({8, 1}));
}

#define CHECK_WS_TENSOR_DIFF_FORMAT(tensorName, storageShape, originShape, dataType, storageFormat, originFormat, \
                                    tensorData)                                                                   \
    aclTensor tensorName(storageShape, originShape, dataType, storageFormat, originFormat, tensorData);           \
    EXPECT_EQ(tensorName.GetViewShape(), originShape);                                                            \
    EXPECT_EQ(tensorName.GetOriginalShape(), originShape);                                                        \
    EXPECT_EQ(tensorName.GetStorageShape(), storageShape);                                                        \
    EXPECT_EQ(tensorName.GetDataType(), dataType);                                                                \
    EXPECT_EQ(tensorName.GetViewFormat(), originFormat);                                                          \
    EXPECT_EQ(tensorName.GetOriginalFormat(), originFormat);                                                      \
    EXPECT_EQ(tensorName.GetStorageAddr(), tensorData);                                                           \
    EXPECT_EQ(tensorName.GetData(), static_cast<char*>(tensorData));                                              \
    EXPECT_EQ(tensorName.IsFromWorkspace(), true);                                                                \
    EXPECT_EQ(tensorName.GetViewOffset(), 0);                                                                     \
    EXPECT_EQ(tensorName.GetStorageAddr(), tensorData);                                                           \
    EXPECT_EQ(tensorName.GetStorageFormat(), storageFormat);                                                      \
    EXPECT_EQ(tensorName.GetPlacement(), TensorPlacement::kOnDeviceHbm);                                          \
    EXPECT_EQ(tensorName.IsView(), false);

TEST_F(CommonTypesTest, CreateWorkspaceTensorDiffFormat)
{
    CHECK_WS_TENSOR_DIFF_FORMAT(a, Shape({0}), Shape({0}), DataType::DT_FLOAT, Format::FORMAT_ND, Format::FORMAT_ND,
                                nullptr);
    CHECK_WS_TENSOR_DIFF_FORMAT(b, Shape({}), Shape({}), DataType::DT_FLOAT, Format::FORMAT_ND, Format::FORMAT_ND,
                                nullptr);
    CHECK_WS_TENSOR_DIFF_FORMAT(c, Shape({16, 16}), Shape({16 * 16}), DataType::DT_FLOAT, Format::FORMAT_FRACTAL_NZ,
                                Format::FORMAT_ND, nullptr);
}

#define CHECK_WS_TENSOR_SAME_FORMAT(tensorName, shape, dataType, format, tensorData) \
    aclTensor tensorName(shape, dataType, format, tensorData);                       \
    EXPECT_EQ(tensorName.GetViewShape(), shape);                                     \
    EXPECT_EQ(tensorName.GetOriginalShape(), shape);                                 \
    EXPECT_EQ(tensorName.GetStorageShape(), shape);                                  \
    EXPECT_EQ(tensorName.GetDataType(), dataType);                                   \
    EXPECT_EQ(tensorName.GetViewFormat(), format);                                   \
    EXPECT_EQ(tensorName.GetOriginalFormat(), format);                               \
    EXPECT_EQ(tensorName.GetStorageAddr(), tensorData);                              \
    EXPECT_EQ(tensorName.GetData(), static_cast<char*>(tensorData));                 \
    EXPECT_EQ(tensorName.IsFromWorkspace(), true);                                   \
    EXPECT_EQ(tensorName.GetViewOffset(), 0);                                        \
    EXPECT_EQ(tensorName.GetStorageAddr(), tensorData);                              \
    EXPECT_EQ(tensorName.GetStorageFormat(), format);                                \
    EXPECT_EQ(tensorName.GetPlacement(), TensorPlacement::kOnDeviceHbm);             \
    EXPECT_EQ(tensorName.IsView(), false);

TEST_F(CommonTypesTest, CreateWorkspaceTensorSameFormat)
{
    CHECK_WS_TENSOR_SAME_FORMAT(a, Shape({0}), DataType::DT_FLOAT, Format::FORMAT_ND, nullptr);
    CHECK_WS_TENSOR_SAME_FORMAT(b, Shape({}), DataType::DT_FLOAT, Format::FORMAT_ND, nullptr);
    CHECK_WS_TENSOR_SAME_FORMAT(c, Shape({16, 16}), DataType::DT_FLOAT, Format::FORMAT_FRACTAL_NZ, nullptr);
}

#define CHECK_VIEW_TENSOR(tensorName, shape, dataType, format, viewShape, viewOffset, tensorData)          \
    aclTensor tensorName##ToView(shape, dataType, format, tensorData);                                     \
    aclTensor tensorName(tensorName##ToView, viewShape, viewOffset);                                       \
    EXPECT_EQ(tensorName.GetViewShape(), viewShape);                                                       \
    EXPECT_EQ(tensorName.GetOriginalShape(), viewShape);                                                   \
    EXPECT_EQ(tensorName.GetStorageShape(), viewShape);                                                    \
    EXPECT_EQ(tensorName.GetDataType(), dataType);                                                         \
    EXPECT_EQ(tensorName.GetViewFormat(), format);                                                         \
    EXPECT_EQ(tensorName.GetOriginalFormat(), format);                                                     \
    EXPECT_EQ(tensorName.GetStorageAddr(), tensorData);                                                    \
    EXPECT_EQ(tensorName.GetData(), static_cast<char*>(tensorData) + viewOffset * op::TypeSize(dataType)); \
    EXPECT_EQ(tensorName.IsFromWorkspace(), true);                                                         \
    EXPECT_EQ(tensorName.GetViewOffset(), viewOffset);                                                     \
    EXPECT_EQ(tensorName.GetStorageAddr(), tensorData);                                                    \
    EXPECT_EQ(tensorName.GetStorageFormat(), format);                                                      \
    EXPECT_EQ(tensorName.GetPlacement(), TensorPlacement::kOnDeviceHbm);                                   \
    EXPECT_EQ(tensorName.Numel(), viewShape.GetShapeSize());                                               \
    EXPECT_EQ(tensorName.IsView(), true);

TEST_F(CommonTypesTest, CreateViewTensor)
{
    CHECK_VIEW_TENSOR(a, Shape({0}), DataType::DT_FLOAT, Format::FORMAT_ND, Shape({0}), 0, nullptr);
    CHECK_VIEW_TENSOR(b, Shape({}), DataType::DT_FLOAT, Format::FORMAT_ND, Shape({}), 0, nullptr);
    CHECK_VIEW_TENSOR(c, Shape({16, 16}), DataType::DT_FLOAT, Format::FORMAT_FRACTAL_NZ, Shape({12, 12}), 3, nullptr);
}

TEST_F(CommonTypesTest, CreateViewTensor2)
{
    aclTensor src(Shape({16, 16}), DataType::DT_FLOAT, Format::FORMAT_FRACTAL_NZ, nullptr);
    aclOpExecutor exe;
    aclTensor* dst = exe.CreateView(&src, Shape({16, 16, 16}), Shape({17, 17, 17}), src.GetViewStrides(), 0);
    EXPECT_EQ(dst->GetOriginalShape(), Shape({16, 16, 16}));
    EXPECT_EQ(dst->GetStorageShape(), Shape({4913}));
    EXPECT_EQ(dst->GetViewStrides(), src.GetViewStrides());
}

#define CHECK_ARRAY(arrayName, value, count, valueType, arrayType)                   \
    acl##arrayType##Array arrayName(value, count);                                   \
    if (value != nullptr && count != 0) {                                            \
        ASSERT_NE(arrayName.GetData(), nullptr);                                     \
        EXPECT_EQ(memcmp(arrayName.GetData(), value, count * sizeof(valueType)), 0); \
        EXPECT_EQ(arrayName.Size(), count);                                          \
    } else {                                                                         \
        EXPECT_EQ(arrayName.GetData(), nullptr);                                     \
        EXPECT_EQ(arrayName.Size(), 0);                                              \
    }

TEST_F(CommonTypesTest, CreateArray)
{
    int64_t intValue[] = {1, 2, 3};
    int64_t* nullInt = nullptr;
    CHECK_ARRAY(intArray1, intValue, 3UL, int64_t, Int);
    for (size_t i = 0; i < intArray1.Size(); i++) {
        EXPECT_EQ(intArray1[i], intValue[i]);
    }

    CHECK_ARRAY(intArray2, nullInt, 0UL, int64_t, Int);
    CHECK_ARRAY(intArray3, intValue, 0UL, int64_t, Int);

    float fpValue[] = {1, 2, 3};
    float* nullFloat = nullptr;
    CHECK_ARRAY(floatArray1, fpValue, 3UL, float, Float);
    for (size_t i = 0; i < floatArray1.Size(); i++) {
        EXPECT_FLOAT_EQ(floatArray1[i], fpValue[i]);
    }
    CHECK_ARRAY(floatArray2, nullFloat, 0UL, float, Float);
    CHECK_ARRAY(floatArray3, fpValue, 0UL, float, Float);

    bool boolValue[] = {true, false, true};
    CHECK_ARRAY(boolArray1, boolValue, 3UL, bool, Bool);
    for (size_t i = 0; i < intArray1.Size(); i++) {
        EXPECT_EQ(boolArray1[i], boolValue[i]);
    }
    bool* nullBool = nullptr;
    CHECK_ARRAY(boolArray2, nullBool, 0UL, bool, Bool);
    CHECK_ARRAY(boolArray3, boolValue, 0UL, bool, Bool);
}

TEST_F(CommonTypesTest, CreateTensorList)
{
    aclTensor* list1[] = {nullptr, nullptr, nullptr};
    aclTensorList aclTensorList1(list1, 0);
    ASSERT_EQ(aclTensorList1.GetData(), nullptr);
    EXPECT_EQ(aclTensorList1.Size(), 0);

    std::vector<int64_t> shape = {1, 2, 3};
    aclTensor tensor1({1, 2, 3}, DataType::DT_FLOAT, Format::FORMAT_ND, nullptr);
    aclTensor tensor2({1, 2, 3}, DataType::DT_FLOAT, Format::FORMAT_ND, nullptr);
    aclTensor tensor3({1, 2, 3}, DataType::DT_FLOAT, Format::FORMAT_ND, nullptr);
    aclTensor* list2[] = {&tensor1, &tensor2, &tensor3};
    aclTensorList aclTensorList2(list2, 3);
    ASSERT_NE(aclTensorList2.GetData(), nullptr);
    EXPECT_EQ(aclTensorList2.Size(), 3);
    for (size_t i = 0; i < aclTensorList2.Size(); i++) {
        EXPECT_EQ(aclTensorList2[i], list2[i]);
    }

    aclTensorList aclTensorList3(nullptr, 3);
    EXPECT_EQ(aclTensorList1.GetData(), nullptr);
    EXPECT_EQ(aclTensorList1.Size(), 0);
}

#define CHECK_SCALAR2TENSOR(scalarName, value, valueType, dataType)                                \
    valueType scalarName##valueType = value;                                                       \
    aclScalar scalarName(&scalarName##valueType, dataType);                                        \
    aclTensor scalarName##Tensor(&scalarName, dataType);                                           \
    ASSERT_NE(scalarName##Tensor.GetData(), nullptr);                                              \
    ASSERT_EQ(scalarName##Tensor.Size(), 1);                                                       \
    EXPECT_EQ(memcmp(scalarName##Tensor.GetData(), &scalarName##valueType, sizeof(valueType)), 0); \
    EXPECT_EQ(scalarName##Tensor.GetDataType(), dataType)

TEST_F(CommonTypesTest, TestAclScalarToAclTensor)
{
    CHECK_SCALAR2TENSOR(fpScalar, 3.2, float, DataType::DT_FLOAT);
    CHECK_SCALAR2TENSOR(fp16Scalar, 3.2, fp16_t, DataType::DT_FLOAT16);
    CHECK_SCALAR2TENSOR(bf16Scalar, 3.2, bfloat16, DataType::DT_BF16);
    CHECK_SCALAR2TENSOR(int8Scalar, 3, int8_t, DataType::DT_INT8);
    CHECK_SCALAR2TENSOR(int16Scalar, 3, int16_t, DataType::DT_INT16);
    CHECK_SCALAR2TENSOR(uint16Scalar, 3, uint16_t, DataType::DT_UINT16);
    CHECK_SCALAR2TENSOR(uint8Scalar, 3, uint8_t, DataType::DT_UINT8);
    CHECK_SCALAR2TENSOR(int32Scalar, 3, int32_t, DataType::DT_INT32);
    CHECK_SCALAR2TENSOR(int64Scalar, 3, int64_t, DataType::DT_INT64);
    CHECK_SCALAR2TENSOR(uint32Scalar, 3, uint32_t, DataType::DT_UINT32);
    CHECK_SCALAR2TENSOR(uint64Scalar, 3, uint64_t, DataType::DT_UINT64);
    CHECK_SCALAR2TENSOR(boolScalar, 1, bool, DataType::DT_BOOL);
    CHECK_SCALAR2TENSOR(doubleScalar, 3, double, DataType::DT_DOUBLE);
    using complex64 = std::complex<float>;
    using complex128 = std::complex<double>;
    CHECK_SCALAR2TENSOR(complex64Scalar, complex64(3.2, 3.2), complex64, DataType::DT_COMPLEX64);
    CHECK_SCALAR2TENSOR(complex128Scalar, complex128(3.2, 3.2), complex128, DataType::DT_COMPLEX128);
    fp16_t fp16Value = 3.2;

    // unsupport dtype
    try {
        aclScalar unsupportedDataTypeScalar(&fp16Value, DataType::DT_QINT16);
        aclTensor unsupportedDataTypeScalarTensor(&unsupportedDataTypeScalar, DataType::DT_QINT16);
        EXPECT_NE(memcmp(unsupportedDataTypeScalarTensor.GetData(), &fp16Value, sizeof(fp16Value)), 0);
    } catch (...) {
    }
}

TEST_F(CommonTypesTest, TestAclIntArrayToAclTensor)
{
    int32_t value = 3;
    std::vector<int64_t> values(3, value);
    aclIntArray array(values.data(), static_cast<uint64_t>(values.size()));
    aclTensor intTensor(&array, op::DataType::DT_INT32);
    EXPECT_EQ(intTensor.Size(), static_cast<int64_t>(values.size()));
    EXPECT_EQ(*static_cast<int32_t*>(intTensor.GetData()), value);
    EXPECT_EQ(*(static_cast<int32_t*>(intTensor.GetData()) + 1), value);
    EXPECT_EQ(*(static_cast<int32_t*>(intTensor.GetData()) + 2), value);

    aclTensor fTensor(&array, op::DataType::DT_FLOAT);
    EXPECT_EQ(fTensor.Size(), static_cast<int64_t>(values.size()));
    EXPECT_EQ(*static_cast<float*>(fTensor.GetData()), static_cast<float>(value));
    EXPECT_EQ(*(static_cast<float*>(fTensor.GetData()) + 1), static_cast<float>(value));
    EXPECT_EQ(*(static_cast<float*>(fTensor.GetData()) + 2), static_cast<float>(value));
}

TEST_F(CommonTypesTest, TestAclFp16ArrayToAclTensor)
{
    float value = 3.1;
    std::vector<op::fp16_t> values(3, value);
    aclFp16Array array(values.data(), static_cast<uint64_t>(values.size()));
    aclTensor intTensor(&array, op::DataType::DT_INT32);
    EXPECT_EQ(intTensor.Size(), static_cast<int64_t>(values.size()));
    EXPECT_EQ(*static_cast<int32_t*>(intTensor.GetData()), static_cast<int32_t>(value));
    EXPECT_EQ(*(static_cast<int32_t*>(intTensor.GetData()) + 1), static_cast<int32_t>(value));
    EXPECT_EQ(*(static_cast<int32_t*>(intTensor.GetData()) + 2), static_cast<int32_t>(value));

    aclTensor fTensor(&array, op::DataType::DT_FLOAT);
    EXPECT_EQ(fTensor.Size(), static_cast<int64_t>(values.size()));
    EXPECT_EQ(*static_cast<float*>(fTensor.GetData()), op::fp16_t(value).toFloat());
    EXPECT_EQ(*(static_cast<float*>(fTensor.GetData()) + 1), op::fp16_t(value).toFloat());
    EXPECT_EQ(*(static_cast<float*>(fTensor.GetData()) + 2), op::fp16_t(value).toFloat());

    aclTensor fp16Tensor(&array, op::DataType::DT_FLOAT16);
    EXPECT_EQ(fp16Tensor.Size(), static_cast<int64_t>(values.size()));
    EXPECT_EQ(static_cast<fp16_t*>(fp16Tensor.GetData())->val, op::fp16_t(value).val);
    EXPECT_EQ((static_cast<fp16_t*>(fp16Tensor.GetData()) + 1)->val, op::fp16_t(value).val);
    EXPECT_EQ((static_cast<fp16_t*>(fp16Tensor.GetData()) + 2)->val, op::fp16_t(value).val);
}

TEST_F(CommonTypesTest, TestAclBf16ArrayToAclTensor)
{
    float value = 3.1;
    std::vector<op::bfloat16> values(3, value);
    aclBf16Array array(values.data(), static_cast<uint64_t>(values.size()));
    aclTensor intTensor(&array, op::DataType::DT_INT32);
    EXPECT_EQ(intTensor.Size(), static_cast<int64_t>(values.size()));
    EXPECT_EQ(*static_cast<int32_t*>(intTensor.GetData()), static_cast<int32_t>(value));
    EXPECT_EQ(*(static_cast<int32_t*>(intTensor.GetData()) + 1), static_cast<int32_t>(value));
    EXPECT_EQ(*(static_cast<int32_t*>(intTensor.GetData()) + 2), static_cast<int32_t>(value));

    aclTensor fTensor(&array, op::DataType::DT_FLOAT);
    EXPECT_EQ(fTensor.Size(), static_cast<int64_t>(values.size()));
    EXPECT_EQ(*static_cast<float*>(fTensor.GetData()), static_cast<float>(op::bfloat16(value)));
    EXPECT_EQ(*(static_cast<float*>(fTensor.GetData()) + 1), static_cast<float>(op::bfloat16(value)));
    EXPECT_EQ(*(static_cast<float*>(fTensor.GetData()) + 2), static_cast<float>(op::bfloat16(value)));

    aclTensor fp16Tensor(&array, op::DataType::DT_BF16);
    EXPECT_EQ(fp16Tensor.Size(), static_cast<int64_t>(values.size()));
    EXPECT_EQ(*static_cast<bfloat16*>(fp16Tensor.GetData()), op::bfloat16(value));
    EXPECT_EQ(*(static_cast<bfloat16*>(fp16Tensor.GetData()) + 1), op::bfloat16(value));
    EXPECT_EQ(*(static_cast<bfloat16*>(fp16Tensor.GetData()) + 2), op::bfloat16(value));
}

TEST_F(CommonTypesTest, TestAclFloatArrayToAclTensor)
{
    float value = 3.1;
    std::vector<float> values(3, value);
    aclFloatArray array(values.data(), static_cast<uint64_t>(values.size()));
    aclTensor intTensor(&array, op::DataType::DT_INT32);
    EXPECT_EQ(intTensor.Size(), static_cast<int64_t>(values.size()));
    EXPECT_EQ(*static_cast<int32_t*>(intTensor.GetData()), static_cast<int32_t>(value));
    EXPECT_EQ(*(static_cast<int32_t*>(intTensor.GetData()) + 1), static_cast<int32_t>(value));
    EXPECT_EQ(*(static_cast<int32_t*>(intTensor.GetData()) + 2), static_cast<int32_t>(value));

    aclTensor fTensor(&array, op::DataType::DT_FLOAT);
    EXPECT_EQ(fTensor.Size(), static_cast<int64_t>(values.size()));
    EXPECT_EQ(*static_cast<float*>(fTensor.GetData()), value);
    EXPECT_EQ(*(static_cast<float*>(fTensor.GetData()) + 1), value);
    EXPECT_EQ(*(static_cast<float*>(fTensor.GetData()) + 2), value);
}

TEST_F(CommonTypesTest, TestAclBoolArrayToAclTensor)
{
    bool values[] = {true, false, true};
    aclBoolArray array(values, static_cast<uint64_t>(3));
    aclTensor intTensor(&array, op::DataType::DT_BOOL);
    EXPECT_EQ(intTensor.Size(), 3);
    EXPECT_EQ(*static_cast<bool*>(intTensor.GetData()), values[0]);
    EXPECT_EQ(*(static_cast<bool*>(intTensor.GetData()) + 1), values[1]);
    EXPECT_EQ(*(static_cast<bool*>(intTensor.GetData()) + 2), values[2]);
}
