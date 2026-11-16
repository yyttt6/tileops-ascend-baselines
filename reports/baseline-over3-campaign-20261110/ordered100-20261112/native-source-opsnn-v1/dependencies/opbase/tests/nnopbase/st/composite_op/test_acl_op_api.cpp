/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn/aclnn_base.h"
#include "gtest/gtest.h"
#include <memory>
#include <vector>
#include <cstdlib>
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "op_dfx_internal.h"
#include "utils/file_faker.h"

using namespace std;
using namespace op;
class AclOpApiTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);
        std::cout << "AclOpApiTest SetUp" << std::endl;
    }

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

#define CHECK_TENSOR(tensorName, viewShape, storageShape, dataType, strides, viewOffset, format, tensorData)       \
    vector<int64_t> _viewShape = viewShape;                                                                        \
    vector<int64_t> _storageShape = storageShape;                                                                  \
    auto tensorName = aclCreateTensor(_viewShape.data(), _viewShape.size(), dataType, strides, viewOffset, format, \
                                      _storageShape.data(), _storageShape.size(), tensorData);                     \
    ASSERT_NE(tensorName, nullptr);                                                                                \
    std::unique_ptr<aclTensor> uniqueTensor(tensorName);                                                           \
    EXPECT_TRUE(ShapeEq(tensorName->GetViewShape(), viewShape));                                                   \
    op::Shape expectStorageShape({1});                                                                             \
    EXPECT_TRUE(ShapeEq(tensorName->GetStorageShape(), storageShape));                                             \
    EXPECT_EQ(tensorName->GetDataType(), op::ToOpDataType(dataType));                                              \
    EXPECT_EQ(tensorName->GetViewFormat(), op::ToOpFormat(format));                                                \
    EXPECT_EQ(tensorName->GetStorageAddr(), tensorData);                                                           \
    EXPECT_EQ(tensorName->GetData(),                                                                               \
              static_cast<char*>(tensorData) + viewOffset * op::TypeSize(op::ToOpDataType(dataType)));             \
    EXPECT_EQ(tensorName->IsFromWorkspace(), false);                                                               \
    EXPECT_EQ(tensorName->GetViewOffset(), viewOffset);                                                            \
    EXPECT_EQ(tensorName->GetStorageAddr(), tensorData);                                                           \
    EXPECT_EQ(tensorName->GetStorageFormat(), op::Format::FORMAT_ND);                                              \
    EXPECT_EQ(tensorName->GetPlacement(), TensorPlacement::kOnDeviceHbm);                                          \
    EXPECT_EQ(tensorName->IsView(), false);

TEST_F(AclOpApiTest, CreateEmptyViewTensor)
{
    std::vector<int64_t> strides = {1};
    CHECK_TENSOR(a, {0}, {1}, aclDataType::ACL_FLOAT, strides.data(), 0, aclFormat::ACL_FORMAT_ND, nullptr);
    EXPECT_EQ(a->GetStorageFormat(), op::Format::FORMAT_ND);
    EXPECT_EQ(a->GetViewStrides(), op::Strides({1}));
}

TEST_F(AclOpApiTest, CreateEmptyStorageTensor)
{
    std::vector<int64_t> strides = {1};
    CHECK_TENSOR(a, {0}, {0}, aclDataType::ACL_FLOAT, strides.data(), 0, aclFormat::ACL_FORMAT_ND, nullptr);
    EXPECT_EQ(a->GetViewStrides(), op::Strides({1}));
}

TEST_F(AclOpApiTest, CreateContiguousTensor001)
{
    std::vector<int64_t> strides = {2, 1};
    CHECK_TENSOR(a, std::vector<int64_t>({4, 2}), std::vector<int64_t>({8}), aclDataType::ACL_FLOAT, strides.data(), 0,
                 aclFormat::ACL_FORMAT_ND, nullptr);
    EXPECT_EQ(a->GetViewStrides(), op::Strides({2, 1}));
}

TEST_F(AclOpApiTest, CreateContiguousTensor002)
{
    CHECK_TENSOR(a, std::vector<int64_t>({1, 4, 2}), std::vector<int64_t>({8}), aclDataType::ACL_FLOAT, nullptr, 0,
                 aclFormat::ACL_FORMAT_ND, nullptr);
    EXPECT_EQ(a->GetViewStrides(), op::Strides({8, 2, 1}));
}

TEST_F(AclOpApiTest, CreateContiguousTensor003)
{
    vector<float> values(8, 3.0);
    CHECK_TENSOR(a, std::vector<int64_t>({1, 4, 2}), std::vector<int64_t>({16}), aclDataType::ACL_FLOAT, nullptr, 3,
                 aclFormat::ACL_FORMAT_ND, static_cast<void*>(values.data()));
    EXPECT_EQ(a->GetViewStrides(), op::Strides({8, 2, 1}));
}

TEST_F(AclOpApiTest, CreateUnContiguousTensor001)
{
    std::vector<int64_t> strides = {8, 1};
    CHECK_TENSOR(a, std::vector<int64_t>({4, 2}), std::vector<int64_t>({32}), aclDataType::ACL_FLOAT, strides.data(), 0,
                 aclFormat::ACL_FORMAT_ND, nullptr);
    EXPECT_EQ(a->GetViewStrides(), op::Strides({8, 1}));
}

#define CHECK_SCALAR(scalarName, value, valueType, dataType)                                  \
    valueType scalarName##valueType = value;                                                  \
    std::unique_ptr<aclScalar> scalarName(aclCreateScalar(&scalarName##valueType, dataType)); \
    ASSERT_NE(scalarName.get(), nullptr);                                                     \
    ASSERT_NE(scalarName->GetData(), nullptr);                                                \
    ASSERT_EQ(scalarName->Size(), sizeof(valueType));                                         \
    EXPECT_EQ(memcmp(scalarName->GetData(), &scalarName##valueType, sizeof(valueType)), 0);   \
    EXPECT_EQ(scalarName->GetDataType(), op::ToOpDataType(dataType));

TEST_F(AclOpApiTest, CreateScalar)
{
    CHECK_SCALAR(fpScalar, 3.2, float, aclDataType::ACL_FLOAT);
    CHECK_SCALAR(fp16Scalar, 3.2, fp16_t, aclDataType::ACL_FLOAT16);
    CHECK_SCALAR(int8Scalar, 3, int8_t, aclDataType::ACL_INT8);
    CHECK_SCALAR(int16Scalar, 3, int16_t, aclDataType::ACL_INT16);
    CHECK_SCALAR(uint16Scalar, 3, uint16_t, aclDataType::ACL_UINT16);
    CHECK_SCALAR(uint8Scalar, 3, uint8_t, aclDataType::ACL_UINT8);
    CHECK_SCALAR(int32Scalar, 3, int32_t, aclDataType::ACL_INT32);
    CHECK_SCALAR(int64Scalar, 3, int64_t, aclDataType::ACL_INT64);
    CHECK_SCALAR(uint32Scalar, 3, uint32_t, aclDataType::ACL_UINT32);
    CHECK_SCALAR(uint64Scalar, 3, uint64_t, aclDataType::ACL_UINT64);
    CHECK_SCALAR(boolScalar, 1, bool, aclDataType::ACL_BOOL);
    CHECK_SCALAR(doubleScalar, 3, double, aclDataType::ACL_DOUBLE);
    using complex64 = std::complex<float>;
    using complex128 = std::complex<double>;
    CHECK_SCALAR(complex64Scalar, complex64(3.2, 3.2), complex64, aclDataType::ACL_COMPLEX64);
    CHECK_SCALAR(complex128Scalar, complex128(3.2, 3.2), complex128, aclDataType::ACL_COMPLEX128);
}

#define CHECK_ARRAY(arrayName, value, count, valueType, arrayType)                               \
    std::unique_ptr<acl##arrayType##Array> arrayName(aclCreate##arrayType##Array(value, count)); \
    ASSERT_NE(arrayName.get(), nullptr);                                                         \
    if (value != nullptr && count != 0) {                                                        \
        ASSERT_NE(arrayName->GetData(), nullptr);                                                \
        EXPECT_EQ(memcmp(arrayName->GetData(), value, count * sizeof(valueType)), 0);            \
        EXPECT_EQ(arrayName->Size(), count);                                                     \
    } else {                                                                                     \
        EXPECT_EQ(arrayName->GetData(), nullptr);                                                \
        EXPECT_EQ(arrayName->Size(), 0);                                                         \
    }

TEST_F(AclOpApiTest, CreateArray)
{
    int64_t intValue[] = {1, 2, 3};
    CHECK_ARRAY(intArray1, intValue, 3UL, int64_t, Int);
    int64_t* nullInt = nullptr;
    CHECK_ARRAY(intArray2, nullInt, 0UL, int64_t, Int);
    CHECK_ARRAY(intArray3, intValue, 0UL, int64_t, Int);

    float fpValue[] = {1, 2, 3};
    float* nullFloat = nullptr;
    CHECK_ARRAY(floatArray1, fpValue, 3UL, float, Float);
    CHECK_ARRAY(floatArray2, nullFloat, 0UL, float, Float);
    CHECK_ARRAY(floatArray3, fpValue, 0UL, float, Float);

    bool boolValue[] = {true, false, true};
    bool* nullBool = nullptr;
    CHECK_ARRAY(boolArray1, boolValue, 3UL, bool, Bool);
    CHECK_ARRAY(boolArray2, nullBool, 0UL, bool, Bool);
    CHECK_ARRAY(boolArray3, boolValue, 0UL, bool, Bool);
}

TEST_F(AclOpApiTest, CreateTensorList)
{
    aclTensor* list1[] = {nullptr, nullptr, nullptr};
    auto aclTensorList1 = aclCreateTensorList(list1, 0);
    ASSERT_NE(aclTensorList1, nullptr);
    EXPECT_EQ(aclTensorList1->GetData(), nullptr);
    EXPECT_EQ(aclTensorList1->Size(), 0);
    aclDestroyTensorList(aclTensorList1);

    std::vector<int64_t> shape = {1, 2, 3};
    aclTensor* list2[] = {aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                          aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr),
                          aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                          aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr),
                          aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                          aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr)};
    auto aclTensorList2 = aclCreateTensorList(list2, 3);
    ASSERT_NE(aclTensorList2, nullptr);
    ASSERT_NE(aclTensorList2->GetData(), nullptr);
    EXPECT_EQ(aclTensorList2->Size(), 3);
    aclDestroyTensorList(aclTensorList2);
}

TEST_F(AclOpApiTest, aclDestroyTensor)
{
    std::vector<int64_t> shape = {1, 2, 3};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    EXPECT_EQ(aclDestroyTensor(tensor), OK);
}

TEST_F(AclOpApiTest, aclDestroyScalar)
{
    float fpValue = 3.2;
    auto* value = aclCreateScalar(&fpValue, aclDataType::ACL_FLOAT);
    EXPECT_EQ(aclDestroyScalar(value), OK);
}

TEST_F(AclOpApiTest, aclIntArray)
{
    int64_t values[] = {3, 4, 5};
    auto* value = aclCreateIntArray(values, sizeof(values) / sizeof(values[0]));
    EXPECT_EQ(aclDestroyIntArray(value), OK);
}

TEST_F(AclOpApiTest, aclFloatArray)
{
    float values[] = {3, 4, 5};
    auto* value = aclCreateFloatArray(values, sizeof(values) / sizeof(values[0]));
    EXPECT_EQ(aclDestroyFloatArray(value), OK);
}

TEST_F(AclOpApiTest, aclBoolArray)
{
    bool values[] = {true, false, true};
    auto* value = aclCreateBoolArray(values, sizeof(values) / sizeof(values[0]));
    EXPECT_EQ(aclDestroyBoolArray(value), OK);
}

TEST_F(AclOpApiTest, aclDestroyTensorList)
{
    aclTensor* list1[] = {nullptr, nullptr, nullptr};
    auto aclTensorList1 = aclCreateTensorList(list1, 0);
    aclDestroyTensorList(aclTensorList1);

    std::vector<int64_t> shape = {1, 2, 3};
    aclTensor* list2[] = {aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                          aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr),
                          aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                          aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr),
                          aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                          aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr)};
    auto aclTensorList2 = aclCreateTensorList(list2, 3);
    aclDestroyTensorList(aclTensorList2);
}

TEST_F(AclOpApiTest, aclGetViewShape)
{
    EXPECT_NE(aclGetViewShape(nullptr, nullptr, nullptr), OK);
    std::vector<int64_t> strides = {8, 1};
    CHECK_TENSOR(a, std::vector<int64_t>({4, 2}), std::vector<int64_t>({32}), aclDataType::ACL_FLOAT, strides.data(), 0,
                 aclFormat::ACL_FORMAT_ND, nullptr);
    int64_t* view_dims = nullptr;
    uint64_t view_dim_num = 0;
    EXPECT_EQ(aclGetViewShape(a, &view_dims, &view_dim_num), OK);
    EXPECT_NE(view_dims, nullptr);
    EXPECT_EQ(view_dim_num, 2);
    EXPECT_EQ(view_dims[0], 4);
    EXPECT_EQ(view_dims[1], 2);
    delete[] view_dims;
}

TEST_F(AclOpApiTest, aclnnInit)
{
    bool flag = false;
    EXPECT_EQ(aclnnInit(nullptr), OK);
    EXPECT_EQ(op::internal::systemConfig.GetEnableDebugKernelFlag(flag), OK);
    EXPECT_EQ(flag, false);

    EXPECT_EQ(aclnnInit((std::string(OP_API_COMMON_UT_SRC_DIR) + "/conf/op_api/not_exit.json").c_str()), OK);
    EXPECT_EQ(op::internal::systemConfig.GetEnableDebugKernelFlag(flag), OK);
    EXPECT_EQ(flag, false);

    EXPECT_EQ(aclnnInit((std::string(OP_API_COMMON_UT_SRC_DIR) + "/conf/op_api/op_debug_config.json").c_str()), OK);
    EXPECT_EQ(op::internal::systemConfig.GetEnableDebugKernelFlag(flag), OK);
    EXPECT_EQ(flag, true);

    EXPECT_EQ(op::internal::systemConfig.SetEnableDebugKernelFlag(false), OK);
    setenv("NPU_COLLECT_PATH", "on", 1); // does overwrite
    EXPECT_EQ(aclnnInit(nullptr), OK);
    EXPECT_EQ(op::internal::systemConfig.GetEnableDebugKernelFlag(flag), OK);
    EXPECT_EQ(flag, true);
}

TEST_F(AclOpApiTest, aclnnFinalize) { EXPECT_EQ(aclnnFinalize(), OK); }

TEST_F(AclOpApiTest, aclGetViewStrides)
{
    EXPECT_NE(aclGetViewStrides(nullptr, nullptr, nullptr), OK);
    std::vector<int64_t> strides = {8, 1};
    CHECK_TENSOR(a, std::vector<int64_t>({4, 2}), std::vector<int64_t>({32}), aclDataType::ACL_FLOAT, strides.data(), 0,
                 aclFormat::ACL_FORMAT_ND, nullptr);
    int64_t* viewStrides = nullptr;
    uint64_t viewStridesNum = 0;
    EXPECT_EQ(aclGetViewStrides(a, &viewStrides, &viewStridesNum), OK);
    EXPECT_NE(viewStrides, nullptr);
    EXPECT_EQ(viewStridesNum, 2);
    EXPECT_EQ(viewStrides[0], 8);
    EXPECT_EQ(viewStrides[1], 1);
    delete[] viewStrides;
}

TEST_F(AclOpApiTest, aclGetViewOffset)
{
    EXPECT_NE(aclGetViewOffset(nullptr, nullptr), OK);
    std::vector<int64_t> strides = {8, 1};
    CHECK_TENSOR(a, std::vector<int64_t>({4, 2}), std::vector<int64_t>({32}), aclDataType::ACL_FLOAT, strides.data(), 0,
                 aclFormat::ACL_FORMAT_ND, nullptr);
    int64_t offset = 0;
    EXPECT_EQ(aclGetViewOffset(a, &offset), OK);
    EXPECT_EQ(offset, 0);
}

TEST_F(AclOpApiTest, aclGetFormat)
{
    EXPECT_NE(aclGetFormat(nullptr, nullptr), OK);
    std::vector<int64_t> strides = {8, 1};
    CHECK_TENSOR(a, std::vector<int64_t>({4, 2}), std::vector<int64_t>({32}), aclDataType::ACL_FLOAT, strides.data(), 0,
                 aclFormat::ACL_FORMAT_ND, nullptr);
    aclFormat formatRes = aclFormat::ACL_FORMAT_UNDEFINED;
    EXPECT_EQ(aclGetFormat(a, &formatRes), OK);
    EXPECT_EQ(formatRes, aclFormat::ACL_FORMAT_ND);
}

TEST_F(AclOpApiTest, aclGetDataType)
{
    EXPECT_NE(aclGetFormat(nullptr, nullptr), OK);
    std::vector<int64_t> strides = {8, 1};
    CHECK_TENSOR(a, std::vector<int64_t>({4, 2}), std::vector<int64_t>({32}), aclDataType::ACL_FLOAT, strides.data(), 0,
                 aclFormat::ACL_FORMAT_ND, nullptr);
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    EXPECT_EQ(aclGetDataType(a, &dataType), OK);
    EXPECT_EQ(dataType, aclDataType::ACL_FLOAT);
}

TEST_F(AclOpApiTest, aclGetIntArraySize)
{
    EXPECT_NE(aclGetIntArraySize(nullptr, nullptr), OK);
    int64_t values[] = {3, 4, 5};
    auto* value = aclCreateIntArray(values, sizeof(values) / sizeof(values[0]));
    uint64_t size = 0;
    EXPECT_EQ(aclGetIntArraySize(value, &size), OK);
    EXPECT_EQ(size, 3);
    EXPECT_EQ(aclDestroyIntArray(value), OK);
}

TEST_F(AclOpApiTest, aclGetFloatArraySize)
{
    EXPECT_NE(aclGetFloatArraySize(nullptr, nullptr), OK);
    float values[] = {3.0, 4.0, 5.0};
    auto* value = aclCreateFloatArray(values, sizeof(values) / sizeof(values[0]));
    uint64_t size = 0;
    EXPECT_EQ(aclGetFloatArraySize(value, &size), OK);
    EXPECT_EQ(size, 3);
    EXPECT_EQ(aclDestroyFloatArray(value), OK);
}

TEST_F(AclOpApiTest, aclGetBoolArraySize)
{
    EXPECT_NE(aclGetBoolArraySize(nullptr, nullptr), OK);
    bool values[] = {true, true, true};
    auto* value = aclCreateBoolArray(values, sizeof(values) / sizeof(values[0]));
    uint64_t size = 0;
    EXPECT_EQ(aclGetBoolArraySize(value, &size), OK);
    EXPECT_EQ(size, 3);
    EXPECT_EQ(aclDestroyBoolArray(value), OK);
}

TEST_F(AclOpApiTest, aclGetTensorListSize)
{
    EXPECT_NE(aclGetTensorListSize(nullptr, nullptr), OK);
    std::vector<int64_t> shape = {1, 2, 3};
    aclTensor* list2[] = {aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                          aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr),
                          aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                          aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr),
                          aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                          aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr)};
    auto aclTensorList2 = aclCreateTensorList(list2, 3);
    ASSERT_NE(aclTensorList2, nullptr);
    ASSERT_NE(aclTensorList2->GetData(), nullptr);
    EXPECT_EQ(aclTensorList2->Size(), 3);

    uint64_t size = 0;
    EXPECT_EQ(aclGetTensorListSize(aclTensorList2, &size), OK);
    EXPECT_EQ(size, 3);

    aclDestroyTensorList(aclTensorList2);
}
