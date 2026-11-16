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
#include "opdev/op_executor.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "thread_local_context.h"

using namespace op;
using namespace std;

class OpExecutorTest : public testing::Test {};

TEST_F(OpExecutorTest, TestOpExecutor)
{
    op::internal::GetThreadLocalContext().cacheHasFull_ = true;
    aclOpExecutor executor;
    bool boolValue[] = {true, false, true};
    auto array1 = executor.AllocBoolArray(boolValue, 3);
    EXPECT_NE(array1, nullptr);
    float floatValue[] = {1, 2, 3};
    auto array2 = executor.AllocFloatArray(floatValue, 3);
    EXPECT_NE(array2, nullptr);
    int64_t intValue[] = {1, 2, 3};
    auto array3 = executor.AllocIntArray(intValue, 3);
    EXPECT_NE(array3, nullptr);

    float fValue = 3.2;
    auto scalar1 = executor.AllocScalar(&fValue, op::DataType::DT_FLOAT);
    EXPECT_NE(scalar1, nullptr);
    auto scalar2 = executor.AllocScalar(float(1.2));
    EXPECT_NE(scalar2, nullptr);
    auto tensor1 = executor.AllocTensor({1, 2}, {1, 2}, DataType::DT_FLOAT, Format::FORMAT_ND, Format::FORMAT_ND);
    EXPECT_NE(tensor1, nullptr);
    auto tensor2 = executor.AllocTensor({1, 2}, DataType::DT_FLOAT);
    EXPECT_NE(tensor2, nullptr);
    auto tensor3 = executor.AllocTensor(DataType::DT_FLOAT, Format::FORMAT_ND, Format::FORMAT_ND);
    EXPECT_NE(tensor3, nullptr);
    auto tensor4 = executor.ConvertToTensor(scalar1, op::DataType::DT_FLOAT);
    EXPECT_NE(tensor4, nullptr);
    auto tensor5 = executor.ConvertToTensor(array1, DataType::DT_BOOL);
    EXPECT_NE(tensor5, nullptr);
    auto tensor6 = executor.ConvertToTensor(array2, DataType::DT_FLOAT);
    EXPECT_NE(tensor6, nullptr);
    auto tensor7 = executor.ConvertToTensor(array3, DataType::DT_INT64);
    EXPECT_NE(tensor7, nullptr);
    bfloat16 bf16Array[4] = {0, 1, 2, 3};
    auto tensor8 = executor.ConvertToTensor(bf16Array, 4, DataType::DT_BF16);
    EXPECT_NE(tensor8, nullptr);
    aclBf16Array bfArray(bf16Array, 4);
    auto tensor9 = executor.ConvertToTensor(&bfArray, DataType::DT_BF16);
    EXPECT_NE(tensor9, nullptr);
    auto tensor10 = executor.AllocHostTensor({100}, DataType::DT_INT64, Format::FORMAT_ND);
    EXPECT_NE(tensor10, nullptr);
    int64_t data[100] = {1};
    tensor10->SetIntData(data, 100, DataType::DT_INT64);
    tensor10->SetIntData(data, 100, DataType::DT_UINT64);
    tensor10->SetIntData(data, 100, DataType::DT_INT32);
    tensor10->SetIntData(data, 100, DataType::DT_UINT32);
    tensor10->SetIntData(data, 100, DataType::DT_INT16);
    tensor10->SetIntData(data, 100, DataType::DT_UINT16);
    tensor10->SetIntData(data, 100, DataType::DT_INT8);
    tensor10->SetIntData(data, 100, DataType::DT_UINT8);
    tensor10->SetIntData(data, 100, DataType::DT_DOUBLE);
    tensor10->SetBoolData((bool*)data, 100, DataType::DT_BOOL);
    tensor10->SetFloatData((float*)data, 100, DataType::DT_FLOAT);
    tensor10->SetFp16Data((fp16_t*)data, 100, DataType::DT_FLOAT16);
    tensor10->SetBf16Data((bfloat16*)data, 100, DataType::DT_BF16);
    auto tensor11 = executor.AllocHostTensor({100}, {100}, DataType::DT_FLOAT, Format::FORMAT_ND, Format::FORMAT_ND);
    EXPECT_NE(tensor11, nullptr);
    vector<int64_t> vec12(100, 1);
    auto tensor12 = executor.AllocHostTensor(vec12.data(), vec12.size(), DataType::DT_INT64);
    EXPECT_NE(tensor12, nullptr);
    bool vec13[100] = {true};
    auto tensor13 = executor.AllocHostTensor(vec13, 100, DataType::DT_BOOL);
    EXPECT_NE(tensor13, nullptr);
    vector<float> vec14(100, 1.0);
    auto tensor14 = executor.AllocHostTensor(vec14.data(), vec14.size(), DataType::DT_FLOAT);
    EXPECT_NE(tensor14, nullptr);
    vector<fp16_t> vec15(100, 1);
    auto tensor15 = executor.AllocHostTensor(vec15.data(), vec15.size(), DataType::DT_FLOAT16);
    EXPECT_NE(tensor15, nullptr);
    vector<bfloat16> vec16(100, 1);
    auto tensor16 = executor.AllocHostTensor(vec16.data(), vec16.size(), DataType::DT_BF16);
    EXPECT_NE(tensor16, nullptr);
    executor.UpdateTensorAddr(nullptr, 512);
    tensor2->SetFromWorkspace(false);
    tensor10->SetDataType(DataType::DT_UINT64);
    EXPECT_EQ(tensor10->GetDataType(), DataType::DT_UINT64);
}

TEST_F(OpExecutorTest, TestOpExecutorWorkspace)
{
    aclOpExecutor executor;
    void* addr = executor.GetWorkspaceAddr();
    EXPECT_EQ(addr, nullptr);
    uint64_t size = executor.GetWorkspaceSize();
    EXPECT_EQ(size, 0);
}

TEST_F(OpExecutorTest, NnopContiguousExecutorCheckOffsetSuccess)
{
    aclOpExecutor executor;
    uint64_t size = executor.GetLinearWorkspaceSize();
    EXPECT_EQ(size, 0);
    op::FVector<uint64_t> ws({1, 2, 3});
    executor.SetWorkspaceOffsets(ws);
    EXPECT_EQ(executor.GetWorkspaceOffsets().size(), 3);
    EXPECT_EQ(executor.GetWorkspaceOffsets()[0], 1);
    EXPECT_EQ(executor.GetWorkspaceOffsets()[1], 2);
    EXPECT_EQ(executor.GetWorkspaceOffsets()[2], 3);
}

TEST_F(OpExecutorTest, OpExecutorImplTest)
{
    aclOpExecutor executor;
    uint64_t size = executor.GetLinearWorkspaceSize();
    EXPECT_EQ(size, 0);
    op::FVector<uint64_t> ws({1, 2, 3});
    executor.SetWorkspaceOffsets(ws);
    EXPECT_EQ(executor.GetWorkspaceOffsets().size(), 3);
    EXPECT_EQ(executor.GetWorkspaceOffsets()[0], 1);
    EXPECT_EQ(executor.GetWorkspaceOffsets()[1], 2);
    EXPECT_EQ(executor.GetWorkspaceOffsets()[2], 3);
}
