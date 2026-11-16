/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include "gtest/gtest.h"
#include "opdev/fp16_t.h"
#include "opdev/op_dfx.h"
#include "opdev/framework_op.h"
#include "opdev/make_op_executor.h"
#include "thread_local_context.h"

OP_TYPE_REGISTER(CopyToNpu);

class FrameworkOpUt : public testing::Test {
protected:
    static void SetUpTestCase() { op::internal::GetThreadLocalContext().cacheHasFull_ = true; }

    static void TearDownTestCase() {}
};

TEST_F(FrameworkOpUt, CopyToNpuUtTest)
{
    auto executor = CREATE_EXECUTOR();
    std::vector<float> value(10, 1);
    auto srcArray = executor.get()->AllocFloatArray(value.data(), value.size());
    auto srcTensor = executor.get()->ConvertToTensor(srcArray, op::DataType::DT_FLOAT);
    auto dstTensor = op::CopyToNpu(srcTensor, executor.get());
    EXPECT_NE(dstTensor, nullptr);
    aclOpExecutor* executorPtr = nullptr;
    executor.ReleaseTo(&executorPtr);
    EXPECT_EQ(executorPtr->CheckLauncherRepeatable(), false);
    auto ret = executorPtr->Run();
    EXPECT_EQ(ret, ACLNN_SUCCESS);
    delete executorPtr;
}

TEST_F(FrameworkOpUt, CopyNpuToNpuUtTest)
{
    auto executor = CREATE_EXECUTOR();
    op::Shape shape = {5, 10};
    auto srcTensor = executor.get()->AllocTensor(shape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND);
    auto dstTensor = executor.get()->AllocTensor(shape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND);
    auto ret = op::CopyNpuToNpu(srcTensor, dstTensor, executor.get());
    EXPECT_EQ(ret, ACLNN_SUCCESS);
    aclOpExecutor* executorPtr = nullptr;
    executor.ReleaseTo(&executorPtr);
    EXPECT_EQ(executorPtr->CheckLauncherRepeatable(), false);
    ret = executorPtr->Run();
    EXPECT_EQ(ret, ACLNN_SUCCESS);
    delete executorPtr;
}

TEST_F(FrameworkOpUt, CopyToNpuSyncTest)
{
    auto executor = CREATE_EXECUTOR();
    std::vector<float> value;
    for (size_t i = 0; i < 75 * 1024 * 256; i++) {
        value.push_back(float(i));
    }
    auto srcArray = executor.get()->AllocFloatArray(value.data(), value.size());
    auto srcTensor = executor.get()->ConvertToTensor(srcArray, op::DataType::DT_FLOAT);
    auto dstTensor = op::CopyToNpuSync(srcTensor, executor.get());
    EXPECT_NE(dstTensor, nullptr);
    EXPECT_EQ(memcmp(srcTensor->GetData(), dstTensor->GetData(), 75 * 1024 * 1024), 0);
    aclOpExecutor* executorPtr = nullptr;
    executor.ReleaseTo(&executorPtr);
    delete[] static_cast<char*>(dstTensor->GetData());
    delete executorPtr;
}
