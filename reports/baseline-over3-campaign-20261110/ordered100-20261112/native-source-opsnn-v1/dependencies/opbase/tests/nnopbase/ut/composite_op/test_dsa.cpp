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
#include "acl/acl.h"
#include "opdev/fp16_t.h"
#include "opdev/op_dfx.h"
#include "dsa_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/shape_utils.h"
#include "thread_local_context.h"

OP_TYPE_REGISTER(DSARandomNormal);
OP_TYPE_REGISTER(DSARandomTruncatedNormal);
OP_TYPE_REGISTER(DSARandomUniform);
OP_TYPE_REGISTER(DSAGenBitMask);

using float32_t = float;

class DSAUt : public testing::Test {
protected:
    static void SetUpTestCase() { op::internal::GetThreadLocalContext().cacheHasFull_ = true; }

    static void TearDownTestCase() {}
};

aclTensor* DSARandomNormal(const aclIntArray* outShape, uint64_t seed, uint64_t offset, const aclScalar* mean,
                           const aclScalar* std, aclOpExecutor* executor)
{
    L0_DFX(DSARandomNormal, outShape, seed, offset, mean, std);
    op::Shape shape;
    op::ToShape(outShape->GetData(), outShape->Size(), shape);
    auto out = executor->AllocTensor(shape, mean->GetDataType(), op::Format::FORMAT_ND);
    auto count = static_cast<uint64_t>(shape.GetShapeSize());
    ADD_TO_LAUNCHER_LIST_DSA(DSARandomNormal, OP_INPUT(count, seed, offset, mean, std), OP_OUTPUT(out), OP_ATTR(0));
    return out;
}

aclTensor* DSARandomNormal(const aclTensor* count, const aclTensor* seed, const aclTensor* offset,
                           const aclTensor* mean, const aclTensor* std, aclOpExecutor* executor)
{
    L0_DFX(DSARandomNormal, count, seed, offset, mean, std);
    auto out = executor->AllocTensor(count->GetViewShape(), mean->GetDataType(), op::Format::FORMAT_ND);
    ADD_TO_LAUNCHER_LIST_DSA(DSARandomNormal, OP_INPUT(count, seed, offset, mean, std), OP_OUTPUT(out), OP_ATTR(0));
    return out;
}

aclTensor* DSARandomNormal(const aclTensor* count, uint64_t seed, const aclTensor* offset, const aclScalar* mean,
                           const aclTensor* std, aclOpExecutor* executor)
{
    L0_DFX(DSARandomNormal, count, seed, offset, mean, std);
    auto out = executor->AllocTensor(count->GetViewShape(), mean->GetDataType(), op::Format::FORMAT_ND);
    ADD_TO_LAUNCHER_LIST_DSA(DSARandomNormal, OP_INPUT(count, seed, offset, mean, std), OP_OUTPUT(out), OP_ATTR(0));
    return out;
}

aclTensor* DSARandomTruncatedNormal(const aclIntArray* outShape, uint64_t seed, uint64_t offset, const aclScalar* mean,
                                    const aclScalar* std, aclOpExecutor* executor)
{
    L0_DFX(DSARandomTruncatedNormal, outShape, seed, offset, mean, std);
    op::Shape shape;
    op::ToShape(outShape->GetData(), outShape->Size(), shape);
    auto out = executor->AllocTensor(shape, mean->GetDataType(), op::Format::FORMAT_ND);
    auto count = static_cast<uint64_t>(shape.GetShapeSize());
    ADD_TO_LAUNCHER_LIST_DSA(DSARandomTruncatedNormal, OP_INPUT(count, seed, offset, mean, std), OP_OUTPUT(out),
                             OP_ATTR(0));
    return out;
}

aclTensor* DSARandomTruncatedNormal(const aclTensor* count, const aclTensor* seed, const aclTensor* offset,
                                    const aclTensor* mean, const aclTensor* std, aclOpExecutor* executor)
{
    L0_DFX(DSARandomTruncatedNormal, count, seed, offset, mean, std);
    auto out = executor->AllocTensor(count->GetViewShape(), mean->GetDataType(), op::Format::FORMAT_ND);
    ADD_TO_LAUNCHER_LIST_DSA(DSARandomTruncatedNormal, OP_INPUT(count, seed, offset, mean, std), OP_OUTPUT(out),
                             OP_ATTR(0));
    return out;
}

aclTensor* DSARandomUniform(const aclIntArray* outShape, uint64_t seed, uint64_t offset, const aclScalar* low,
                            const aclScalar* high, aclOpExecutor* executor)
{
    L0_DFX(DSARandomUniform, outShape, seed, offset, low, high);
    op::Shape shape;
    op::ToShape(outShape->GetData(), outShape->Size(), shape);
    auto out = executor->AllocTensor(shape, low->GetDataType(), op::Format::FORMAT_ND);
    auto count = static_cast<uint64_t>(shape.GetShapeSize());
    ADD_TO_LAUNCHER_LIST_DSA(DSARandomUniform, OP_INPUT(count, seed, offset, low, high), OP_OUTPUT(out), OP_ATTR(0));
    return out;
}

aclTensor* DSARandomUniform(const aclTensor* count, const aclTensor* seed, const aclTensor* offset,
                            const aclTensor* low, const aclTensor* high, aclOpExecutor* executor)
{
    L0_DFX(DSARandomUniform, count, seed, offset, low, high);
    auto out = executor->AllocTensor(count->GetViewShape(), low->GetDataType(), op::Format::FORMAT_ND);
    ADD_TO_LAUNCHER_LIST_DSA(DSARandomUniform, OP_INPUT(count, seed, offset, low, high), OP_OUTPUT(out), OP_ATTR(0));
    return out;
}

aclTensor* DSAGenBitMask(const aclIntArray* outShape, uint64_t seed, uint64_t offset, const aclScalar* dropout,
                         aclOpExecutor* executor)
{
    L0_DFX(DSAGenBitMask, outShape, seed, offset, dropout);
    op::Shape shape;
    op::ToShape(outShape->GetData(), outShape->Size(), shape);
    auto out = executor->AllocTensor(shape, dropout->GetDataType(), op::Format::FORMAT_ND);
    auto count = static_cast<uint64_t>(shape.GetShapeSize());
    ADD_TO_LAUNCHER_LIST_DSA(DSAGenBitMask, OP_INPUT(count, seed, offset, dropout), OP_OUTPUT(out), OP_ATTR(0));
    return out;
}

aclTensor* DSAGenBitMask(const aclTensor* count, const aclTensor* seed, const aclTensor* offset,
                         const aclTensor* dropout, aclOpExecutor* executor)
{
    L0_DFX(DSAGenBitMask, count, seed, offset, dropout);
    auto out = executor->AllocTensor(dropout->GetViewShape(), dropout->GetDataType(), op::Format::FORMAT_ND);
    ADD_TO_LAUNCHER_LIST_DSA(DSAGenBitMask, OP_INPUT(count, seed, offset, dropout), OP_OUTPUT(out), OP_ATTR(0));
    return out;
}

aclTensor* GetInt32AclTensor(aclOpExecutor* executor, int32_t value)
{
    auto tensorPtr = executor->AllocTensor({4}, op::DataType::DT_INT32);
    tensorPtr->SetFromWorkspace(false);
    void* ptr = nullptr;
    aclrtMalloc(&ptr, 4, ACL_MEM_MALLOC_HUGE_FIRST);
    tensorPtr->SetStorageAddr(ptr);
    aclrtMemcpy(ptr, 4, &value, 4, ACL_MEMCPY_HOST_TO_DEVICE);

    return tensorPtr;
}

aclTensor* GetUint32AclTensor(aclOpExecutor* executor, uint32_t value)
{
    auto tensorPtr = executor->AllocTensor({4}, op::DataType::DT_UINT32);
    tensorPtr->SetFromWorkspace(false);
    void* ptr = nullptr;
    aclrtMalloc(&ptr, 4, ACL_MEM_MALLOC_HUGE_FIRST);
    tensorPtr->SetStorageAddr(ptr);
    aclrtMemcpy(ptr, 4, &value, 4, ACL_MEMCPY_HOST_TO_DEVICE);

    return tensorPtr;
}

aclTensor* GetInt64AclTensor(aclOpExecutor* executor, int64_t value)
{
    auto tensorPtr = executor->AllocTensor({8}, op::DataType::DT_INT64);
    tensorPtr->SetFromWorkspace(false);
    void* ptr = nullptr;
    aclrtMalloc(&ptr, 8, ACL_MEM_MALLOC_HUGE_FIRST);
    tensorPtr->SetStorageAddr(ptr);
    aclrtMemcpy(ptr, 8, &value, 8, ACL_MEMCPY_HOST_TO_DEVICE);

    return tensorPtr;
}

aclTensor* GetUint64AclTensor(aclOpExecutor* executor, uint64_t value)
{
    auto tensorPtr = executor->AllocTensor({8}, op::DataType::DT_UINT64);
    tensorPtr->SetFromWorkspace(false);
    void* ptr = nullptr;
    aclrtMalloc(&ptr, 8, ACL_MEM_MALLOC_HUGE_FIRST);
    tensorPtr->SetStorageAddr(ptr);
    aclrtMemcpy(ptr, 8, &value, 8, ACL_MEMCPY_HOST_TO_DEVICE);

    return tensorPtr;
}

aclTensor* GetFP16AclTensor(aclOpExecutor* executor, op::fp16_t value)
{
    auto tensorPtr = executor->AllocTensor({2}, op::DataType::DT_FLOAT16);
    tensorPtr->SetFromWorkspace(false);
    void* ptr = nullptr;
    aclrtMalloc(&ptr, 2, ACL_MEM_MALLOC_HUGE_FIRST);
    tensorPtr->SetStorageAddr(ptr);
    aclrtMemcpy(ptr, 2, &value, 2, ACL_MEMCPY_HOST_TO_DEVICE);

    return tensorPtr;
}

aclTensor* GetFP32AclTensor(aclOpExecutor* executor, float32_t value)
{
    auto tensorPtr = executor->AllocTensor({4}, op::DataType::DT_FLOAT);
    tensorPtr->SetFromWorkspace(false);
    void* ptr = nullptr;
    aclrtMalloc(&ptr, 4, ACL_MEM_MALLOC_HUGE_FIRST);
    tensorPtr->SetStorageAddr(ptr);
    aclrtMemcpy(ptr, 4, &value, 4, ACL_MEMCPY_HOST_TO_DEVICE);

    return tensorPtr;
}

TEST_F(DSAUt, DSARandomNormalTestCase0)
{
    auto rc = aclrtSetDevice(0);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    aclrtContext ctx = nullptr;
    rc = aclrtGetCurrentContext(&ctx);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    rc = aclrtSetCurrentContext(ctx);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    aclrtStream stream = nullptr;
    rc = aclrtCreateStream(&stream);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    auto uniqueExecutor = CREATE_EXECUTOR();
    aclOpExecutor* executor = nullptr;
    uniqueExecutor.ReleaseTo(&executor);

    std::array<int64_t, 1> dims{100};
    auto outShape = executor->AllocIntArray(dims.data(), dims.size());
    auto mean = executor->AllocScalar(0.0f);
    auto std = executor->AllocScalar(1.0f);

    auto out = DSARandomNormal(outShape, 0, 100, mean, std, executor);
    ASSERT_NE(out, nullptr);
    out->SetFromWorkspace(false);

    void* workspacePtr = nullptr;
    auto workspaceSize = 512;
    rc = aclrtMalloc(&workspacePtr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    executor->UpdateTensorAddr(workspacePtr, workspaceSize);

    void* outputPtr = nullptr;
    rc = aclrtMalloc(&outputPtr, 1024, ACL_MEM_MALLOC_HUGE_FIRST);
    ASSERT_EQ(rc, ACLNN_SUCCESS);
    out->SetStorageAddr(outputPtr);

    executor->SetStream(stream);
    rc = executor->Run();
    ASSERT_EQ(rc, ACLNN_SUCCESS);
    delete executor;
    rc = aclrtSynchronizeStream(stream);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    std::vector<float> resultData(100, 0);
    rc = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outputPtr, 400,
                     ACL_MEMCPY_DEVICE_TO_HOST);
    ASSERT_EQ(rc, ACLNN_SUCCESS);
}

TEST_F(DSAUt, DSARandomNormalTestCase1)
{
    auto rc = aclrtSetDevice(0);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    aclrtContext ctx = nullptr;
    rc = aclrtGetCurrentContext(&ctx);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    rc = aclrtSetCurrentContext(ctx);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    aclrtStream stream = nullptr;
    rc = aclrtCreateStream(&stream);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    auto uniqueExecutor = CREATE_EXECUTOR();
    aclOpExecutor* executor = nullptr;
    uniqueExecutor.ReleaseTo(&executor);

    auto count = GetUint64AclTensor(executor, 100);
    auto seed = GetUint64AclTensor(executor, 0);
    auto counter = GetUint64AclTensor(executor, 100);
    auto mean = GetFP32AclTensor(executor, 0.0f);
    auto std = GetFP32AclTensor(executor, 1.0f);

    auto out = DSARandomNormal(count, seed, counter, mean, std, executor);
    ASSERT_NE(out, nullptr);
    out->SetFromWorkspace(false);

    void* workspacePtr = nullptr;
    auto workspaceSize = 512;
    rc = aclrtMalloc(&workspacePtr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    executor->UpdateTensorAddr(workspacePtr, workspaceSize);

    void* outputPtr = nullptr;
    rc = aclrtMalloc(&outputPtr, 1024, ACL_MEM_MALLOC_HUGE_FIRST);
    ASSERT_EQ(rc, ACLNN_SUCCESS);
    out->SetStorageAddr(outputPtr);

    executor->SetStream(stream);
    rc = executor->Run();
    ASSERT_EQ(rc, ACLNN_SUCCESS);
    delete executor;
    rc = aclrtSynchronizeStream(stream);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    std::vector<float> resultData(100, 0);
    rc = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outputPtr, 400,
                     ACL_MEMCPY_DEVICE_TO_HOST);
    ASSERT_EQ(rc, ACLNN_SUCCESS);
}

TEST_F(DSAUt, DSARandomNormalTestCase2)
{
    auto rc = aclrtSetDevice(0);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    aclrtContext ctx = nullptr;
    rc = aclrtGetCurrentContext(&ctx);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    rc = aclrtSetCurrentContext(ctx);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    aclrtStream stream = nullptr;
    rc = aclrtCreateStream(&stream);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    auto uniqueExecutor = CREATE_EXECUTOR();
    aclOpExecutor* executor = nullptr;
    uniqueExecutor.ReleaseTo(&executor);

    auto count = GetUint64AclTensor(executor, 100);
    auto seed = GetUint64AclTensor(executor, 0);
    auto counter = GetUint64AclTensor(executor, 100);
    auto mean = GetFP16AclTensor(executor, op::fp16_t(0.0));
    auto std = GetFP16AclTensor(executor, op::fp16_t(1.0));

    auto out = DSARandomNormal(count, seed, counter, mean, std, executor);
    ASSERT_NE(out, nullptr);
    out->SetFromWorkspace(false);

    void* workspacePtr = nullptr;
    auto workspaceSize = 512;
    rc = aclrtMalloc(&workspacePtr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    executor->UpdateTensorAddr(workspacePtr, workspaceSize);

    void* outputPtr = nullptr;
    rc = aclrtMalloc(&outputPtr, 1024, ACL_MEM_MALLOC_HUGE_FIRST);
    ASSERT_EQ(rc, ACLNN_SUCCESS);
    out->SetStorageAddr(outputPtr);

    executor->SetStream(stream);
    rc = executor->Run();
    ASSERT_EQ(rc, ACLNN_SUCCESS);
    delete executor;
    rc = aclrtSynchronizeStream(stream);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    std::vector<op::fp16_t> resultData(100, 0);
    rc = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outputPtr, 200,
                     ACL_MEMCPY_DEVICE_TO_HOST);
    ASSERT_EQ(rc, ACLNN_SUCCESS);
}

TEST_F(DSAUt, DSARandomNormalTestCase3)
{
    auto rc = aclrtSetDevice(0);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    aclrtContext ctx = nullptr;
    rc = aclrtGetCurrentContext(&ctx);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    rc = aclrtSetCurrentContext(ctx);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    aclrtStream stream = nullptr;
    rc = aclrtCreateStream(&stream);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    auto uniqueExecutor = CREATE_EXECUTOR();
    aclOpExecutor* executor = nullptr;
    uniqueExecutor.ReleaseTo(&executor);

    auto count = GetUint64AclTensor(executor, 100);
    // auto seed = GetUint64AclTensor(executor, 0);
    auto counter = GetUint64AclTensor(executor, 100);
    auto mean = executor->AllocScalar(0.0f);
    auto std = GetFP32AclTensor(executor, 1.0f);

    auto out = DSARandomNormal(count, 0, counter, mean, std, executor);
    ASSERT_NE(out, nullptr);
    out->SetFromWorkspace(false);

    void* workspacePtr = nullptr;
    auto workspaceSize = 512;
    rc = aclrtMalloc(&workspacePtr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    executor->UpdateTensorAddr(workspacePtr, workspaceSize);

    void* outputPtr = nullptr;
    rc = aclrtMalloc(&outputPtr, 1024, ACL_MEM_MALLOC_HUGE_FIRST);
    ASSERT_EQ(rc, ACLNN_SUCCESS);
    out->SetStorageAddr(outputPtr);

    executor->SetStream(stream);
    rc = executor->Run();
    ASSERT_EQ(rc, ACLNN_SUCCESS);
    delete executor;
    rc = aclrtSynchronizeStream(stream);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    std::vector<float> resultData(100, 0);
    rc = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outputPtr, 400,
                     ACL_MEMCPY_DEVICE_TO_HOST);
    ASSERT_EQ(rc, ACLNN_SUCCESS);
}

TEST_F(DSAUt, DSARandomTruncatedNormalTestCase0)
{
    auto rc = aclrtSetDevice(0);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    aclrtContext ctx = nullptr;
    rc = aclrtGetCurrentContext(&ctx);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    rc = aclrtSetCurrentContext(ctx);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    aclrtStream stream = nullptr;
    rc = aclrtCreateStream(&stream);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    auto uniqueExecutor = CREATE_EXECUTOR();
    aclOpExecutor* executor = nullptr;
    uniqueExecutor.ReleaseTo(&executor);

    std::array<int64_t, 1> dims{100};
    auto outShape = executor->AllocIntArray(dims.data(), dims.size());
    auto mean = executor->AllocScalar(0.0f);
    auto std = executor->AllocScalar(1.0f);

    auto out = DSARandomTruncatedNormal(outShape, 0, 100, mean, std, executor);
    ASSERT_NE(out, nullptr);
    out->SetFromWorkspace(false);

    void* workspacePtr = nullptr;
    auto workspaceSize = 512;
    rc = aclrtMalloc(&workspacePtr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    executor->UpdateTensorAddr(workspacePtr, workspaceSize);

    void* outputPtr = nullptr;
    rc = aclrtMalloc(&outputPtr, 1024, ACL_MEM_MALLOC_HUGE_FIRST);
    ASSERT_EQ(rc, ACLNN_SUCCESS);
    out->SetStorageAddr(outputPtr);

    executor->SetStream(stream);
    rc = executor->Run();
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    rc = aclrtSynchronizeStream(stream);
    ASSERT_EQ(rc, ACLNN_SUCCESS);
    delete executor;
    std::vector<float> resultData(100, 0);
    rc = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outputPtr, 400,
                     ACL_MEMCPY_DEVICE_TO_HOST);
    ASSERT_EQ(rc, ACLNN_SUCCESS);
}

TEST_F(DSAUt, DSARandomTruncatedNormalTestCase1)
{
    auto rc = aclrtSetDevice(0);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    aclrtContext ctx = nullptr;
    rc = aclrtGetCurrentContext(&ctx);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    rc = aclrtSetCurrentContext(ctx);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    aclrtStream stream = nullptr;
    rc = aclrtCreateStream(&stream);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    auto uniqueExecutor = CREATE_EXECUTOR();
    aclOpExecutor* executor = nullptr;
    uniqueExecutor.ReleaseTo(&executor);

    auto count = GetUint64AclTensor(executor, 100);
    auto seed = GetUint64AclTensor(executor, 0);
    auto counter = GetUint64AclTensor(executor, 100);
    auto mean = GetFP32AclTensor(executor, 0.0f);
    auto std = GetFP32AclTensor(executor, 1.0f);

    auto out = DSARandomTruncatedNormal(count, seed, counter, mean, std, executor);
    ASSERT_NE(out, nullptr);
    out->SetFromWorkspace(false);

    void* workspacePtr = nullptr;
    auto workspaceSize = 512;
    rc = aclrtMalloc(&workspacePtr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    executor->UpdateTensorAddr(workspacePtr, workspaceSize);

    void* outputPtr = nullptr;
    rc = aclrtMalloc(&outputPtr, 1024, ACL_MEM_MALLOC_HUGE_FIRST);
    ASSERT_EQ(rc, ACLNN_SUCCESS);
    out->SetStorageAddr(outputPtr);

    executor->SetStream(stream);
    rc = executor->Run();
    ASSERT_EQ(rc, ACLNN_SUCCESS);
    delete executor;
    rc = aclrtSynchronizeStream(stream);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    std::vector<float> resultData(100, 0);
    rc = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outputPtr, 400,
                     ACL_MEMCPY_DEVICE_TO_HOST);
    ASSERT_EQ(rc, ACLNN_SUCCESS);
}

TEST_F(DSAUt, DSARandomTruncatedNormalTestCase2)
{
    auto rc = aclrtSetDevice(0);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    aclrtContext ctx = nullptr;
    rc = aclrtGetCurrentContext(&ctx);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    rc = aclrtSetCurrentContext(ctx);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    aclrtStream stream = nullptr;
    rc = aclrtCreateStream(&stream);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    auto uniqueExecutor = CREATE_EXECUTOR();
    aclOpExecutor* executor = nullptr;
    uniqueExecutor.ReleaseTo(&executor);

    auto count = GetUint64AclTensor(executor, 100);
    auto seed = GetUint64AclTensor(executor, 0);
    auto counter = GetUint64AclTensor(executor, 100);
    auto mean = GetFP16AclTensor(executor, op::fp16_t(0.0));
    auto std = GetFP16AclTensor(executor, op::fp16_t(0.1));

    auto out = DSARandomTruncatedNormal(count, seed, counter, mean, std, executor);
    ASSERT_NE(out, nullptr);
    out->SetFromWorkspace(false);

    void* workspacePtr = nullptr;
    auto workspaceSize = 512;
    rc = aclrtMalloc(&workspacePtr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    executor->UpdateTensorAddr(workspacePtr, workspaceSize);

    void* outputPtr = nullptr;
    rc = aclrtMalloc(&outputPtr, 1024, ACL_MEM_MALLOC_HUGE_FIRST);
    ASSERT_EQ(rc, ACLNN_SUCCESS);
    out->SetStorageAddr(outputPtr);

    executor->SetStream(stream);
    rc = executor->Run();
    ASSERT_EQ(rc, ACLNN_SUCCESS);
    delete executor;
    rc = aclrtSynchronizeStream(stream);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    std::vector<op::fp16_t> resultData(100, 0);
    rc = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outputPtr, 200,
                     ACL_MEMCPY_DEVICE_TO_HOST);
    ASSERT_EQ(rc, ACLNN_SUCCESS);
}

TEST_F(DSAUt, DSARandomUniformTestCase0)
{
    auto rc = aclrtSetDevice(0);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    aclrtContext ctx = nullptr;
    rc = aclrtGetCurrentContext(&ctx);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    rc = aclrtSetCurrentContext(ctx);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    aclrtStream stream = nullptr;
    rc = aclrtCreateStream(&stream);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    auto uniqueExecutor = CREATE_EXECUTOR();
    aclOpExecutor* executor = nullptr;
    uniqueExecutor.ReleaseTo(&executor);

    std::array<int64_t, 1> dims{100};
    auto outShape = executor->AllocIntArray(dims.data(), dims.size());
    auto low = executor->AllocScalar(10);
    auto high = executor->AllocScalar(100);

    auto out = DSARandomUniform(outShape, 0, 100, low, high, executor);
    ASSERT_NE(out, nullptr);
    out->SetFromWorkspace(false);

    void* workspacePtr = nullptr;
    auto workspaceSize = 512;
    rc = aclrtMalloc(&workspacePtr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    executor->UpdateTensorAddr(workspacePtr, workspaceSize);

    void* outputPtr = nullptr;
    rc = aclrtMalloc(&outputPtr, 1024, ACL_MEM_MALLOC_HUGE_FIRST);
    ASSERT_EQ(rc, ACLNN_SUCCESS);
    out->SetStorageAddr(outputPtr);

    executor->SetStream(stream);
    rc = executor->Run();
    ASSERT_EQ(rc, ACLNN_SUCCESS);
    delete executor;
    rc = aclrtSynchronizeStream(stream);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    std::vector<int32_t> resultData(100, 0);
    rc = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outputPtr, 400,
                     ACL_MEMCPY_DEVICE_TO_HOST);
    ASSERT_EQ(rc, ACLNN_SUCCESS);
}

TEST_F(DSAUt, DSARandomUniformTestCase1)
{
    auto rc = aclrtSetDevice(0);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    aclrtContext ctx = nullptr;
    rc = aclrtGetCurrentContext(&ctx);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    rc = aclrtSetCurrentContext(ctx);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    aclrtStream stream = nullptr;
    rc = aclrtCreateStream(&stream);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    auto uniqueExecutor = CREATE_EXECUTOR();
    aclOpExecutor* executor = nullptr;
    uniqueExecutor.ReleaseTo(&executor);

    auto count = GetUint64AclTensor(executor, 100);
    auto seed = GetUint64AclTensor(executor, 0);
    auto counter = GetUint64AclTensor(executor, 100);
    auto low = GetFP32AclTensor(executor, 100.0);
    auto high = GetFP32AclTensor(executor, 200.0);

    auto out = DSARandomUniform(count, seed, counter, low, high, executor);
    ASSERT_NE(out, nullptr);
    out->SetFromWorkspace(false);

    void* workspacePtr = nullptr;
    auto workspaceSize = 512;
    rc = aclrtMalloc(&workspacePtr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    executor->UpdateTensorAddr(workspacePtr, workspaceSize);

    void* outputPtr = nullptr;
    rc = aclrtMalloc(&outputPtr, 1024, ACL_MEM_MALLOC_HUGE_FIRST);
    ASSERT_EQ(rc, ACLNN_SUCCESS);
    out->SetStorageAddr(outputPtr);

    executor->SetStream(stream);
    rc = executor->Run();
    ASSERT_EQ(rc, ACLNN_SUCCESS);
    delete executor;
    rc = aclrtSynchronizeStream(stream);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    std::vector<float> resultData(100, 0);
    rc = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outputPtr, 400,
                     ACL_MEMCPY_DEVICE_TO_HOST);
    ASSERT_EQ(rc, ACLNN_SUCCESS);
}

TEST_F(DSAUt, DSARandomUniformTestCase2)
{
    auto rc = aclrtSetDevice(0);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    aclrtContext ctx = nullptr;
    rc = aclrtGetCurrentContext(&ctx);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    rc = aclrtSetCurrentContext(ctx);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    aclrtStream stream = nullptr;
    rc = aclrtCreateStream(&stream);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    auto uniqueExecutor = CREATE_EXECUTOR();
    aclOpExecutor* executor = nullptr;
    uniqueExecutor.ReleaseTo(&executor);

    auto count = GetUint64AclTensor(executor, 100);
    auto seed = GetUint64AclTensor(executor, 0);
    auto counter = GetUint64AclTensor(executor, 100);
    auto low = GetFP16AclTensor(executor, op::fp16_t(100.0));
    auto high = GetFP16AclTensor(executor, op::fp16_t(200.0));

    auto out = DSARandomUniform(count, seed, counter, low, high, executor);
    ASSERT_NE(out, nullptr);
    out->SetFromWorkspace(false);

    void* workspacePtr = nullptr;
    auto workspaceSize = 512;
    rc = aclrtMalloc(&workspacePtr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    executor->UpdateTensorAddr(workspacePtr, workspaceSize);

    void* outputPtr = nullptr;
    rc = aclrtMalloc(&outputPtr, 1024, ACL_MEM_MALLOC_HUGE_FIRST);
    ASSERT_EQ(rc, ACLNN_SUCCESS);
    out->SetStorageAddr(outputPtr);

    executor->SetStream(stream);
    rc = executor->Run();
    ASSERT_EQ(rc, ACLNN_SUCCESS);
    delete executor;
    rc = aclrtSynchronizeStream(stream);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    std::vector<op::fp16_t> resultData(100, 0);
    rc = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outputPtr, 200,
                     ACL_MEMCPY_DEVICE_TO_HOST);
    ASSERT_EQ(rc, ACLNN_SUCCESS);
}

TEST_F(DSAUt, DSARandomUniformTestCase3)
{
    auto rc = aclrtSetDevice(0);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    aclrtContext ctx = nullptr;
    rc = aclrtGetCurrentContext(&ctx);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    rc = aclrtSetCurrentContext(ctx);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    aclrtStream stream = nullptr;
    rc = aclrtCreateStream(&stream);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    auto uniqueExecutor = CREATE_EXECUTOR();
    aclOpExecutor* executor = nullptr;
    uniqueExecutor.ReleaseTo(&executor);

    auto count = GetUint64AclTensor(executor, 100);
    auto seed = GetUint64AclTensor(executor, 0);
    auto counter = GetUint64AclTensor(executor, 100);
    auto low = GetInt64AclTensor(executor, -100);
    auto high = GetInt64AclTensor(executor, 200);

    auto out = DSARandomUniform(count, seed, counter, low, high, executor);
    ASSERT_NE(out, nullptr);
    out->SetFromWorkspace(false);

    void* workspacePtr = nullptr;
    auto workspaceSize = 512;
    rc = aclrtMalloc(&workspacePtr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    executor->UpdateTensorAddr(workspacePtr, workspaceSize);

    void* outputPtr = nullptr;
    rc = aclrtMalloc(&outputPtr, 1024, ACL_MEM_MALLOC_HUGE_FIRST);
    ASSERT_EQ(rc, ACLNN_SUCCESS);
    out->SetStorageAddr(outputPtr);

    executor->SetStream(stream);
    rc = executor->Run();
    ASSERT_EQ(rc, ACLNN_SUCCESS);
    delete executor;
    rc = aclrtSynchronizeStream(stream);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    std::vector<int64_t> resultData(100, 0);
    rc = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outputPtr, 800,
                     ACL_MEMCPY_DEVICE_TO_HOST);
    ASSERT_EQ(rc, ACLNN_SUCCESS);
}

TEST_F(DSAUt, DSAGenBitMaskTestCase0)
{
    auto rc = aclrtSetDevice(0);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    aclrtContext ctx = nullptr;
    rc = aclrtGetCurrentContext(&ctx);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    rc = aclrtSetCurrentContext(ctx);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    aclrtStream stream = nullptr;
    rc = aclrtCreateStream(&stream);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    auto uniqueExecutor = CREATE_EXECUTOR();
    aclOpExecutor* executor = nullptr;
    uniqueExecutor.ReleaseTo(&executor);

    std::array<int64_t, 1> dims{32};
    auto outShape = executor->AllocIntArray(dims.data(), dims.size());
    auto dropout = executor->AllocScalar(0.5f);

    auto out = DSAGenBitMask(outShape, 0, 0, dropout, executor);
    ASSERT_NE(out, nullptr);
    out->SetFromWorkspace(false);

    void* workspacePtr = nullptr;
    auto workspaceSize = 512;
    rc = aclrtMalloc(&workspacePtr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    executor->UpdateTensorAddr(workspacePtr, workspaceSize);

    void* outputPtr = nullptr;
    rc = aclrtMalloc(&outputPtr, 4, ACL_MEM_MALLOC_HUGE_FIRST);
    ASSERT_EQ(rc, ACLNN_SUCCESS);
    out->SetStorageAddr(outputPtr);

    executor->SetStream(stream);
    rc = executor->Run();
    ASSERT_EQ(rc, ACLNN_SUCCESS);
    delete executor;
    rc = aclrtSynchronizeStream(stream);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    std::vector<uint32_t> resultData(1, 0);
    rc = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outputPtr, 4,
                     ACL_MEMCPY_DEVICE_TO_HOST);
    ASSERT_EQ(rc, ACLNN_SUCCESS);
}

TEST_F(DSAUt, DSAGenBitMaskTestCase1)
{
    auto rc = aclrtSetDevice(0);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    aclrtContext ctx = nullptr;
    rc = aclrtGetCurrentContext(&ctx);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    rc = aclrtSetCurrentContext(ctx);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    aclrtStream stream = nullptr;
    rc = aclrtCreateStream(&stream);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    auto uniqueExecutor = CREATE_EXECUTOR();
    aclOpExecutor* executor = nullptr;
    uniqueExecutor.ReleaseTo(&executor);

    auto count = GetUint64AclTensor(executor, 32);
    auto seed = GetUint64AclTensor(executor, 0);
    auto counter = GetUint64AclTensor(executor, 0);
    auto dropout = GetFP32AclTensor(executor, 0.5f);

    auto out = DSAGenBitMask(count, seed, counter, dropout, executor);
    ASSERT_NE(out, nullptr);
    out->SetFromWorkspace(false);

    void* workspacePtr = nullptr;
    auto workspaceSize = 512;
    rc = aclrtMalloc(&workspacePtr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    executor->UpdateTensorAddr(workspacePtr, workspaceSize);

    void* outputPtr = nullptr;
    rc = aclrtMalloc(&outputPtr, 4, ACL_MEM_MALLOC_HUGE_FIRST);
    ASSERT_EQ(rc, ACLNN_SUCCESS);
    out->SetStorageAddr(outputPtr);

    executor->SetStream(stream);
    rc = executor->Run();
    ASSERT_EQ(rc, ACLNN_SUCCESS);
    delete executor;
    rc = aclrtSynchronizeStream(stream);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    std::vector<uint32_t> resultData(1, 0);
    rc = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outputPtr, 4,
                     ACL_MEMCPY_DEVICE_TO_HOST);
    ASSERT_EQ(rc, ACLNN_SUCCESS);
}

TEST_F(DSAUt, DSAGenBitMaskTestCase2)
{
    auto rc = aclrtSetDevice(0);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    aclrtContext ctx = nullptr;
    rc = aclrtGetCurrentContext(&ctx);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    rc = aclrtSetCurrentContext(ctx);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    aclrtStream stream = nullptr;
    rc = aclrtCreateStream(&stream);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    auto uniqueExecutor = CREATE_EXECUTOR();
    aclOpExecutor* executor = nullptr;
    uniqueExecutor.ReleaseTo(&executor);

    auto count = GetUint64AclTensor(executor, 64);
    auto seed = GetUint64AclTensor(executor, 0);
    auto counter = GetUint64AclTensor(executor, 0);
    auto dropout = GetFP16AclTensor(executor, op::fp16_t(0.5));

    auto out = DSAGenBitMask(count, seed, counter, dropout, executor);
    ASSERT_NE(out, nullptr);
    out->SetFromWorkspace(false);

    void* workspacePtr = nullptr;
    auto workspaceSize = 512;
    rc = aclrtMalloc(&workspacePtr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    executor->UpdateTensorAddr(workspacePtr, workspaceSize);

    void* outputPtr = nullptr;
    rc = aclrtMalloc(&outputPtr, 4, ACL_MEM_MALLOC_HUGE_FIRST);
    ASSERT_EQ(rc, ACLNN_SUCCESS);
    out->SetStorageAddr(outputPtr);

    executor->SetStream(stream);
    rc = executor->Run();
    ASSERT_EQ(rc, ACLNN_SUCCESS);
    delete executor;
    rc = aclrtSynchronizeStream(stream);
    ASSERT_EQ(rc, ACLNN_SUCCESS);

    std::vector<uint32_t> resultData(2, 0);
    rc = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outputPtr, 8,
                     ACL_MEMCPY_DEVICE_TO_HOST);
    ASSERT_EQ(rc, ACLNN_SUCCESS);
}

TEST_F(DSAUt, AbnormalNoCacheDSA)
{
    bool usePTAHash = op::internal::GetThreadLocalContext().usePTAHash_;
    op::internal::GetThreadLocalContext().usePTAHash_ = false;
    op::Shape tShape{1, 2, 3};
    auto self = std::make_unique<aclTensor>(tShape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND, nullptr);
    auto out = std::make_unique<aclTensor>(tShape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND, nullptr);

    int seed = 2;
    int offset = 2;

    std::string apiName = "aclnnInplaceRandom";
    uint32_t OpTypeId = op::OpTypeDict::ToOpType("DSARandomUniform");

    op::internal::GetThreadLocalContext().logInfo_.l2ApiName = "aclnnInplaceRandom";
    op::internal::GetThreadLocalContext().logInfo_.l2SequenceCounter = op::internal::OpGetLogSequence();
    op::internal::GetThreadLocalContext().cacheHasFull_ = false;
    auto uniqueExecutor = CREATE_EXECUTOR();
    auto inputShape = op::ToShapeVector(tShape);
    auto inputShapeArray = uniqueExecutor.get()->AllocIntArray(inputShape.data(), inputShape.size());
    auto low = uniqueExecutor.get()->AllocScalar(static_cast<float>(1));
    auto high = uniqueExecutor.get()->AllocScalar(static_cast<float>(2));
    auto in0 = std::make_tuple(3, seed, offset, low, high);
    auto out0 = std::make_tuple(out.get());

    op::internal::OpCacheKey key;
    // op::internal::OpExecCache::GenerateOpCacheKey(key, apiName, in0, out0);
    op::internal::AddParamToBuf(apiName);
    op::internal::CalculateHashKey(in0);
    op::internal::CalculateHashKey(out0);
    SetOpCacheKey(key);
    EXPECT_NE(key.buf, nullptr);
    EXPECT_NE(key.len, 0);
    auto cache = op::internal::GetOpExecCache(key);
    EXPECT_EQ(cache, nullptr);

    aclOpExecutor* executor = uniqueExecutor.get();
    EXPECT_NE(executor->GetOpExecCache(), nullptr);
    op::internal::OpCacheKey opCacheKey = executor->GetOpExecCache()->GetOpCacheKey();
    op::internal::GetLauncherCtx().ClearTilingCache();

    // DSA 禁止cache
    ADD_TO_LAUNCHER_LIST_DSA(DSARandomUniform, OP_INPUT(3, seed, offset, low, high), OP_OUTPUT(out.get()), OP_ATTR(0));

    EXPECT_NE(executor->GetOpExecCache(), nullptr);
    EXPECT_EQ(opCacheKey.buf, nullptr);
    EXPECT_EQ(opCacheKey.len, 0);

    uniqueExecutor.ReleaseTo(&executor);

    op::internal::GetThreadLocalContext().usePTAHash_ = usePTAHash;
    op::internal::GetThreadLocalContext().cacheHasFull_ = true;
    delete executor->GetOpExecCache();
    delete executor;
}
