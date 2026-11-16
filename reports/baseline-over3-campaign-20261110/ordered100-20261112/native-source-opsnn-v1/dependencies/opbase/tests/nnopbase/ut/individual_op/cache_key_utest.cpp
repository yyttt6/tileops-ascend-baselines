/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include "mockcpp/mockcpp.hpp"
#include "executor/indv_bininfo.h"
#include "executor/indv_collector.h"
#include "executor/indv_executor.h"
#include "executor/indv_cache_key_builder.h"
#include "utils/file_faker.h"
#include "utils/indv_soc.h"
#include "utils/thread_var_container.h"
#include "individual_op_api.h"
#include "individual_op_internal.h"
#include "op_cache_internal.h"
#include "depends/op/op_stub.h"
#include "depends/dump/dump_stub.h"
#include "depends/mmpa/mmpa_stub.h"
#include "depends/runtime/runtime_stub.h"
#include "depends/op/aclnn_bninference_d_kernel_stub.h"
#include "opdev/op_executor.h"
#include "utils/indv_lib_wrapper.h"
#include "utils/indv_types.h"
#define private public
#include "executor/indv_args_pool.h"
#include "register/op_binary_resource_manager.h"
#undef private

#ifdef __cplusplus
extern "C" {
#endif
extern aclnnStatus NnopbaseInit();
#ifdef __cplusplus
}
#endif

namespace {
constexpr size_t kExpKeyBufBytes = 10240U;

class NnopbaseCacheKeyUnitTest : public testing::Test {
protected:
    void SetUp()
    {
        setenv("ASCEND_C", "1", 1);
        NnopbaseExecutorSetGlobalConfig();
        op::internal::opProfilingSwitch.recordOpArgFlag = false;
    }
    void TearDown()
    {
        unsetenv("ASCEND_C");
        op::internal::opProfilingSwitch.recordOpArgFlag = false;
        GlobalMockObject::verify();
    }
};

void GetCacheTestExecutor(NnopbaseExecutor*& executor, const char* opType = "bninference_d_kernel",
                          std::vector<int64_t> shape = {1, 1, 1, 1, 1})
{
    static NnopbaseBinCollector* gBinCollector = nullptr;
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);
    if (gBinCollector == nullptr) {
        gBinCollector = new NnopbaseBinCollector;
        ASSERT_NE(gBinCollector, nullptr);
        ASSERT_EQ(NnopbaseCollectorInit(gBinCollector), OK);
        ASSERT_EQ(NnopbaseCollectorWork(gBinCollector), OK);
    }
    executor = new NnopbaseExecutor;
    ASSERT_NE(executor, nullptr);
    char inputDesc[] = {1, 1, 1};
    char outputDesc[] = {1};
    char attrDesc[] = {};
    ASSERT_EQ(
        NnopbaseExecutorInit(executor, {inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                        sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char)}),
        OK);
    executor->space = new NnopbaseExecutorSpace();
    NnopbaseExecutorSetCollector(executor, gBinCollector);
    ASSERT_EQ(NnopbaseExecutorSetRegInfo(executor, opType), OK);

    aclTensor* tensor = aclCreateTensor(&shape[0], shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, &shape[0], shape.size(), nullptr);
    ASSERT_EQ(NnopbaseExecutorUpdateTensorsIndex(&(executor->ownArgs.inputs), 0), OK);
    ASSERT_EQ(NnopbaseExecutorAddTensor(executor, tensor, 0, true, false), OK);
    ASSERT_EQ(NnopbaseExecutorUpdateTensorsIndex(&(executor->ownArgs.inputs), 1), OK);
    ASSERT_EQ(NnopbaseExecutorAddTensor(executor, tensor, 1, true, false), OK);
    ASSERT_EQ(NnopbaseExecutorUpdateTensorsIndex(&(executor->ownArgs.inputs), 2), OK);
    ASSERT_EQ(NnopbaseExecutorAddTensor(executor, tensor, 2, true, false), OK);
    ASSERT_EQ(NnopbaseExecutorUpdateTensorsIndex(&(executor->ownArgs.outputs), 0), OK);
    ASSERT_EQ(NnopbaseExecutorAddTensor(executor, tensor, 0, false, false), OK);
    aclDestroyTensor(tensor);
}

void GetCacheTestExecutorWithAttr(NnopbaseExecutor*& executor)
{
    static NnopbaseBinCollector* gBinCollector = nullptr;
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);
    if (gBinCollector == nullptr) {
        gBinCollector = new NnopbaseBinCollector;
        ASSERT_NE(gBinCollector, nullptr);
        ASSERT_EQ(NnopbaseCollectorInit(gBinCollector), OK);
        ASSERT_EQ(NnopbaseCollectorWork(gBinCollector), OK);
    }
    executor = new NnopbaseExecutor;
    ASSERT_NE(executor, nullptr);
    char inputDesc[] = {1, 1, 1};
    char outputDesc[] = {1};
    char attrDesc[] = {1};
    ASSERT_EQ(
        NnopbaseExecutorInit(executor, {inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                        sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char)}),
        OK);
    NnopbaseExecutorSetCollector(executor, gBinCollector);
    NnopbaseExecutorSpace space;
    executor->space = &space;
    ASSERT_EQ(NnopbaseExecutorSetRegInfo(executor, "bninference_d_kernel"), OK);
    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(&shape[0], shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, &shape[0], shape.size(), nullptr);
    ASSERT_EQ(NnopbaseExecutorUpdateTensorsIndex(&(executor->ownArgs.inputs), 0), OK);
    ASSERT_EQ(NnopbaseExecutorAddTensor(executor, tensor, 0, true, false), OK);
    ASSERT_EQ(NnopbaseExecutorUpdateTensorsIndex(&(executor->ownArgs.inputs), 1), OK);
    ASSERT_EQ(NnopbaseExecutorAddTensor(executor, tensor, 1, true, false), OK);
    ASSERT_EQ(NnopbaseExecutorUpdateTensorsIndex(&(executor->ownArgs.inputs), 2), OK);
    ASSERT_EQ(NnopbaseExecutorAddTensor(executor, tensor, 2, true, false), OK);
    ASSERT_EQ(NnopbaseExecutorUpdateTensorsIndex(&(executor->ownArgs.outputs), 0), OK);
    ASSERT_EQ(NnopbaseExecutorAddTensor(executor, tensor, 0, false, false), OK);

    static int64_t bias1[1] = {1};
    NnopbaseAddAttrWithDtype(executor, bias1, sizeof(int64_t), 0, kNnopbaseInt);
    aclDestroyTensor(tensor);
}

void* GetDynamicCacheExecutor(void* executorSpace)
{
    const char* opType = "bninference_d_kernel";
    char inputDesc[] = {2, 2, 2};
    char outputDesc[] = {2};
    char attrDesc[] = {};
    void* executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                         sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    std::vector<const aclTensor*> tensorListA;
    tensorListA.push_back(tensor);
    aclTensorList* aclTensorTestList = aclCreateTensorList(tensorListA.data(), tensorListA.size());
    (void)NnopbaseAddDynamicInput(executor, aclTensorTestList, 0);
    (void)NnopbaseAddDynamicInput(executor, aclTensorTestList, 1);
    (void)NnopbaseAddDynamicInput(executor, aclTensorTestList, 2);
    (void)NnopbaseAddDynamicOutput(executor, aclTensorTestList, 0);
    aclDestroyTensorList((const aclTensorList*)aclTensorTestList);
    return executor;
}

NnopbaseUChar* AppendV1TensorToExp(NnopbaseUChar* key, const NnopbaseTensor& extTensor)
{
    if (extTensor.isNull) {
        key = NnopbaseAppend1Byte(key, '/');
        return key;
    }
    const auto& rt2Tensor = extTensor.rt2Tensor;
    key = NnopbaseAppend1Byte(key, static_cast<NnopbaseUChar>(rt2Tensor.GetDataType()));
    key = NnopbaseAppend1Byte(key, static_cast<NnopbaseUChar>(rt2Tensor.GetStorageFormat()));
    const auto& shape = rt2Tensor.GetStorageShape();
    key = NnopbaseAppend1Byte(key, static_cast<NnopbaseUChar>(shape.GetDimNum()));
    for (size_t j = 0; j < shape.GetDimNum(); j++) {
        key = static_cast<uint8_t*>(NnopbaseAppend8Byte(key, static_cast<uint64_t>(shape.GetDim(j))));
    }
    const auto& stride = rt2Tensor.GetStride();
    key = NnopbaseAppend1Byte(key, static_cast<NnopbaseUChar>(stride.GetDimNum()));
    for (size_t j = 0; j < stride.GetDimNum(); j++) {
        key = static_cast<uint8_t*>(NnopbaseAppend8Byte(key, static_cast<uint64_t>(stride.GetStride(j))));
    }
    key = static_cast<uint8_t*>(NnopbaseAppend8Byte(key, static_cast<uint64_t>(rt2Tensor.GetOffset())));
    return key;
}

NnopbaseUChar* BuildV1ExpKey(NnopbaseExecutor* executor, NnopbaseUChar* key)
{
    key = static_cast<uint8_t*>(
        NnopbaseAppendBinary(key, strlen(executor->opType), executor->opType, strlen(executor->opType)));
    auto& inputs = executor->ownArgs.inputs;
    for (uint32_t i = 0; i < inputs.num; i++) {
        key = AppendV1TensorToExp(key, inputs.extTensors[i]);
    }
    key = NnopbaseAppend1Byte(key, '/');
    auto& outputs = executor->ownArgs.outputs;
    for (uint32_t i = 0; i < outputs.num; i++) {
        key = AppendV1TensorToExp(key, outputs.extTensors[i]);
    }
    key = NnopbaseAppend1Byte(key, '/');
    return key;
}

NnopbaseUChar* AppendCoreNumExp(NnopbaseUChar* key, const NnopbaseCoreNum& coreNumInfo)
{
    key = NnopbaseAppend1Byte(key, '/');
    key = static_cast<uint8_t*>(
        NnopbaseAppendBinary(key, sizeof(NnopbaseCoreNum), &coreNumInfo, sizeof(NnopbaseCoreNum)));
    return key;
}

NnopbaseUChar* AppendTailExp(NnopbaseUChar* key, uint32_t rankId = 0U, bool deterministic = false)
{
    key = NnopbaseAppend1Byte(key, '/');
    key = static_cast<uint8_t*>(NnopbaseAppendBinary(key, sizeof(uint32_t), &rankId, sizeof(uint32_t)));
    key = NnopbaseAppend1Byte(key, '/');
    key = NnopbaseAppend1Byte(key, static_cast<NnopbaseUChar>(deterministic));
    return key;
}

NnopbaseUChar* AppendIntAttrExp(NnopbaseUChar* key)
{
    static int64_t bias1[1] = {1};
    return static_cast<uint8_t*>(NnopbaseAppendBinary(key, sizeof(int64_t), &bias1[0], sizeof(int64_t)));
}
} // namespace

// ===== migrated from executor_utest.cpp: V1 path, no attr =====
TEST_F(NnopbaseCacheKeyUnitTest, NnopbaseCacheNoAttrKey)
{
    NnopbaseExecutor* executor = nullptr;
    GetCacheTestExecutor(executor);
    ASSERT_NE(executor, nullptr);

    executor->coreNum.aicNum = 24;
    executor->coreNum.aivNum = 24;
    nnopbase::ArgsPool::GetInstance().MatchArgs(executor);

    std::vector<NnopbaseUChar> exp(kExpKeyBufBytes, '\0');
    auto key = &exp[0U];
    key = BuildV1ExpKey(executor, key);
    NnopbaseCoreNum coreNumInfo = {24, 24};
    key = AppendCoreNumExp(key, coreNumInfo);
    key = AppendTailExp(key);

    auto keyLen = key - &exp[0U];
    ASSERT_EQ(static_cast<size_t>(keyLen), executor->ownArgs.keyLen);
    for (size_t i = 0; i < executor->ownArgs.keyLen; ++i) {
        ASSERT_EQ(executor->ownArgs.inputKey[i], exp[i]) << "cache key byte mismatch at offset " << i;
    }

    NnopbaseExecutorGcSpace((void*)executor->space);
    NnopbaseExecutorDeInit(executor);
    delete executor;
    NnopbaseUnsetEnvAndClearFolder();
}

// ===== migrated from executor_utest.cpp: V1 path, with attr =====
TEST_F(NnopbaseCacheKeyUnitTest, NnopbaseCacheWithAttrKey)
{
    NnopbaseExecutor* executor = nullptr;
    GetCacheTestExecutorWithAttr(executor);
    ASSERT_NE(executor, nullptr);
    executor->space = new NnopbaseExecutorSpace();

    executor->coreNum.aicNum = 24;
    executor->coreNum.aivNum = 24;
    nnopbase::ArgsPool::GetInstance().MatchArgs(executor);

    std::vector<NnopbaseUChar> exp(kExpKeyBufBytes, '\0');
    auto key = &exp[0U];
    key = BuildV1ExpKey(executor, key);
    key = AppendIntAttrExp(key);
    NnopbaseCoreNum coreNumInfo = {24, 24};
    key = AppendCoreNumExp(key, coreNumInfo);
    key = AppendTailExp(key);

    auto keyLen = key - &exp[0U];
    ASSERT_EQ(static_cast<size_t>(keyLen), executor->ownArgs.keyLen);
    for (size_t i = 0; i < executor->ownArgs.keyLen; ++i) {
        ASSERT_EQ(executor->ownArgs.inputKey[i], exp[i]) << "cache key byte mismatch at offset " << i;
    }

    NnopbaseExecutorGcSpace((void*)executor->space);
    NnopbaseExecutorDeInit(executor);
    delete executor;
    NnopbaseUnsetEnvAndClearFolder();
}

// ===== migrated from api_utest.cpp: V1 path, dynamic IO =====
TEST_F(NnopbaseCacheKeyUnitTest, NnopBaseCacheDynamicIoKey)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);
    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);
    void* executor = GetDynamicCacheExecutor(executorSpace);
    ASSERT_NE(executor, nullptr);
    NnopbaseExecutor* opExecutor = (NnopbaseExecutor*)executor;
    opExecutor->coreNum.aicNum = 24;
    opExecutor->coreNum.aivNum = 24;
    nnopbase::ArgsPool::GetInstance().MatchArgs(opExecutor);

    std::vector<NnopbaseUChar> exp(kExpKeyBufBytes, '\0');
    auto key = &exp[0U];
    key = BuildV1ExpKey(opExecutor, key);
    NnopbaseCoreNum coreNumInfo = {24, 24};
    key = AppendCoreNumExp(key, coreNumInfo);
    key = AppendTailExp(key);

    auto keyLen = key - &exp[0U];
    ASSERT_EQ(static_cast<size_t>(keyLen), opExecutor->ownArgs.keyLen);
    for (size_t i = 0; i < opExecutor->ownArgs.keyLen; ++i) {
        ASSERT_EQ(opExecutor->ownArgs.inputKey[i], exp[i]) << "cache key byte mismatch at offset " << i;
    }

    NnopbaseExecutorGcSpace(executorSpace);
    NnopbaseUnsetEnvAndClearFolder();
}

// ===== V2 path helpers =====
namespace {
constexpr NnopbaseUChar kSeparator = '/';

NnopbaseUChar* AppendV2OpType(NnopbaseUChar* key, const char* opType)
{
    return static_cast<NnopbaseUChar*>(NnopbaseAppendBinary(key, strlen(opType), opType, strlen(opType)));
}

NnopbaseUChar* AppendV2TensorShapeInfo(NnopbaseUChar* key, const aclTensor* tensor)
{
    const op::Shape& shape = tensor->GetViewShape();
    const size_t dimNum = shape.GetDimNum();
    const auto& strides = tensor->GetViewStrides();
    const size_t strideNum = strides.size();
    key = NnopbaseAppend1Byte(key, static_cast<NnopbaseUChar>(tensor->GetDataType()));
    key = NnopbaseAppend1Byte(key, static_cast<NnopbaseUChar>(tensor->GetStorageFormat()));
    key = NnopbaseAppend1Byte(key, static_cast<NnopbaseUChar>(dimNum));
    for (size_t j = 0U; j < dimNum; j++) {
        key = NnopbaseAppend8Byte(key, static_cast<uint64_t>(shape.GetDim(j)));
    }
    key = NnopbaseAppend1Byte(key, static_cast<NnopbaseUChar>(strideNum));
    for (size_t j = 0U; j < strideNum; j++) {
        key = NnopbaseAppend8Byte(key, static_cast<uint64_t>(strides[j]));
    }
    key = NnopbaseAppend8Byte(key, static_cast<uint64_t>(tensor->GetViewOffset()));
    return key;
}

NnopbaseUChar* AppendV2ValueDependTensor(NnopbaseUChar* key, const void* addr, uint64_t dim, uint64_t dataLen,
                                         ge::DataType dType)
{
    key = NnopbaseAppend1Byte(key, static_cast<NnopbaseUChar>(dType));
    key = NnopbaseAppend1Byte(key, static_cast<NnopbaseUChar>(ge::FORMAT_ND));
    key = NnopbaseAppend1Byte(key, static_cast<NnopbaseUChar>(1U));
    key = NnopbaseAppend8Byte(key, dim);
    key = static_cast<NnopbaseUChar*>(NnopbaseAppendBinary(key, dataLen, addr, dataLen));
    return key;
}

NnopbaseUChar* AppendV2ScalarInfo(NnopbaseUChar* key, ge::DataType dtype)
{
    key = NnopbaseAppend1Byte(key, static_cast<NnopbaseUChar>(dtype));
    key = NnopbaseAppend1Byte(key, static_cast<NnopbaseUChar>(ge::FORMAT_ND));
    key = NnopbaseAppend1Byte(key, static_cast<NnopbaseUChar>(0U));
    return key;
}

NnopbaseUChar* AppendV2ScalarListInfo(NnopbaseUChar* key, ge::DataType dtype, uint64_t size)
{
    key = NnopbaseAppend1Byte(key, static_cast<NnopbaseUChar>(dtype));
    key = NnopbaseAppend1Byte(key, static_cast<NnopbaseUChar>(ge::FORMAT_ND));
    key = NnopbaseAppend1Byte(key, static_cast<NnopbaseUChar>(1U));
    key = NnopbaseAppend8Byte(key, size);
    return key;
}

NnopbaseUChar* AppendV2Placeholder(NnopbaseUChar* key) { return NnopbaseAppend1Byte(key, kSeparator); }

NnopbaseUChar* AppendV2CoreNum(NnopbaseUChar* key, const NnopbaseCoreNum& coreNum)
{
    key = NnopbaseAppend1Byte(key, kSeparator);
    key = static_cast<NnopbaseUChar*>(
        NnopbaseAppendBinary(key, sizeof(NnopbaseCoreNum), &coreNum, sizeof(NnopbaseCoreNum)));
    return key;
}

NnopbaseUChar* AppendV2Mc2RankId(NnopbaseUChar* key, uint32_t rankId)
{
    key = NnopbaseAppend1Byte(key, kSeparator);
    key = static_cast<NnopbaseUChar*>(NnopbaseAppendBinary(key, sizeof(uint32_t), &rankId, sizeof(uint32_t)));
    return key;
}

NnopbaseUChar* AppendV2Deterministic(NnopbaseUChar* key, bool deterministic)
{
    key = NnopbaseAppend1Byte(key, kSeparator);
    key = NnopbaseAppend1Byte(key, static_cast<NnopbaseUChar>(deterministic));
    return key;
}

void* PrepareV2Executor(void*& executorSpace, const char* opType, char* inputDesc, size_t inputNum, char* outputDesc,
                        size_t outputNum, char* attrDesc, size_t attrNum)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);
    if (executorSpace == nullptr) {
        EXPECT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);
    }
    void* executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, inputNum, outputDesc, outputNum, attrDesc,
                                         attrNum);
    if (executor != nullptr) {
        NnopbaseSetMatchArgsFlag(executor);
    }
    return executor;
}

void FinishV2MatchArgs(NnopbaseExecutor* executor)
{
    executor->coreNum.aicNum = 24;
    executor->coreNum.aivNum = 24;
    uint64_t workspaceLen = 0U;
    (void)NnopbaseMatchArgs(executor, &workspaceLen);
}
} // namespace

// ===== V2 path: regular static-tensor inputs/outputs =====
TEST_F(NnopbaseCacheKeyUnitTest, NnopbaseV2CacheKeyRegularTensor)
{
    void* executorSpace = nullptr;
    const char* opType = "bninference_d_kernel";
    char inputDesc[] = {1, 1, 1};
    char outputDesc[] = {1};
    char attrDesc[] = {};
    void* executor = PrepareV2Executor(executorSpace, opType, inputDesc, sizeof(inputDesc), outputDesc,
                                       sizeof(outputDesc), attrDesc, sizeof(attrDesc));
    ASSERT_NE(executor, nullptr);

    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    ASSERT_EQ(NnopbaseAddInput(executor, tensor, 0), 0);
    ASSERT_EQ(NnopbaseAddInput(executor, tensor, 1), 0);
    ASSERT_EQ(NnopbaseAddInput(executor, tensor, 2), 0);
    ASSERT_EQ(NnopbaseAddOutput(executor, tensor, 0), 0);

    NnopbaseExecutor* opExecutor = static_cast<NnopbaseExecutor*>(executor);
    FinishV2MatchArgs(opExecutor);

    std::vector<NnopbaseUChar> exp(kExpKeyBufBytes, '\0');
    auto key = &exp[0U];
    key = AppendV2OpType(key, opType);
    key = AppendV2TensorShapeInfo(key, tensor);
    key = AppendV2TensorShapeInfo(key, tensor);
    key = AppendV2TensorShapeInfo(key, tensor);
    key = AppendV2Placeholder(key);
    key = AppendV2TensorShapeInfo(key, tensor);
    NnopbaseCoreNum coreNumInfo = {24, 24};
    key = AppendV2CoreNum(key, coreNumInfo);
    key = AppendV2Mc2RankId(key, 0U);
    key = AppendV2Deterministic(key, false);

    auto keyLen = key - &exp[0U];
    ASSERT_EQ(static_cast<size_t>(keyLen), opExecutor->ownArgs.keyLen);
    for (size_t i = 0; i < opExecutor->ownArgs.keyLen; ++i) {
        ASSERT_EQ(opExecutor->ownArgs.inputKey[i], exp[i]) << "v2 cache key byte mismatch at offset " << i;
    }

    aclDestroyTensor(tensor);
    NnopbaseExecutorGcSpace(executorSpace);
    NnopbaseUnsetEnvAndClearFolder();
}

// ===== V2 path: null input is encoded as a single '/' placeholder =====
TEST_F(NnopbaseCacheKeyUnitTest, NnopbaseV2CacheKeyNullTensor)
{
    void* executorSpace = nullptr;
    const char* opType = "bninference_d_kernel";
    char inputDesc[] = {1, 0, 1};
    char outputDesc[] = {1};
    char attrDesc[] = {};
    void* executor = PrepareV2Executor(executorSpace, opType, inputDesc, sizeof(inputDesc), outputDesc,
                                       sizeof(outputDesc), attrDesc, sizeof(attrDesc));
    ASSERT_NE(executor, nullptr);

    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    ASSERT_EQ(NnopbaseAddInput(executor, tensor, 0), 0);
    ASSERT_EQ(NnopbaseAddInput(executor, nullptr, 1), 0);
    ASSERT_EQ(NnopbaseAddInput(executor, tensor, 2), 0);
    ASSERT_EQ(NnopbaseAddOutput(executor, tensor, 0), 0);

    NnopbaseExecutor* opExecutor = static_cast<NnopbaseExecutor*>(executor);
    FinishV2MatchArgs(opExecutor);

    std::vector<NnopbaseUChar> exp(kExpKeyBufBytes, '\0');
    auto key = &exp[0U];
    key = AppendV2OpType(key, opType);
    key = AppendV2TensorShapeInfo(key, tensor);
    key = AppendV2Placeholder(key);
    key = AppendV2TensorShapeInfo(key, tensor);
    key = AppendV2Placeholder(key);
    key = AppendV2TensorShapeInfo(key, tensor);
    NnopbaseCoreNum coreNumInfo = {24, 24};
    key = AppendV2CoreNum(key, coreNumInfo);
    key = AppendV2Mc2RankId(key, 0U);
    key = AppendV2Deterministic(key, false);

    auto keyLen = key - &exp[0U];
    ASSERT_EQ(static_cast<size_t>(keyLen), opExecutor->ownArgs.keyLen);
    for (size_t i = 0; i < opExecutor->ownArgs.keyLen; ++i) {
        ASSERT_EQ(opExecutor->ownArgs.inputKey[i], exp[i]) << "v2 cache key byte mismatch at offset " << i;
    }

    aclDestroyTensor(tensor);
    NnopbaseExecutorGcSpace(executorSpace);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseCacheKeyUnitTest, NnopbaseV2CacheKeyAppendsRawDeterministicLevel)
{
    NnopbaseExecutorArgs args;
    args.keyLen = 0U;
    args.remainKeyLen = 1U;
    const uint8_t deterministicLevel = 2U;

    (void)Indv::CacheKeyBuilder::AppendDeterministicLevel(&args, &deterministicLevel);

    ASSERT_EQ(args.keyLen, sizeof(uint8_t) + 1U);
    ASSERT_EQ(args.inputKey[0], kSeparator);
    ASSERT_EQ(args.inputKey[1], deterministicLevel);
}

// ===== V2 path: IntArray input is value-dependent; raw bytes are appended =====
TEST_F(NnopbaseCacheKeyUnitTest, NnopbaseV2CacheKeyIntArray)
{
    void* executorSpace = nullptr;
    const char* opType = "bninference_d_kernel";
    char inputDesc[] = {1, 1, 0};
    char outputDesc[] = {1};
    char attrDesc[] = {};
    void* executor = PrepareV2Executor(executorSpace, opType, inputDesc, sizeof(inputDesc), outputDesc,
                                       sizeof(outputDesc), attrDesc, sizeof(attrDesc));
    ASSERT_NE(executor, nullptr);

    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    int64_t intValues[] = {3, 4, 5};
    auto* intArray = aclCreateIntArray(intValues, sizeof(intValues) / sizeof(intValues[0]));

    ASSERT_EQ(NnopbaseAddInput(executor, tensor, 0), 0);
    ASSERT_EQ(NnopbaseAddInput(executor, tensor, 1), 0);
    ASSERT_EQ(NnopbaseAddIntArrayInput(executor, intArray, 2), OK);
    ASSERT_EQ(NnopbaseAddOutput(executor, tensor, 0), 0);

    NnopbaseExecutor* opExecutor = static_cast<NnopbaseExecutor*>(executor);
    FinishV2MatchArgs(opExecutor);

    std::vector<NnopbaseUChar> exp(kExpKeyBufBytes, '\0');
    auto key = &exp[0U];
    key = AppendV2OpType(key, opType);
    key = AppendV2TensorShapeInfo(key, tensor);
    key = AppendV2TensorShapeInfo(key, tensor);
    key = AppendV2ValueDependTensor(key, intValues, sizeof(intValues) / sizeof(intValues[0]), sizeof(intValues),
                                    ge::DataType::DT_INT64);
    key = AppendV2Placeholder(key);
    key = AppendV2TensorShapeInfo(key, tensor);
    NnopbaseCoreNum coreNumInfo = {24, 24};
    key = AppendV2CoreNum(key, coreNumInfo);
    key = AppendV2Mc2RankId(key, 0U);
    key = AppendV2Deterministic(key, false);

    auto keyLen = key - &exp[0U];
    ASSERT_EQ(static_cast<size_t>(keyLen), opExecutor->ownArgs.keyLen);
    for (size_t i = 0; i < opExecutor->ownArgs.keyLen; ++i) {
        ASSERT_EQ(opExecutor->ownArgs.inputKey[i], exp[i]) << "v2 cache key byte mismatch at offset " << i;
    }

    aclDestroyIntArray(intArray);
    aclDestroyTensor(tensor);
    NnopbaseExecutorGcSpace(executorSpace);
    NnopbaseUnsetEnvAndClearFolder();
}

// ===== V2 path: scalar input encodes only dtype/format/dimNum=0 =====
TEST_F(NnopbaseCacheKeyUnitTest, NnopbaseV2CacheKeyScalar)
{
    void* executorSpace = nullptr;
    const char* opType = "bninference_d_kernel";
    char inputDesc[] = {1, 1, 0};
    char outputDesc[] = {1};
    char attrDesc[] = {};
    void* executor = PrepareV2Executor(executorSpace, opType, inputDesc, sizeof(inputDesc), outputDesc,
                                       sizeof(outputDesc), attrDesc, sizeof(attrDesc));
    ASSERT_NE(executor, nullptr);

    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    float scalarValue = 5.0f;
    auto* scalar = aclCreateScalar(&scalarValue, aclDataType::ACL_FLOAT);

    ASSERT_EQ(NnopbaseAddInput(executor, tensor, 0), 0);
    ASSERT_EQ(NnopbaseAddInput(executor, tensor, 1), 0);
    ASSERT_EQ(NnopbaseAddScalarInput(executor, scalar, 2, -1, ge::DT_UNDEFINED), 0);
    ASSERT_EQ(NnopbaseAddOutput(executor, tensor, 0), 0);

    NnopbaseExecutor* opExecutor = static_cast<NnopbaseExecutor*>(executor);
    FinishV2MatchArgs(opExecutor);

    std::vector<NnopbaseUChar> exp(kExpKeyBufBytes, '\0');
    auto key = &exp[0U];
    key = AppendV2OpType(key, opType);
    key = AppendV2TensorShapeInfo(key, tensor);
    key = AppendV2TensorShapeInfo(key, tensor);
    key = AppendV2ScalarInfo(key, ge::DataType::DT_FLOAT);
    key = AppendV2Placeholder(key);
    key = AppendV2TensorShapeInfo(key, tensor);
    NnopbaseCoreNum coreNumInfo = {24, 24};
    key = AppendV2CoreNum(key, coreNumInfo);
    key = AppendV2Mc2RankId(key, 0U);
    key = AppendV2Deterministic(key, false);

    auto keyLen = key - &exp[0U];
    ASSERT_EQ(static_cast<size_t>(keyLen), opExecutor->ownArgs.keyLen);
    for (size_t i = 0; i < opExecutor->ownArgs.keyLen; ++i) {
        ASSERT_EQ(opExecutor->ownArgs.inputKey[i], exp[i]) << "v2 cache key byte mismatch at offset " << i;
    }

    aclDestroyScalar(scalar);
    aclDestroyTensor(tensor);
    NnopbaseExecutorGcSpace(executorSpace);
    NnopbaseUnsetEnvAndClearFolder();
}

// ===== V2 path: scalar list encodes dtype/format/dimNum=1 + 8B size =====
TEST_F(NnopbaseCacheKeyUnitTest, NnopbaseV2CacheKeyScalarList)
{
    void* executorSpace = nullptr;
    const char* opType = "bninference_d_kernel";
    char inputDesc[] = {1, 1, 0};
    char outputDesc[] = {1};
    char attrDesc[] = {};
    void* executor = PrepareV2Executor(executorSpace, opType, inputDesc, sizeof(inputDesc), outputDesc,
                                       sizeof(outputDesc), attrDesc, sizeof(attrDesc));
    ASSERT_NE(executor, nullptr);

    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    float scalarValue = 5.0f;
    auto* scalar = aclCreateScalar(&scalarValue, aclDataType::ACL_FLOAT);
    auto* scalarList = aclCreateScalarList(&scalar, 1);

    ASSERT_EQ(NnopbaseAddInput(executor, tensor, 0), 0);
    ASSERT_EQ(NnopbaseAddInput(executor, tensor, 1), 0);
    ASSERT_EQ(NnopbaseAddScalarListInput(executor, scalarList, 2, -1, ge::DT_UNDEFINED), 0);
    ASSERT_EQ(NnopbaseAddOutput(executor, tensor, 0), 0);

    NnopbaseExecutor* opExecutor = static_cast<NnopbaseExecutor*>(executor);
    FinishV2MatchArgs(opExecutor);

    std::vector<NnopbaseUChar> exp(kExpKeyBufBytes, '\0');
    auto key = &exp[0U];
    key = AppendV2OpType(key, opType);
    key = AppendV2TensorShapeInfo(key, tensor);
    key = AppendV2TensorShapeInfo(key, tensor);
    key = AppendV2ScalarListInfo(key, ge::DataType::DT_FLOAT, 1U);
    key = AppendV2Placeholder(key);
    key = AppendV2TensorShapeInfo(key, tensor);
    NnopbaseCoreNum coreNumInfo = {24, 24};
    key = AppendV2CoreNum(key, coreNumInfo);
    key = AppendV2Mc2RankId(key, 0U);
    key = AppendV2Deterministic(key, false);

    auto keyLen = key - &exp[0U];
    ASSERT_EQ(static_cast<size_t>(keyLen), opExecutor->ownArgs.keyLen);
    for (size_t i = 0; i < opExecutor->ownArgs.keyLen; ++i) {
        ASSERT_EQ(opExecutor->ownArgs.inputKey[i], exp[i]) << "v2 cache key byte mismatch at offset " << i;
    }

    aclDestroyScalarList(scalarList);
    aclDestroyTensor(tensor);
    NnopbaseExecutorGcSpace(executorSpace);
    NnopbaseUnsetEnvAndClearFolder();
}

// ===== V2 path: dynamic input (TensorList of homogeneous tensors) collapses to shape + count =====
TEST_F(NnopbaseCacheKeyUnitTest, NnopbaseV2CacheKeyDynamicInput)
{
    void* executorSpace = nullptr;
    const char* opType = "bninference_d_kernel";
    char inputDesc[] = {2, 2, 2};
    char outputDesc[] = {2};
    char attrDesc[] = {};
    void* executor = PrepareV2Executor(executorSpace, opType, inputDesc, sizeof(inputDesc), outputDesc,
                                       sizeof(outputDesc), attrDesc, sizeof(attrDesc));
    ASSERT_NE(executor, nullptr);

    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    std::vector<const aclTensor*> tensorVec;
    tensorVec.push_back(tensor);
    aclTensorList* tensorList = aclCreateTensorList(tensorVec.data(), tensorVec.size());

    ASSERT_EQ(NnopbaseAddDynamicInput(executor, tensorList, 0), 0);
    ASSERT_EQ(NnopbaseAddDynamicInput(executor, tensorList, 1), 0);
    ASSERT_EQ(NnopbaseAddDynamicInput(executor, tensorList, 2), 0);
    ASSERT_EQ(NnopbaseAddDynamicOutput(executor, tensorList, 0), 0);

    NnopbaseExecutor* opExecutor = static_cast<NnopbaseExecutor*>(executor);
    FinishV2MatchArgs(opExecutor);

    std::vector<NnopbaseUChar> exp(kExpKeyBufBytes, '\0');
    auto key = &exp[0U];
    key = AppendV2OpType(key, opType);
    for (uint32_t i = 0; i < 3U; ++i) {
        key = AppendV2TensorShapeInfo(key, tensor);
        key = NnopbaseAppend1Byte(key, static_cast<NnopbaseUChar>(1U));
    }
    key = AppendV2Placeholder(key);
    key = AppendV2TensorShapeInfo(key, tensor);
    key = NnopbaseAppend1Byte(key, static_cast<NnopbaseUChar>(1U));
    NnopbaseCoreNum coreNumInfo = {24, 24};
    key = AppendV2CoreNum(key, coreNumInfo);
    key = AppendV2Mc2RankId(key, 0U);
    key = AppendV2Deterministic(key, false);

    auto keyLen = key - &exp[0U];
    ASSERT_EQ(static_cast<size_t>(keyLen), opExecutor->ownArgs.keyLen);
    for (size_t i = 0; i < opExecutor->ownArgs.keyLen; ++i) {
        ASSERT_EQ(opExecutor->ownArgs.inputKey[i], exp[i]) << "v2 cache key byte mismatch at offset " << i;
    }

    aclDestroyTensorList(tensorList);
    NnopbaseExecutorGcSpace(executorSpace);
    NnopbaseUnsetEnvAndClearFolder();
}
