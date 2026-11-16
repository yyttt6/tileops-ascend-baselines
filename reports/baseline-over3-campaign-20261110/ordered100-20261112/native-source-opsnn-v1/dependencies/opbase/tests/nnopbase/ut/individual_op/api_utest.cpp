/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include "mockcpp/mockcpp.hpp"
#include "executor/indv_executor.h"
#include "executor/indv_args_pool.h"
#include "individual_op_api.h"
#include "individual_op_internal.h"
#include "op_cache_internal.h"
#include "executor/indv_bininfo.h"
#include "executor/indv_collector.h"
#include "utils/file_faker.h"
#include "utils/indv_soc.h"
#include "utils/thread_var_container.h"
#include "depends/dump/dump_stub.h"
#include "depends/mmpa/mmpa_stub.h"
#include "depends/op/op_stub.h"
#include "depends/op/aclnn_bninference_d_kernel_stub.h"
#include "opdev/op_executor.h"
#include "utils/indv_lib_wrapper.h"
#include "depends/runtime/runtime_stub.h"
#define private public
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
static const NnopbaseCoreNum coreNum{24, 48};

void SetSocVersion(const std::string& version)
{
    auto& soc = nnopbase::IndvSoc::GetInstance();
    soc.socVersion = version;
    soc.isInit = true;
}
} // namespace

class NnopbaseUnitTest : public testing::Test {
public:
    aclTensor* CreateAclTensor(vector<int64_t> view_shape, vector<int64_t> stride, int64_t offset,
                               vector<int64_t> storage_shape, aclDataType dataType = ACL_FLOAT, int64_t dataOffset = 0)
    {
        return aclCreateTensor(view_shape.data(), view_shape.size(), dataType, stride.data(), offset, ACL_FORMAT_ND,
                               storage_shape.data(), storage_shape.size(), data + dataOffset);
    }

protected:
    void SetUp()
    {
        setenv("ASCEND_C", "1", 1);
        setenv("GE_PROFILING_TO_STD_OUT", "1", 1);
        setenv("ASCEND_MC2_DEBUG_MODE", "2", 1);
        NnopbaseExecutorSetGlobalConfig();
        op::internal::opProfilingSwitch.recordOpArgFlag = true;
    }

    void TearDown()
    {
        unsetenv("ASCEND_C");
        unsetenv("GE_PROFILING_TO_STD_OUT");
        unsetenv("ASCEND_MC2_DEBUG_MODE");
        op::internal::opProfilingSwitch.recordOpArgFlag = false;
        GlobalMockObject::verify();
    }

    int64_t data[1024 * 1024] = {0};
    void TestHcclServerType(std::function<void(void*)> setHcclServerTypeFunc, const char* socVersion);
};

TEST_F(NnopbaseUnitTest, NnopbaseGetExecutorNullPtr)
{
    void* executorSpace = nullptr;

    const char* opType = "bninference_d_kernel";
    char inputDesc[] = {1, 1, 1};
    char outputDesc[] = {1};
    char attrDesc[] = {};
    void* executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                         sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    ASSERT_EQ(executor, nullptr);
}

TEST_F(NnopbaseUnitTest, NnopBaseRunSuccessWithoutAttr)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);
    NnopbaseInit();

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char* opType = "bninference_d_kernel";
    char inputDesc[] = {1, 1, 1};
    char outputDesc[] = {1, 0};
    char attrDesc[] = {};
    void* executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                         sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    ASSERT_NE(executor, nullptr);

    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    (void)NnopbaseAddInput(executor, tensor, 0);
    (void)NnopbaseAddInput(executor, tensor, 1);
    (void)NnopbaseAddInput(executor, tensor, 2);
    (void)NnopbaseAddOutput(executor, tensor, 0);
    (void)NnopbaseAddOutput(executor, nullptr, 1);

    size_t workspaceLen = 0U;
    ASSERT_EQ(NnopbaseRunForWorkspace(executor, &workspaceLen), OK);
    void* stream = nullptr;
    void* workspace = nullptr;
    if (workspaceLen > 0U) {
        workspace = (void*)malloc(workspaceLen);
    }
    ASSERT_EQ(NnopbaseRunWithWorkspace(executor, stream, workspace, workspaceLen), OK);
    if (workspaceLen > 0U) {
        free(workspace);
    }

    NnopbaseExecutorGcSpace(executorSpace);
    aclDestroyTensor(tensor);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseUnitTest, NnopBaseRunSuccessWithAttr)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char* opType = "bninference_d_kernel";
    char inputDesc[] = {1, 1, 1};
    char outputDesc[] = {1};
    char attrDesc[] = {1, 1};
    void* executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                         sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    ASSERT_NE(executor, nullptr);

    (void)NnopbaseSetFormatMatchMode(executor, 1);

    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT16, nullptr, 0,
                                        aclFormat::ACL_FORMAT_NCDHW, shape.data(), shape.size(), nullptr);
    (void)NnopbaseAddInput(executor, tensor, 0);
    (void)NnopbaseAddInput(executor, tensor, 1);
    (void)NnopbaseAddInput(executor, tensor, 2);
    (void)NnopbaseAddOutput(executor, tensor, 0);
    ASSERT_EQ(((NnopbaseExecutor*)executor)->ownArgs.inputs.extTensors[0].rt2Tensor.GetStorageFormat(),
              ge::FORMAT_NCDHW);

    int64_t bias1[] = {1, 2, 3};
    size_t bias1_size = 3U;
    if (bias1) {
        NnopbaseAddAttr(executor, bias1, bias1_size, 0);
    } else {
        static int64_t bias1_def[] = {1, 2};
        static size_t bias1_size_def = 2 * sizeof(int64_t);
        NnopbaseAddAttr(executor, bias1_def, bias1_size_def, 0);
    }
    bool bias2[] = {true, true, false};
    if (bias2) {
        NnopbaseAddAttrWithDtype(executor, bias2, bias1_size, 1, kNnopbaseBool);
    } else {
        static bool bias2_def[] = {true, false};
        static size_t bias2_size_def = 2 * sizeof(bool);
        NnopbaseAddAttrWithDtype(executor, bias2_def, bias2_size_def, 1, kNnopbaseBool);
    }

    size_t workspaceLen = 0U;
    ASSERT_EQ(NnopbaseRunForWorkspace(executor, &workspaceLen), OK);

    void* attrAddr = nullptr;
    ASSERT_EQ(NnopbaseGetAttrAddr(executor, 0, &attrAddr, &bias1_size), OK);
    uint64_t dataLen = 0U;
    NnopbaseGetInputTensorAddr(executor, 0, &attrAddr);
    NnopbaseGetOutputTensorAddr(executor, 0, &attrAddr);
    NnopbaseGetTilingData(executor, &attrAddr, &dataLen);

    void* stream = nullptr;
    void* workspace = nullptr;
    if (workspaceLen > 0U) {
        workspace = (void*)malloc(workspaceLen);
    }
    ASSERT_EQ(NnopbaseRunWithWorkspace(executor, stream, workspace, workspaceLen), OK);
    if (workspaceLen > 0U) {
        free(workspace);
    }
    NnopbaseExecutorGcSpace(executorSpace);
    aclDestroyTensor(tensor);
    NnopbaseUnsetEnvAndClearFolder();
}

void* GetDynamicInputExecutor(void* executorSpace)
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

    std::vector<const aclTensor*> tensor_list_a;
    tensor_list_a.push_back(tensor);
    aclTensorList* aclTensorTestList = aclCreateTensorList(tensor_list_a.data(), tensor_list_a.size());

    (void)NnopbaseAddDynamicInput(executor, aclTensorTestList, 0);
    (void)NnopbaseAddDynamicInput(executor, aclTensorTestList, 1);
    (void)NnopbaseAddDynamicInput(executor, aclTensorTestList, 2);
    (void)NnopbaseAddDynamicOutput(executor, aclTensorTestList, 0);
    aclDestroyTensorList((const aclTensorList*)aclTensorTestList);
    return executor;
}

TEST_F(NnopbaseUnitTest, NnopBaseRunSuccessWithDynamicInput)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);
    void* executor = GetDynamicInputExecutor(executorSpace);
    ASSERT_NE(executor, nullptr);

    size_t workspaceLen = 0U;
    ASSERT_EQ(NnopbaseRunForWorkspace(executor, &workspaceLen), OK);
    void* stream = nullptr;
    void* workspace = nullptr;
    if (workspaceLen > 0U) {
        workspace = (void*)malloc(workspaceLen);
    }
    ASSERT_EQ(NnopbaseRunWithWorkspace(executor, stream, workspace, workspaceLen), OK);
    if (workspaceLen > 0U) {
        free(workspace);
    }
    NnopbaseExecutorGcSpace(executorSpace);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseUnitTest, NnopBaseDynamicTensorAddrGet)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char* opType = "bninference_d_kernel";
    char inputDesc[] = {2, 1, 2};
    char outputDesc[] = {2, 1, 2};
    char attrDesc[] = {};
    void* executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                         sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    ASSERT_NE(executor, nullptr);

    std::vector<int64_t> shape1 = {1, 1, 1, 1, 1};
    std::vector<int64_t> shape2 = {1, 1, 1, 1, 2};
    std::vector<int64_t> shape3 = {1, 1, 1, 1, 3};
    aclTensor* tensor1 = aclCreateTensor(shape1.data(), shape1.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                         aclFormat::ACL_FORMAT_ND, shape1.data(), shape1.size(), nullptr);
    aclTensor* tensor2 = aclCreateTensor(shape2.data(), shape2.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                         aclFormat::ACL_FORMAT_ND, shape2.data(), shape2.size(), nullptr);
    aclTensor* tensor3 = aclCreateTensor(shape3.data(), shape3.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                         aclFormat::ACL_FORMAT_ND, shape3.data(), shape3.size(), nullptr);
    std::vector<const aclTensor*> tensor_list_a;
    std::vector<const aclTensor*> tensor_list_b;
    tensor_list_a.push_back(tensor1);
    tensor_list_b.push_back(tensor2);
    tensor_list_b.push_back(tensor3);
    aclTensorList* aclTensorTestLista = aclCreateTensorList(tensor_list_a.data(), tensor_list_a.size());
    aclTensorList* aclTensorTestListb = aclCreateTensorList(tensor_list_b.data(), tensor_list_b.size());

    ASSERT_EQ(NnopbaseAddDynamicInput(executor, aclTensorTestLista, 0), 0);
    ASSERT_EQ(NnopbaseAddDynamicOutput(executor, aclTensorTestListb, 0), 0);
    ASSERT_EQ(NnopbaseAddInput(executor, tensor1, 1), 0);
    ASSERT_EQ(NnopbaseAddOutput(executor, tensor2, 1), 0);
    ASSERT_EQ(NnopbaseAddDynamicInput(executor, aclTensorTestListb, 2), 0);
    ASSERT_EQ(NnopbaseAddDynamicOutput(executor, aclTensorTestLista, 2), 0);

    size_t workspaceLen = 0U;
    ASSERT_EQ(NnopbaseRunForWorkspace(executor, &workspaceLen), ACLNN_ERR_INNER_FIND_KERNEL_ERROR);

    uint32_t count = 0;
    std::vector<void*> addrs(2);
    addrs[0] = (void*)tensor1;
    ASSERT_EQ(NnopbaseGetDynamicInputTensorAddrs(executor, 0, &addrs[0], 1, &count), OK);
    ASSERT_EQ(count, 1);
    ASSERT_EQ(addrs[0], nullptr);
    addrs[0] = (void*)tensor1;
    addrs[1] = (void*)tensor1;
    ASSERT_EQ(NnopbaseGetDynamicInputTensorAddrs(executor, 2, &addrs[0], 2, &count), OK);
    ASSERT_EQ(count, 2);
    ASSERT_EQ(addrs[0], nullptr);
    ASSERT_EQ(addrs[1], nullptr);
    addrs[0] = (void*)tensor1;
    ASSERT_EQ(NnopbaseGetDynamicInputTensorAddrs(executor, 2, &addrs[0], 1, &count), OK);
    ASSERT_EQ(count, 1);
    ASSERT_EQ(addrs[0], nullptr);
    addrs[0] = (void*)tensor1;
    ASSERT_EQ(NnopbaseGetDynamicInputTensorAddrs(executor, 0, &addrs[0], 3, &count), OK);
    ASSERT_EQ(count, 1);
    ASSERT_EQ(addrs[0], nullptr);

    addrs[0] = (void*)tensor1;
    ASSERT_EQ(NnopbaseGetDynamicOutputTensorAddrs(executor, 2, &addrs[0], 1, &count), OK);
    ASSERT_EQ(count, 1);
    ASSERT_EQ(addrs[0], nullptr);
    addrs[0] = (void*)tensor1;
    addrs[1] = (void*)tensor1;
    ASSERT_EQ(NnopbaseGetDynamicOutputTensorAddrs(executor, 0, &addrs[0], 2, &count), OK);
    ASSERT_EQ(count, 2);
    ASSERT_EQ(addrs[0], nullptr);
    ASSERT_EQ(addrs[1], nullptr);
    addrs[0] = (void*)tensor1;
    ASSERT_EQ(NnopbaseGetDynamicOutputTensorAddrs(executor, 0, &addrs[0], 1, &count), OK);
    ASSERT_EQ(count, 1);
    ASSERT_EQ(addrs[0], nullptr);
    addrs[0] = (void*)tensor1;
    ASSERT_EQ(NnopbaseGetDynamicOutputTensorAddrs(executor, 2, &addrs[0], 3, &count), OK);
    ASSERT_EQ(count, 1);
    ASSERT_EQ(addrs[0], nullptr);

    ASSERT_EQ(NnopbaseGetDynamicInputTensorAddrs(nullptr, 0, &addrs[0], 1, &count), ACLNN_ERR_PARAM_NULLPTR);
    ASSERT_EQ(NnopbaseGetDynamicOutputTensorAddrs(nullptr, 0, &addrs[0], 1, &count), ACLNN_ERR_PARAM_NULLPTR);
    ASSERT_EQ(NnopbaseGetDynamicInputTensorAddrs(executor, 0, nullptr, 1, &count), ACLNN_ERR_PARAM_NULLPTR);
    ASSERT_EQ(NnopbaseGetDynamicOutputTensorAddrs(executor, 0, nullptr, 1, &count), ACLNN_ERR_PARAM_NULLPTR);
    ASSERT_EQ(NnopbaseGetDynamicInputTensorAddrs(executor, 0, &addrs[0], 0, nullptr), ACLNN_ERR_PARAM_NULLPTR);
    ASSERT_EQ(NnopbaseGetDynamicOutputTensorAddrs(executor, 0, &addrs[0], 0, nullptr), ACLNN_ERR_PARAM_NULLPTR);
    ASSERT_EQ(NnopbaseGetDynamicInputTensorAddrs(executor, 3, &addrs[0], 1, &count), ACLNN_ERR_PARAM_INVALID);
    ASSERT_EQ(NnopbaseGetDynamicOutputTensorAddrs(executor, 3, &addrs[0], 1, &count), ACLNN_ERR_PARAM_INVALID);
    ASSERT_EQ(NnopbaseGetDynamicInputTensorAddrs(executor, 1, &addrs[0], 0, &count), ACLNN_ERR_PARAM_INVALID);
    ASSERT_EQ(NnopbaseGetDynamicOutputTensorAddrs(executor, 1, &addrs[0], 0, &count), ACLNN_ERR_PARAM_INVALID);

    ASSERT_EQ(NnopbaseGetInputTensorCount(executor, 0, &count), OK);
    ASSERT_EQ(count, 1);
    ASSERT_EQ(NnopbaseGetInputTensorCount(executor, 2, &count), OK);
    ASSERT_EQ(count, 2);
    ASSERT_EQ(NnopbaseGetOutputTensorCount(executor, 2, &count), OK);
    ASSERT_EQ(count, 1);
    ASSERT_EQ(NnopbaseGetOutputTensorCount(executor, 0, &count), OK);
    ASSERT_EQ(count, 2);
    ASSERT_EQ(NnopbaseGetInputTensorCount(nullptr, 0, &count), ACLNN_ERR_PARAM_NULLPTR);
    ASSERT_EQ(NnopbaseGetInputTensorCount(executor, 0, nullptr), ACLNN_ERR_PARAM_NULLPTR);
    ASSERT_EQ(NnopbaseGetInputTensorCount(executor, 4, &count), ACLNN_ERR_PARAM_INVALID);
    ASSERT_EQ(NnopbaseGetOutputTensorCount(nullptr, 0, &count), ACLNN_ERR_PARAM_NULLPTR);
    ASSERT_EQ(NnopbaseGetOutputTensorCount(executor, 0, nullptr), ACLNN_ERR_PARAM_NULLPTR);
    ASSERT_EQ(NnopbaseGetOutputTensorCount(executor, 4, &count), ACLNN_ERR_PARAM_INVALID);

    aclDestroyTensorList((const aclTensorList*)aclTensorTestLista);
    aclDestroyTensorList((const aclTensorList*)aclTensorTestListb);
    NnopbaseExecutorGcSpace(executorSpace);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseUnitTest, NnopBaseTensorAddrSet)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);
    void* executor = GetDynamicInputExecutor(executorSpace);
    ASSERT_NE(executor, nullptr);

    size_t workspaceLen = 0U;
    (void)NnopbaseRunForWorkspace(executor, &workspaceLen);

    aclTensor* setTensor = nullptr;
    ASSERT_EQ(NnopbaseSetDynamicInputTensorAddr(executor, 0, 0, (void*)setTensor), OK);
    ASSERT_EQ(NnopbaseSetDynamicInputTensorAddr(executor, 1, 0, (void*)setTensor), OK);
    ASSERT_EQ(NnopbaseSetDynamicOutputTensorAddr(executor, 0, 0, (void*)setTensor), OK);
    ASSERT_EQ(NnopbaseSetDynamicInputTensorAddr(executor, 0, 1, (void*)setTensor), ACLNN_ERR_PARAM_INVALID);
    ASSERT_EQ(NnopbaseSetDynamicOutputTensorAddr(executor, 0, 1, (void*)setTensor), ACLNN_ERR_PARAM_INVALID);
    ASSERT_EQ(NnopbaseSetDynamicInputTensorAddr(executor, 3, 0, (void*)setTensor), ACLNN_ERR_PARAM_INVALID);
    ASSERT_EQ(NnopbaseSetDynamicOutputTensorAddr(executor, 1, 0, (void*)setTensor), ACLNN_ERR_PARAM_INVALID);
    ASSERT_EQ(NnopbaseSetInputTensorAddr(executor, 2, (void*)setTensor), OK);
    ASSERT_EQ(NnopbaseSetOutputTensorAddr(executor, 0, (void*)setTensor), OK);
    ASSERT_EQ(NnopbaseSetInputTensorAddr(executor, 3, (void*)setTensor), ACLNN_ERR_PARAM_INVALID);
    ASSERT_EQ(NnopbaseSetOutputTensorAddr(executor, 1, (void*)setTensor), ACLNN_ERR_PARAM_INVALID);

    void* getTensor = executor;
    NnopbaseGetInputTensorAddr(executor, 0, &getTensor);
    ASSERT_EQ(getTensor, nullptr);
    getTensor = executor;
    NnopbaseGetInputTensorAddr(executor, 1, &getTensor);
    ASSERT_EQ(getTensor, nullptr);
    getTensor = executor;
    NnopbaseGetInputTensorAddr(executor, 2, &getTensor);
    ASSERT_EQ(getTensor, nullptr);
    getTensor = executor;
    NnopbaseGetOutputTensorAddr(executor, 0, &getTensor);
    ASSERT_EQ(getTensor, nullptr);

    NnopbaseExecutorGcSpace(executorSpace);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseUnitTest, NnopBaseRunSuccessWithWorkSpace)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char* opType = "bninference_d_kernel";
    char inputDesc[] = {1, 1, 1};
    char outputDesc[] = {1};
    char attrDesc[] = {};

    void* executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                         sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    ASSERT_NE(executor, nullptr);

    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    (void)NnopbaseAddInput(executor, tensor, 0);
    (void)NnopbaseAddInput(executor, tensor, 1);
    (void)NnopbaseAddInput(executor, tensor, 2);
    (void)NnopbaseAddOutput(executor, tensor, 0);

    const aclTensorList* inUnContTensors = nullptr;
    NnopbaseGetUnContiguousTensors(executor, &inUnContTensors);
    ASSERT_EQ(inUnContTensors, nullptr);

    size_t workspaceLen = 0U;
    ASSERT_EQ(NnopbaseRunForWorkspace(executor, &workspaceLen), OK);
    void* stream = nullptr;
    void* workspace = nullptr;
    if (workspaceLen > 0U) {
        workspace = (void*)malloc(workspaceLen);
    }
    ASSERT_EQ(NnopbaseRunWithWorkspace(executor, stream, workspace, workspaceLen), OK);
    NnopbaseExecutorGcSpace(executorSpace);
    if (workspaceLen > 0U) {
        free(workspace);
    }
    aclDestroyTensor(tensor);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseUnitTest, NnopbaseGetExecutorFailed)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char* opType = "bninferenced";
    char inputDesc[] = {1, 1, 1};
    char outputDesc[] = {1};
    char attrDesc[] = {};
    void* executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                         sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    ASSERT_EQ(executor, nullptr);

    void* stream = nullptr;
    size_t workspaceLen = 0U;
    ASSERT_EQ(NnopbaseRunForWorkspace(executor, &workspaceLen), ACLNN_ERR_PARAM_NULLPTR);
    NnopbaseOpLogE(ACLNN_ERR_PARAM_NULLPTR, "executor != nullptr");

    NnopbaseExecutorGcSpace(executorSpace);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseUnitTest, NnopbaseNoExecutor)
{
    void* executor = nullptr;
    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    ASSERT_EQ(NnopbaseAddInput(executor, tensor, 0), ACLNN_ERR_PARAM_NULLPTR);
    ASSERT_EQ(NnopbaseAddOutput(executor, tensor, 0), ACLNN_ERR_PARAM_NULLPTR);

    std::vector<const aclTensor*> tensor_list_a;
    tensor_list_a.push_back(tensor);
    aclTensorList* aclTensorTestList = aclCreateTensorList(tensor_list_a.data(), tensor_list_a.size());
    ASSERT_EQ(NnopbaseAddDynamicInput(executor, aclTensorTestList, 0), ACLNN_ERR_PARAM_NULLPTR);
    ASSERT_EQ(NnopbaseAddDynamicOutput(executor, aclTensorTestList, 0), ACLNN_ERR_PARAM_NULLPTR);
    static float momentum_def[] = {1};
    ASSERT_EQ(NnopbaseAddAttrWithDtype(executor, momentum_def, sizeof(float), 0, kNnopbaseFloat),
              ACLNN_ERR_PARAM_NULLPTR);

    size_t workspaceLen = 0U;
    ASSERT_EQ(NnopbaseRunForWorkspace(executor, &workspaceLen), ACLNN_ERR_PARAM_NULLPTR);

    void* stream = nullptr;
    void* workspace = nullptr;
    void* handle = executor;
    ASSERT_EQ(NnopbaseRunWithWorkspace(handle, stream, workspace, workspaceLen), ACLNN_ERR_PARAM_NULLPTR);
    if (workspaceLen > 0U) {
        free(workspace);
    }
    aclDestroyTensorList((const aclTensorList*)aclTensorTestList);
}

TEST_F(NnopbaseUnitTest, NnopbaseParamCheckFailed)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char* opType = "bninference_d_kernel";
    char inputDesc[] = {1, 1, 1};
    char outputDesc[] = {1};
    char attrDesc[] = {};

    void* executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                         sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    ASSERT_NE(executor, nullptr);

    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    (void)NnopbaseAddInput(executor, tensor, 0);
    (void)NnopbaseAddInput(executor, tensor, 1);
    (void)NnopbaseAddInput(executor, tensor, 2);
    (void)NnopbaseAddOutput(executor, tensor, 0);

    ASSERT_EQ(NnopbaseRunForWorkspace(executor, nullptr), ACLNN_ERR_PARAM_NULLPTR);
    size_t workspaceLen = 0U;
    ASSERT_EQ(NnopbaseRunForWorkspace(executor, &workspaceLen), OK);

    NnopbaseExecutor* nnopExecutor = (NnopbaseExecutor*)executor;
    nnopExecutor->workspaces.length = workspaceLen + 2U;
    void* stream = nullptr;
    void* handle = executor;
    void* workspace = nullptr;
    if (workspaceLen > 0U) {
        workspace = (void*)malloc(workspaceLen);
    }
    ASSERT_EQ(NnopbaseRunWithWorkspace(handle, stream, workspace, workspaceLen), ACLNN_ERR_PARAM_INVALID);
    if (workspaceLen > 0U) {
        free(workspace);
    }
    NnopbaseExecutorGcSpace(executorSpace);
    aclDestroyTensor(tensor);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseUnitTest, NnopBaseRunFailed)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char* opType = "bninference_d_kernel";
    char inputDesc[] = {1, 1, 1};
    char outputDesc[] = {1};
    char attrDesc[] = {};
    void* executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                         sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    ASSERT_NE(executor, nullptr);

    TensorDesc inputDesc1[3] = {
        {ge::DT_FLOAT16, ge::FORMAT_ND}, {ge::DT_FLOAT, ge::FORMAT_ND}, {ge::DT_FLOAT, ge::FORMAT_ND}};
    TensorDesc outputDesc1[1] = {{ge::DT_FLOAT, ge::FORMAT_ND}};
    SupportInfo list1 = {inputDesc1, 3, outputDesc1, 1};
    SupportInfo supportInfo0[1] = {list1};
    SupportInfo supportInfo1[1] = {list1};
    OpSocSupportInfo socSupportInfo0 = {supportInfo0, 1};
    OpSocSupportInfo socSupportInfo1 = {supportInfo1, 1};
    OpSocSupportInfo opSocSupportList[2] = {socSupportInfo0, socSupportInfo1};
    OpSupportList supportList = {opSocSupportList, 2};
    uint32_t socSupportList[] = {nnopbase::SOC_VERSION_ASCEND910A, nnopbase::SOC_VERSION_ASCEND910B};
    ASSERT_EQ(NnopbaseAddSupportList(executor, &supportList, socSupportList, 2), OK);

    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT16, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    (void)NnopbaseAddInput(executor, tensor, 0);
    (void)NnopbaseAddInput(executor, tensor, 1);
    (void)NnopbaseAddInput(executor, tensor, 2);
    (void)NnopbaseAddOutput(executor, tensor, 0);
    (void)NnopbaseAddOutput(executor, tensor, 1);
    ASSERT_EQ(NnopbaseAddParamName(executor, 0, "x1", true), OK);
    ASSERT_EQ(NnopbaseAddParamName(executor, 1, "x2", true), OK);
    ASSERT_EQ(NnopbaseAddParamName(executor, 2, "x3", true), OK);
    ASSERT_EQ(NnopbaseAddParamName(executor, 0, "y", false), OK);

    size_t workspaceLen = 0U;
    ASSERT_EQ(NnopbaseRunForWorkspace(executor, &workspaceLen), ACLNN_ERR_PARAM_INVALID);
    aclDestroyTensor(tensor);
    NnopbaseExecutorGcSpace(executorSpace);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseUnitTest, NnopBaseDynamicCheckFailed)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char* opType = "bninference_d_kernel";
    char inputDesc[] = {2, 1, 0, 1};
    char outputDesc[] = {1};
    char attrDesc[] = {};

    void* executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                         sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    ASSERT_NE(executor, nullptr);

    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor1 = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT16, nullptr, 0,
                                         aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    aclTensor* tensor2 = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT16, nullptr, 0,
                                         aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    std::vector<const aclTensor*> tensor_list;
    tensor_list.push_back(tensor1);
    tensor_list.push_back(tensor2);
    aclTensorList* tensorTestList = aclCreateTensorList(tensor_list.data(), tensor_list.size());

    ASSERT_EQ(NnopbaseAddDynamicInput(executor, tensorTestList, 0), OK);
    ASSERT_EQ(NnopbaseAddInput(executor, tensor1, 1), OK);
    ASSERT_EQ(NnopbaseAddInput(executor, tensor1, 2), OK);
    ASSERT_EQ(NnopbaseAddInput(executor, tensor1, 3), OK);
    ASSERT_EQ(NnopbaseAddOutput(executor, tensor1, 0), OK);

    TensorDesc inputDesc1[4] = {{ge::DT_FLOAT, ge::FORMAT_ND},
                                {ge::DT_FLOAT16, ge::FORMAT_ND},
                                {ge::DT_FLOAT16, ge::FORMAT_ND},
                                {ge::DT_FLOAT16, ge::FORMAT_ND}};
    TensorDesc outputDesc1[1] = {{ge::DT_FLOAT16, ge::FORMAT_ND}};
    SupportInfo list1 = {inputDesc1, 4, outputDesc1, 1};
    SupportInfo supportInfo0[1] = {list1};
    SupportInfo supportInfo1[1] = {list1};
    OpSocSupportInfo socSupportInfo0 = {supportInfo0, 1};
    OpSocSupportInfo socSupportInfo1 = {supportInfo1, 1};
    OpSocSupportInfo opSocSupportList[2] = {socSupportInfo0, socSupportInfo1};
    OpSupportList supportList = {opSocSupportList, 2};
    uint32_t socSupportList[] = {nnopbase::SOC_VERSION_ASCEND910A, nnopbase::SOC_VERSION_ASCEND910B};
    ASSERT_EQ(NnopbaseAddSupportList(executor, &supportList, socSupportList, 2), OK);

    size_t workspaceLen = 0U;
    ASSERT_EQ(NnopbaseRunForWorkspace(executor, &workspaceLen), ACLNN_ERR_PARAM_INVALID);
    aclDestroyTensorList((const aclTensorList*)tensorTestList);
    NnopbaseExecutorGcSpace(executorSpace);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseUnitTest, NnopbaseCheckFailedWithUnsupportOutputDtype)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char* opType = "bninference_d_kernel";
    char inputDesc[] = {1, 1, 1};
    char outputDesc[] = {0, 1};
    char attrDesc[] = {};

    void* executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                         sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    ASSERT_NE(executor, nullptr);

    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT16, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    (void)NnopbaseAddInput(executor, tensor, 0);
    (void)NnopbaseAddInput(executor, tensor, 1);
    (void)NnopbaseAddInput(executor, tensor, 2);
    (void)NnopbaseAddOutput(executor, tensor, 1);

    TensorDesc inputDesc1[3] = {
        {ge::DT_FLOAT16, ge::FORMAT_ND}, {ge::DT_FLOAT16, ge::FORMAT_ND}, {ge::DT_FLOAT16, ge::FORMAT_ND}};
    TensorDesc outputDesc1[2] = {{ge::DT_FLOAT, ge::FORMAT_ND}, {ge::DT_FLOAT, ge::FORMAT_ND}};
    SupportInfo list1 = {inputDesc1, 3, outputDesc1, 2};
    SupportInfo supportInfo0[1] = {list1};
    SupportInfo supportInfo1[1] = {list1};
    OpSocSupportInfo socSupportInfo0 = {supportInfo0, 1};
    OpSocSupportInfo socSupportInfo1 = {supportInfo1, 1};
    OpSocSupportInfo opSocSupportList[2] = {socSupportInfo0, socSupportInfo1};
    OpSupportList supportList = {opSocSupportList, 2};
    uint32_t socSupportList[] = {nnopbase::SOC_VERSION_ASCEND910A, nnopbase::SOC_VERSION_ASCEND910B};
    ASSERT_EQ(NnopbaseAddSupportList(executor, &supportList, socSupportList, 2), OK);

    size_t workspaceLen = 0U;
    ASSERT_EQ(NnopbaseRunForWorkspace(executor, &workspaceLen), ACLNN_ERR_PARAM_INVALID);
    aclDestroyTensor(tensor);
    NnopbaseExecutorGcSpace(executorSpace);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseUnitTest, NnopBaseNotFindKernel)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char* opType = "bninference_d_kernel";
    char inputDesc[] = {1, 2, 1};
    char outputDesc[] = {1};
    char attrDesc[] = {};

    void* executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                         sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    ASSERT_NE(executor, nullptr);

    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor1 = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT16, nullptr, 0,
                                         aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    aclTensor* tensor2 = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT16, nullptr, 0,
                                         aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    std::vector<const aclTensor*> tensor_list;
    tensor_list.push_back(tensor1);
    tensor_list.push_back(tensor2);
    aclTensorList* tensorTestList = aclCreateTensorList(tensor_list.data(), tensor_list.size());

    ASSERT_EQ(NnopbaseAddInput(executor, tensor1, 0), OK);
    ASSERT_EQ(NnopbaseAddDynamicInput(executor, tensorTestList, 1), OK);
    ASSERT_EQ(NnopbaseAddInput(executor, tensor1, 2), OK);
    ASSERT_EQ(NnopbaseAddOutput(executor, tensor1, 0), OK);

    TensorDesc inputDesc1[3] = {
        {ge::DT_FLOAT16, ge::FORMAT_ND}, {ge::DT_FLOAT16, ge::FORMAT_ND}, {ge::DT_FLOAT16, ge::FORMAT_ND}};
    TensorDesc outputDesc1[1] = {{ge::DT_FLOAT16, ge::FORMAT_ND}};
    SupportInfo list1 = {inputDesc1, 3, outputDesc1, 1};
    SupportInfo supportInfo0[1] = {list1};
    SupportInfo supportInfo1[1] = {list1};
    OpSocSupportInfo socSupportInfo0 = {supportInfo0, 1};
    OpSocSupportInfo socSupportInfo1 = {supportInfo1, 1};
    OpSocSupportInfo opSocSupportList[2] = {socSupportInfo0, socSupportInfo1};
    OpSupportList supportList = {opSocSupportList, 2};
    uint32_t socSupportList[] = {nnopbase::SOC_VERSION_ASCEND910A, nnopbase::SOC_VERSION_ASCEND910B};
    ASSERT_EQ(NnopbaseAddSupportList(executor, &supportList, socSupportList, 2), OK);

    size_t workspaceLen = 0U;
    ASSERT_EQ(NnopbaseRunForWorkspace(executor, &workspaceLen), ACLNN_ERR_INNER_FIND_KERNEL_ERROR);
    aclDestroyTensorList((const aclTensorList*)tensorTestList);
    NnopbaseExecutorGcSpace(executorSpace);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseUnitTest, NnopbaseUnlimitedMaxNumExecutor)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char* opType = "bninference_d_kernel";
    char inputDesc[] = {1, 1, 1};
    char outputDesc[] = {1};
    char attrDesc[] = {};
    void* executor = nullptr;
    for (int i = 0; i < 100; i++) {
        executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                       sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
        ASSERT_NE(executor, nullptr);
    }

    executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                   sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    ASSERT_NE(executor, nullptr);

    NnopbaseExecutorGcSpace(executorSpace);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseUnitTest, TestNnopbaseAddSupportList)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char* opType = "bninference_d_kernel";
    char inputDesc[] = {1, 1, 1};
    char outputDesc[] = {1};
    char attrDesc[] = {};
    void* executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                         sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    ASSERT_NE(executor, nullptr);

    TensorDesc inputDesc1[3] = {
        {ge::DT_FLOAT, ge::FORMAT_ND}, {ge::DT_FLOAT, ge::FORMAT_ND}, {ge::DT_FLOAT, ge::FORMAT_ND}};
    TensorDesc outputDesc1[1] = {{ge::DT_FLOAT, ge::FORMAT_ND}};
    SupportInfo list1 = {inputDesc1, 2, outputDesc1, 1};
    SupportInfo supportInfo0[1] = {list1};
    SupportInfo supportInfo1[1] = {list1};
    OpSocSupportInfo socSupportInfo0 = {supportInfo0, 1};
    OpSocSupportInfo socSupportInfo1 = {supportInfo1, 1};
    OpSocSupportInfo opSocSupportList[2] = {socSupportInfo0, socSupportInfo1};
    OpSupportList supportList = {opSocSupportList, 2};
    uint32_t socSupportList[] = {nnopbase::SOC_VERSION_ASCEND910A, nnopbase::SOC_VERSION_ASCEND910B};
    ASSERT_EQ(NnopbaseAddSupportList(executor, &supportList, socSupportList, 2), OK);
    NnopbaseExecutorGcSpace(executorSpace);
}

TEST_F(NnopbaseUnitTest, TestNnopbaseAddOpTypeId)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char* opType = "bninference_d_kernel";
    char inputDesc[] = {1, 1, 1};
    char outputDesc[] = {1};
    char attrDesc[] = {};
    void* executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                         sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    ASSERT_NE(executor, nullptr);

    NnopbaseAddOpTypeId(executor, 32);
    ASSERT_EQ(((NnopbaseExecutor*)executor)->opTypeId, 32);

    NnopbaseExecutorGcSpace(executorSpace);
}

TEST_F(NnopbaseUnitTest, NnopBaseRunSuccessWithNclFormat)
{
#if 0 // comment out for ge decoupling
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    void *executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char *opType = "bninference_d_kernel";
    char inputDesc[] = {1, 1, 1};
    char outputDesc[] = {1};
    char attrDesc[] = {};

    void *executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                         sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    ASSERT_NE(executor, nullptr);

    TensorDesc inputDesc1[3] = {{ge::DT_FLOAT, ge::FORMAT_ND}, {ge::DT_FLOAT, ge::FORMAT_ND}, {ge::DT_FLOAT, ge::FORMAT_ND}};
    TensorDesc outputDesc1[1] = {{ge::DT_FLOAT, ge::FORMAT_ND}};
    SupportInfo list1 = {inputDesc1, 2, outputDesc1, 1};
    SupportInfo supportInfo0[1] = {list1};
    SupportInfo supportInfo1[1] = {list1};
    OpSocSupportInfo socSupportInfo0 = {supportInfo0, 1};
    OpSocSupportInfo socSupportInfo1 = {supportInfo1, 1};
    OpSocSupportInfo opSocSupportList[2] = {socSupportInfo0, socSupportInfo1};
    OpSupportList supportList = {opSocSupportList, 2};
    uint32_t socSupportList[] = {nnopbase::SOC_VERSION_ASCEND910A, nnopbase::SOC_VERSION_ASCEND910B};
    ASSERT_EQ(NnopbaseAddSupportList(executor, &supportList, socSupportList, 2), OK);

    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    std::vector<int64_t> storageShape = {64};
    aclTensor *tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT,
                                          nullptr, 0, aclFormat::ACL_FORMAT_NCL, storageShape.data(), storageShape.size(), nullptr);
    (void) NnopbaseAddInput(executor, tensor, 0);
    (void) NnopbaseAddInput(executor, tensor, 1);
    (void) NnopbaseAddInput(executor, tensor, 2);
    (void) NnopbaseAddOutput(executor, tensor, 0);

    size_t workspaceLen = 0U;
    ASSERT_EQ(NnopbaseRunForWorkspace(executor, &workspaceLen), OK);
    void *stream = nullptr;
    void *workspace = nullptr;
    if (workspaceLen > 0U) {
        workspace = (void *) malloc(workspaceLen);
    }
    ASSERT_EQ(NnopbaseRunWithWorkspace(executor, stream, workspace, workspaceLen), OK);
    const size_t tilingDataSize =
        ((NnopbaseTilingData *)(((NnopbaseExecutor *)executor)->args->tilingInfo.tilingData))->GetDataSize();
    const size_t alignTilingDataSize =
            ((tilingDataSize % 8U) != 0) ? (tilingDataSize / 8U + 1U) * 8U : tilingDataSize; // 8byte对齐
    uint64_t *ptr = (uint64_t *)((NnopbaseUChar *)((NnopbaseTilingData *)(((NnopbaseExecutor *)executor)
                                                                              ->args->tilingInfo.tilingData))
                                     ->GetData() +
                                 alignTilingDataSize);
    // check io size
    size_t expIOSize[] = {32, 32, 32, 32, 0};
    for (size_t i = 0; i < 5; i++) {
        ASSERT_EQ(ptr[i], expIOSize[i]);
    }
    NnopbaseExecutorGcSpace(executorSpace);
    if (workspaceLen > 0U) {
        free(workspace);
    }
    aclDestroyTensor(tensor);
    NnopbaseUnsetEnvAndClearFolder();
#endif
}

TEST_F(NnopbaseUnitTest, NnopbaseFindSupportInfoSuccess)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char* opType = "bninference_d_kernel";
    char inputDesc[] = {2, 1, 1};
    char outputDesc[] = {1};
    char attrDesc[] = {};

    void* executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                         sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    ASSERT_NE(executor, nullptr);

    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    aclTensor* tensor1 = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT16, nullptr, 0,
                                         aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);

    std::vector<const aclTensor*> tensor_list_a;
    tensor_list_a.push_back(tensor);
    tensor_list_a.push_back(tensor1);
    aclTensorList* aclTensorTestList = aclCreateTensorList(tensor_list_a.data(), tensor_list_a.size());

    (void)NnopbaseAddDynamicInput(executor, aclTensorTestList, 0);
    (void)NnopbaseAddInput(executor, tensor, 1);
    (void)NnopbaseAddInput(executor, tensor, 2);
    (void)NnopbaseAddOutput(executor, tensor, 0);

    TensorDesc inputDesc0[3] = {
        {ge::DT_FLOAT, ge::FORMAT_ND}, {ge::DT_FLOAT, ge::FORMAT_ND}, {ge::DT_FLOAT, ge::FORMAT_ND}};
    TensorDesc outputDesc0[1] = {{ge::DT_FLOAT, ge::FORMAT_ND}};
    SupportInfo list0 = {inputDesc0, 3, outputDesc0, 1};
    SupportInfo supportInfo0[1] = {list0};
    SupportInfo supportInfo1[1] = {list0};
    OpSocSupportInfo socSupportInfo0 = {supportInfo0, 1};
    OpSocSupportInfo socSupportInfo1 = {supportInfo1, 1};
    OpSocSupportInfo opSocSupportList[2] = {socSupportInfo0, socSupportInfo1};
    OpSupportList supportList = {opSocSupportList, 2};
    uint32_t socSupportList[] = {nnopbase::SOC_VERSION_ASCEND910A, nnopbase::SOC_VERSION_ASCEND910B};
    ASSERT_EQ(NnopbaseAddSupportList(executor, &supportList, socSupportList, 2), OK);

    size_t workspaceLen = 0U;
    (void)NnopbaseRunForWorkspace(executor, &workspaceLen);

    EXPECT_EQ(CheckSocVersionAndParam((NnopbaseExecutor*)executor, NnopbaseParamCheckMode::kCheckRequiredIo), OK);
    aclDestroyTensorList((const aclTensorList*)aclTensorTestList);
    NnopbaseExecutorGcSpace(executorSpace);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseUnitTest, NnopbaseAddUnContiguousTensor)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char* opType = "bninference_d_kernel";
    char inputDesc[] = {2, 1};
    char outputDesc[] = {1};
    char attrDesc[] = {};

    void* executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                         sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    ASSERT_NE(executor, nullptr);

    std::vector<int64_t> shape = {1, 3, 1};
    std::vector<int64_t> shape1 = {1, 2, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, shape1.data(), 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    aclTensor* tensor1 = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT16, shape1.data(), 0,
                                         aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);

    std::vector<const aclTensor*> tensor_list_a;
    tensor_list_a.push_back(tensor);
    tensor_list_a.push_back(tensor1);
    aclTensorList* aclTensorTestList = aclCreateTensorList(tensor_list_a.data(), tensor_list_a.size());

    ASSERT_EQ(NnopbaseAddDynamicInput(executor, aclTensorTestList, 0), OK);
    ASSERT_EQ(NnopbaseAddIgnoreContinuesInput(executor, tensor, 1), OK);

    aclDestroyTensorList((const aclTensorList*)aclTensorTestList);
    NnopbaseExecutorGcSpace(executorSpace);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseUnitTest, NnopContiguousUpdateAddrSuccess)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);
    NnopbaseInit();

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char* opType = "bninference_d_kernel";
    char inputDesc[] = {1, 1, 1};
    char outputDesc[] = {1};
    char attrDesc[] = {};
    void* executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                         sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    ASSERT_NE(executor, nullptr);

    std::vector<int64_t> shape = {1, 3, 1, 1, 1};
    std::vector<int64_t> shape1 = {1, 2, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, shape1.data(), 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);

    ASSERT_EQ(NnopbaseSetRef(executor, 0, 0), OK);
    ASSERT_EQ(NnopbaseAddInput(executor, tensor, 0), OK);
    ASSERT_EQ(NnopbaseAddInput(executor, tensor, 1), OK);
    ASSERT_EQ(NnopbaseAddInput(executor, tensor, 2), OK);
    ASSERT_EQ(NnopbaseAddOutput(executor, tensor, 0), OK);

    const aclTensorList* inUnContTensors = nullptr;
    NnopbaseGetUnContiguousTensors(executor, &inUnContTensors);
    aclOpExecutor* aclInExecutor = new aclOpExecutor;
    op::FVector<uint64_t> ws({8, 80, 160});
    aclInExecutor->SetWorkspaceOffsets(ws);
    uint64_t inContWorkspaceSize = 1024;
    ASSERT_EQ(NnopbaseSetUnContExecutor(executor, aclInExecutor, inContWorkspaceSize), OK);
    size_t workspaceLen = 0U;
    ASSERT_EQ(NnopbaseRunForWorkspace(executor, &workspaceLen), OK);

    aclOpExecutor* viewcopyExecutor = new aclOpExecutor;
    const aclTensorList* unContTensors = nullptr;
    const aclTensorList* contTensors = nullptr;
    ASSERT_EQ(NnopbaseGetRefUnContiguousTensors(executor, &unContTensors, &contTensors), OK);
    if (unContTensors != nullptr) {
        ASSERT_EQ(NnopbaseSetViewCopyExecutor(executor, viewcopyExecutor), OK);
    }

    uint64_t handleExpectValue = 1;
    void* handle = &handleExpectValue;
    NnopbaseSetUserHandle(executor, handle);
    uint64_t handleActualValue = *((uintptr_t*)NnopbaseGetUserHandle(executor));
    ASSERT_EQ(handleActualValue, handleExpectValue);
    ASSERT_EQ(NnopbaseGetUserHandle(nullptr), nullptr);

    auto opExe = (NnopbaseExecutor*)executor;
    ASSERT_EQ(opExe->args->inputs.unContiguousTensors.tensors.size(), 3);
    ASSERT_EQ(opExe->args->inputs.unContiguousTensors.tensors[0], tensor);
    ASSERT_EQ(opExe->args->inputs.unContiguousTensors.tensors[1], tensor);
    ASSERT_EQ(opExe->args->inputs.unContiguousTensors.tensors[2], tensor);

    // ASSERT_EQ(aclSetAclOpExecutorRepeatable((aclOpExecutor *)executor), OK);

    // aclOpExecutor *aclRunInExecutor = nullptr;
    // uint64_t inRunContWorkspaceSize = 0;
    // ASSERT_EQ(NnopbaseGetUnContExecutor(executor, &aclRunInExecutor, &inRunContWorkspaceSize), OK);
    // ASSERT_EQ(aclRunInExecutor, aclInExecutor);
    // ASSERT_EQ(inRunContWorkspaceSize, inContWorkspaceSize);

    // void *stream = nullptr;
    // void *workspace = (void *) malloc(workspaceLen + inContWorkspaceSize);
    // ASSERT_EQ(NnopbaseRunWithWorkspace(executor, stream, workspace, workspaceLen), OK);
    // char *inWorkspace = (char *)workspace + workspaceLen;
    // for (size_t i = 0; i < 3; ++i) {
    //     auto opExe = (NnopbaseExecutor *)executor;
    //     auto &rt2Tensor = opExe->args->inputs.extTensors[i].rt2Tensor;
    //     ASSERT_EQ(rt2Tensor.GetAddr(), inWorkspace + ws[i]);
    // }

    // const aclTensorList *viewcopyTensors = nullptr;
    // ASSERT_EQ(NnopbaseReleaseRefContiguousTensors(executor, &viewcopyTensors), OK);
    // if (viewcopyExecutor != nullptr) {
    //     ASSERT_EQ(aclDestroyTensorList(viewcopyTensors), OK);
    // }

    // int *ptr = new int;
    // void *addr = reinterpret_cast<void *>(ptr);
    // ASSERT_EQ(AclSetInputTensorAddr((aclOpExecutor *)executor, 0, tensor, addr), OK);
    // ASSERT_EQ(((NnopbaseExecutor *)executor)->args->inputs.extTensors[0].rt2Tensor.GetAddr(), addr);
    // ASSERT_EQ(aclDestroyAclOpExecutor((aclOpExecutor *)executor), OK);

    // delete ptr;
    // free(workspace);
    NnopbaseExecutorGcSpace(executorSpace);
    aclDestroyTensor(tensor);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseUnitTest, NnopContiguousUpdateAddrFailed)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);
    NnopbaseInit();

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char* opType = "bninference_d_kernel";
    char inputDesc[] = {1, 1, 1};
    char outputDesc[] = {1};
    char attrDesc[] = {};
    void* executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                         sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    ASSERT_NE(executor, nullptr);

    std::vector<int64_t> shape = {1, 3, 1, 1, 1};
    std::vector<int64_t> shape1 = {1, 2, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, shape1.data(), 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);

    (void)NnopbaseAddInput(executor, tensor, 0);
    (void)NnopbaseAddInput(executor, tensor, 1);
    (void)NnopbaseAddInput(executor, tensor, 2);
    (void)NnopbaseAddOutput(executor, tensor, 0);

    const aclTensorList* inUnContTensors = nullptr;
    NnopbaseGetUnContiguousTensors(executor, &inUnContTensors);
    aclOpExecutor aclInExecutor;
    op::FVector<uint64_t> ws({8, 80, 160});
    aclInExecutor.SetWorkspaceOffsets(ws);
    uint64_t inContWorkspaceSize = 1024;
    ASSERT_EQ(NnopbaseSetUnContExecutor(executor, &aclInExecutor, inContWorkspaceSize), OK);

    size_t workspaceLen = 0U;
    ASSERT_EQ(NnopbaseRunForWorkspace(executor, &workspaceLen), OK);

    op::internal::OpExecCache* cache = ((NnopbaseExecutor*)executor)->inUnContExe->GetOpExecCache();
    op::internal::OpExecCacheWrap* cacheWrap = op::internal::CreateCacheWrap(cache);
    aclOpExecutor* fakeExecutor = reinterpret_cast<aclOpExecutor*>(cacheWrap);
    ((NnopbaseExecutor*)executor)->inUnContExe = fakeExecutor;
    ASSERT_EQ(aclSetAclOpExecutorRepeatable((aclOpExecutor*)executor), ACLNN_ERR_INNER);
    ASSERT_EQ(aclDestroyAclOpExecutor((aclOpExecutor*)executor), ACLNN_ERR_INNER);

    delete cacheWrap;
    NnopbaseExecutorGcSpace(executorSpace);
    aclDestroyTensor(tensor);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseUnitTest, NnopUnContiguousUpdateAddrFailed)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char* opType = "bninference_d_kernel";
    char inputDesc[] = {1, 2, 1};
    char outputDesc[] = {1};
    char attrDesc[] = {};
    void* executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                         sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    ASSERT_NE(executor, nullptr);

    std::vector<int64_t> shape = {1, 3, 1, 1, 1};
    std::vector<int64_t> shape1 = {1, 2, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, shape1.data(), 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    aclTensor* tensor1 = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, shape1.data(), 0,
                                         aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    std::vector<int64_t> shape2 = {1, 1, 1, 1, 1};
    aclTensor* tensor2 = aclCreateTensor(shape2.data(), shape2.size(), aclDataType::ACL_FLOAT, shape2.data(), 0,
                                         aclFormat::ACL_FORMAT_ND, shape2.data(), shape2.size(), nullptr);

    std::vector<const aclTensor*> tensor_list;
    tensor_list.push_back(tensor);
    tensor_list.push_back(tensor1);
    aclTensorList* aclTensorTestList = aclCreateTensorList(tensor_list.data(), tensor_list.size());

    (void)NnopbaseAddInput(executor, tensor, 0);
    ASSERT_EQ(NnopbaseAddDynamicInput(executor, aclTensorTestList, 1), OK);
    (void)NnopbaseAddInput(executor, tensor, 2);
    (void)NnopbaseAddOutput(executor, tensor2, 0);

    ASSERT_EQ(NnopbaseAddParamName(executor, 0, "x1", true), OK);
    ASSERT_EQ(NnopbaseAddParamName(executor, 1, "x2", true), OK);
    ASSERT_EQ(NnopbaseAddParamName(executor, 2, "x3", true), OK);
    ASSERT_EQ(NnopbaseAddParamName(executor, 0, "y", false), OK);

    size_t workspaceLen = 0U;
    ASSERT_EQ(NnopbaseRunForWorkspace(executor, &workspaceLen), OK);
    void* stream = nullptr;
    void* workspace = nullptr;
    if (workspaceLen > 0U) {
        workspace = (void*)malloc(workspaceLen);
    }
    ASSERT_EQ(NnopbaseRunWithWorkspace(executor, stream, workspace, workspaceLen), ACLNN_ERR_INNER);
    if (workspaceLen > 0U) {
        free(workspace);
    }

    NnopbaseExecutorGcSpace(executorSpace);
    aclDestroyTensor(tensor2);
    aclDestroyTensorList((const aclTensorList*)aclTensorTestList);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseUnitTest, NnopBaseRunWithArrayAttrSuccess)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char* opType = "bninference_d_kernel";
    char inputDesc[] = {1, 1, 1};
    char outputDesc[] = {1};
    char attrDesc[] = {1, 1, 1};
    void* executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                         sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    ASSERT_NE(executor, nullptr);

    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    (void)NnopbaseAddInput(executor, tensor, 0);
    (void)NnopbaseAddInput(executor, tensor, 1);
    (void)NnopbaseAddInput(executor, tensor, 2);
    (void)NnopbaseAddOutput(executor, tensor, 0);

    bool boolValues[] = {true, false, true};
    auto* bias0 = aclCreateBoolArray(boolValues, sizeof(boolValues) / sizeof(boolValues[0]));
    float floatValues[] = {3, 4, 5};
    auto* bias1 = aclCreateFloatArray(floatValues, sizeof(floatValues) / sizeof(floatValues[0]));
    int64_t intValues[] = {3, 4, 5};
    auto* bias2 = aclCreateIntArray(intValues, sizeof(intValues) / sizeof(intValues[0]));
    ASSERT_EQ(NnopbaseAddBoolArrayAttr(executor, bias0, 0), OK);
    ASSERT_EQ(NnopbaseAddFloatArrayAttr(executor, bias1, 1), OK);
    ASSERT_EQ(NnopbaseAddIntArrayAttr(executor, bias2, 2), OK);

    size_t workspaceLen = 0U;
    ASSERT_EQ(NnopbaseRunForWorkspace(executor, &workspaceLen), OK);
    void* stream = nullptr;
    void* workspace = nullptr;
    if (workspaceLen > 0U) {
        workspace = (void*)malloc(workspaceLen);
    }
    ASSERT_EQ(NnopbaseRunWithWorkspace(executor, stream, workspace, workspaceLen), OK);
    NnopbaseExecutorGcSpace(executorSpace);
    if (workspaceLen > 0U) {
        free(workspace);
    }
    aclDestroyTensor(tensor);
    aclDestroyBoolArray(bias0);
    aclDestroyFloatArray(bias1);
    aclDestroyIntArray(bias2);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseUnitTest, NnopBaseRunStaticKernelSuccess)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char* opType = "Flash";
    char inputDesc[] = {1, 1, 1};
    char outputDesc[] = {1};
    char attrDesc[] = {};
    void* executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                         sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    ASSERT_NE(executor, nullptr);

    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    (void)NnopbaseAddInput(executor, tensor, 0);
    (void)NnopbaseAddInput(executor, tensor, 1);
    (void)NnopbaseAddInput(executor, tensor, 2);
    (void)NnopbaseAddOutput(executor, tensor, 0);

    size_t workspaceLen = 0U;
    ASSERT_EQ(NnopbaseRunForWorkspace(executor, &workspaceLen), OK);
    void* stream = nullptr;
    void* workspace = nullptr;
    if (workspaceLen > 0U) {
        workspace = (void*)malloc(workspaceLen);
    }
    ASSERT_EQ(NnopbaseRunWithWorkspace(executor, stream, workspace, workspaceLen), OK);

    // 测试多次执行
    (void)NnopbaseAddInput(executor, tensor, 0);
    (void)NnopbaseAddInput(executor, tensor, 1);
    (void)NnopbaseAddInput(executor, tensor, 2);
    (void)NnopbaseAddOutput(executor, tensor, 0);
    ASSERT_EQ(NnopbaseRunForWorkspace(executor, &workspaceLen), OK);
    ASSERT_EQ(NnopbaseRunWithWorkspace(executor, stream, workspace, workspaceLen), OK);

    if (workspaceLen > 0U) {
        free(workspace);
    }

    NnopbaseExecutorGcSpace(executorSpace);
    aclDestroyTensor(tensor);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseUnitTest, NnopBaseFindStaticKernelFailed)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    const char* opType = "Flash1";
    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    const aclTensor* tensors[] = {tensor, tensor, tensor, tensor};
    int64_t numTensors = 4;
    const int64_t dynamicIndex[] = {};
    const int64_t dynamicCount[] = {};
    int64_t numDynamic = 0;
    const NnopbaseAttrAddr* attrs[] = {};
    int64_t numAttrs = 0;
    int64_t implMode = 0;
    int64_t deterMin = 0;
    const int64_t valueDepend[] = {};
    int64_t numValueDepend = 0;
    NnopbaseStaticTensorNumInfo tensorNumInfo;
    tensorNumInfo.numTensors = numTensors;
    tensorNumInfo.numDynamic = numDynamic;
    tensorNumInfo.numAttrs = numAttrs;
    tensorNumInfo.numValueDepend = numValueDepend;
    NnopbaseStaticRuntimeInfo staticRuntimeInfo;
    staticRuntimeInfo.opType = std::string(opType);
    staticRuntimeInfo.aicNum = coreNum.aicNum;
    staticRuntimeInfo.aivNum = coreNum.aivNum;
    staticRuntimeInfo.implMode = implMode;
    staticRuntimeInfo.deterMode = deterMin;

    const char* path = NnopbaseFindStaticKernel(tensors, attrs, valueDepend, &tensorNumInfo, &staticRuntimeInfo);
    ASSERT_EQ(path, nullptr);
    aclDestroyTensor(tensor);
    NnopbaseUnsetEnvAndClearFolder();
    GlobalMockObject::verify();
}

TEST_F(NnopbaseUnitTest, NnopBaseFindStaticKernelOpTypeFailed)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    const char* opType = "Flash1";
    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    const aclTensor* tensors[] = {tensor, tensor, tensor, tensor};
    int64_t numTensors = 4;
    const int64_t dynamicIndex[] = {};
    const int64_t dynamicCount[] = {};
    int64_t numDynamic = 0;
    const NnopbaseAttrAddr* attrs[] = {};
    int64_t numAttrs = 0;
    int64_t implMode = 0;
    int64_t deterMin = 0;
    const int64_t valueDepend[] = {};
    int64_t numValueDepend = 0;
    NnopbaseStaticTensorNumInfo tensorNumInfo;
    tensorNumInfo.numTensors = numTensors;
    tensorNumInfo.numDynamic = numDynamic;
    tensorNumInfo.numAttrs = numAttrs;
    tensorNumInfo.numValueDepend = numValueDepend;
    NnopbaseStaticRuntimeInfo staticRuntimeInfo;
    staticRuntimeInfo.opType = std::string(opType);
    staticRuntimeInfo.aicNum = coreNum.aicNum;
    staticRuntimeInfo.aivNum = coreNum.aivNum;
    staticRuntimeInfo.implMode = implMode;
    staticRuntimeInfo.deterMode = deterMin;

    const char* path = NnopbaseFindStaticKernel(tensors, attrs, valueDepend, &tensorNumInfo, &staticRuntimeInfo);
    ASSERT_EQ(path, nullptr);
    aclDestroyTensor(tensor);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseUnitTest, NnopBaseFindStaticKernelRefreshFailed)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);
    EXPECT_EQ(NnopbaseCollectorDeleteStaticBins(nullptr), OK);

    const char* opType = "Flash";
    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    const aclTensor* tensors[] = {tensor, tensor, tensor, tensor};
    int64_t numTensors = 4;
    const int64_t dynamicIndex[] = {};
    const int64_t dynamicCount[] = {};
    int64_t numDynamic = 0;
    const NnopbaseAttrAddr* attrs[] = {};
    int64_t numAttrs = 0;
    int64_t implMode = 0;
    int64_t deterMin = 0;
    const int64_t valueDepend[] = {};
    int64_t numValueDepend = 0;
    NnopbaseStaticTensorNumInfo tensorNumInfo;
    tensorNumInfo.numTensors = numTensors;
    tensorNumInfo.numDynamic = numDynamic;
    tensorNumInfo.numAttrs = numAttrs;
    tensorNumInfo.numValueDepend = numValueDepend;
    NnopbaseStaticRuntimeInfo staticRuntimeInfo;
    staticRuntimeInfo.opType = std::string(opType);
    staticRuntimeInfo.aicNum = coreNum.aicNum;
    staticRuntimeInfo.aivNum = coreNum.aivNum;
    staticRuntimeInfo.implMode = implMode;
    staticRuntimeInfo.deterMode = deterMin;

    const char* path = NnopbaseFindStaticKernel(tensors, attrs, valueDepend, &tensorNumInfo, &staticRuntimeInfo);
    ASSERT_NE(path, nullptr);
    aclDestroyTensor(tensor);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseUnitTest, NnopBaseFindStaticKernelSuccess)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    const char* opType = "Flash";
    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    const aclTensor* tensors[] = {tensor, tensor, tensor, tensor};
    int64_t numTensors = 4;
    const int64_t dynamicIndex[] = {};
    const int64_t dynamicCount[] = {};
    int64_t numDynamic = 0;
    const NnopbaseAttrAddr* attrs[] = {};
    int64_t numAttrs = 0;
    int64_t implMode = 0;
    int64_t deterMin = 0;
    const int64_t valueDepend[] = {};
    int64_t numValueDepend = 0;
    NnopbaseStaticTensorNumInfo tensorNumInfo;
    tensorNumInfo.numTensors = numTensors;
    tensorNumInfo.numDynamic = numDynamic;
    tensorNumInfo.numAttrs = numAttrs;
    tensorNumInfo.numValueDepend = numValueDepend;
    NnopbaseStaticRuntimeInfo staticRuntimeInfo;
    staticRuntimeInfo.opType = std::string(opType);
    staticRuntimeInfo.aicNum = coreNum.aicNum;
    staticRuntimeInfo.aivNum = coreNum.aivNum;
    staticRuntimeInfo.implMode = implMode;
    staticRuntimeInfo.deterMode = deterMin;

    const char* path = NnopbaseFindStaticKernel(tensors, attrs, valueDepend, &tensorNumInfo, &staticRuntimeInfo);
    ASSERT_NE(path, nullptr);
    aclDestroyTensor(tensor);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseUnitTest, NnopBaseFindStaticKernelWithValueDependSuccess)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    const char* opType = "Flash";
    std::vector<int64_t> shape = {5};
    int32_t tensorData[] = {27, 27, 27, 27, 39856}; // 16进制->1b
    void* ptr = (void*)tensorData;
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_INT32, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), ptr);
    const aclTensor* tensors[] = {tensor, tensor, tensor, tensor};
    int64_t numTensors = 4;
    const int64_t dynamicIndex[] = {};
    const int64_t dynamicCount[] = {};
    int64_t numDynamic = 0;
    const NnopbaseAttrAddr* attrs[] = {};
    int64_t numAttrs = 0;
    int64_t implMode = 0;
    int64_t deterMin = 0;
    const int64_t valueDepend[] = {0};
    int64_t numValueDepend = 1;
    NnopbaseStaticTensorNumInfo tensorNumInfo;
    tensorNumInfo.numTensors = numTensors;
    tensorNumInfo.numDynamic = numDynamic;
    tensorNumInfo.numAttrs = numAttrs;
    tensorNumInfo.numValueDepend = numValueDepend;
    NnopbaseStaticRuntimeInfo staticRuntimeInfo;
    staticRuntimeInfo.opType = std::string(opType);
    staticRuntimeInfo.aicNum = coreNum.aicNum;
    staticRuntimeInfo.aivNum = coreNum.aivNum;
    staticRuntimeInfo.implMode = implMode;
    staticRuntimeInfo.deterMode = deterMin;

    const char* path = NnopbaseFindStaticKernel(tensors, attrs, valueDepend, &tensorNumInfo, &staticRuntimeInfo);
    ASSERT_NE(path, nullptr);
    aclDestroyTensor(tensor);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseUnitTest, NnopBaseFindStaticKernelWithFloatAttrSuccess)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    const char* opType = "Flash";
    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    const aclTensor* tensors[] = {tensor, tensor, tensor, tensor};
    int64_t numTensors = 4;
    const int64_t dynamicIndex[] = {};
    const int64_t dynamicCount[] = {};
    int64_t numDynamic = 0;
    NnopbaseAttrAddr floatAddr;
    float a = 0.8;
    float* pa = &a;
    floatAddr.addr = (void*)pa;
    floatAddr.elementSize = sizeof(float);
    floatAddr.isVector = false;
    floatAddr.size = 4;
    const NnopbaseAttrAddr* attrs[1] = {&floatAddr};
    int64_t numAttrs = 1;
    int64_t implMode = 0;
    int64_t deterMin = 0;
    const int64_t valueDepend[] = {};
    int64_t numValueDepend = 0;
    NnopbaseStaticTensorNumInfo tensorNumInfo;
    tensorNumInfo.numTensors = numTensors;
    tensorNumInfo.numDynamic = numDynamic;
    tensorNumInfo.numAttrs = numAttrs;
    tensorNumInfo.numValueDepend = numValueDepend;
    NnopbaseStaticRuntimeInfo staticRuntimeInfo;
    staticRuntimeInfo.opType = std::string(opType);
    staticRuntimeInfo.aicNum = coreNum.aicNum;
    staticRuntimeInfo.aivNum = coreNum.aivNum;
    staticRuntimeInfo.implMode = implMode;
    staticRuntimeInfo.deterMode = deterMin;

    const char* path = NnopbaseFindStaticKernel(tensors, attrs, valueDepend, &tensorNumInfo, &staticRuntimeInfo);
    ASSERT_NE(path, nullptr);
    aclDestroyTensor(tensor);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseUnitTest, NnopBaseFindStaticKernelWithArrayAttrSuccess)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    const char* opType = "Flash";
    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    const aclTensor* tensors[] = {tensor, tensor, tensor, tensor};
    int64_t numTensors = 4;
    const int64_t dynamicIndex[] = {};
    const int64_t dynamicCount[] = {};
    int64_t numDynamic = 0;
    NnopbaseAttrAddr floatAddr;
    float attr1[2] = {0.8, 0.8};
    float* pa = attr1;
    floatAddr.addr = (void*)pa;
    floatAddr.elementSize = sizeof(float);
    floatAddr.isVector = true;
    floatAddr.size = 8;
    const NnopbaseAttrAddr* attrs[1] = {&floatAddr};
    int64_t numAttrs = 1;
    int64_t implMode = 0;
    int64_t deterMin = 0;
    const int64_t valueDepend[] = {};
    int64_t numValueDepend = 0;
    NnopbaseStaticTensorNumInfo tensorNumInfo;
    tensorNumInfo.numTensors = numTensors;
    tensorNumInfo.numDynamic = numDynamic;
    tensorNumInfo.numAttrs = numAttrs;
    tensorNumInfo.numValueDepend = numValueDepend;
    NnopbaseStaticRuntimeInfo staticRuntimeInfo;
    staticRuntimeInfo.opType = std::string(opType);
    staticRuntimeInfo.aicNum = coreNum.aicNum;
    staticRuntimeInfo.aivNum = coreNum.aivNum;
    staticRuntimeInfo.implMode = implMode;
    staticRuntimeInfo.deterMode = deterMin;

    const char* path = NnopbaseFindStaticKernel(tensors, attrs, valueDepend, &tensorNumInfo, &staticRuntimeInfo);
    ASSERT_NE(path, nullptr);
    aclDestroyTensor(tensor);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseUnitTest, NnopBaseFindStaticKernelWithArrayAttrNullptrSuccess)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    const char* opType = "Flash";
    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    const aclTensor* tensors[] = {tensor, tensor, tensor, tensor};
    int64_t numTensors = 4;
    const int64_t dynamicIndex[] = {};
    const int64_t dynamicCount[] = {};
    int64_t numDynamic = 0;
    NnopbaseAttrAddr floatAddr;
    float attr1[2] = {0.8, 0.8};
    float* pa = attr1;
    floatAddr.addr = (void*)pa;
    floatAddr.elementSize = sizeof(float);
    floatAddr.isVector = true;
    floatAddr.size = 8;
    const NnopbaseAttrAddr* attrs[1] = {nullptr};
    int64_t numAttrs = 1;
    int64_t implMode = 0;
    int64_t deterMin = 0;
    const int64_t valueDepend[] = {};
    int64_t numValueDepend = 0;
    NnopbaseStaticTensorNumInfo tensorNumInfo;
    tensorNumInfo.numTensors = numTensors;
    tensorNumInfo.numDynamic = numDynamic;
    tensorNumInfo.numAttrs = numAttrs;
    tensorNumInfo.numValueDepend = numValueDepend;
    NnopbaseStaticRuntimeInfo staticRuntimeInfo;
    staticRuntimeInfo.opType = std::string(opType);
    staticRuntimeInfo.aicNum = coreNum.aicNum;
    staticRuntimeInfo.aivNum = coreNum.aivNum;
    staticRuntimeInfo.implMode = implMode;
    staticRuntimeInfo.deterMode = deterMin;

    const char* path = NnopbaseFindStaticKernel(tensors, attrs, valueDepend, &tensorNumInfo, &staticRuntimeInfo);
    ASSERT_EQ(path, nullptr);
    aclDestroyTensor(tensor);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseUnitTest, NnopBaseFindStaticKernelOptionalNullSuccess)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    const char* opType = "Flash";
    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    const aclTensor* tensors[] = {tensor, NULL, tensor, tensor};
    int64_t numTensors = 4;
    const int64_t dynamicIndex[] = {};
    const int64_t dynamicCount[] = {};
    int64_t numDynamic = 0;
    const NnopbaseAttrAddr* attrs[] = {};
    int64_t numAttrs = 0;
    int64_t implMode = 0;
    int64_t deterMin = 0;
    const int64_t valueDepend[] = {};
    int64_t numValueDepend = 0;
    NnopbaseStaticTensorNumInfo tensorNumInfo;
    tensorNumInfo.numTensors = numTensors;
    tensorNumInfo.numDynamic = numDynamic;
    tensorNumInfo.numAttrs = numAttrs;
    tensorNumInfo.numValueDepend = numValueDepend;
    NnopbaseStaticRuntimeInfo staticRuntimeInfo;
    staticRuntimeInfo.opType = std::string(opType);
    staticRuntimeInfo.aicNum = coreNum.aicNum;
    staticRuntimeInfo.aivNum = coreNum.aivNum;
    staticRuntimeInfo.implMode = implMode;
    staticRuntimeInfo.deterMode = deterMin;

    const char* path = NnopbaseFindStaticKernel(tensors, attrs, valueDepend, &tensorNumInfo, &staticRuntimeInfo);
    ASSERT_NE(path, nullptr);
    aclDestroyTensor(tensor);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseUnitTest, NnopBaseFindStaticKernelFail)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    const char* opType = "Flash";
    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    const aclTensor* tensors[] = {tensor, tensor, tensor, tensor, tensor};
    int64_t numTensors = 5;
    const int64_t dynamicIndex[] = {};
    const int64_t dynamicCount[] = {};
    int64_t numDynamic = 0;
    const NnopbaseAttrAddr* attrs[] = {};
    int64_t numAttrs = 0;
    int64_t implMode = 0;
    int64_t deterMin = 0;
    const int64_t valueDepend[] = {};
    int64_t numValueDepend = 0;
    NnopbaseStaticTensorNumInfo tensorNumInfo;
    tensorNumInfo.numTensors = numTensors;
    tensorNumInfo.numDynamic = numDynamic;
    tensorNumInfo.numAttrs = numAttrs;
    tensorNumInfo.numValueDepend = numValueDepend;
    NnopbaseStaticRuntimeInfo staticRuntimeInfo;
    staticRuntimeInfo.opType = std::string(opType);
    staticRuntimeInfo.aicNum = coreNum.aicNum;
    staticRuntimeInfo.aivNum = coreNum.aivNum;
    staticRuntimeInfo.implMode = implMode;
    staticRuntimeInfo.deterMode = deterMin;

    const char* path = NnopbaseFindStaticKernel(tensors, attrs, valueDepend, &tensorNumInfo, &staticRuntimeInfo);
    ASSERT_EQ(path, nullptr);
    aclDestroyTensor(tensor);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseUnitTest, NnopBaseRunStaticKernelWithAttrSuccess)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char* opType = "Flash";
    char inputDesc[] = {1, 1, 1};
    char outputDesc[] = {1};
    char attrDesc[] = {1, 1, 1, 1};
    void* executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                         sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    ASSERT_NE(executor, nullptr);

    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    (void)NnopbaseAddInput(executor, tensor, 0);
    (void)NnopbaseAddInput(executor, tensor, 1);
    (void)NnopbaseAddInput(executor, tensor, 2);
    (void)NnopbaseAddOutput(executor, tensor, 0);

    bool boolValues[] = {true, false, true};
    auto* bias0 = aclCreateBoolArray(boolValues, sizeof(boolValues) / sizeof(boolValues[0]));
    float floatValues[] = {0.8, 0.8, 0.8};
    auto* bias1 = aclCreateFloatArray(floatValues, sizeof(floatValues) / sizeof(floatValues[0]));
    int64_t intValues[] = {3, 40, 5};
    auto* bias2 = aclCreateIntArray(intValues, sizeof(intValues) / sizeof(intValues[0]));
    char* bias3 = "abce";
    size_t bias3_size = strlen(bias3) + 1;
    ASSERT_EQ(NnopbaseAddBoolArrayAttr(executor, bias0, 0), OK);
    ASSERT_EQ(NnopbaseAddIntArrayAttr(executor, bias2, 1), OK);
    ASSERT_EQ(NnopbaseAddAttrWithDtype(executor, bias3, bias3_size, 2, kNnopbaseString), OK);
    ASSERT_EQ(NnopbaseAddFloatArrayAttr(executor, bias1, 3), OK);

    size_t workspaceLen = 0U;
    ASSERT_EQ(NnopbaseRunForWorkspace(executor, &workspaceLen), OK);
    void* stream = nullptr;
    void* workspace = nullptr;
    if (workspaceLen > 0U) {
        workspace = (void*)malloc(workspaceLen);
    }
    ASSERT_EQ(NnopbaseRunWithWorkspace(executor, stream, workspace, workspaceLen), OK);
    NnopbaseExecutorGcSpace(executorSpace);
    if (workspaceLen > 0U) {
        free(workspace);
    }
    aclDestroyTensor(tensor);
    aclDestroyBoolArray(bias0);
    aclDestroyFloatArray(bias1);
    aclDestroyIntArray(bias2);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseUnitTest, NnopBaseRunStaticKernelWithDynamicInputSuccess)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char* opType = "Flash";
    char inputDesc[] = {1, 2, 1};
    char outputDesc[] = {2};
    char attrDesc[] = {};
    void* executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                         sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    ASSERT_NE(executor, nullptr);

    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    std::vector<int64_t> shape2 = {1, 1, 1, 1, 1};
    aclTensor* tensor2 = aclCreateTensor(shape2.data(), shape2.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                         aclFormat::ACL_FORMAT_ND, shape2.data(), shape2.size(), nullptr);

    std::vector<const aclTensor*> tensor_list_a;
    tensor_list_a.push_back(tensor);
    tensor_list_a.push_back(tensor2);
    aclTensorList* aclTensorTestList = aclCreateTensorList(tensor_list_a.data(), tensor_list_a.size());
    (void)NnopbaseAddInput(executor, tensor, 0);
    (void)NnopbaseAddDynamicInput(executor, aclTensorTestList, 1);
    (void)NnopbaseAddInput(executor, tensor, 2);
    (void)NnopbaseAddDynamicOutput(executor, aclTensorTestList, 0);

    size_t workspaceLen = 0U;
    ASSERT_EQ(NnopbaseRunForWorkspace(executor, &workspaceLen), OK);
    void* stream = nullptr;
    void* workspace = nullptr;
    if (workspaceLen > 0U) {
        workspace = (void*)malloc(workspaceLen);
    }
    ASSERT_EQ(NnopbaseRunWithWorkspace(executor, stream, workspace, workspaceLen), OK);
    if (workspaceLen > 0U) {
        free(workspace);
    }

    NnopbaseExecutorGcSpace(executorSpace);
    aclDestroyTensorList((const aclTensorList*)aclTensorTestList);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseUnitTest, NnopBaseRunStaticKernelWithOptionalInputSuccess)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char* opType = "Flash";
    char inputDesc[] = {1, 2, 1, 0};
    char outputDesc[] = {2};
    char attrDesc[] = {};
    void* executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                         sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    ASSERT_NE(executor, nullptr);

    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    std::vector<int64_t> shape2 = {1, 1, 1, 1, 1};
    aclTensor* tensor2 = aclCreateTensor(shape2.data(), shape2.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                         aclFormat::ACL_FORMAT_ND, shape2.data(), shape2.size(), nullptr);

    std::vector<const aclTensor*> tensor_list_a;
    tensor_list_a.push_back(tensor);
    tensor_list_a.push_back(tensor2);
    aclTensorList* aclTensorTestList = aclCreateTensorList(tensor_list_a.data(), tensor_list_a.size());
    (void)NnopbaseAddInput(executor, tensor, 0);
    (void)NnopbaseAddDynamicInput(executor, aclTensorTestList, 1);
    (void)NnopbaseAddInput(executor, tensor, 2);
    (void)NnopbaseAddDynamicOutput(executor, aclTensorTestList, 0);

    size_t workspaceLen = 0U;
    ASSERT_EQ(NnopbaseRunForWorkspace(executor, &workspaceLen), OK);
    void* stream = nullptr;
    void* workspace = nullptr;
    if (workspaceLen > 0U) {
        workspace = (void*)malloc(workspaceLen);
    }
    ASSERT_EQ(NnopbaseRunWithWorkspace(executor, stream, workspace, workspaceLen), OK);
    if (workspaceLen > 0U) {
        free(workspace);
    }

    NnopbaseExecutorGcSpace(executorSpace);
    aclDestroyTensorList((const aclTensorList*)aclTensorTestList);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseUnitTest, NnopBaseRunStaticKernelWithValueDependSuccess)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char* opType = "Flash";
    char inputDesc[] = {1, 1, 1};
    char outputDesc[] = {1};
    char attrDesc[] = {1, 1, 1, 1};
    void* executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                         sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    ASSERT_NE(executor, nullptr);
    float tensorData[] = {0.8, 0.8, 0.8, 0.8, 0.8}; // 16进制->1b
    void* ptr = (void*)tensorData;
    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), ptr);
    (void)NnopbaseAddInput(executor, tensor, 0);
    (void)NnopbaseAddInput(executor, tensor, 1);
    (void)NnopbaseAddInput(executor, tensor, 2);
    (void)NnopbaseAddOutput(executor, tensor, 0);

    bool boolValues[] = {true, false, true};
    auto* bias0 = aclCreateBoolArray(boolValues, sizeof(boolValues) / sizeof(boolValues[0]));
    float floatValues[] = {0.8, 0.8, 0.8};
    auto* bias1 = aclCreateFloatArray(floatValues, sizeof(floatValues) / sizeof(floatValues[0]));
    int64_t intValues[] = {3, 40, 5};
    auto* bias2 = aclCreateIntArray(intValues, sizeof(intValues) / sizeof(intValues[0]));
    char* bias3 = "abce";
    size_t bias3_size = strlen(bias3) + 1;
    ASSERT_EQ(NnopbaseAddBoolArrayAttr(executor, bias0, 0), OK);
    ASSERT_EQ(NnopbaseAddIntArrayAttr(executor, bias2, 1), OK);
    ASSERT_EQ(NnopbaseAddAttrWithDtype(executor, bias3, bias3_size, 2, kNnopbaseString), OK);
    ASSERT_EQ(NnopbaseAddFloatArrayAttr(executor, bias1, 3), OK);
    // set index0 is valueDepend input
    ((NnopbaseExecutor*)executor)->ownArgs.inputs.extTensors[0].valueDepend = true;
    size_t workspaceLen = 0U;
    ASSERT_EQ(NnopbaseRunForWorkspace(executor, &workspaceLen), OK);
    void* stream = nullptr;
    void* workspace = nullptr;
    if (workspaceLen > 0U) {
        workspace = (void*)malloc(workspaceLen);
    }
    ASSERT_EQ(NnopbaseRunWithWorkspace(executor, stream, workspace, workspaceLen), OK);
    NnopbaseExecutorGcSpace(executorSpace);
    if (workspaceLen > 0U) {
        free(workspace);
    }
    aclDestroyTensor(tensor);
    aclDestroyBoolArray(bias0);
    aclDestroyFloatArray(bias1);
    aclDestroyIntArray(bias2);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseUnitTest, NnopBaseCheckWorkspaceSuccess)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char* opType = "custom_op1";
    char inputDesc[] = {1, 1};
    char outputDesc[] = {1, 1};
    char attrDesc[] = {};
    void* executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                         sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    ASSERT_NE(executor, nullptr);

    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    (void)NnopbaseAddInput(executor, tensor, 0);
    (void)NnopbaseAddInput(executor, tensor, 1);
    (void)NnopbaseAddOutput(executor, tensor, 0);
    (void)NnopbaseAddOutput(executor, tensor, 1);

    size_t workspaceLen = 300U;
    ASSERT_EQ(NnopbaseRunForWorkspace(executor, &workspaceLen), OK);
    ASSERT_EQ(workspaceLen, 200U);
    NnopbaseExecutorGcSpace(executorSpace);
    aclDestroyTensor(tensor);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseUnitTest, NnopBaseFindStaticKernelWithStringAttrSuccess)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    const char* opType = "Flash";
    std::vector<int64_t> shape = {5};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    const aclTensor* tensors[] = {tensor, tensor, tensor, tensor};
    int64_t numTensors = 4;
    const int64_t dynamicIndex[] = {};
    const int64_t dynamicCount[] = {};
    int64_t numDynamic = 0;

    NnopbaseAttrAddr stringAddr;
    const char* attr1 = "FRACTAL_Z";
    const char* pa = attr1;
    stringAddr.addr = (void*)pa;
    stringAddr.elementSize = sizeof(char);
    stringAddr.isVector = false;
    stringAddr.size = 9;
    const NnopbaseAttrAddr* attrs[1] = {&stringAddr};

    int64_t numAttrs = 1;
    int64_t implMode = 0;
    int64_t deterMin = 0;
    const int64_t valueDepend[] = {};
    int64_t numValueDepend = 0;
    NnopbaseStaticTensorNumInfo tensorNumInfo;
    tensorNumInfo.numTensors = numTensors;
    tensorNumInfo.numDynamic = numDynamic;
    tensorNumInfo.numAttrs = numAttrs;
    tensorNumInfo.numValueDepend = numValueDepend;
    NnopbaseStaticRuntimeInfo staticRuntimeInfo;
    staticRuntimeInfo.opType = std::string(opType);
    staticRuntimeInfo.aicNum = coreNum.aicNum;
    staticRuntimeInfo.aivNum = coreNum.aivNum;
    staticRuntimeInfo.implMode = implMode;
    staticRuntimeInfo.deterMode = deterMin;

    const char* path = NnopbaseFindStaticKernel(tensors, attrs, valueDepend, &tensorNumInfo, &staticRuntimeInfo);
    ASSERT_NE(path, nullptr);
    aclDestroyTensor(tensor);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseUnitTest, NnopBaseRunStaticKernelWithOptionalOutputNullSuccess)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char* opType = "Flash";
    char inputDesc[] = {1, 2, 1};
    char outputDesc[] = {0, 2};
    char attrDesc[] = {};
    void* executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                         sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    ASSERT_NE(executor, nullptr);

    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    std::vector<int64_t> shape2 = {1, 1, 1, 1, 1};
    aclTensor* tensor2 = aclCreateTensor(shape2.data(), shape2.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                         aclFormat::ACL_FORMAT_ND, shape2.data(), shape2.size(), nullptr);

    std::vector<const aclTensor*> tensor_list_a;
    tensor_list_a.push_back(tensor);
    tensor_list_a.push_back(tensor2);
    aclTensorList* aclTensorTestList = aclCreateTensorList(tensor_list_a.data(), tensor_list_a.size());
    (void)NnopbaseAddInput(executor, tensor, 0);
    (void)NnopbaseAddDynamicInput(executor, aclTensorTestList, 1);
    (void)NnopbaseAddInput(executor, tensor, 2);
    (void)NnopbaseAddDynamicOutput(executor, aclTensorTestList, 1);

    size_t workspaceLen = 0U;
    ASSERT_EQ(NnopbaseRunForWorkspace(executor, &workspaceLen), OK);
    void* stream = nullptr;
    void* workspace = nullptr;
    if (workspaceLen > 0U) {
        workspace = (void*)malloc(workspaceLen);
    }
    ASSERT_EQ(NnopbaseRunWithWorkspace(executor, stream, workspace, workspaceLen), OK);
    if (workspaceLen > 0U) {
        free(workspace);
    }

    NnopbaseExecutorGcSpace(executorSpace);
    aclDestroyTensorList((const aclTensorList*)aclTensorTestList);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseUnitTest, NnopBaseAddArrayInputSuccess)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char* opType = "AddTik2";
    char inputDesc[] = {1, 1, 0, 0, 0};
    char outputDesc[] = {1};
    char attrDesc[] = {};
    void* executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                         sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    ASSERT_NE(executor, nullptr);

    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    std::vector<float> vec(2048 * 4, 2.0);
    auto* floatArray = aclCreateFloatArray(vec.data(), vec.size());
    int64_t intValues[] = {3, 4, 5};
    auto* intArray = aclCreateIntArray(intValues, sizeof(intValues) / sizeof(intValues[0]));
    bool boolValues[] = {true, false, true};
    auto* boolArray = aclCreateBoolArray(boolValues, sizeof(boolValues) / sizeof(boolValues[0]));
    ASSERT_EQ(NnopbaseAddInput(executor, tensor, 0), OK);
    ASSERT_EQ(NnopbaseAddInput(executor, tensor, 1), OK);
    ASSERT_EQ(NnopbaseAddFloatArrayInput(executor, floatArray, 2), OK);
    ASSERT_EQ(NnopbaseAddIntArrayInput(executor, intArray, 3), OK);
    ASSERT_EQ(NnopbaseAddBoolArrayInput(executor, boolArray, 4), OK);
    ASSERT_EQ(NnopbaseAddOutput(executor, tensor, 0), OK);

    auto* tensors = &(((NnopbaseExecutor*)executor)->ownArgs.inputs);
    auto* floatTensor = &(tensors->extTensors[2]).rt2Tensor;
    ASSERT_NE(floatTensor, nullptr);
    auto floatdata = floatTensor->GetData<float>();
    ASSERT_EQ(floatdata[0], 2.0);
    auto* intTensor = &(tensors->extTensors[3]).rt2Tensor;
    ASSERT_NE(intTensor, nullptr);
    auto intdata = intTensor->GetData<int64_t>();
    ASSERT_EQ(intdata[0], 3);
    auto* boolTensor = &(tensors->extTensors[4]).rt2Tensor;
    ASSERT_NE(boolTensor, nullptr);
    auto booldata = boolTensor->GetData<bool>();
    ASSERT_EQ(booldata[0], true);

    size_t workspaceLen = 0U;
    ASSERT_EQ(NnopbaseRunForWorkspace(executor, &workspaceLen), OK);

    void* stream = nullptr;
    void* workspace = nullptr;
    if (workspaceLen > 0U) {
        workspace = (void*)malloc(workspaceLen);
    }
    ASSERT_EQ(NnopbaseRunWithWorkspace(executor, stream, workspace, workspaceLen), OK);
    if (workspaceLen > 0U) {
        free(workspace);
    }

    NnopbaseExecutorGcSpace(executorSpace);
    aclDestroyTensor(tensor);
    aclDestroyFloatArray(floatArray);
    aclDestroyIntArray(intArray);
    aclDestroyBoolArray(boolArray);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseUnitTest, NnopBaseRunWithHostAndDynamicInput)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char* opType = "AddTik2";
    char inputDesc[] = {1, 2, 0};
    char outputDesc[] = {1};
    char attrDesc[] = {};
    void* executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                         sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    ASSERT_NE(executor, nullptr);

    std::vector<int64_t> shape = {1, 1, 1, 1, 1, 1, 1, 1};
    aclTensor* tensor1 = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                         aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    std::vector<aclTensor*> tensor_list;
    for (size_t i = 0U; i < 80; i++) {
        aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                            aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
        tensor_list.push_back(tensor);
    }
    aclTensorList* aclTensorTestList = aclCreateTensorList(tensor_list.data(), tensor_list.size());
    std::vector<float> vec(2048 * 4, 2.0);
    auto* floatArray = aclCreateFloatArray(vec.data(), vec.size());

    ASSERT_EQ(NnopbaseAddInput(executor, tensor1, 0), OK);
    ASSERT_EQ(NnopbaseAddDynamicInput(executor, aclTensorTestList, 1), OK);
    ASSERT_EQ(NnopbaseAddFloatArrayInput(executor, floatArray, 2), OK);
    ASSERT_EQ(NnopbaseAddOutput(executor, tensor1, 0), OK);

    size_t workspaceLen = 0U;
    ASSERT_EQ(NnopbaseRunForWorkspace(executor, &workspaceLen), OK);

    void* stream = nullptr;
    void* workspace = nullptr;
    if (workspaceLen > 0U) {
        workspace = (void*)malloc(workspaceLen);
    }
    ASSERT_EQ(NnopbaseRunWithWorkspace(executor, stream, workspace, workspaceLen), OK);
    if (workspaceLen > 0U) {
        free(workspace);
    }

    NnopbaseExecutorGcSpace(executorSpace);
    aclDestroyTensor(tensor1);
    aclDestroyTensorList((const aclTensorList*)aclTensorTestList);
    aclDestroyFloatArray(floatArray);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseUnitTest, NnopBaseAllOutputIsEmptyTensorSuccess)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char* opType = "bninference_d_kernel";
    char inputDesc[] = {1, 1, 1};
    char outputDesc[] = {1, 0};
    char attrDesc[] = {};
    void* executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                         sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    ASSERT_NE(executor, nullptr);

    std::vector<int64_t> shape1 = {1, 1, 1, 1, 1};
    aclTensor* tensor1 = aclCreateTensor(shape1.data(), shape1.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                         aclFormat::ACL_FORMAT_ND, shape1.data(), shape1.size(), nullptr);
    std::vector<int64_t> shape2 = {0};
    aclTensor* tensor2 = aclCreateTensor(shape2.data(), shape2.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                         aclFormat::ACL_FORMAT_ND, shape2.data(), shape2.size(), nullptr);
    ASSERT_EQ(tensor2->IsEmpty(), true);
    ASSERT_EQ(NnopbaseAddInput(executor, tensor1, 0), OK);
    ASSERT_EQ(NnopbaseAddInput(executor, tensor1, 1), OK);
    ASSERT_EQ(NnopbaseAddInput(executor, tensor1, 2), OK);
    ASSERT_EQ(NnopbaseAddOutput(executor, tensor2, 0), OK);
    ASSERT_EQ(NnopbaseAddOutput(executor, tensor2, 1), OK);

    uint64_t workspaceLen = 0U;
    ASSERT_EQ(NnopbaseRunForWorkspace(executor, &workspaceLen), OK);
    ASSERT_EQ(workspaceLen, 0U);
    void* stream = nullptr;
    void* workspace = nullptr;
    ASSERT_EQ(NnopbaseRunWithWorkspace(executor, stream, workspace, workspaceLen), OK);

    NnopbaseExecutorGcSpace(executorSpace);
    aclDestroyTensor(tensor1);
    aclDestroyTensor(tensor2);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseUnitTest, NnopBaseAllOutputIsEmptyTensorWithDynamicSuccess)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char* opType = "bninference_d_kernel";
    char inputDesc[] = {1, 1, 1};
    char outputDesc[] = {0, 2};
    char attrDesc[] = {};
    void* executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                         sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    ASSERT_NE(executor, nullptr);

    std::vector<int64_t> shape1 = {1, 1, 1, 1, 1};
    aclTensor* tensor1 = aclCreateTensor(shape1.data(), shape1.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                         aclFormat::ACL_FORMAT_ND, shape1.data(), shape1.size(), nullptr);
    std::vector<int64_t> shape2 = {0};
    aclTensor* tensor2 = aclCreateTensor(shape2.data(), shape2.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                         aclFormat::ACL_FORMAT_ND, shape2.data(), shape2.size(), nullptr);
    std::vector<const aclTensor*> tensorList;
    tensorList.push_back(tensor2);
    aclTensorList* aclTensorTestList = aclCreateTensorList(tensorList.data(), tensorList.size());
    ASSERT_EQ(tensor2->IsEmpty(), true);
    ASSERT_EQ(NnopbaseAddInput(executor, tensor1, 0), OK);
    ASSERT_EQ(NnopbaseAddInput(executor, tensor1, 1), OK);
    ASSERT_EQ(NnopbaseAddInput(executor, tensor1, 2), OK);
    ASSERT_EQ(NnopbaseAddOutput(executor, tensor2, 0), OK);
    ASSERT_EQ(NnopbaseAddDynamicOutput(executor, aclTensorTestList, 1), OK);

    uint64_t workspaceLen = 0U;
    ASSERT_EQ(NnopbaseRunForWorkspace(executor, &workspaceLen), OK);
    ASSERT_EQ(workspaceLen, 0U);
    void* stream = nullptr;
    void* workspace = nullptr;
    ASSERT_EQ(NnopbaseRunWithWorkspace(executor, stream, workspace, workspaceLen), OK);

    NnopbaseExecutorGcSpace(executorSpace);
    aclDestroyTensor(tensor1);
    aclDestroyTensorList((const aclTensorList*)aclTensorTestList);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseUnitTest, NnopBaseAddOptionalArrayAttrSuccess)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char* opType = "bninference_d_kernel";
    char inputDesc[] = {};
    char outputDesc[] = {};
    char attrDesc[] = {0, 0, 0};
    void* executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                         sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    ASSERT_NE(executor, nullptr);

    static bool bias2Def[] = {true, false};
    static size_t bias2Len = 2;
    // 无dtype接口也可以正常跑
    ASSERT_EQ(NnopbaseAddArrayAttr(executor, static_cast<void*>(bias2Def), bias2Len, sizeof(bool), 0), OK);

    static float bias3Def[] = {0.1, 0.2};
    static size_t bias3Len = 2;
    ASSERT_EQ(NnopbaseAddArrayAttrWithDtype(executor, static_cast<void*>(bias3Def), bias3Len, sizeof(float), 1,
                                            kNnopbaseFloat),
              OK);

    static int64_t bias4Def[] = {1, 2};
    static size_t bias4Len = 2;
    ASSERT_EQ(NnopbaseAddArrayAttrWithDtype(executor, static_cast<void*>(bias4Def), bias4Len, sizeof(int64_t), 2,
                                            kNnopbaseInt),
              OK);

    NnopbaseExecutorGcSpace(executorSpace);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseUnitTest, NnopBaseRunSuccessWithScalarInput)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char* opType = "bninference_d_kernel";
    char inputDesc[] = {1, 1, 1};
    char outputDesc[] = {1};
    char attrDesc[] = {};
    void* executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                         sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    ASSERT_NE(executor, nullptr);

    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);

    int32_t scalar_value = 5;
    auto* scalar = aclCreateScalar(&scalar_value, aclDataType::ACL_FLOAT);
    (void)NnopbaseAddInput(executor, tensor, 0);
    (void)NnopbaseAddScalarInput(executor, scalar, 1, -1, ge::DT_UNDEFINED);
    (void)NnopbaseAddScalarInput(executor, scalar, 2, -1, ge::DT_UNDEFINED);
    (void)NnopbaseAddOutput(executor, tensor, 0);

    uint64_t workspaceLen = 0U;
    ASSERT_EQ(NnopbaseRunForWorkspace(executor, &workspaceLen), OK);
#if 0 // comment out for ge decoupling
    ASSERT_EQ(workspaceLen, 0U);
#endif
    void* stream = nullptr;
    void* workspace = nullptr;
    if (workspaceLen > 0U) {
        workspace = (void*)malloc(workspaceLen);
    }
    ASSERT_EQ(NnopbaseRunWithWorkspace(executor, stream, workspace, workspaceLen), OK);

    NnopbaseExecutorGcSpace(executorSpace);
    if (workspaceLen > 0U) {
        free(workspace);
    }
    aclDestroyTensor(tensor);
    aclDestroyScalar(scalar);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseUnitTest, NnopBaseRunSuccessWithDoubleDtypeScalarInput)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char* opType = "bninference_d_kernel";
    char inputDesc[] = {1, 1, 1};
    char outputDesc[] = {1};
    char attrDesc[] = {};
    void* executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                         sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    ASSERT_NE(executor, nullptr);

    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);

    double scalar_value = 5;
    auto* scalar = aclCreateScalar(&scalar_value, aclDataType::ACL_DOUBLE);
    (void)NnopbaseAddInput(executor, tensor, 0);
    (void)NnopbaseAddScalarInput(executor, scalar, 1, -1, ge::DT_UNDEFINED);
    (void)NnopbaseAddScalarInput(executor, scalar, 2, -1, ge::DT_UNDEFINED);
    (void)NnopbaseAddOutput(executor, tensor, 0);

    uint64_t workspaceLen = 0U;
    ASSERT_EQ(NnopbaseRunForWorkspace(executor, &workspaceLen), OK);
#if 0 // comment out for ge decoupling
    ASSERT_EQ(workspaceLen, 0U);
#endif
    void* stream = nullptr;
    void* workspace = nullptr;
    if (workspaceLen > 0U) {
        workspace = (void*)malloc(workspaceLen);
    }
    ASSERT_EQ(NnopbaseRunWithWorkspace(executor, stream, workspace, workspaceLen), OK);

    NnopbaseExecutorGcSpace(executorSpace);
    if (workspaceLen > 0U) {
        free(workspace);
    }
    aclDestroyTensor(tensor);
    aclDestroyScalar(scalar);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseUnitTest, NnopBaseRunSuccessWithScalarListInput)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char* opType = "bninference_d_kernel";
    char inputDesc[] = {1, 1, 1};
    char outputDesc[] = {1};
    char attrDesc[] = {};
    void* executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                         sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    ASSERT_NE(executor, nullptr);

    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);

    float scalar_value = 5;
    auto* scalar = aclCreateScalar(&scalar_value, aclDataType::ACL_FLOAT);
    auto scalarList = aclCreateScalarList(&scalar, 1);
    (void)NnopbaseAddInput(executor, tensor, 0);
    (void)NnopbaseAddScalarListInput(executor, scalarList, 1, -1, ge::DT_UNDEFINED);
    (void)NnopbaseAddScalarListInput(executor, scalarList, 2, -1, ge::DT_UNDEFINED);
    (void)NnopbaseAddOutput(executor, tensor, 0);

    uint64_t workspaceLen = 0U;
    ASSERT_EQ(NnopbaseRunForWorkspace(executor, &workspaceLen), OK);
#if 0 // comment out for ge decoupling
    ASSERT_EQ(workspaceLen, 0U);
#endif
    void* stream = nullptr;
    void* workspace = nullptr;
    if (workspaceLen > 0U) {
        workspace = (void*)malloc(workspaceLen);
    }
    ASSERT_EQ(NnopbaseRunWithWorkspace(executor, stream, workspace, workspaceLen), OK);

    NnopbaseExecutorGcSpace(executorSpace);
    if (workspaceLen > 0U) {
        free(workspace);
    }
    aclDestroyTensor(tensor);
    aclDestroyScalarList(scalarList);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseUnitTest, NnopBaseRunSuccessWithDoubleDtypeScalarListInput)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char* opType = "bninference_d_kernel";
    char inputDesc[] = {1, 1, 1};
    char outputDesc[] = {1};
    char attrDesc[] = {};
    void* executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                         sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    ASSERT_NE(executor, nullptr);

    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);

    double scalar_value = 5;
    auto* scalar = aclCreateScalar(&scalar_value, aclDataType::ACL_DOUBLE);
    auto scalarList = aclCreateScalarList(&scalar, 1);
    (void)NnopbaseAddInput(executor, tensor, 0);
    (void)NnopbaseAddScalarListInput(executor, scalarList, 1, -1, ge::DT_UNDEFINED);
    (void)NnopbaseAddScalarListInput(executor, scalarList, 2, -1, ge::DT_UNDEFINED);
    (void)NnopbaseAddOutput(executor, tensor, 0);

    uint64_t workspaceLen = 0U;
    NnopbaseRunForWorkspace(executor, &workspaceLen);

    NnopbaseExecutorGcSpace(executorSpace);
    aclDestroyTensor(tensor);
    aclDestroyScalarList(scalarList);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseUnitTest, NnopBaseRunSuccessWithDynamicInputOver50)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char* opType = "bninference_d_kernel";
    char inputDesc[] = {2, 2, 2};
    char outputDesc[] = {1};
    char attrDesc[] = {};
    void* executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                         sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    ASSERT_NE(executor, nullptr);

    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    std::vector<aclTensor*> tensor_list_a;
    for (size_t i = 0U; i < 30; i++) {
        aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                            aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
        tensor_list_a.push_back(tensor);
    }
    aclTensorList* aclTensorTestList = aclCreateTensorList(tensor_list_a.data(), tensor_list_a.size());
    aclTensor* tensor2 = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                         aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    (void)NnopbaseAddDynamicInput(executor, aclTensorTestList, 0);
    (void)NnopbaseAddDynamicInput(executor, aclTensorTestList, 1);
    (void)NnopbaseAddDynamicInput(executor, aclTensorTestList, 2);
    (void)NnopbaseAddOutput(executor, tensor2, 0);

    uint64_t workspaceLen = 0U;
    ASSERT_EQ(NnopbaseRunForWorkspace(executor, &workspaceLen), OK);

    void* stream = nullptr;
    void* workspace = nullptr;
    if (workspaceLen > 0U) {
        workspace = (void*)malloc(workspaceLen);
    }
    ASSERT_EQ(NnopbaseRunWithWorkspace(executor, stream, workspace, workspaceLen), OK);

    NnopbaseExecutorGcSpace(executorSpace);
    if (workspaceLen > 0U) {
        free(workspace);
    }

    aclDestroyTensor(tensor2);
    aclDestroyTensorList((const aclTensorList*)aclTensorTestList);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseUnitTest, NnopBaseRunSuccessWithDynamicInputMaxLen)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char* opType = "bninference_d_kernel";
    char inputDesc[] = {2, 2, 2};
    char outputDesc[] = {1};
    char attrDesc[] = {};
    void* executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                         sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    ASSERT_NE(executor, nullptr);

    std::vector<int64_t> shape = {1, 1, 1, 1, 1, 1, 1, 1};
    std::vector<aclTensor*> tensor_list_a;
    for (size_t i = 0U; i < 16; i++) {
        aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                            aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
        tensor_list_a.push_back(tensor);
    }
    aclTensorList* aclTensorTestList = aclCreateTensorList(tensor_list_a.data(), tensor_list_a.size());
    aclTensor* tensor2 = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                         aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    (void)NnopbaseAddDynamicInput(executor, aclTensorTestList, 0);
    (void)NnopbaseAddDynamicInput(executor, aclTensorTestList, 1);
    (void)NnopbaseAddDynamicInput(executor, aclTensorTestList, 2);
    (void)NnopbaseAddOutput(executor, tensor2, 0);

    uint64_t workspaceLen = 0U;
    ASSERT_EQ(NnopbaseRunForWorkspace(executor, &workspaceLen), OK);
    ASSERT_EQ(((NnopbaseExecutor*)executor)->args->argsBuf.size() > 1024 * 10, true);
    void* stream = nullptr;
    void* workspace = nullptr;
    if (workspaceLen > 0U) {
        workspace = (void*)malloc(workspaceLen);
    }
    ASSERT_EQ(NnopbaseRunWithWorkspace(executor, stream, workspace, workspaceLen), OK);

    NnopbaseExecutorGcSpace(executorSpace);
    if (workspaceLen > 0U) {
        free(workspace);
    }

    aclDestroyTensor(tensor2);
    aclDestroyTensorList((const aclTensorList*)aclTensorTestList);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseUnitTest, NnopbaseSupportScalarWithDtype)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char* opType = "bninference_d_kernel";
    char inputDesc[] = {1, 1, 0, 1};
    char outputDesc[] = {1};
    char attrDesc[] = {};
    void* executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                         sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    ASSERT_NE(executor, nullptr);

    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);

    int32_t scalar_value = 5;
    auto* scalar = aclCreateScalar(&scalar_value, aclDataType::ACL_INT32);
    (void)NnopbaseAddInput(executor, tensor, 0);
    (void)NnopbaseAddInput(executor, tensor, 1);
    (void)NnopbaseAddScalarInput(executor, scalar, 2, -1, ge::DT_FLOAT);
    (void)NnopbaseAddInput(executor, tensor, 3);
    (void)NnopbaseAddOutput(executor, tensor, 0);
    auto inputs = &(((NnopbaseExecutor*)executor)->ownArgs.inputs);
    auto tensors = &inputs->extTensors[2];
    gert::TensorV2* rt2Tensor = &tensors->rt2Tensor;
    ASSERT_EQ(rt2Tensor->GetDataType(), ge::DT_FLOAT);

    uint64_t workspaceLen = 0U;
    ASSERT_EQ(NnopbaseRunForWorkspace(executor, &workspaceLen), OK);
#if 0 // comment out for ge decoupling
    ASSERT_EQ(workspaceLen, 0U);
#endif
    void* stream = nullptr;
    void* workspace = nullptr;
    if (workspaceLen > 0U) {
        workspace = (void*)malloc(workspaceLen);
    }
    ASSERT_EQ(NnopbaseRunWithWorkspace(executor, stream, workspace, workspaceLen), OK);

    NnopbaseExecutorGcSpace(executorSpace);
    if (workspaceLen > 0U) {
        free(workspace);
    }
    aclDestroyTensor(tensor);
    aclDestroyScalar(scalar);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseUnitTest, NnopbaseSupportScalarListWithDtype)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char* opType = "bninference_d_kernel";
    char inputDesc[] = {1, 1, 0, 1};
    char outputDesc[] = {1};
    char attrDesc[] = {};
    void* executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                         sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    ASSERT_NE(executor, nullptr);

    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);

    int32_t scalar_value = 5;
    auto* scalar = aclCreateScalar(&scalar_value, aclDataType::ACL_INT32);
    auto scalarList = aclCreateScalarList(&scalar, 1);
    (void)NnopbaseAddInput(executor, tensor, 0);
    (void)NnopbaseAddInput(executor, tensor, 1);
    (void)NnopbaseAddScalarListInput(executor, scalarList, 2, -1, ge::DT_FLOAT);
    (void)NnopbaseAddInput(executor, tensor, 3);
    (void)NnopbaseAddOutput(executor, tensor, 0);
    auto inputs = &(((NnopbaseExecutor*)executor)->ownArgs.inputs);
    auto tensors = &inputs->extTensors[2];
    gert::TensorV2* rt2Tensor = &tensors->rt2Tensor;
    ASSERT_EQ(rt2Tensor->GetDataType(), ge::DT_FLOAT);

    uint64_t workspaceLen = 0U;
    ASSERT_EQ(NnopbaseRunForWorkspace(executor, &workspaceLen), OK);
#if 0 // comment out for ge decoupling
    ASSERT_EQ(workspaceLen, 0U);
#endif
    void* stream = nullptr;
    void* workspace = nullptr;
    if (workspaceLen > 0U) {
        workspace = (void*)malloc(workspaceLen);
    }
    ASSERT_EQ(NnopbaseRunWithWorkspace(executor, stream, workspace, workspaceLen), OK);

    NnopbaseExecutorGcSpace(executorSpace);
    if (workspaceLen > 0U) {
        free(workspace);
    }
    aclDestroyTensor(tensor);
    aclDestroyScalarList(scalarList);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseUnitTest, NnopbaseSupportScalarWithName)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char* opType = "bninference_d_kernel";
    char inputDesc[] = {1, 1, 0, 0, 1};
    char outputDesc[] = {1};
    char attrDesc[] = {};
    void* executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                         sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    ASSERT_NE(executor, nullptr);

    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);

    int32_t scalar_value = 5;
    auto* scalar = aclCreateScalar(&scalar_value, aclDataType::ACL_INT32);
    (void)NnopbaseAddInput(executor, tensor, 0);
    (void)NnopbaseAddInput(executor, tensor, 1);
    (void)NnopbaseAddScalarInput(executor, scalar, 2, 1, ge::DT_UNDEFINED);
    (void)NnopbaseAddScalarInput(executor, scalar, 3, 4, ge::DT_UNDEFINED);
    (void)NnopbaseAddInput(executor, tensor, 4);
    (void)NnopbaseAddOutput(executor, tensor, 0);
    auto inputs = &(((NnopbaseExecutor*)executor)->ownArgs.inputs);
    auto tensors1 = &inputs->extTensors[2];
    gert::TensorV2* rt2Tensor1 = &tensors1->rt2Tensor;
    ASSERT_EQ(rt2Tensor1->GetDataType(), ge::DT_FLOAT);
    auto tensors2 = &inputs->extTensors[2];
    gert::TensorV2* rt2Tensor2 = &tensors2->rt2Tensor;
    ASSERT_EQ(rt2Tensor2->GetDataType(), ge::DT_FLOAT);

    uint64_t workspaceLen = 0U;
    ASSERT_EQ(NnopbaseRunForWorkspace(executor, &workspaceLen), OK);
#if 0 // comment out for ge decoupling
    ASSERT_EQ(workspaceLen, 0U);
#endif
    void* stream = nullptr;
    void* workspace = nullptr;
    if (workspaceLen > 0U) {
        workspace = (void*)malloc(workspaceLen);
    }
    ASSERT_EQ(NnopbaseRunWithWorkspace(executor, stream, workspace, workspaceLen), OK);

    NnopbaseExecutorGcSpace(executorSpace);
    if (workspaceLen > 0U) {
        free(workspace);
    }
    aclDestroyTensor(tensor);
    aclDestroyScalar(scalar);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseUnitTest, NnopbaseSupportScalarListWithName)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char* opType = "bninference_d_kernel";
    char inputDesc[] = {1, 0, 0, 0, 1, 0, 2};
    char outputDesc[] = {1};
    char attrDesc[] = {};
    void* executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                         sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    ASSERT_NE(executor, nullptr);

    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    std::vector<const aclTensor*> tensor_list_a;
    tensor_list_a.push_back(tensor);
    aclTensorList* aclTensorTestList = aclCreateTensorList(tensor_list_a.data(), tensor_list_a.size());

    int32_t scalar_value = 5;
    auto* scalar = aclCreateScalar(&scalar_value, aclDataType::ACL_INT32);
    auto scalarList = aclCreateScalarList(&scalar, 1);
    (void)NnopbaseAddInput(executor, tensor, 0);
    (void)NnopbaseAddScalarListInput(executor, nullptr, 1, -1, ge::DT_UNDEFINED);
    (void)NnopbaseAddScalarListInput(executor, scalarList, 2, 0, ge::DT_UNDEFINED);
    (void)NnopbaseAddScalarListInput(executor, scalarList, 3, 4, ge::DT_UNDEFINED);
    (void)NnopbaseAddInput(executor, tensor, 4);
    (void)NnopbaseAddScalarListInput(executor, scalarList, 5, 6, ge::DT_UNDEFINED);
    (void)NnopbaseAddDynamicInput(executor, aclTensorTestList, 6);
    (void)NnopbaseAddOutput(executor, tensor, 0);
    auto inputs = &(((NnopbaseExecutor*)executor)->ownArgs.inputs);
    auto tensors = &inputs->extTensors[2];
    gert::TensorV2* rt2Tensor = &tensors->rt2Tensor;
    ASSERT_EQ(rt2Tensor->GetDataType(), ge::DT_FLOAT);

    uint64_t workspaceLen = 0U;
    ASSERT_EQ(NnopbaseRunForWorkspace(executor, &workspaceLen), OK);
#if 0 // comment out for ge decoupling
    ASSERT_EQ(workspaceLen, 0U);
#endif
    void* stream = nullptr;
    void* workspace = nullptr;
    if (workspaceLen > 0U) {
        workspace = (void*)malloc(workspaceLen);
    }
    ASSERT_EQ(NnopbaseRunWithWorkspace(executor, stream, workspace, workspaceLen), OK);

    NnopbaseExecutorGcSpace(executorSpace);
    if (workspaceLen > 0U) {
        free(workspace);
    }

    aclDestroyScalarList(scalarList);
    aclDestroyTensorList((const aclTensorList*)aclTensorTestList);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseUnitTest, NnopbaseAddScalarInputWithErrorDtype)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char* opType = "bninference_d_kernel";
    char inputDesc[] = {1, 1, 0, 1};
    char outputDesc[] = {1};
    char attrDesc[] = {};
    void* executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                         sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    ASSERT_NE(executor, nullptr);

    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);

    float scalar_value = 5;
    auto* scalar = aclCreateScalar(&scalar_value, aclDataType::ACL_FLOAT);
    (void)NnopbaseAddInput(executor, tensor, 0);
    (void)NnopbaseAddInput(executor, tensor, 1);
    (void)NnopbaseAddScalarInput(executor, scalar, 2, 3, ge::DT_UNDEFINED);
    auto inputs = &(((NnopbaseExecutor*)executor)->ownArgs.inputs);
    auto tensors = &inputs->extTensors[2];
    gert::TensorV2* rt2Tensor = &tensors->rt2Tensor;
    rt2Tensor->SetDataType(ge::DataType::DT_STRING);
    ASSERT_EQ(NnopbaseAddInput(executor, tensor, 3), ACLNN_ERR_PARAM_INVALID);

    NnopbaseExecutorGcSpace(executorSpace);
    aclDestroyTensor(tensor);
    aclDestroyScalar(scalar);
    NnopbaseUnsetEnvAndClearFolder();
}

aclrtStream mainStream;
aclrtStream subStream;
aclrtEvent eventA;
aclrtEvent eventB;

TEST_F(NnopbaseUnitTest, NnopbaseFirstGetStreamAndEvent)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);
    EXPECT_EQ(aclrtCreateStreamWithConfig(&mainStream, 0, ACL_STREAM_FAST_LAUNCH | ACL_STREAM_FAST_SYNC), ACL_SUCCESS);
    std::shared_ptr<std::mutex> ptr;
    ASSERT_EQ(NnopbaseGetStreamAndEvent(mainStream, &subStream, &eventA, &eventB, ptr), OK);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseUnitTest, NnopbaseGetStreamAndEventWithoutMainStream)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);
    aclrtStream stream = (void*)(new uint8_t[1]);
    aclrtStream secondStream;
    aclrtEvent event1;
    aclrtEvent event2;
    std::shared_ptr<std::mutex> ptr;
    ASSERT_EQ(NnopbaseGetStreamAndEvent(stream, &secondStream, &event1, &event2, ptr), OK);
    ASSERT_NE(stream, mainStream);
    ASSERT_NE(secondStream, subStream);
    ASSERT_NE(event1, eventA);
    ASSERT_NE(event2, eventB);
    NnopbaseDestroyStreamCallback(stream, false);
    rtStreamDestroy(stream);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseUnitTest, NnopbaseGetStreamAndEvent)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);
    aclrtStream secondStream;
    aclrtEvent event1;
    aclrtEvent event2;
    std::shared_ptr<std::mutex> ptr;
    ASSERT_EQ(NnopbaseGetStreamAndEvent(mainStream, &secondStream, &event1, &event2, ptr), OK);
    ASSERT_EQ(secondStream, subStream);
    ASSERT_EQ(event1, eventA);
    ASSERT_EQ(event2, eventB);
    NnopbaseDestroyStreamCallback(mainStream, false);
    rtStreamDestroy(mainStream);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseUnitTest, NnopbaseGetStreamAndEventWithSameMainStream)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);
    aclrtStream stream;
    EXPECT_EQ(aclrtCreateStreamWithConfig(&stream, 0, ACL_STREAM_FAST_LAUNCH | ACL_STREAM_FAST_SYNC), ACL_SUCCESS);
    aclrtStream secondStream;
    aclrtEvent event1;
    aclrtEvent event2;
    std::shared_ptr<std::mutex> ptr1;
    ASSERT_EQ(NnopbaseGetStreamAndEvent(stream, &secondStream, &event1, &event2, ptr1), OK);
    ASSERT_NE(secondStream, nullptr);
    ASSERT_NE(event1, nullptr);
    ASSERT_NE(event2, nullptr);
    aclrtStream thirdStream;
    aclrtEvent event3;
    aclrtEvent event4;
    std::shared_ptr<std::mutex> ptr2;
    ASSERT_EQ(NnopbaseGetStreamAndEvent(stream, &thirdStream, &event3, &event4, ptr2), OK);
    ASSERT_EQ(secondStream, thirdStream);
    ASSERT_EQ(event1, event3);
    ASSERT_EQ(event2, event4);
    ASSERT_EQ(ptr1, ptr2);
    NnopbaseDestroyStreamCallback(stream, false);
    rtStreamDestroy(stream);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseUnitTest, NnopbaseSetMc2SuccessWithAiCPU)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char* opType = "bninference_d_kernel";
    char inputDesc[] = {1, 1, 1};
    char outputDesc[] = {1};
    char attrDesc[] = {};
    void* executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                         sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    ASSERT_NE(executor, nullptr);

    NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_AICPU);
    ASSERT_EQ(((NnopbaseExecutor*)executor)->mc2.serverType, NNOPBASE_HCCL_SERVER_TYPE_AICPU);
    ASSERT_EQ(NnopbaseSetMc2(executor), OK);
    NnopbaseExecutorGcSpace(executorSpace);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseUnitTest, NnopbaseSetMc2SuccessForascend950WithAiCPU)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char* opType = "bninference_d_kernel";
    char inputDesc[] = {1, 1, 1};
    char outputDesc[] = {1};
    char attrDesc[] = {};
    void* executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                         sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    ASSERT_NE(executor, nullptr);

    NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_AICPU);

    auto oriSocVersion = nnopbase::IndvSoc::GetInstance().GetCurSocVersion();
    struct SocVersionGuard {
        std::string originalVersion;
        ~SocVersionGuard() { SetSocVersion(originalVersion); }
    } socGuard{oriSocVersion};
    SetSocVersion(nnopbase::OPS_SUBPATH_ASCEND950);
    ASSERT_EQ(nnopbase::IndvSoc::GetInstance().GetCurSocVersion(), nnopbase::OPS_SUBPATH_ASCEND950);
    ASSERT_EQ(((NnopbaseExecutor*)executor)->mc2.serverType, NNOPBASE_HCCL_SERVER_TYPE_AICPU);
    ASSERT_EQ(NnopbaseSetMc2(executor), OK);

    // SOC version will be restored by SocVersionGuard destructor
    NnopbaseExecutorGcSpace(executorSpace);
    NnopbaseUnsetEnvAndClearFolder();
}
