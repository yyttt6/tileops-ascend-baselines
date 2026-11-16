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
#include "executor/indv_args.h"
#include "individual_op_api.h"
#include "individual_op_internal.h"
#include "op_cache_internal.h"
#include "executor/indv_collector.h"
#include "utils/file_faker.h"
#include "utils/indv_soc.h"
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
constexpr const char* ASCEND_MILAN_SOC_VERSION = "1";
constexpr const char* ASCEND_DAVID_SOC_VERSION = "2";
constexpr const char* ASCEND_DC_SOC_VERSION = "3";

namespace {
void SetSocVersion(const std::string& version)
{
    auto& soc = nnopbase::IndvSoc::GetInstance();
    soc.socVersion = version;
    soc.isInit = true;
}

const NnopbaseHcclCommParamDesc* GetMc2ParamDesc(const NnopbaseExecutor* const executor)
{
    if (nnopbase::IndvSoc::GetInstance().NnopbaseEnableCcuLaunch(executor->mc2.serverType)) {
        if (executor->mc2.fusionArgs.args == nullptr) {
            return nullptr;
        }
        const size_t kfcArgsFmtOffset = executor->mc2.fusionArgs.aicpuArgs[0].kfcArgsFmtOffset * sizeof(void*);
        return reinterpret_cast<const NnopbaseHcclCommParamDesc*>(
            static_cast<const NnopbaseUChar*>(executor->mc2.fusionArgs.args) + kfcArgsFmtOffset);
    }
    return reinterpret_cast<const NnopbaseHcclCommParamDesc*>(executor->mc2.aicpuArgs.args);
}
} // namespace

class NnopbaseExtUnitTest : public testing::Test {
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
        setenv("ASCEND_C", ASCEND_MILAN_SOC_VERSION, 1);
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

TEST_F(NnopbaseExtUnitTest, TestDynamicLaunchUtForAscend310P)
{
    setenv("ASCEND_C", ASCEND_DC_SOC_VERSION, 1);
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
    size_t workspaceLen = 0U;
    ASSERT_EQ(NnopbaseRunForWorkspace(executor, &workspaceLen), OK);
    void* stream = nullptr;
    void* workspace = nullptr;
    if (workspaceLen > 0U) {
        workspace = (void*)malloc(workspaceLen);
    }
    auto oriSocVersion = nnopbase::IndvSoc::GetInstance().GetCurSocVersion();
    struct SocVersionGuard {
        std::string originalVersion;
        ~SocVersionGuard() { SetSocVersion(originalVersion); }
    } socGuard{oriSocVersion};
    SetSocVersion(std::string(nnopbase::OPS_SUBPATH_ASCEND310P));
    ASSERT_EQ(NnopbaseRunWithWorkspace(executor, stream, workspace, workspaceLen), OK);
    if (workspaceLen > 0U) {
        free(workspace);
    }
    NnopbaseExecutorGcSpace(executorSpace);
    aclDestroyTensor(tensor);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseExtUnitTest, TestDynamicLaunchUtForAscend310PMixAic)
{
    setenv("ASCEND_C", ASCEND_DC_SOC_VERSION, 1);
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

    size_t workspaceLen = 0U;
    ASSERT_EQ(NnopbaseRunForWorkspace(executor, &workspaceLen), OK);
    void* stream = nullptr;
    void* workspace = nullptr;
    if (workspaceLen > 0U) {
        workspace = (void*)malloc(workspaceLen);
    }

    auto oriSocVersion = nnopbase::IndvSoc::GetInstance().GetCurSocVersion();
    struct SocVersionGuard {
        std::string originalVersion;
        ~SocVersionGuard() { SetSocVersion(originalVersion); }
    } socGuard{oriSocVersion};
    auto oriCoreType = ((NnopbaseExecutor*)executor)->args->binInfo->coreType;
    SetSocVersion(std::string(nnopbase::OPS_SUBPATH_ASCEND310P));
    ((NnopbaseExecutor*)executor)->args->binInfo->coreType = kMixAiCore;
    ASSERT_EQ(NnopbaseRunWithWorkspace(executor, stream, workspace, workspaceLen), OK);
    ((NnopbaseExecutor*)executor)->args->binInfo->coreType = oriCoreType;
    if (workspaceLen > 0U) {
        free(workspace);
    }

    NnopbaseDestroyStreamCallback(stream, false);
    NnopbaseExecutorGcSpace(executorSpace);
    aclDestroyTensor(tensor);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseExtUnitTest, TestDynamicLaunchUtForAscend310PMixAiv)
{
    setenv("ASCEND_C", ASCEND_DC_SOC_VERSION, 1);
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

    size_t workspaceLen = 0U;
    ASSERT_EQ(NnopbaseRunForWorkspace(executor, &workspaceLen), OK);
    void* stream = nullptr;
    void* workspace = nullptr;
    if (workspaceLen > 0U) {
        workspace = (void*)malloc(workspaceLen);
    }
    auto oriSocVersion = nnopbase::IndvSoc::GetInstance().GetCurSocVersion();
    struct SocVersionGuard {
        std::string originalVersion;
        ~SocVersionGuard() { SetSocVersion(originalVersion); }
    } socGuard{oriSocVersion};
    auto oriCoreType = ((NnopbaseExecutor*)executor)->args->binInfo->coreType;
    SetSocVersion(std::string(nnopbase::OPS_SUBPATH_ASCEND310P));
    ((NnopbaseExecutor*)executor)->args->binInfo->coreType = kMixAiv;
    ASSERT_EQ(NnopbaseRunWithWorkspace(executor, stream, workspace, workspaceLen), OK);
    ((NnopbaseExecutor*)executor)->args->binInfo->coreType = oriCoreType;
    if (workspaceLen > 0U) {
        free(workspace);
    }
    NnopbaseDestroyStreamCallback(stream, false);
    NnopbaseExecutorGcSpace(executorSpace);
    aclDestroyTensor(tensor);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseExtUnitTest, TestDynamicLaunchUtForAscend310PMixAivBlockDim)
{
    setenv("ASCEND_C", ASCEND_DC_SOC_VERSION, 1);
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

    size_t workspaceLen = 0U;
    ASSERT_EQ(NnopbaseRunForWorkspace(executor, &workspaceLen), OK);
    void* stream = nullptr;
    void* workspace = nullptr;
    if (workspaceLen > 0U) {
        workspace = (void*)malloc(workspaceLen);
    }
    auto oriSocVersion = nnopbase::IndvSoc::GetInstance().GetCurSocVersion();
    struct SocVersionGuard {
        std::string originalVersion;
        ~SocVersionGuard() { SetSocVersion(originalVersion); }
    } socGuard{oriSocVersion};
    auto oriCoreType = ((NnopbaseExecutor*)executor)->args->binInfo->coreType;
    SetSocVersion(std::string(nnopbase::OPS_SUBPATH_ASCEND310P));
    ((NnopbaseExecutor*)executor)->args->binInfo->coreType = kMixAiv;

    // blockDim > aivBlockDim + aicBlockDim
    auto oriBlockDim = ((NnopbaseExecutor*)executor)->args->tilingInfo.numBlocks;
    ((NnopbaseExecutor*)executor)->args->tilingInfo.numBlocks = 18;
    ASSERT_EQ(NnopbaseRunWithWorkspace(executor, stream, workspace, workspaceLen), OK);

    ((NnopbaseExecutor*)executor)->args->binInfo->coreType = oriCoreType;
    ((NnopbaseExecutor*)executor)->args->tilingInfo.numBlocks = oriBlockDim;
    if (workspaceLen > 0U) {
        free(workspace);
    }
    NnopbaseDestroyStreamCallback(stream, false);
    NnopbaseExecutorGcSpace(executorSpace);
    aclDestroyTensor(tensor);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseExtUnitTest, TestDynamicLaunchUtForAscend310PMixAivBlockDim2)
{
    setenv("ASCEND_C", ASCEND_DC_SOC_VERSION, 1);
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

    size_t workspaceLen = 0U;
    ASSERT_EQ(NnopbaseRunForWorkspace(executor, &workspaceLen), OK);
    void* stream = nullptr;
    void* workspace = nullptr;
    if (workspaceLen > 0U) {
        workspace = (void*)malloc(workspaceLen);
    }
    auto oriSocVersion = nnopbase::IndvSoc::GetInstance().GetCurSocVersion();
    struct SocVersionGuard {
        std::string originalVersion;
        ~SocVersionGuard() { SetSocVersion(originalVersion); }
    } socGuard{oriSocVersion};
    auto oriCoreType = ((NnopbaseExecutor*)executor)->args->binInfo->coreType;
    SetSocVersion(std::string(nnopbase::OPS_SUBPATH_ASCEND310P));
    ((NnopbaseExecutor*)executor)->args->binInfo->coreType = kMixAiv;

    // blockDim > aivBlockDim + aicBlockDim
    auto oriBlockDim = ((NnopbaseExecutor*)executor)->args->tilingInfo.numBlocks;
    ((NnopbaseExecutor*)executor)->args->tilingInfo.numBlocks = 12;
    ASSERT_EQ(NnopbaseRunWithWorkspace(executor, stream, workspace, workspaceLen), OK);

    ((NnopbaseExecutor*)executor)->args->binInfo->coreType = oriCoreType;
    ((NnopbaseExecutor*)executor)->args->tilingInfo.numBlocks = oriBlockDim;
    if (workspaceLen > 0U) {
        free(workspace);
    }
    NnopbaseDestroyStreamCallback(stream, false);
    NnopbaseExecutorGcSpace(executorSpace);
    aclDestroyTensor(tensor);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseExtUnitTest, TestStaticLaunchUtForAscend310PMixAic)
{
    setenv("ASCEND_C", ASCEND_DC_SOC_VERSION, 1);
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
    auto oriSocVersion = nnopbase::IndvSoc::GetInstance().GetCurSocVersion();
    struct SocVersionGuard {
        std::string originalVersion;
        ~SocVersionGuard() { SetSocVersion(originalVersion); }
    } socGuard{oriSocVersion};
    auto oriCoreType = ((NnopbaseExecutor*)executor)->args->binInfo->coreType;
    SetSocVersion(std::string(nnopbase::OPS_SUBPATH_ASCEND310P));
    ((NnopbaseExecutor*)executor)->args->binInfo->coreType = kMixAiCore;
    ASSERT_EQ(NnopbaseRunWithWorkspace(executor, stream, workspace, workspaceLen), OK);
    ((NnopbaseExecutor*)executor)->args->binInfo->coreType = oriCoreType;
    if (workspaceLen > 0U) {
        free(workspace);
    }
    NnopbaseDestroyStreamCallback(stream, false);
    NnopbaseExecutorGcSpace(executorSpace);
    aclDestroyTensor(tensor);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseExtUnitTest, TestStaticLaunchUtForAscend310PMixAiv)
{
    setenv("ASCEND_C", ASCEND_DC_SOC_VERSION, 1);
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
    auto oriSocVersion = nnopbase::IndvSoc::GetInstance().GetCurSocVersion();
    struct SocVersionGuard {
        std::string originalVersion;
        ~SocVersionGuard() { SetSocVersion(originalVersion); }
    } socGuard{oriSocVersion};
    auto oriCoreType = ((NnopbaseExecutor*)executor)->args->binInfo->coreType;
    SetSocVersion(std::string(nnopbase::OPS_SUBPATH_ASCEND310P));
    ((NnopbaseExecutor*)executor)->args->binInfo->coreType = kMixAiv;
    ASSERT_EQ(NnopbaseRunWithWorkspace(executor, stream, workspace, workspaceLen), OK);
    ((NnopbaseExecutor*)executor)->args->binInfo->coreType = oriCoreType;
    if (workspaceLen > 0U) {
        free(workspace);
    }
    NnopbaseDestroyStreamCallback(stream, false);
    NnopbaseExecutorGcSpace(executorSpace);
    aclDestroyTensor(tensor);
    NnopbaseUnsetEnvAndClearFolder();
}

// 尝试多次设置 多次调用
TEST_F(NnopbaseExtUnitTest, TestStaticLaunchUtForAscend310PBlockDim)
{
    setenv("ASCEND_C", ASCEND_DC_SOC_VERSION, 1);
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
    auto oriSocVersion = nnopbase::IndvSoc::GetInstance().GetCurSocVersion();
    struct SocVersionGuard {
        std::string originalVersion;
        ~SocVersionGuard() { SetSocVersion(originalVersion); }
    } socGuard{oriSocVersion};
    auto oriBlockDim = ((NnopbaseExecutor*)executor)->args->tilingInfo.numBlocks;
    auto oriCoreType = ((NnopbaseExecutor*)executor)->args->binInfo->coreType;
    ((NnopbaseExecutor*)executor)->args->binInfo->coreType = kMixAiv;
    // blockDim < aicBlockDim
    ((NnopbaseExecutor*)executor)->args->tilingInfo.numBlocks = 7;
    SetSocVersion(std::string(nnopbase::OPS_SUBPATH_ASCEND310P));
    ASSERT_EQ(NnopbaseRunWithWorkspace(executor, stream, workspace, workspaceLen), OK);

    ((NnopbaseExecutor*)executor)->args->tilingInfo.numBlocks = oriBlockDim;
    ((NnopbaseExecutor*)executor)->args->binInfo->coreType = oriCoreType;
    if (workspaceLen > 0U) {
        free(workspace);
    }
    NnopbaseDestroyStreamCallback(stream, false);
    NnopbaseExecutorGcSpace(executorSpace);
    aclDestroyTensor(tensor);
    NnopbaseUnsetEnvAndClearFolder();
}

// 打桩只修改算子名 但是不修改算子输入输出 算子输入输出和bninference_d_kernel一样
void RunCommonOp(std::string opName, CoreType coreType, std::string socVersion, uint64_t tilingKey,
                 MixRationParam goldenData, bool flag)
{
    op::internal::opProfilingSwitch.reportFlag = true;
    op::internal::opProfilingSwitch.additionInfoFlag = true;
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);
    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char* opType = opName.c_str();
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

    if (socVersion == "ascend910b") {
        ASSERT_EQ(workspaceLen, 200U + 75 * 1024 * 1024);
    }

    auto oriSocVersion = nnopbase::IndvSoc::GetInstance().GetCurSocVersion();
    struct SocVersionGuard {
        std::string originalVersion;
        ~SocVersionGuard() { SetSocVersion(originalVersion); }
    } socGuard{oriSocVersion};
    auto oriCoreType = ((NnopbaseExecutor*)executor)->args->binInfo->coreType;
    auto oriTilingKey = ((NnopbaseExecutor*)executor)->args->tilingInfo.tilingKey;
    SetSocVersion(socVersion);
    ((NnopbaseExecutor*)executor)->args->binInfo->coreType = coreType;
    ((NnopbaseExecutor*)executor)->args->tilingInfo.tilingKey = tilingKey;
    if (flag) {
        ASSERT_EQ(NnopbaseRunWithWorkspace(executor, stream, workspace, workspaceLen), ACLNN_ERR_PARAM_INVALID);
    } else {
        ASSERT_EQ(NnopbaseRunWithWorkspace(executor, stream, workspace, workspaceLen), OK);
        ASSERT_EQ(((NnopbaseExecutor*)executor)->args->binInfo->multiKernelType == 1, true);
        ASSERT_EQ(((NnopbaseExecutor*)executor)->args->binInfo->tilingKeyInfo[tilingKey].crossCoreSync ==
                      goldenData.crossCoreSync,
                  true);
        ASSERT_EQ(((NnopbaseExecutor*)executor)->args->binInfo->tilingKeyInfo[tilingKey].coreType, goldenData.coreType);
        ASSERT_EQ(
            ((NnopbaseExecutor*)executor)->args->binInfo->tilingKeyInfo[tilingKey].taskRation == goldenData.taskRation,
            true);
    }

    ((NnopbaseExecutor*)executor)->args->binInfo->coreType = oriCoreType;
    ((NnopbaseExecutor*)executor)->args->tilingInfo.tilingKey = oriTilingKey;

    if (workspaceLen > 0U) {
        free(workspace);
    }

    NnopbaseExecutorGcSpace(executorSpace);
    aclDestroyTensor(tensor);
    NnopbaseUnsetEnvAndClearFolder();
    op::internal::opProfilingSwitch.reportFlag = false;
    op::internal::opProfilingSwitch.additionInfoFlag = false;
}

class UtNnopbaseExceptionDump : public Adx::DumpStub {
public:
    void AdumpPrintAndGetTimeStampInfo(const void* workSpaceAddr, const size_t dumpWorkSpaceSize, aclrtStream stream,
                                       const char* opType, std::vector<MsprofAicTimeStampInfo>& timeStampInfo)
    {
        MsprofAicTimeStampInfo timeInfo = {8662162037790U, 0U, 10U, 20619064410912U};
        timeStampInfo.push_back(timeInfo);
        return;
    }
};

// test for 1971
TEST_F(NnopbaseExtUnitTest, Test1971MixJsonInfo1)
{
    UtNnopbaseExceptionDump dumpStub;
    Adx::DumpStub::GetInstance()->Install(&dumpStub);
    op::internal::opProfilingSwitch.timeStampFlag = true;
    MixRationParam goldenData;
    goldenData.coreType = kMix;
    goldenData.taskRation = kRation10;
    goldenData.crossCoreSync = true;
    int64_t tilingKey = 6U;
    RunCommonOp("1971_for_mix_normal", kMix, "ascend910b", tilingKey, goldenData, false);
    op::internal::opProfilingSwitch.timeStampFlag = false;
    Adx::DumpStub::GetInstance()->UnInstall();
}

TEST_F(NnopbaseExtUnitTest, Test1971MixJsonInfo2)
{
    MixRationParam goldenData;
    goldenData.coreType = kMix;
    goldenData.taskRation = kRation01;
    goldenData.crossCoreSync = true;
    uint64_t tilingKey = 1U;
    RunCommonOp("1971_for_mix_normal", kMix, "ascend910b", tilingKey, goldenData, false);
}

TEST_F(NnopbaseExtUnitTest, Test1971MixJsonInfo3)
{
    MixRationParam goldenData;
    goldenData.coreType = kMix;
    goldenData.taskRation = kRation10;
    goldenData.crossCoreSync = false;
    uint64_t tilingKey = 2U;
    RunCommonOp("1971_for_mix_normal", kMix, "ascend910b", tilingKey, goldenData, false);
}

TEST_F(NnopbaseExtUnitTest, Test1971MixJsonInfo4)
{
    MixRationParam goldenData;
    goldenData.coreType = kMix;
    goldenData.taskRation = kRation01;
    goldenData.crossCoreSync = false;
    uint64_t tilingKey = 3U;
    RunCommonOp("1971_for_mix_normal", kMix, "ascend910b", tilingKey, goldenData, false);
}

TEST_F(NnopbaseExtUnitTest, Test1971MixJsonInfo5)
{
    MixRationParam goldenData;
    goldenData.coreType = kMix;
    goldenData.taskRation = kRation11;
    goldenData.crossCoreSync = true;
    uint64_t tilingKey = 4U;
    RunCommonOp("1971_for_mix_normal", kMix, "ascend910b", tilingKey, goldenData, false);
}

TEST_F(NnopbaseExtUnitTest, Test1971MixJsonInfo6)
{
    MixRationParam goldenData;
    goldenData.coreType = kMix;
    goldenData.taskRation = kRation12;
    goldenData.crossCoreSync = true;
    uint64_t tilingKey = 5U;
    RunCommonOp("1971_for_mix_normal", kMix, "ascend910b", tilingKey, goldenData, false);
}

// test for 310p aic
TEST_F(NnopbaseExtUnitTest, Test310pMixAicJsonInfo)
{
    MixRationParam goldenData;
    goldenData.coreType = kMixAiCore;
    goldenData.taskRation = kRation10;
    goldenData.crossCoreSync = true;
    uint64_t tilingKey = 0U;
    RunCommonOp("310p_for_mix_aic_normal", kMixAiCore, "ascend310p", tilingKey, goldenData, false);
}

// test for 310p aiv
TEST_F(NnopbaseExtUnitTest, Test310pMixAivJsonInfo)
{
    MixRationParam goldenData;
    goldenData.coreType = kMixAiv;
    goldenData.taskRation = kRation10;
    goldenData.crossCoreSync = true;
    uint64_t tilingKey = 0U;
    RunCommonOp("310p_for_mix_aiv_normal", kMixAiv, "ascend310p", tilingKey, goldenData, false);
}

// not find tilingKey
TEST_F(NnopbaseExtUnitTest, Test310pMixAivForNotFindTilingKey)
{
    MixRationParam goldenData;
    goldenData.coreType = kMixAiv;
    goldenData.taskRation = kRation10;
    goldenData.crossCoreSync = true;
    uint64_t tilingKey = 10U;
    RunCommonOp("310p_for_mix_aiv_normal", kMixAiv, "ascend310p", tilingKey, goldenData, true);
}

void RunCommonOpForProfiling(std::string opName, CoreType coreType, std::string socVersion, uint64_t tilingKey,
                             uint32_t goldenData, bool flag)
{
    op::internal::opProfilingSwitch.reportFlag = true;
    op::internal::opProfilingSwitch.additionInfoFlag = true;
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);
    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char* opType = opName.c_str();
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

    ((NnopbaseExecutor*)executor)->args->tilingInfo.tilingKey = tilingKey; // tilingKey
    uint32_t numBlocks = ((NnopbaseExecutor*)executor)->args->binInfo->numBlocks;
    uint32_t taskType = NnopbaseExecutorGetTaskType(((NnopbaseExecutor*)executor)->args->binInfo->coreType,
                                                    ((NnopbaseExecutor*)executor)->args->binInfo->taskRation);
    uint64_t timeStamp = MsprofSysCycleTime();
    if (flag) {
        NnopbaseReportContextIdInfoByRation((NnopbaseExecutor*)executor, timeStamp, numBlocks, taskType);
    } else {
        NnopbaseReportContextIdInfoByRation((NnopbaseExecutor*)executor, timeStamp, numBlocks, taskType);
        ASSERT_EQ(numBlocks, goldenData);
    }

    if (workspaceLen > 0U) {
        free(workspace);
    }
    NnopbaseExecutorGcSpace(executorSpace);
    aclDestroyTensor(tensor);
    NnopbaseUnsetEnvAndClearFolder();
    op::internal::opProfilingSwitch.reportFlag = false;
    op::internal::opProfilingSwitch.additionInfoFlag = false;
}

TEST_F(NnopbaseExtUnitTest, Test1971Profiling0)
{
    uint32_t goldenData = 0U;
    uint64_t tilingKey = 6U; // taskRation 1:0  crossCoreSync: 1
    RunCommonOpForProfiling("1971_for_mix_normal", kMixAiv, "ascend910b", tilingKey, goldenData, false);
}

TEST_F(NnopbaseExtUnitTest, Test1971Profiling1)
{
    uint32_t goldenData = 0U;
    uint64_t tilingKey = 1U; // taskRation 0:1  crossCoreSync: 1
    RunCommonOpForProfiling("1971_for_mix_normal", kMixAiv, "ascend910b", tilingKey, goldenData, false);
}

TEST_F(NnopbaseExtUnitTest, Test1971Profiling2)
{
    uint32_t goldenData = 0u;
    uint64_t tilingKey = 2U; // taskRation 1:0 crossCoreSync: 0
    RunCommonOpForProfiling("1971_for_mix_normal", kMixAiv, "ascend910b", tilingKey, goldenData, false);
}

TEST_F(NnopbaseExtUnitTest, Test1971Profiling3)
{
    uint32_t goldenData = 65536u;
    uint64_t tilingKey = 4U; // taskRation 1:1 crossCoreSync: 1
    RunCommonOpForProfiling("1971_for_mix_normal", kMixAiv, "ascend910b", tilingKey, goldenData, false);
}

TEST_F(NnopbaseExtUnitTest, Test1971ProfilingForNotFindTilingKey4)
{
    uint32_t goldenData = 0u;
    uint64_t tilingKey = 10U;
    RunCommonOpForProfiling("1971_for_mix_normal", kMixAiv, "ascend910b", tilingKey, goldenData, true);
}

TEST_F(NnopbaseExtUnitTest, Test1971Profiling5)
{
    uint32_t goldenData = 0u;
    uint64_t tilingKey = 3U; // taskRation 1:0 crossCoreSync: 0
    RunCommonOpForProfiling("1971_for_mix_normal", kMixAiv, "ascend910b", tilingKey, goldenData, false);
}

void CheckSizeInfoAddr(void* sizeInfoAddr)
{
    uint64_t* infoAddr = reinterpret_cast<uint64_t*>(sizeInfoAddr);
    ASSERT_EQ(*infoAddr, 1);
    infoAddr++;
    uint64_t addrNum = *infoAddr;
    bool hasFftsAddr = false;
    hasFftsAddr = (((*infoAddr) >> 32U) == 1ULL) ? true : false;
    // 标记ffts地址
    if (hasFftsAddr) {
        addrNum &= 0x00000000ffffffff;
    }
    infoAddr++;
    for (size_t i = 0; i < addrNum; i++) {
        if (((*infoAddr) >> 56) == 4U) {
            uint64_t workSpaceSize = (*infoAddr) & 0x00ffffffffffffff;
            ASSERT_EQ(workSpaceSize, 76800);
            break;
        }
        ++infoAddr;
    }
}

// 测试静态库流程
void RunCommonOpForStaticBin(std::string op, aclnnStatus res1, aclnnStatus res2, aclDataType dataType, bool isSetStub,
                             bool isPrint, bool hasTilingKey, std::string binPath)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);
    if (!isSetStub) {
        nnopbase::OpBinaryResourceManager::GetInstance().pathToBinary_.clear();
    }
    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char* opType = op.c_str();
    char inputDesc[] = {1, 1};
    char outputDesc[] = {1};
    char attrDesc[] = {};
    void* executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                         sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    ASSERT_NE(executor, nullptr);

    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_ND,
                                        shape.data(), shape.size(), nullptr);
    (void)NnopbaseAddInput(executor, tensor, 0);
    (void)NnopbaseAddInput(executor, tensor, 1);
    (void)NnopbaseAddOutput(executor, tensor, 0);

    size_t workspaceLen = 0U;
    if (res1 != OK) {
        ASSERT_NE(NnopbaseRunForWorkspace(executor, &workspaceLen), OK);
        NnopbaseExecutorGcSpace(executorSpace);
        aclDestroyTensor(tensor);
        NnopbaseUnsetEnvAndClearFolder();
        return;
    } else {
        ASSERT_EQ(NnopbaseRunForWorkspace(executor, &workspaceLen), OK);
    }
    void* stream = nullptr;
    void* workspace = nullptr;
    if (workspaceLen > 0U) {
        workspace = (void*)malloc(workspaceLen);
    }

    ASSERT_EQ(NnopbaseRunWithWorkspace(executor, stream, workspace, workspaceLen), res2);
    if (isPrint) {
        ASSERT_EQ(((NnopbaseExecutor*)executor)->args->binInfo->dfxInfo.isAssertEnable, true);
        uint32_t atomicIndex = 1;
        void* sizeInfoAddr = Adx::DumpStub::GetInstance()->AdumpGetSizeInfoAddr(3, atomicIndex);
        std::cout << "enter CheckSizeInfoAddr" << std::endl;
        CheckSizeInfoAddr(sizeInfoAddr);
    }

    if (hasTilingKey) {
        ASSERT_EQ(((NnopbaseExecutor*)executor)->args->binInfo->tilingKeyInfo[1].crossCoreSync, true);
        ASSERT_EQ(((NnopbaseExecutor*)executor)->args->binInfo->tilingKeyInfo[1].coreType, kVectorcore);
        ASSERT_EQ(((NnopbaseExecutor*)executor)->args->binInfo->tilingKeyInfo[1].taskRation, kRation01);
        ASSERT_EQ(((NnopbaseExecutor*)executor)->args->binInfo->coreType, kVectorcore);
    }
    if (!binPath.empty()) {
        ASSERT_EQ(((NnopbaseExecutor*)executor)->args->binInfo->binPath, binPath);
    }
    if (workspaceLen > 0U) {
        free(workspace);
    }

    NnopbaseExecutorGcSpace(executorSpace);
    aclDestroyTensor(tensor);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseExtUnitTest, TestStaticBin01) // 正常场景
{
    RunCommonOpForStaticBin("AddCustom", OK, OK, aclDataType::ACL_INT32, true, false, false,
                            "ascend910b/add_custom/AddCustom_402e355eb717124771cfc7dbebfe946c.o");
}

TEST_F(NnopbaseExtUnitTest, TestStaticBin02) // 不支持场景
{
    RunCommonOpForStaticBin("AddCustom", ACLNN_ERR_PARAM_INVALID, ACLNN_ERR_PARAM_NULLPTR, aclDataType::ACL_FLOAT, true,
                            false, false, "");
}

TEST_F(NnopbaseExtUnitTest, TestStaticBin03) // isPrint + hasTilingKey
{
    RunCommonOpForStaticBin("AddCustom", OK, OK, aclDataType::ACL_FLOAT16, true, true, true,
                            "ascend910b/add_custom/AddCustom_1e04ee05ab491cc5ae9c3d5c9ee8950b.o");
}

TEST_F(NnopbaseExtUnitTest, TestStaticBin04) // 不支持场景
{
    RunCommonOpForStaticBin("AddCustom", ACLNN_ERR_INNER_FIND_KERNEL_ERROR, ACLNN_ERR_PARAM_NULLPTR,
                            aclDataType::ACL_UINT8, true, false, false, "");
}

TEST_F(NnopbaseExtUnitTest, TestStaticBin05) // 不支持场景
{
    RunCommonOpForStaticBin("AddCustom", ACLNN_ERR_PARAM_INVALID, ACLNN_ERR_PARAM_NULLPTR, aclDataType::ACL_FLOAT,
                            false, false, false, "");
}

class MmpaNormalStub;
extern MmpaNormalStub mmpaNormallStub;

struct MmpaNormalStubGuard {
    MmpaNormalStubGuard() { Adx::MmpaStub::GetInstance()->Install((Adx::MmpaStub*)&mmpaNormallStub); }
    ~MmpaNormalStubGuard() { Adx::MmpaStub::GetInstance()->UnInstall(); }
};

TEST_F(NnopbaseExtUnitTest, NnopBaseMc2SetSuccess)
{
    MmpaNormalStubGuard mmpaGuard;
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char* opType = "bninference_d_kernel";
    char inputDesc[] = {1, 1};
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
    (void)NnopbaseAddOutput(executor, tensor, 0);

    char* group = "123";
    ASSERT_EQ(NnopbaseSetHcomGroup(executor, group), ACLNN_ERR_PARAM_INVALID);

    ASSERT_EQ(NnopbaseSetMc2(executor), OK);
    ASSERT_EQ(((NnopbaseExecutor*)executor)->mc2.enabled, true);

    ASSERT_EQ(NnopbaseSetHcomGroup(nullptr, group), ACLNN_ERR_PARAM_NULLPTR);
    ASSERT_EQ(NnopbaseSetHcomGroup(executor, nullptr), OK);

    ASSERT_EQ(NnopbaseSetHcomGroup(executor, group), OK);
    ASSERT_EQ(NnopbaseSetHcomGroup(executor, group), OK);

    NnopbaseExecutorGcSpace(executorSpace);
    aclDestroyTensor(tensor);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseExtUnitTest, NnopBaseMC2RunSuccess02)
{
    MmpaNormalStubGuard mmpaGuard;
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char* opType = "AddCustom";
    char inputDesc[] = {1, 1};
    char outputDesc[] = {1};
    char attrDesc[] = {1, 0};

    void* executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                         sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    ASSERT_NE(executor, nullptr);

    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_INT32, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    ASSERT_EQ(NnopbaseSetMc2(executor), OK);
    ASSERT_EQ(NnopbaseAddInput(executor, tensor, 0), OK);
    ASSERT_EQ(NnopbaseAddInput(executor, tensor, 1), OK);
    ASSERT_EQ(NnopbaseAddOutput(executor, tensor, 0), OK);

    char* group = "123";
    ASSERT_EQ(NnopbaseAddAttrWithDtype(executor, static_cast<void*>(group), strlen(group) + 1, 0, kNnopbaseString), OK);
    ASSERT_EQ(NnopbaseSetHcomGroup(executor, group), OK);
    char* groupTp = "123";
    ASSERT_EQ(NnopbaseAddAttrWithDtype(executor, static_cast<void*>(groupTp), strlen(groupTp) + 1, 1, kNnopbaseString),
              OK);
    ASSERT_EQ(NnopbaseSetHcomGroup(executor, groupTp), OK);

    size_t workspaceLen = 0U;
    ASSERT_EQ(NnopbaseRunForWorkspace(executor, &workspaceLen), OK);
    void* stream = nullptr;
    void* workspace = nullptr;
    if (workspaceLen > 0U) {
        workspace = (void*)malloc(workspaceLen);
    }

    auto oriSocVersion = nnopbase::IndvSoc::GetInstance().GetCurSocVersion();
    struct SocVersionGuard {
        std::string originalVersion;
        ~SocVersionGuard() { SetSocVersion(originalVersion); }
    } socGuard{oriSocVersion};
    SetSocVersion(std::string(nnopbase::OPS_SUBPATH_ASCEND910B));
    ASSERT_EQ(NnopbaseRunWithWorkspace(executor, stream, workspace, workspaceLen), OK);
    NnopbaseExecutorGcSpace(executorSpace);
    if (workspaceLen > 0U) {
        free(workspace);
    }
    aclDestroyTensor(tensor);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseExtUnitTest, NnopBaseMC2RunSuccess03)
{
    MmpaNormalStubGuard mmpaGuard;
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char* opType = "AddCustom";
    char inputDesc[] = {1, 1};
    char outputDesc[] = {1};
    char attrDesc[] = {0, 0};

    void* executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                         sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    ASSERT_NE(executor, nullptr);

    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_INT32, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    ASSERT_EQ(NnopbaseSetMc2(executor), OK);
    ASSERT_EQ(NnopbaseAddInput(executor, tensor, 0), OK);
    ASSERT_EQ(NnopbaseAddInput(executor, tensor, 1), OK);
    ASSERT_EQ(NnopbaseAddOutput(executor, tensor, 0), OK);

    char* groupEp = "123";
    ASSERT_EQ(NnopbaseAddAttrWithDtype(executor, static_cast<void*>(groupEp), strlen(groupEp) + 1, 0, kNnopbaseString),
              OK);
    ASSERT_EQ(NnopbaseSetHcomGroup(executor, groupEp), OK);
    static char* groupOptionalDef = "";
    ASSERT_EQ(NnopbaseAddAttrWithDtype(executor, static_cast<void*>(groupOptionalDef), strlen(groupOptionalDef) + 1, 1,
                                       kNnopbaseString),
              OK);
    ASSERT_EQ(NnopbaseSetHcomGroup(executor, nullptr), OK);

    size_t workspaceLen = 0U;
    ASSERT_EQ(NnopbaseRunForWorkspace(executor, &workspaceLen), OK);
    void* stream = nullptr;
    void* workspace = nullptr;
    if (workspaceLen > 0U) {
        workspace = (void*)malloc(workspaceLen);
    }

    NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
    ASSERT_EQ(NnopbaseRunWithWorkspace(executor, stream, workspace, workspaceLen), OK);
    NnopbaseExecutorGcSpace(executorSpace);
    if (workspaceLen > 0U) {
        free(workspace);
    }
    aclDestroyTensor(tensor);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseExtUnitTest, NnopBaseMC2RunSuccess04)
{
    MmpaNormalStubGuard mmpaGuard;
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char* opType = "AddCustom";
    char inputDesc[] = {1, 1};
    char outputDesc[] = {1};
    char attrDesc[] = {0, 0};

    void* executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                         sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    ASSERT_NE(executor, nullptr);

    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_INT32, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    ASSERT_EQ(NnopbaseSetMc2(executor), OK);
    ASSERT_EQ(NnopbaseAddInput(executor, tensor, 0), OK);
    ASSERT_EQ(NnopbaseAddInput(executor, tensor, 1), OK);
    ASSERT_EQ(NnopbaseAddOutput(executor, tensor, 0), OK);

    char* groupEp = "123";
    ASSERT_EQ(NnopbaseAddAttrWithDtype(executor, static_cast<void*>(groupEp), strlen(groupEp) + 1, 0, kNnopbaseString),
              OK);
    ASSERT_EQ(NnopbaseSetHcomGroup(executor, groupEp), OK);
    static char* groupOptionalDef = "";
    ASSERT_EQ(NnopbaseAddAttrWithDtype(executor, static_cast<void*>(groupOptionalDef), strlen(groupOptionalDef) + 1, 1,
                                       kNnopbaseString),
              OK);
    ASSERT_EQ(NnopbaseSetHcomGroup(executor, nullptr), OK);

    NnopbaseHcclServerType hcclServerTypeList[] = {NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_AICPU,
                                                   NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_MTE};
    uint32_t socSupportList[] = {0, 1};
    uint32_t socSupportListLen = 2;
    NnopbaseSetHcclServerTypeList(nullptr, hcclServerTypeList, socSupportList, socSupportListLen);
    NnopbaseSetHcclServerTypeList(executor, nullptr, socSupportList, socSupportListLen);
    NnopbaseSetHcclServerTypeList(executor, hcclServerTypeList, nullptr, socSupportListLen);
    NnopbaseSetHcclServerTypeList(executor, hcclServerTypeList, socSupportList, socSupportListLen);

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

TEST_F(NnopbaseExtUnitTest, NnopBaseMC2RunSuccessWithHostInput)
{
    MmpaNormalStubGuard mmpaGuard;
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);
    setenv("HCCL_EXEC_TIMEOUT", "10", 1);

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char* opType = "bninference_d_kernel";
    char inputDesc[] = {1, 1, 1, 0};
    char outputDesc[] = {1};
    char attrDesc[] = {1};

    void* executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                         sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    ASSERT_NE(executor, nullptr);

    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    std::vector<int64_t> shape2 = {1, 1, 1, 1, 1};
    aclTensor* tensor2 = aclCreateTensor(shape2.data(), shape2.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                         aclFormat::ACL_FORMAT_ND, shape2.data(), shape2.size(), nullptr);
    std::vector<float> vec(2048 * 4, 1.0);
    auto* floatArray = aclCreateFloatArray(vec.data(), vec.size());
    ASSERT_EQ(NnopbaseSetMc2(executor), OK);
    ASSERT_EQ(NnopbaseAddInput(executor, tensor, 0), OK);
    ASSERT_EQ(NnopbaseAddFloatArrayInput(executor, floatArray, 1), OK);
    ASSERT_EQ(NnopbaseAddInput(executor, tensor, 2), OK);
    ASSERT_EQ(NnopbaseAddOutput(executor, tensor2, 0), OK);

    char* group = "123";
    ASSERT_EQ(NnopbaseAddAttrWithDtype(executor, static_cast<void*>(group), strlen(group) + 1, 0, kNnopbaseString), OK);
    ASSERT_EQ(NnopbaseSetHcomGroup(executor, group), OK);

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
    aclDestroyFloatArray(floatArray);
    aclDestroyTensor(tensor);
    aclDestroyTensor(tensor2);
    NnopbaseUnsetEnvAndClearFolder();
    unsetenv("HCCL_EXEC_TIMEOUT");
}

TEST_F(NnopbaseExtUnitTest, NnopBaseRunSuccessWithHostInput)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char* opType = "bninference_d_kernel";
    char inputDesc[] = {1, 1, 1, 0};
    char outputDesc[] = {1};
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
    std::vector<float> vec(2048 * 4, 1.0);
    auto* floatArray = aclCreateFloatArray(vec.data(), vec.size());
    ASSERT_EQ(NnopbaseAddInput(executor, tensor, 0), OK);
    ASSERT_EQ(NnopbaseAddFloatArrayInput(executor, floatArray, 1), OK);
    ASSERT_EQ(NnopbaseAddInput(executor, tensor, 2), OK);
    ASSERT_EQ(NnopbaseAddOutput(executor, tensor2, 0), OK);

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
    aclDestroyFloatArray(floatArray);
    aclDestroyTensor(tensor);
    aclDestroyTensor(tensor2);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseExtUnitTest, TestRepeat01)
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
    (void)NnopbaseSetRef(executor, 0, 0);

    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    (void)NnopbaseAddInput(executor, tensor, 0);
    (void)NnopbaseAddInput(executor, tensor, 1);
    (void)NnopbaseAddInput(executor, nullptr, 2);
    (void)NnopbaseAddInput(executor, tensor, 3);
    (void)NnopbaseAddOutput(executor, tensor, 0);

    size_t workspaceLen = 0U;
    ASSERT_EQ(NnopbaseRunForWorkspace(executor, &workspaceLen), OK);
    void* stream = nullptr;
    void* workspace = nullptr;
    if (workspaceLen > 0U) {
        workspace = (void*)malloc(workspaceLen);
    }

    ASSERT_EQ(aclSetAclOpExecutorRepeatable((aclOpExecutor*)executor), OK);
    ASSERT_EQ(NnopbaseRunWithWorkspace(executor, stream, workspace, workspaceLen), OK);

    ASSERT_EQ(NnopbaseDisableOptionalInput(executor, 4), ACLNN_ERR_PARAM_INVALID);
    ASSERT_EQ(NnopbaseDisableOptionalInput(executor, 2), OK);

    int* ptr = new int;
    void* addr = reinterpret_cast<void*>(ptr);

    ASSERT_EQ(AclSetInputTensorAddr((aclOpExecutor*)executor, 0, tensor, addr), OK);
    ASSERT_EQ(((NnopbaseExecutor*)executor)->args->inputs.extTensors[0].rt2Tensor.GetAddr(), addr);
    ASSERT_EQ(((NnopbaseExecutor*)executor)->args->outputs.extTensors[0].rt2Tensor.GetAddr(), addr);
    ASSERT_EQ(AclSetOutputTensorAddr((aclOpExecutor*)executor, 0, tensor, nullptr), OK);
    ASSERT_EQ(((NnopbaseExecutor*)executor)->args->outputs.extTensors[0].rt2Tensor.GetAddr(), nullptr);
    ASSERT_EQ(NnopbaseRunWithWorkspace((aclOpExecutor*)executor, stream, workspace, workspaceLen), OK);

    ASSERT_EQ(AclSetTensorAddr((aclOpExecutor*)executor, 0, tensor, nullptr), OK);
    ASSERT_EQ(AclSetTensorAddr((aclOpExecutor*)executor, 3, tensor, nullptr), OK);
    ASSERT_EQ(((NnopbaseExecutor*)executor)->args->inputs.extTensors[0].rt2Tensor.GetAddr(), nullptr);
    ASSERT_EQ(((NnopbaseExecutor*)executor)->args->outputs.extTensors[0].rt2Tensor.GetAddr(), nullptr);
    ASSERT_EQ(AclSetTensorAddr((aclOpExecutor*)executor, 2, tensor, nullptr), OK);
    ASSERT_EQ(((NnopbaseExecutor*)executor)->args->inputs.extTensors[3].rt2Tensor.GetAddr(), nullptr);
    ASSERT_EQ(NnopbaseRunWithWorkspace((aclOpExecutor*)executor, stream, workspace, workspaceLen), OK);

    ASSERT_EQ(aclDestroyAclOpExecutor((aclOpExecutor*)executor), OK);

    if (workspaceLen > 0U) {
        free(workspace);
    }

    delete ptr;
    NnopbaseExecutorGcSpace(executorSpace);
    aclDestroyTensor(tensor);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseExtUnitTest, TestRepeat02)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char* opType = "bninference_d_kernel";
    char inputDesc[] = {2, 0, 1, 1};
    char outputDesc[] = {2, 0};
    char attrDesc[] = {};
    void* executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                         sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    ASSERT_NE(executor, nullptr);
    (void)NnopbaseSetRef(executor, 0, 0);

    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor1 = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                         aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    aclTensor* tensor2 = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                         aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    std::vector<const aclTensor*> tensor_list;
    tensor_list.push_back(tensor1);
    tensor_list.push_back(tensor2);
    aclTensorList* aclTensorTestList = aclCreateTensorList(tensor_list.data(), tensor_list.size());

    (void)NnopbaseAddDynamicInput(executor, aclTensorTestList, 0);
    (void)NnopbaseAddInput(executor, nullptr, 1);
    (void)NnopbaseAddInput(executor, tensor1, 2);
    (void)NnopbaseAddInput(executor, tensor1, 3);
    (void)NnopbaseAddDynamicOutput(executor, aclTensorTestList, 0);
    (void)NnopbaseAddOutput(executor, tensor1, 1);

    size_t workspaceLen = 0U;
    ASSERT_EQ(NnopbaseRunForWorkspace(executor, &workspaceLen), OK);
    void* stream = nullptr;
    void* workspace = nullptr;
    if (workspaceLen > 0U) {
        workspace = (void*)malloc(workspaceLen);
    }

    ASSERT_EQ(aclSetAclOpExecutorRepeatable((aclOpExecutor*)executor), OK);
    ASSERT_EQ(NnopbaseRunWithWorkspace(executor, stream, workspace, workspaceLen), OK);
    ASSERT_EQ(NnopbaseDisableOptionalInput(executor, 4), ACLNN_ERR_PARAM_INVALID);
    ASSERT_EQ(NnopbaseDisableOptionalInput(executor, 1), OK);

    int* ptr = new int;
    void* addr = reinterpret_cast<void*>(ptr);

    ASSERT_EQ(AclSetDynamicInputTensorAddr((aclOpExecutor*)executor, 0, 0, aclTensorTestList, addr), OK);
    ASSERT_EQ(((NnopbaseExecutor*)executor)->args->inputs.extTensors[0].rt2Tensor.GetAddr(), addr);
    ASSERT_EQ(((NnopbaseExecutor*)executor)->args->outputs.extTensors[0].rt2Tensor.GetAddr(), addr);
    ASSERT_EQ(AclSetDynamicOutputTensorAddr((aclOpExecutor*)executor, 0, 0, aclTensorTestList, nullptr), OK);
    ASSERT_EQ(((NnopbaseExecutor*)executor)->args->outputs.extTensors[0].rt2Tensor.GetAddr(), nullptr);
    ASSERT_EQ(AclSetInputTensorAddr((aclOpExecutor*)executor, 2, tensor1, addr), OK);
    ASSERT_EQ(AclSetOutputTensorAddr((aclOpExecutor*)executor, 2, tensor1, addr), OK);
    ASSERT_EQ(((NnopbaseExecutor*)executor)->args->inputs.extTensors[3].rt2Tensor.GetAddr(), addr);
    ASSERT_EQ(((NnopbaseExecutor*)executor)->args->outputs.extTensors[2].rt2Tensor.GetAddr(), addr);
    ASSERT_EQ(NnopbaseRunWithWorkspace((aclOpExecutor*)executor, stream, workspace, workspaceLen), OK);

    int* ptr1 = new int;
    void* addr1 = reinterpret_cast<void*>(ptr1);
    ASSERT_EQ(AclSetTensorAddr((aclOpExecutor*)executor, 3, tensor1, addr1), OK);
    ASSERT_EQ(((NnopbaseExecutor*)executor)->args->inputs.extTensors[4].rt2Tensor.GetAddr(), addr1);
    ASSERT_EQ(AclSetTensorAddr((aclOpExecutor*)executor, 5, tensor1, addr1), OK);
    ASSERT_EQ(((NnopbaseExecutor*)executor)->args->outputs.extTensors[1].rt2Tensor.GetAddr(), addr1);
    ASSERT_EQ(AclSetDynamicTensorAddr((aclOpExecutor*)executor, 0, 0, aclTensorTestList, addr1), OK);
    ASSERT_EQ(((NnopbaseExecutor*)executor)->args->inputs.extTensors[0].rt2Tensor.GetAddr(), addr1);
    ASSERT_EQ(AclSetDynamicTensorAddr((aclOpExecutor*)executor, 3, 0, aclTensorTestList, addr1), OK);
    ASSERT_EQ(((NnopbaseExecutor*)executor)->args->outputs.extTensors[0].rt2Tensor.GetAddr(), addr1);
    ASSERT_EQ(NnopbaseRunWithWorkspace((aclOpExecutor*)executor, stream, workspace, workspaceLen), OK);

    ASSERT_EQ(aclDestroyAclOpExecutor((aclOpExecutor*)executor), OK);

    if (workspaceLen > 0U) {
        free(workspace);
    }

    delete ptr1;
    delete ptr;
    NnopbaseExecutorGcSpace(executorSpace);
    aclDestroyTensorList(aclTensorTestList);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseExtUnitTest, TestRepeatErr01)
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

    size_t workspaceLen = 0U;
    ASSERT_EQ(NnopbaseRunForWorkspace(executor, &workspaceLen), OK);
    void* stream = nullptr;
    void* workspace = nullptr;
    if (workspaceLen > 0U) {
        workspace = (void*)malloc(workspaceLen);
    }

    ASSERT_EQ(aclSetAclOpExecutorRepeatable((aclOpExecutor*)executor), OK);

    ASSERT_EQ(AclSetInputTensorAddr((aclOpExecutor*)executor, 4, tensor, nullptr), ACLNN_ERR_PARAM_INVALID);
    ASSERT_EQ(AclSetOutputTensorAddr((aclOpExecutor*)executor, 1, tensor, nullptr), ACLNN_ERR_PARAM_INVALID);
    ASSERT_EQ(AclSetTensorAddr((aclOpExecutor*)executor, 4, tensor, nullptr), ACLNN_ERR_PARAM_INVALID);

    ((NnopbaseExecutor*)executor)->magicNum = 12;
    ASSERT_EQ(aclDestroyAclOpExecutor((aclOpExecutor*)executor), ACLNN_ERR_INNER);

    if (workspaceLen > 0U) {
        free(workspace);
    }

    NnopbaseExecutorGcSpace(executorSpace);
    aclDestroyTensor(tensor);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseExtUnitTest, TestRepeatErr02)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char* opType = "bninference_d_kernel";
    char inputDesc[] = {2, 1, 1};
    char outputDesc[] = {2};
    char attrDesc[] = {};
    void* executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                         sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    ASSERT_NE(executor, nullptr);

    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    std::vector<const aclTensor*> tensor_list;
    tensor_list.push_back(tensor);
    aclTensorList* aclTensorTestList = aclCreateTensorList(tensor_list.data(), tensor_list.size());

    (void)NnopbaseAddDynamicInput(executor, aclTensorTestList, 0);
    (void)NnopbaseAddInput(executor, tensor, 1);
    (void)NnopbaseAddInput(executor, tensor, 2);
    (void)NnopbaseAddDynamicOutput(executor, aclTensorTestList, 0);

    size_t workspaceLen = 0U;
    ASSERT_EQ(NnopbaseRunForWorkspace(executor, &workspaceLen), OK);
    void* stream = nullptr;
    void* workspace = nullptr;
    if (workspaceLen > 0U) {
        workspace = (void*)malloc(workspaceLen);
    }

    ASSERT_EQ(aclSetAclOpExecutorRepeatable((aclOpExecutor*)executor), OK);

    ASSERT_EQ(AclSetDynamicInputTensorAddr((aclOpExecutor*)executor, 0, 1, aclTensorTestList, nullptr),
              ACLNN_ERR_PARAM_INVALID);
    ASSERT_EQ(AclSetDynamicOutputTensorAddr((aclOpExecutor*)executor, 1, 0, aclTensorTestList, nullptr),
              ACLNN_ERR_PARAM_INVALID);

    ASSERT_EQ(AclSetDynamicTensorAddr((aclOpExecutor*)executor, 0, 1, aclTensorTestList, nullptr),
              ACLNN_ERR_PARAM_INVALID);
    ASSERT_EQ(AclSetDynamicTensorAddr((aclOpExecutor*)executor, 4, 0, aclTensorTestList, nullptr),
              ACLNN_ERR_PARAM_INVALID);

    ASSERT_EQ(aclDestroyAclOpExecutor((aclOpExecutor*)executor), OK);

    if (workspaceLen > 0U) {
        free(workspace);
    }

    NnopbaseExecutorGcSpace(executorSpace);
    aclDestroyTensorList(aclTensorTestList);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseExtUnitTest, NnopBaseRunSuccessProfilingMixAic)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);
    NnopbaseInit();

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char* opType = "test_profiling_mix_aic";
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
    ASSERT_EQ(((NnopbaseExecutor*)executor)->args->binInfo->taskRation == kRationEnd, true);
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

TEST_F(NnopbaseExtUnitTest, NnopBaseRunSuccessProfilingMixAiv)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);
    NnopbaseInit();

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char* opType = "test_profiling_mix_aiv";
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
    ASSERT_EQ(((NnopbaseExecutor*)executor)->args->binInfo->taskRation == kRation01, true);
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

TEST_F(NnopbaseExtUnitTest, NnopBaseRunSuccess1971MixFailed)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);
    NnopbaseInit();

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char* opType = "1971_for_mix_unnormal";
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
    ASSERT_EQ(NnopbaseRunForWorkspace(executor, &workspaceLen), ACLNN_ERR_PARAM_INVALID);
    NnopbaseExecutorGcSpace(executorSpace);
    aclDestroyTensor(tensor);
    NnopbaseUnsetEnvAndClearFolder();
}

aclnnStatus AclnnContiguousGetWorkspace(const aclTensorList* list, uint64_t* num, aclOpExecutor** exe) { return OK; }

class ApiStub : public Adx::MmpaStub {
public:
    void* mmDlsym(void* handle, const char* funcName)
    {
        if (std::string(funcName) == "AclnnContiguousGetWorkspace") {
            return (void*)AclnnContiguousGetWorkspace;
        } else {
            return nullptr;
        }
    }

    void* mmDlopen(const char* fileName, int32_t mode) { return &count; }

    static uint64_t count;
};

uint64_t ApiStub::count = 0;

TEST_F(NnopbaseExtUnitTest, TestApiFunc)
{
    ApiStub mmpaStub;
    Adx::MmpaStub::GetInstance()->Install(&mmpaStub);
    ASSERT_EQ(nnopbase::ApiWrapper::GetInstance().LoadFunctions(), OK);
    ASSERT_EQ(NnopbaseGetApiFunc("AclnnContiguousGetWorkspace"), (void*)AclnnContiguousGetWorkspace);
    ASSERT_EQ(NnopbaseGetApiFunc(nullptr), nullptr);
    ASSERT_EQ(NnopbaseGetApiFunc("AclnnXxx"), nullptr);
    Adx::MmpaStub::GetInstance()->UnInstall();
}

TEST_F(NnopbaseExtUnitTest, NnopBaseEmptyTensorSuccessWithMultiKernelType)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char* opType = "1971_for_mix_normal";
    char inputDesc[] = {1, 1, 1};
    char outputDesc[] = {1};
    char attrDesc[] = {};
    void* executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                         sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    ASSERT_NE(executor, nullptr);

    std::vector<int64_t> shape = {0};
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
    if (workspaceLen > 0U) {
        free(workspace);
    }

    NnopbaseExecutorGcSpace(executorSpace);
    aclDestroyTensor(tensor);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseExtUnitTest, aclnnBninferenceDKernelRef)
{
    auto unContTensor = CreateAclTensor({4, 5, 6, 7, 1}, {210, 42, 1, 7, 1}, 0, {4, 5, 7, 6, 1});
    auto contTensor = CreateAclTensor({4, 5, 6, 7, 1}, {210, 42, 7, 1, 1}, 0, {4, 5, 6, 7, 1});
    ASSERT_EQ(RunOp(aclnnBninferenceDKernelRef, unContTensor, contTensor, contTensor), OK);
}

TEST_F(NnopbaseExtUnitTest, refOpTestSetOutputAddr)
{
    auto unContTensor = CreateAclTensor({4, 5, 6, 7, 1}, {210, 42, 1, 7, 1}, 0, {4, 5, 7, 6, 1});
    auto contTensor = CreateAclTensor({4, 5, 6, 7, 1}, {210, 42, 7, 1, 1}, 0, {4, 5, 6, 7, 1});
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    ASSERT_EQ(
        aclnnBninferenceDKernelRefGetWorkspaceSize(unContTensor, contTensor, contTensor, &workspace_size, &executor),
        OK);
    NnopbaseExecutor* nnopExecutor = (NnopbaseExecutor*)executor;
    size_t offset = nnopExecutor->args->inputs.unContiguousTensors.workspaceOffsets[0];

    void* workspace_addr = nullptr;
    if (workspace_size != 0) {
        workspace_addr = malloc(workspace_size);
    }

    ASSERT_EQ(aclnnBninferenceDKernelRef(workspace_addr, workspace_size, executor, nullptr), OK);

    void* addr;
    NnopbaseGetOutputTensorAddr(executor, 0, &addr);
#if 0 // comment out for ge decoupling
    ASSERT_EQ(addr, workspace_addr + offset);
#endif
    if (workspace_addr != nullptr) {
        free(workspace_addr);
    }
}

TEST_F(NnopbaseExtUnitTest, TestAddOutputShapeTensorSuccess)
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
    ASSERT_EQ(NnopbaseAddInput(executor, tensor, 0), OK);
    ASSERT_EQ(NnopbaseAddInput(executor, tensor, 1), OK);
    ASSERT_EQ(NnopbaseAddInput(executor, tensor, 2), OK);
    ASSERT_EQ(NnopbaseAddOutputShapeDependTensor(executor, tensor, 0), OK);
    NnopbaseExecutorGcSpace(executorSpace);
    aclDestroyTensor(tensor);
    NnopbaseUnsetEnvAndClearFolder();
}

class ThirdOpRuntimeStub : public RuntimeStub {
public:
    rtError_t rtKernelLaunchWithHandleV2(void* hdl, const uint64_t tilingKey, uint32_t blockDim, rtArgsEx_t* argsInfo,
                                         rtSmDesc_t* smDesc, aclrtStream stm, const rtTaskCfgInfo_t* cfgInfo)
    {
        uint64_t** outPutShape = (uint64_t**)((uint8_t*)(argsInfo->args) +
                                              5 * sizeof(uint64_t*)); // 5 is input and output
        (*outPutShape)[0] = 5 | 0x80;
        (*outPutShape)[1] = 2;
        (*outPutShape)[2] = 3;
        (*outPutShape)[3] = 4;
        (*outPutShape)[4] = 5;
        (*outPutShape)[5] = 6;

        (*outPutShape)[9] = 3 | 0x80;
        (*outPutShape)[10] = 6;
        (*outPutShape)[11] = 7;
        (*outPutShape)[12] = 8;

        return RT_ERROR_NONE;
    }

    rtError_t rtsLaunchKernelWithHostArgs(rtFuncHandle funcHandle, uint32_t blockDim, aclrtStream stm,
                                          rtKernelLaunchCfg_t* cfg, void* hostArgs, uint32_t argsSize,
                                          aclrtPlaceHolderInfo* placeHolderArray, uint32_t placeHolderNum)
    {
        uint64_t** outPutShape = (uint64_t**)((uint8_t*)(hostArgs) + 5 * sizeof(uint64_t*)); // 5 is input and output
        (*outPutShape)[0] = 5 | 0x80;
        (*outPutShape)[1] = 2;
        (*outPutShape)[2] = 3;
        (*outPutShape)[3] = 4;
        (*outPutShape)[4] = 5;
        (*outPutShape)[5] = 6;

        (*outPutShape)[9] = 3 | 0x80;
        (*outPutShape)[10] = 6;
        (*outPutShape)[11] = 7;
        (*outPutShape)[12] = 8;
        return RT_ERROR_NONE;
    }
};

TEST_F(NnopbaseExtUnitTest, TestAddTwoOutputShapeTensorFailed)
{
    setenv("MEMCPY_ENV", "1", 1);
    ThirdOpRuntimeStub runtimeStub;
    RuntimeStub::GetInstance()->Install(&runtimeStub);
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

    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    aclTensor* out1 = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                      aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    aclTensor* out2 = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                      aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    ASSERT_EQ(NnopbaseAddInput(executor, tensor, 0), OK);
    ASSERT_EQ(NnopbaseAddInput(executor, tensor, 1), OK);
    ASSERT_EQ(NnopbaseAddInput(executor, tensor, 2), OK);
    ASSERT_EQ(NnopbaseAddOutputShapeDependTensor(executor, out1, 0), OK);
    ASSERT_EQ(NnopbaseAddOutputShapeDependTensor(executor, out2, 1), OK);

    size_t workspaceLen = 0U;
    ASSERT_EQ(NnopbaseRunForWorkspace(executor, &workspaceLen), OK);
#if 0 // comment out for ge decoupling
    ASSERT_EQ(workspaceLen, 144);
#endif

    void* stream = nullptr;
    void* workspace = nullptr;
    if (workspaceLen > 0U) {
        workspace = (void*)malloc(workspaceLen);
    }

    ASSERT_NE(NnopbaseRunWithWorkspace(executor, stream, workspace, workspaceLen), OK);
    if (workspaceLen > 0U) {
        free(workspace);
    }
    NnopbaseExecutorGcSpace(executorSpace);
    aclDestroyTensor(tensor);
    aclDestroyTensor(out1);
    aclDestroyTensor(out2);
    NnopbaseUnsetEnvAndClearFolder();
    RuntimeStub::GetInstance()->UnInstall();
    unsetenv("MEMCPY_ENV");
}

TEST_F(NnopbaseExtUnitTest, TestOutputAtomicCleanErrorWithInput)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char* opType = "AutomicClean";
    char inputDesc[] = {1, 0, 1};
    char outputDesc[] = {1, 0, 0, 0, 0, 0, 0, 0, 0};
    char attrDesc[] = {};
    void* executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                         sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    ASSERT_NE(executor, nullptr);

    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    ASSERT_EQ(NnopbaseAddInput(executor, tensor, 0), OK);
    ASSERT_EQ(NnopbaseAddInput(executor, nullptr, 1), OK);
    ASSERT_EQ(NnopbaseAddInput(executor, tensor, 2), OK);
    ASSERT_EQ(NnopbaseAddOutput(executor, tensor, 0), OK);
    ASSERT_EQ(NnopbaseAddOutput(executor, nullptr, 1), OK);
    ASSERT_EQ(NnopbaseAddOutput(executor, nullptr, 2), OK);
    ASSERT_EQ(NnopbaseAddOutput(executor, nullptr, 3), OK);
    ASSERT_EQ(NnopbaseAddOutput(executor, nullptr, 4), OK);
    ASSERT_EQ(NnopbaseAddOutput(executor, nullptr, 5), OK);
    ASSERT_EQ(NnopbaseAddOutput(executor, nullptr, 6), OK);
    ASSERT_EQ(NnopbaseAddOutput(executor, nullptr, 7), OK);
    ASSERT_EQ(NnopbaseAddOutput(executor, nullptr, 8), OK);

    size_t workspaceLen = 0U;
    ASSERT_EQ(NnopbaseRunForWorkspace(executor, &workspaceLen), ACLNN_ERR_PARAM_INVALID);

    NnopbaseExecutorGcSpace(executorSpace);
    aclDestroyTensor(tensor);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseExtUnitTest, TestOutputAtomicCleanErrorWithDynamicOutput)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char* opType = "AutomicClean";
    char inputDesc[] = {1, 1};
    char outputDesc[] = {2, 0, 0, 0, 0, 0, 0, 0, 0, 0};
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

    ASSERT_EQ(NnopbaseAddInput(executor, tensor, 0), OK);
    ASSERT_EQ(NnopbaseAddInput(executor, tensor, 1), OK);
    ASSERT_EQ(NnopbaseAddDynamicOutput(executor, aclTensorTestList, 0), OK);
    ASSERT_EQ(NnopbaseAddOutput(executor, nullptr, 1), OK);
    ASSERT_EQ(NnopbaseAddOutput(executor, nullptr, 2), OK);
    ASSERT_EQ(NnopbaseAddOutput(executor, nullptr, 3), OK);
    ASSERT_EQ(NnopbaseAddOutput(executor, nullptr, 4), OK);
    ASSERT_EQ(NnopbaseAddOutput(executor, nullptr, 5), OK);
    ASSERT_EQ(NnopbaseAddOutput(executor, nullptr, 6), OK);
    ASSERT_EQ(NnopbaseAddOutput(executor, nullptr, 7), OK);
    ASSERT_EQ(NnopbaseAddOutput(executor, nullptr, 8), OK);
    ASSERT_EQ(NnopbaseAddOutput(executor, nullptr, 9), OK);

    size_t workspaceLen = 0U;
    ASSERT_EQ(NnopbaseRunForWorkspace(executor, &workspaceLen), ACLNN_ERR_PARAM_INVALID);
    NnopbaseExecutorGcSpace(executorSpace);
    aclDestroyTensorList((const aclTensorList*)aclTensorTestList);
    NnopbaseUnsetEnvAndClearFolder();
}

#if 0 // comment out for ge decoupling
TEST_F(NnopbaseExtUnitTest, TestOutputAtomicCleanSuccess)
{
    op::internal::opProfilingSwitch.reportFlag = true;
    op::internal::opProfilingSwitch.additionInfoFlag = true;
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    void *executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char *opType = "AutomicClean";
    char inputDesc[] = {1, 1};
    char outputDesc[] = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    char attrDesc[] = {};
    void *executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                         sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    ASSERT_NE(executor, nullptr);

    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor *tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT,
                                        nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    ASSERT_EQ(NnopbaseAddInput(executor, tensor, 0), OK);
    ASSERT_EQ(NnopbaseAddInput(executor, tensor, 1), OK);
    ASSERT_EQ(NnopbaseAddOutput(executor, tensor, 0), OK);
    ASSERT_EQ(NnopbaseAddOutput(executor, nullptr, 1), OK);
    ASSERT_EQ(NnopbaseAddOutput(executor, nullptr, 2), OK);
    ASSERT_EQ(NnopbaseAddOutput(executor, nullptr, 3), OK);
    ASSERT_EQ(NnopbaseAddOutput(executor, nullptr, 4), OK);
    ASSERT_EQ(NnopbaseAddOutput(executor, nullptr, 5), OK);
    ASSERT_EQ(NnopbaseAddOutput(executor, nullptr, 6), OK);
    ASSERT_EQ(NnopbaseAddOutput(executor, nullptr, 7), OK);
    ASSERT_EQ(NnopbaseAddOutput(executor, nullptr, 8), OK);
    ASSERT_EQ(NnopbaseAddOutput(executor, nullptr, 9), OK);

    size_t workspaceLen = 0U;
    ASSERT_EQ(NnopbaseRunForWorkspace(executor, &workspaceLen), OK);

    void *stream = nullptr;
    void *workspace = nullptr;
    if (workspaceLen > 0U) {
        workspace = (void *) malloc(workspaceLen);
    }

    ASSERT_EQ(NnopbaseRunWithWorkspace(executor, stream, workspace, workspaceLen), OK);

    const size_t tilingAddrOffset = ((NnopbaseExecutor *)executor)->args->binInfo->initValues.size() * sizeof(void *);
    const size_t tilingDataOffset = tilingAddrOffset + 2U * sizeof(void *);
    NnopbaseTilingData *tilingdata = (NnopbaseTilingData *)(((NnopbaseExecutor *)executor)->args->binInfo->memsetInfo->tilingData);
    const size_t tilingDataSize = tilingdata->GetDataSize();
    const size_t alignTilingDataSize =
            ((tilingDataSize % 8U) != 0) ? (tilingDataSize / 8U + 1U) * 8U : tilingDataSize; // 8byte对齐
    ASSERT_EQ(alignTilingDataSize + tilingDataOffset, ((NnopbaseExecutor *)executor)->args->binInfo->memsetInfo->argsExt.argsSize);

   ASSERT_EQ(((NnopbaseExecutor *)executor)->args->binInfo->memsetInfo->argsExt.tilingDataOffset, tilingDataOffset);
   ASSERT_EQ(((NnopbaseExecutor *)executor)->args->binInfo->memsetInfo->argsExt.tilingAddrOffset, tilingAddrOffset);

    if (workspaceLen > 0U) {
        free(workspace);
    }
    NnopbaseExecutorGcSpace(executorSpace);
    aclDestroyTensor(tensor);
    NnopbaseUnsetEnvAndClearFolder();

    op::internal::opProfilingSwitch.reportFlag = false;
    op::internal::opProfilingSwitch.additionInfoFlag = false;
}

TEST_F(NnopbaseExtUnitTest, TestOutputAtomicCleanCacheSuccess)
{
    op::internal::opProfilingSwitch.recordOpArgFlag = false;
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    void *executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char *opType = "AutomicClean";
    char inputDesc[] = {1, 1};
    char outputDesc[] = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    char attrDesc[] = {};
    void *executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                         sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    ASSERT_NE(executor, nullptr);

    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor *tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT,
                                        nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    ASSERT_EQ(NnopbaseAddInput(executor, tensor, 0), OK);
    ASSERT_EQ(NnopbaseAddInput(executor, tensor, 1), OK);
    ASSERT_EQ(NnopbaseAddOutput(executor, tensor, 0), OK);
    ASSERT_EQ(NnopbaseAddOutput(executor, nullptr, 1), OK);
    ASSERT_EQ(NnopbaseAddOutput(executor, nullptr, 2), OK);
    ASSERT_EQ(NnopbaseAddOutput(executor, nullptr, 3), OK);
    ASSERT_EQ(NnopbaseAddOutput(executor, nullptr, 4), OK);
    ASSERT_EQ(NnopbaseAddOutput(executor, nullptr, 5), OK);
    ASSERT_EQ(NnopbaseAddOutput(executor, nullptr, 6), OK);
    ASSERT_EQ(NnopbaseAddOutput(executor, nullptr, 7), OK);
    ASSERT_EQ(NnopbaseAddOutput(executor, nullptr, 8), OK);
    ASSERT_EQ(NnopbaseAddOutput(executor, nullptr, 9), OK);

    size_t workspaceLen = 0U;
    void *stream = nullptr;
    for (size_t i = 0; i < 3; i++) {
        ASSERT_EQ(NnopbaseRunForWorkspace(executor, &workspaceLen), OK);

        void *workspace = nullptr;
        if (workspaceLen > 0U) {
            workspace = (void *) malloc(workspaceLen);
        }

        ASSERT_EQ(NnopbaseRunWithWorkspace(executor, stream, workspace, workspaceLen), OK);
        if (workspaceLen > 0U) {
            free(workspace);
        }
    }

    NnopbaseExecutorGcSpace(executorSpace);
    aclDestroyTensor(tensor);
    NnopbaseUnsetEnvAndClearFolder();
    op::internal::opProfilingSwitch.recordOpArgFlag = true;
}
#endif

TEST_F(NnopbaseExtUnitTest, TestOutputShapeTensorWithOOM)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char* opType = "bninference_d_kernel";
    char inputDesc[] = {1, 2, 0, 0, 1};
    char outputDesc[] = {1, 0};
    char attrDesc[] = {};
    void* executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                         sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    ASSERT_NE(executor, nullptr);

    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    aclTensor* out = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                     aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);

    std::vector<const aclTensor*> tensor_list_a;
    tensor_list_a.push_back(tensor);
    tensor_list_a.push_back(out);
    aclTensorList* aclTensorTestList = aclCreateTensorList(tensor_list_a.data(), tensor_list_a.size());
    ASSERT_EQ(NnopbaseAddInput(executor, tensor, 0), OK);
    ASSERT_EQ(NnopbaseAddDynamicInput(executor, aclTensorTestList, 1), OK);
    ASSERT_EQ(NnopbaseAddInput(executor, nullptr, 2), OK);
    ASSERT_EQ(NnopbaseAddInput(executor, nullptr, 3), OK);
    ASSERT_EQ(NnopbaseAddInput(executor, tensor, 4), OK);
    ASSERT_EQ(NnopbaseAddOutputShapeDependTensor(executor, out, 0), OK);
    ASSERT_EQ(NnopbaseAddOutput(executor, tensor, 1), OK);

    size_t workspaceLen = 0U;
    ASSERT_EQ(NnopbaseRunForWorkspace(executor, &workspaceLen), OK);
#if 0 // comment out for ge decoupling
    uint64_t arr[] = {32, 120, 0, 0, 32, 32, 32, 96, 0, 5, 1, 1, 1, 1, 1, 5, 1, 1, 1, 1, 1, 5, 1, 1, 1, 1, 1};
    for (size_t i = 0U; i < ((NnopbaseExecutor *)executor)->args->dfxInfo.size(); ++i) {
        ASSERT_EQ(((NnopbaseExecutor *)executor)->args->dfxInfo[i], arr[i]);
    }
#endif
    void* stream = nullptr;
    void* workspace = nullptr;
    if (workspaceLen > 0U) {
        workspace = (void*)malloc(workspaceLen);
    }

    ASSERT_EQ(NnopbaseRunWithWorkspace(executor, stream, workspace, workspaceLen), OK);

    const size_t tilingDataSize = ((NnopbaseTilingData*)(((NnopbaseExecutor*)executor)->args->tilingInfo.tilingData))
                                      ->GetDataSize();
    const size_t alignTilingDataSize = ((tilingDataSize % 8U) != 0) ? (tilingDataSize / 8U + 1U) * 8U :
                                                                      tilingDataSize; // 8byte对齐
    const size_t workspaceNum = ((NnopbaseExecutor*)executor)->workspaces.num == 0U ?
                                    1U :
                                    ((NnopbaseExecutor*)executor)->workspaces.num;
    const size_t offset = alignTilingDataSize +
                          (((NnopbaseExecutor*)executor)->args->inputs.paramDescs.count +
                           ((NnopbaseExecutor*)executor)->args->outputs.paramDescs.count + workspaceNum + 1U) *
                              sizeof(void*); // 1 is outputShape
    uint64_t* ptr = (uint64_t*)((NnopbaseUChar*)((NnopbaseTilingData*)(((NnopbaseExecutor*)executor)
                                                                           ->args->tilingInfo.tilingData))
                                    ->GetData() +
                                offset);
#if 0 // comment out for ge decoupling
    ASSERT_EQ(*ptr, 5525U);
#endif

    if (workspaceLen > 0U) {
        free(workspace);
    }
    NnopbaseExecutorGcSpace(executorSpace);
    aclDestroyTensorList((const aclTensorList*)aclTensorTestList);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseExtUnitTest, NnopBaseReportAttrAndHostInfo)
{
    op::internal::opProfilingSwitch.level2ProfilingFlag = true;
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char* opType = "bninference_d_kernel";
    char inputDesc[] = {1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    char outputDesc[] = {1};
    char attrDesc[] = {0, 0, 0, 0};

    void* executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                         sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    ASSERT_NE(executor, nullptr);

    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    ASSERT_EQ(NnopbaseAddInput(executor, tensor, 0), OK);
    ASSERT_EQ(NnopbaseAddInput(executor, tensor, 1), OK);

    aclOpExecutor opExecutor;
    aclTensor* tensors[2] = {nullptr};
    vector<float> vec(2, 2.0);
    for (auto& tensor : tensors) {
        tensor = opExecutor.AllocHostTensor(vec.data(), vec.size(), op::DataType::DT_FLOAT);
    }
    aclTensorList* tensorList = opExecutor.AllocTensorList(reinterpret_cast<const aclTensor**>(&tensors), 2);
    ASSERT_EQ(NnopbaseAddDynamicInput(executor, tensorList, 2), OK);

    vector<int32_t> vec1(2, -2);
    auto intTensor = opExecutor.AllocHostTensor(vec1.data(), vec1.size(), op::DataType::DT_INT32);
    ASSERT_EQ(NnopbaseAddInput(executor, intTensor, 3), OK);

    vector<op::fp16_t> vec2(2, 5.0);
    auto fp16Tensor = opExecutor.AllocHostTensor(vec2.data(), vec2.size(), op::DataType::DT_FLOAT16);
    ASSERT_EQ(NnopbaseAddInput(executor, fp16Tensor, 4), OK);

    vector<op::bfloat16> vec3(2, 3.0);
    auto bf16Tensor = opExecutor.AllocHostTensor(vec3.data(), vec3.size(), op::DataType::DT_BF16);
    ASSERT_EQ(NnopbaseAddInput(executor, bf16Tensor, 5), OK);

    vector<double> vec4(2, 3.0);
    auto doubleTensor = opExecutor.AllocHostTensor(vec4.data(), vec4.size(), op::DataType::DT_DOUBLE);
    ASSERT_EQ(NnopbaseAddInput(executor, doubleTensor, 6), OK);

    bool vec5[] = {true, false, true};
    auto boolTensor = opExecutor.AllocHostTensor(vec5, 3, op::DataType::DT_BOOL);
    ASSERT_EQ(NnopbaseAddInput(executor, boolTensor, 7), OK);

    vector<int64_t> vec6(2, 8);
    auto int64Tensor = opExecutor.AllocHostTensor(vec6.data(), vec6.size(), op::DataType::DT_INT64);
    ASSERT_EQ(NnopbaseAddInput(executor, int64Tensor, 8), OK);

    vector<uint64_t> vec7(2, 3);
    auto uint64Tensor = opExecutor.AllocHostTensor(vec7.data(), vec7.size(), op::DataType::DT_UINT64);
    ASSERT_EQ(NnopbaseAddInput(executor, uint64Tensor, 9), OK);

    auto floatArray = opExecutor.AllocFloatArray(vec.data(), vec.size());
    auto boolArray = opExecutor.AllocBoolArray(vec5, 3);
    auto intArray = opExecutor.AllocIntArray(vec6.data(), vec6.size());
    ASSERT_EQ(NnopbaseAddFloatArrayInput(executor, floatArray, 10), OK);
    ASSERT_EQ(NnopbaseAddBoolArrayInput(executor, boolArray, 11), OK);
    ASSERT_EQ(NnopbaseAddIntArrayInput(executor, intArray, 12), OK);

    auto stringTensor = opExecutor.AllocHostTensor(vec7.data(), vec7.size(), op::DataType::DT_UINT64);
    ASSERT_EQ(NnopbaseAddInput(executor, stringTensor, 13), OK);
    ((NnopbaseExecutor*)executor)->ownArgs.inputs.extTensors[14].rt2Tensor.SetDataType(op::DataType::DT_COMPLEX64);

    vector<int8_t> vec8(2, 1);
    auto int8Tensor = opExecutor.AllocHostTensor(vec8.data(), vec8.size(), op::DataType::DT_INT8);
    ASSERT_EQ(NnopbaseAddInput(executor, int8Tensor, 14), OK);

    vector<uint8_t> vec9(2, 2);
    auto uint8Tensor = opExecutor.AllocHostTensor(vec9.data(), vec9.size(), op::DataType::DT_UINT8);
    ASSERT_EQ(NnopbaseAddInput(executor, uint8Tensor, 15), OK);

    vector<uint16_t> vec10(2, 4);
    auto uint16Tensor = opExecutor.AllocHostTensor(vec10.data(), vec10.size(), op::DataType::DT_UINT16);
    ASSERT_EQ(NnopbaseAddInput(executor, uint16Tensor, 16), OK);

    vector<int16_t> vec11(2, 7);
    auto int16Tensor = opExecutor.AllocHostTensor(vec11.data(), vec11.size(), op::DataType::DT_INT16);
    ASSERT_EQ(NnopbaseAddInput(executor, int16Tensor, 17), OK);

    ASSERT_EQ(NnopbaseAddInput(executor, nullptr, 18), OK);

    vector<uint32_t> vec12(2, 9);
    auto uint32Tensor = opExecutor.AllocHostTensor(vec12.data(), vec12.size(), op::DataType::DT_UINT32);
    ASSERT_EQ(NnopbaseAddInput(executor, uint32Tensor, 19), OK);

    ASSERT_EQ(NnopbaseAddOutput(executor, tensor, 0), OK);

    NnopbaseAddFloatArrayAttr(executor, floatArray, 0);
    NnopbaseAddIntArrayAttr(executor, intArray, 1);
    NnopbaseAddBoolArrayAttr(executor, boolArray, 2);
    char* bias = "ssss";
    NnopbaseAddAttrWithDtype(executor, static_cast<void*>(bias), strlen(bias) + 1, 3, kNnopbaseString);

    size_t workspaceLen = 0U;
    ASSERT_EQ(NnopbaseRunForWorkspace(executor, &workspaceLen), OK);

    std::string expectAttrStr = "attr_0:2.000000,2.000000|attr_1:8,8|attr_2:1,0,1|attr_3:ssss";
    std::string expectInputStr =
        "input_2_0:2.000000,2.000000|input_2_1:2.000000,2.000000|input_3:-2,-2|input_4:5.000000,5.000000|input_5:3."
        "000000,3.000000|input_6:3.000000,3.000000|input_7:1,0,1|input_8:8,8|input_9:3,3|input_10:2.000000,2.000000|"
        "input_11:1,0,1|input_12:8,8|input_13:0x03 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x03 0x00 0x00 0x00 0x00 0x00 "
        "0x00 0x00|input_14:1,1|input_15:2,2|input_16:4,4|input_17:7,7|input_19:9,9";
    std::string attrStr = NnopbaseGetAttrVal(((NnopbaseExecutor*)executor)->attrs);
    std::string inputStr = NnopbaseGetHostInfoStr(((NnopbaseExecutor*)executor)->args->inputs);
    ASSERT_EQ(expectAttrStr, attrStr);
    ASSERT_EQ(expectInputStr, inputStr);

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
    NnopbaseUnsetEnvAndClearFolder();
    op::internal::opProfilingSwitch.level2ProfilingFlag = false;
}

TEST_F(NnopbaseExtUnitTest, NnopBaseReportAttrRepeat)
{
    op::internal::opProfilingSwitch.recordOpArgFlag = false;
    op::internal::opProfilingSwitch.level2ProfilingFlag = true;
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char* opType = "bninference_d_kernel";
    char inputDesc[] = {1, 1, 1};
    char outputDesc[] = {1};
    char attrDesc[] = {0, 0, 0, 0, 0};

    void* executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                         sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    ASSERT_NE(executor, nullptr);

    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    int64_t a = 2;
    float b = 0.244;
    size_t workspaceLen = 0U;
    void* workspace = nullptr;
    void* stream = nullptr;
    for (size_t i = 0; i < 3; i++) {
        NnopbaseAddInput(executor, tensor, 0);
        NnopbaseAddInput(executor, tensor, 1);
        NnopbaseAddInput(executor, tensor, 2);
        NnopbaseAddOutput(executor, tensor, 0);
        NnopbaseAddAttrWithDtype(executor, static_cast<void*>(&a), sizeof(int64_t), 0, kNnopbaseInt);
        NnopbaseAddAttrWithDtype(executor, static_cast<void*>(&a), sizeof(int64_t), 1, kNnopbaseInt);
        NnopbaseAddAttrWithDtype(executor, static_cast<void*>(&b), sizeof(float), 2, kNnopbaseFloat);
        NnopbaseAddAttrWithDtype(executor, static_cast<void*>(&b), sizeof(float), 3, kNnopbaseFloat);
        NnopbaseAddAttrWithDtype(executor, static_cast<void*>(&a), sizeof(int64_t), 4, kNnopbaseInt);
        ASSERT_EQ(NnopbaseRunForWorkspace(executor, &workspaceLen), OK);
        if (workspaceLen > 0U) {
            workspace = (void*)malloc(workspaceLen);
        }
        ASSERT_EQ(NnopbaseRunWithWorkspace(executor, stream, workspace, workspaceLen), OK);

        if (workspaceLen > 0U) {
            free(workspace);
        }
    }

    NnopbaseExecutorGcSpace(executorSpace);
    NnopbaseUnsetEnvAndClearFolder();
    op::internal::opProfilingSwitch.level2ProfilingFlag = false;
    op::internal::opProfilingSwitch.recordOpArgFlag = true;
}

TEST_F(NnopbaseExtUnitTest, TestConvert)
{
    gert::Tensor a_tensor = {{{1, 2, 3, 4}, {1, 2, 3, 4}},
                             {ge::FORMAT_ND, ge::FORMAT_ND, {}},
                             gert::kOnDeviceHbm,
                             ge::DT_FLOAT16,
                             (void*)0x123};
    aclTensor* a = NnopbaseConvertTensor(&a_tensor);
    ASSERT_NE(a, nullptr);

    gert::Tensor tensor = {
        {{1}, {1}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, gert::kOnDeviceHbm, ge::DT_FLOAT16, (void*)0x234};
    std::vector<const gert::Tensor*> list;
    list.push_back(&tensor);
    aclTensorList* tensorList = NnopbaseConvertTensorList(list);
    ASSERT_NE(tensorList, nullptr);

    void* ptr = malloc(100);
    gert::Tensor b_tensor = {{{1}, {1}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, gert::kOnHost, ge::DT_INT64, ptr};
    aclIntArray* b = NnopbaseCovertIntArray(&b_tensor);
    ASSERT_NE(b, nullptr);

    gert::Tensor c_tensor = {{{1}, {1}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, gert::kOnHost, ge::DT_FLOAT, ptr};
    aclFloatArray* c = NnopbaseCovertFloatArray(&c_tensor);
    ASSERT_NE(c, nullptr);

    gert::Tensor d_tensor = {{{1}, {1}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, gert::kOnHost, ge::DT_BOOL, ptr};
    aclBoolArray* d = NnopbaseCovertBoolArray(&d_tensor);
    ASSERT_NE(d, nullptr);

    gert::Tensor e_tensor = {{{1}, {1}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, gert::kOnHost, ge::DT_BOOL, ptr};
    aclScalar* e = NnopbaseConvertScalar(&e_tensor);
    ASSERT_NE(e, nullptr);

    gert::Tensor f_tensor = {{{1}, {1}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, gert::kOnHost, ge::DT_BOOL, ptr};
    aclScalarList* f = NnopbaseConvertScalarList(&f_tensor);
    ASSERT_NE(f, nullptr);

    auto vec_holder = gert::ContinuousVector::Create<size_t>(16);
    auto vec = reinterpret_cast<gert::ContinuousVector*>(vec_holder.get());
    vec->SetSize(16);
    auto int_vec = reinterpret_cast<const gert::TypedContinuousVector<int64_t>*>(vec);
    aclIntArray* intArr = NnopbaseCovertIntArrayAttr(int_vec);
    ASSERT_NE(intArr, nullptr);

    auto bool_vec = reinterpret_cast<const gert::TypedContinuousVector<bool>*>(vec);
    aclBoolArray* boolArr = NnopbaseCovertBoolArrayAttr(bool_vec);
    ASSERT_NE(boolArr, nullptr);

    auto float_vec = reinterpret_cast<const gert::TypedContinuousVector<float>*>(vec);
    aclFloatArray* floatArr = NnopbaseCovertFloatArrayAttr(float_vec);
    ASSERT_NE(floatArr, nullptr);

    NnopbaseDestroyTensor(a);
    NnopbaseDestroyTensorList(tensorList);
    NnopbaseDestroyIntArray(b);
    NnopbaseDestroyBoolArray(d);
    NnopbaseDestroyFloatArray(c);
    NnopbaseDestroyScalar(e);
    NnopbaseDestroyScalarList(f);
    NnopbaseDestroyIntArray(intArr);
    NnopbaseDestroyBoolArray(boolArr);
    NnopbaseDestroyFloatArray(floatArr);
    free(ptr);
}

aclnnStatus aclnnFallBackGetWorkspaceSize(const aclTensorList* list, uint64_t* num, aclOpExecutor** exe) { return OK; }

aclnnStatus aclnnFallBack(void* workspace, uint64_t workspaceSize, aclOpExecutor* exe, aclrtStream stream)
{
    return OK;
}

class ApiFuncStub : public Adx::MmpaStub {
public:
    void* mmDlsym(void* handle, const char* funcName)
    {
        if (std::string(funcName) == "aclnnFallBackGetWorkspaceSize") {
            return (void*)aclnnFallBackGetWorkspaceSize;
        } else if (std::string(funcName) == "aclnnFallBack") {
            return (void*)aclnnFallBack;
        } else {
            return nullptr;
        }
    }
};

TEST_F(NnopbaseExtUnitTest, TestGetOpApiFunc)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);
    ApiFuncStub mmpaStub;
    Adx::MmpaStub::GetInstance()->Install(&mmpaStub);
    ASSERT_EQ(NnopbaseGetOpApiFunc("aclnnFallBackGetWorkspaceSize"), (void*)aclnnFallBackGetWorkspaceSize);
    ASSERT_EQ(NnopbaseGetOpApiFunc("aclnnFallBack"), (void*)aclnnFallBack);
    // // 重复执行命中map
    ASSERT_EQ(NnopbaseGetOpApiFunc("aclnnFallBackGetWorkspaceSize"), (void*)aclnnFallBackGetWorkspaceSize);
    ASSERT_EQ(NnopbaseGetOpApiFunc("aclnnFallBack"), (void*)aclnnFallBack);
    ASSERT_EQ(NnopbaseGetOpApiFunc("aclnnXxx"), nullptr);
    Adx::MmpaStub::GetInstance()->UnInstall();
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseExtUnitTest, NnopBaseMC2RunSuccessForDavidWithHostInput)
{
    setenv("ASCEND_C", ASCEND_DAVID_SOC_VERSION, 1);
    Adx::MmpaStub::GetInstance()->Install((Adx::MmpaStub*)&mmpaNormallStub);
    ASSERT_EQ(nnopbase::IndvHcclWrapper::GetInstance().IndvHcclWrapperInit("libhccl.so", true), OK);
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char* opType = "bninference_d_kernel";
    char inputDesc[] = {1, 1, 1, 0};
    char outputDesc[] = {1, 0};
    char attrDesc[] = {1};

    void* executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                         sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    ASSERT_NE(executor, nullptr);

    int* ptr = new int;
    void* addr = reinterpret_cast<void*>(ptr);
    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), addr);
    std::vector<float> vec(2048, 1.0);
    auto* floatArray = aclCreateFloatArray(vec.data(), vec.size());
    auto oriSocVersion = nnopbase::IndvSoc::GetInstance().GetCurSocVersion();
    struct SocVersionGuard {
        std::string originalVersion;
        ~SocVersionGuard() { SetSocVersion(originalVersion); }
    } socGuard{oriSocVersion};
    SetSocVersion(nnopbase::OPS_SUBPATH_ASCEND950);
    ASSERT_EQ(nnopbase::IndvSoc::GetInstance().GetCurSocVersion(), nnopbase::OPS_SUBPATH_ASCEND950);
    ASSERT_EQ(NnopbaseSetMc2(executor), OK);
    ASSERT_EQ(NnopbaseAddInput(executor, tensor, 0), OK);
    ASSERT_EQ(NnopbaseAddFloatArrayInput(executor, floatArray, 1), OK);
    ASSERT_EQ(NnopbaseAddInput(executor, tensor, 2), OK);
    ASSERT_EQ(NnopbaseAddInput(executor, nullptr, 3), OK);
    ASSERT_EQ(NnopbaseAddOutput(executor, tensor, 0), OK);
    ASSERT_EQ(NnopbaseAddOutput(executor, nullptr, 1), OK);

    char* group = "123";
    ASSERT_EQ(NnopbaseAddAttr(executor, static_cast<void*>(group), strlen(group) + 1, 0), OK);
    ASSERT_EQ(NnopbaseSetHcomGroup(executor, group), OK);

    auto oriMc2FusionLaunchFlag = ((NnopbaseExecutor*)executor)->collector->isMc2FusionLaunch;
    ((NnopbaseExecutor*)executor)->collector->isMc2FusionLaunch = true;
    size_t workspaceLen = 0U;
    ASSERT_EQ(NnopbaseRunForWorkspace(executor, &workspaceLen), OK);
    void* stream = nullptr;
    void* workspace = nullptr;
    if (workspaceLen > 0U) {
        workspace = (void*)malloc(workspaceLen);
    }

    ASSERT_EQ(NnopbaseRunWithWorkspace(executor, stream, workspace, workspaceLen), OK);
    // SOC version will be restored by SocVersionGuard destructor
    ((NnopbaseExecutor*)executor)->collector->isMc2FusionLaunch = oriMc2FusionLaunchFlag;

    const auto* desc = GetMc2ParamDesc(static_cast<NnopbaseExecutor*>(executor));
    ASSERT_NE(desc, nullptr);
    ASSERT_EQ(desc->groupNum, 1U);
    ASSERT_EQ(desc->version, 1U);
    ASSERT_EQ(desc->hasFfts, 0U);
#if 0 // comment out for ge decoupling
    ASSERT_EQ(desc->tilingOff, 8U);
#endif
    ASSERT_EQ(desc->isDyn, 0U);

    const size_t paramSize = (((NnopbaseExecutor*)executor)->args->inputs.paramDescs.count +
                              ((NnopbaseExecutor*)executor)->args->outputs.paramDescs.count +
                              (desc->groupNum + 1)) * // 1是workspace
                             sizeof(void*);
    const size_t tilingDataSize = ((NnopbaseTilingData*)(((NnopbaseExecutor*)executor)->args->tilingInfo.tilingData))
                                      ->GetDataSize();
    const size_t alignTilingDataSize = ((tilingDataSize % 8U) != 0) ? (tilingDataSize / 8U + 1U) * 8U :
                                                                      tilingDataSize; // 8byte对齐
    const size_t hostInfoOffset = ((NnopbaseExecutor*)executor)->args->binInfo->oomFlag ?
                                      alignTilingDataSize + 8 + paramSize :
                                      tilingDataSize + 8; // 8字节是atomicIndex
    NnopbaseUChar* tilingdata = (NnopbaseUChar*)(((NnopbaseTilingData*)(((NnopbaseExecutor*)executor)
                                                                            ->args->tilingInfo.tilingData))
                                                     ->GetData());
    rtHostInputInfo_t* hostInputInfo = (rtHostInputInfo_t*)(tilingdata + hostInfoOffset);

#if 0 // comment out for ge decoupling
    ASSERT_EQ(hostInputInfo->addrOffset, 64U);
    ASSERT_EQ(hostInputInfo->dataOffset, 88U);
#endif

    NnopbaseExecutorGcSpace(executorSpace);
    if (workspaceLen > 0U) {
        free(workspace);
    }
    delete ptr;
    aclDestroyFloatArray(floatArray);
    aclDestroyTensor(tensor);
    NnopbaseUnsetEnvAndClearFolder();
    Adx::MmpaStub::GetInstance()->UnInstall();
}

TEST_F(NnopbaseExtUnitTest, NnopBaseMC2RunSuccessForDavidWithDynamicInput)
{
    setenv("ASCEND_C", ASCEND_DAVID_SOC_VERSION, 1);
    Adx::MmpaStub::GetInstance()->Install((Adx::MmpaStub*)&mmpaNormallStub);
    ASSERT_EQ(nnopbase::IndvHcclWrapper::GetInstance().IndvHcclWrapperInit("libhccl.so", true), OK);
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char* opType = "bninference_d_kernel";
    char inputDesc[] = {1, 2, 1, 0};
    char outputDesc[] = {2, 0};
    char attrDesc[] = {1};

    void* executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                         sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    ASSERT_NE(executor, nullptr);

    int* ptr = new int;
    void* addr = reinterpret_cast<void*>(ptr);
    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), addr);
    aclTensor* tensor1 = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                         aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), addr);
    std::vector<const aclTensor*> tensor_list_a;
    tensor_list_a.push_back(tensor);
    tensor_list_a.push_back(tensor1);
    aclTensorList* tensorList = aclCreateTensorList(tensor_list_a.data(), tensor_list_a.size());
    auto oriSocVersion = nnopbase::IndvSoc::GetInstance().GetCurSocVersion();
    struct SocVersionGuard {
        std::string originalVersion;
        ~SocVersionGuard() { SetSocVersion(originalVersion); }
    } socGuard{oriSocVersion};
    SetSocVersion(nnopbase::OPS_SUBPATH_ASCEND950);
    ASSERT_EQ(NnopbaseSetMc2(executor), OK);
    ASSERT_EQ(NnopbaseAddInput(executor, tensor, 0), OK);
    ASSERT_EQ(NnopbaseAddDynamicInput(executor, tensorList, 1), OK);
    ASSERT_EQ(NnopbaseAddInput(executor, tensor, 2), OK);
    ASSERT_EQ(NnopbaseAddInput(executor, nullptr, 3), OK);
    ASSERT_EQ(NnopbaseAddDynamicOutput(executor, tensorList, 0), OK);
    ASSERT_EQ(NnopbaseAddOutput(executor, nullptr, 1), OK);

    char* group = "123";
    ASSERT_EQ(NnopbaseAddAttr(executor, static_cast<void*>(group), strlen(group) + 1, 0), OK);
    ASSERT_EQ(NnopbaseSetHcomGroup(executor, group), OK);

    auto oriMc2FusionLaunchFlag = ((NnopbaseExecutor*)executor)->collector->isMc2FusionLaunch;
    ((NnopbaseExecutor*)executor)->collector->isMc2FusionLaunch = true;
    size_t workspaceLen = 0U;
    ASSERT_EQ(NnopbaseRunForWorkspace(executor, &workspaceLen), OK);
    void* stream = nullptr;
    void* workspace = nullptr;
    if (workspaceLen > 0U) {
        workspace = (void*)malloc(workspaceLen);
    }

    ASSERT_EQ(NnopbaseRunWithWorkspace(executor, stream, workspace, workspaceLen), OK);
    // SOC version will be restored by SocVersionGuard destructor
    ((NnopbaseExecutor*)executor)->collector->isMc2FusionLaunch = oriMc2FusionLaunchFlag;

    const auto* desc = GetMc2ParamDesc(static_cast<NnopbaseExecutor*>(executor));
    ASSERT_NE(desc, nullptr);
    ASSERT_EQ(desc->groupNum, 1U);
    ASSERT_EQ(desc->version, 1U);
    ASSERT_EQ(desc->hasFfts, 0U);
#if 0 // comment out for ge decoupling
    ASSERT_EQ(desc->tilingOff, 8U);
#endif
    ASSERT_EQ(desc->isDyn, 18U);

    const size_t paramSize = (((NnopbaseExecutor*)executor)->args->inputs.paramDescs.count +
                              ((NnopbaseExecutor*)executor)->args->outputs.paramDescs.count +
                              (desc->groupNum + 1)) * // 1是workspace
                             sizeof(void*);
    const size_t tilingDataSize = ((NnopbaseTilingData*)(((NnopbaseExecutor*)executor)->args->tilingInfo.tilingData))
                                      ->GetDataSize();
    const size_t alignTilingDataSize = ((tilingDataSize % 8U) != 0) ? (tilingDataSize / 8U + 1U) * 8U :
                                                                      tilingDataSize; // 8byte对齐
    const size_t hostInfoOffset = ((NnopbaseExecutor*)executor)->args->binInfo->oomFlag ?
                                      alignTilingDataSize + 8 + paramSize :
                                      tilingDataSize + 8; // 8字节是atomicIndex
    NnopbaseUChar* tilingdata = (NnopbaseUChar*)(((NnopbaseTilingData*)(((NnopbaseExecutor*)executor)
                                                                            ->args->tilingInfo.tilingData))
                                                     ->GetData());
    rtHostInputInfo_t* hostInputInfo = (rtHostInputInfo_t*)(tilingdata + hostInfoOffset);
#if 0 // comment out for ge decoupling
    ASSERT_EQ(hostInputInfo->addrOffset, 64U);
    ASSERT_EQ(hostInputInfo->dataOffset, 88U);
#endif

    NnopbaseExecutorGcSpace(executorSpace);
    if (workspaceLen > 0U) {
        free(workspace);
    }
    delete ptr;
    aclDestroyTensorList((const aclTensorList*)tensorList);
    NnopbaseUnsetEnvAndClearFolder();
    Adx::MmpaStub::GetInstance()->UnInstall();
}

void NnopbaseExtUnitTest::TestHcclServerType(std::function<void(void*)> setHcclServerTypeFunc, const char* socVersion)
{
    MmpaNormalStubGuard mmpaGuard;
    // RAII guard to ensure SOC version is always restored, even if test fails
    auto oriSocVersion = nnopbase::IndvSoc::GetInstance().GetCurSocVersion();
    struct SocVersionGuard {
        std::string originalVersion;
        ~SocVersionGuard() { SetSocVersion(originalVersion); }
    } socGuard{oriSocVersion};

    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char* opType = "bninference_d_kernel";
    char inputDesc[] = {1, 2, 1, 0};
    char outputDesc[] = {2, 0};
    char attrDesc[] = {1};

    void* executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                         sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    ASSERT_NE(executor, nullptr);

    auto oriMc2FusionLaunchFlag = ((NnopbaseExecutor*)executor)->collector->isMc2FusionLaunch;
    SetSocVersion(socVersion);
    if ((socVersion == nnopbase::OPS_SUBPATH_ASCEND950 || socVersion == nnopbase::OPS_SUBPATH_ASCEND910_96)) {
        ((NnopbaseExecutor*)executor)->collector->isMc2FusionLaunch = true;
    } else {
        ((NnopbaseExecutor*)executor)->collector->isMc2FusionLaunch = false;
    }

    int* ptr = new int;
    void* addr = reinterpret_cast<void*>(ptr);
    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), addr);
    aclTensor* tensor1 = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                         aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), addr);
    std::vector<const aclTensor*> tensor_list_a;
    tensor_list_a.push_back(tensor);
    tensor_list_a.push_back(tensor1);
    aclTensorList* tensorList = aclCreateTensorList(tensor_list_a.data(), tensor_list_a.size());
    ASSERT_EQ(NnopbaseSetMc2(executor), OK);
    ASSERT_EQ(NnopbaseAddInput(executor, tensor, 0), OK);
    ASSERT_EQ(NnopbaseAddDynamicInput(executor, tensorList, 1), OK);
    ASSERT_EQ(NnopbaseAddInput(executor, tensor, 2), OK);
    ASSERT_EQ(NnopbaseAddInput(executor, nullptr, 3), OK);
    ASSERT_EQ(NnopbaseAddDynamicOutput(executor, tensorList, 0), OK);
    ASSERT_EQ(NnopbaseAddOutput(executor, nullptr, 1), OK);

    char* group = "123";
    ASSERT_EQ(NnopbaseAddAttr(executor, static_cast<void*>(group), strlen(group) + 1, 0), OK);
    ASSERT_EQ(NnopbaseSetHcomGroup(executor, group), OK);
    setHcclServerTypeFunc(executor);

    size_t workspaceLen = 0U;
    ASSERT_EQ(NnopbaseRunForWorkspace(executor, &workspaceLen), OK);
    void* stream = nullptr;
    void* workspace = nullptr;
    if (workspaceLen > 0U) {
        workspace = (void*)malloc(workspaceLen);
    }

    ASSERT_EQ(NnopbaseRunWithWorkspace(executor, stream, workspace, workspaceLen), OK);
    // SOC version will be restored by SocVersionGuard destructor
    ((NnopbaseExecutor*)executor)->collector->isMc2FusionLaunch = oriMc2FusionLaunchFlag;

    NnopbaseExecutorGcSpace(executorSpace);
    if (workspaceLen > 0U) {
        free(workspace);
    }
    delete ptr;
    aclDestroyTensorList((const aclTensorList*)tensorList);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseExtUnitTest, NnopBaseMC2RunSuccessWithServerTypeDefault)
{
    TestHcclServerType([](void* executor) {}, nnopbase::OPS_SUBPATH_ASCEND910);
}

TEST_F(NnopbaseExtUnitTest, NnopBaseMC2RunSuccessWithServerTypeMTE)
{
    TestHcclServerType([](void* executor) { NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE); },
                       nnopbase::OPS_SUBPATH_ASCEND910);
}

TEST_F(NnopbaseExtUnitTest, NnopBaseMC2RunSuccessWithServerTypeACIPUWithEmptyOut)
{
    // is out empty
    TestHcclServerType(
        [](void* executor) {
            ((NnopbaseExecutor*)executor)->repeatFlag = true;
            ((NnopbaseExecutor*)executor)->isOutEmpty = true;
            NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_AICPU);
        },
        nnopbase::OPS_SUBPATH_ASCEND910);
}

TEST_F(NnopbaseExtUnitTest, NnopBaseMC2RunSuccessWithServerTypeACIPUWithNoEmptyOut)
{
    // is out empty
    TestHcclServerType(
        [](void* executor) {
            ((NnopbaseExecutor*)executor)->repeatFlag = false;
            NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_AICPU);
        },
        nnopbase::OPS_SUBPATH_ASCEND910);
}

TEST_F(NnopbaseExtUnitTest, NnopBaseMC2RunSuccessForDavidWithServerTypDefault)
{
    TestHcclServerType([](void* executor) {}, nnopbase::OPS_SUBPATH_ASCEND950);
}

TEST_F(NnopbaseExtUnitTest, NnopBaseMC2RunSuccessForDavidWithServerTypeCCU)
{
    TestHcclServerType([](void* executor) { NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_CCU); },
                       nnopbase::OPS_SUBPATH_ASCEND950);
}

TEST_F(NnopbaseExtUnitTest, NnopBaseRunSuccessWithNotEnableCache)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    std::vector<const aclTensor*> tensor_list_a;
    tensor_list_a.push_back(tensor);
    aclTensorList* aclTensorTestList = aclCreateTensorList(tensor_list_a.data(), tensor_list_a.size());

    size_t workspaceLen = 0U;
    void* workspace = nullptr;
    void* stream = nullptr;
    static void* executorSpace = nullptr;
    void* executor = nullptr;

    for (size_t i = 0; i < 2; i++) {
        ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

        const char* opType = "bninference_d_kernel";
        char inputDesc[] = {1, 1, 1, 0};
        char outputDesc[] = {2};
        char attrDesc[] = {};

        executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                       sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
        ASSERT_NE(executor, nullptr);
        NnopbaseSetMatchArgsFlag(executor);
        ASSERT_EQ(NnopbaseAddInput(executor, tensor, 0), 0);
        ASSERT_EQ(NnopbaseAddInput(executor, tensor, 1), 0);
        ASSERT_EQ(NnopbaseAddIgnoreContinuesInput(executor, tensor, 2), 0);
        if (i == 0) {
            ASSERT_EQ(NnopbaseAddInput(executor, tensor, 3), 0);
        } else {
            ASSERT_EQ(NnopbaseAddInput(executor, nullptr, 3), 0);
        }
        ASSERT_EQ(NnopbaseAddDynamicOutput(executor, aclTensorTestList, 0), 0);
        ASSERT_EQ(NnopbaseMatchArgs(executor, &workspaceLen), false);
        ASSERT_EQ(NnopbaseRunForWorkspace(executor, &workspaceLen), OK);
        if ((workspace == nullptr) && (workspaceLen > 0U)) {
            workspace = (void*)malloc(workspaceLen);
        }
        ASSERT_EQ(NnopbaseRunWithWorkspace(executor, stream, workspace, workspaceLen), OK);
    }
    if (workspaceLen > 0U) {
        free(workspace);
    }
    aclDestroyTensorList(aclTensorTestList);
    NnopbaseExecutorGcSpace(executorSpace);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseExtUnitTest, NnopBaseRunNewCacheSuccess)
{
    op::internal::opProfilingSwitch.recordOpArgFlag = false;
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    int* ptr = new int;
    void* addr = reinterpret_cast<void*>(ptr);
    aclTensor* tensor1 = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                         aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), addr);

    std::vector<const aclTensor*> tensor_list_a;
    for (uint32_t i = 0; i < 3; i++) {
        aclTensor* tmp = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                         aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
        tensor_list_a.push_back(tmp);
    }
    aclTensorList* tensorList1 = aclCreateTensorList(tensor_list_a.data(), tensor_list_a.size());

    std::vector<const aclTensor*> tensor_list_b;
    for (uint32_t i = 0; i < 3; i++) {
        aclTensor* tmp = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                         aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), addr);
        tensor_list_b.push_back(tmp);
    }
    aclTensorList* tensorList2 = aclCreateTensorList(tensor_list_b.data(), tensor_list_b.size());

    std::vector<const aclTensor*> tensor_list_c;
    tensor_list_c.push_back(tensor);
    tensor_list_c.push_back(tensor1);
    aclTensorList* tensorList3 = aclCreateTensorList(tensor_list_c.data(), tensor_list_c.size());

    std::vector<float> vec(2048 * 4, 2.0);
    auto* floatArray = aclCreateFloatArray(vec.data(), vec.size());
    int64_t intValues[] = {3, 4, 5};
    auto* intArray = aclCreateIntArray(intValues, sizeof(intValues) / sizeof(intValues[0]));
    bool boolValues[] = {true, false, true};
    auto* boolArray = aclCreateBoolArray(boolValues, sizeof(boolValues) / sizeof(boolValues[0]));

    float scalar_value = 5.0;
    auto* scalar = aclCreateScalar(&scalar_value, aclDataType::ACL_FLOAT);
    auto scalarList = aclCreateScalarList(&scalar, 1);

    int64_t a = 2;
    char* str = "test";
    size_t workspaceLen = 0U;
    void* workspace = nullptr;
    void* stream = nullptr;
    static void* executorSpace = nullptr;
    void* executor = nullptr;

    for (size_t i = 0; i < 4; i++) {
        ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

        const char* opType = "bninference_d_kernel";
        char inputDesc[] = {1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        char outputDesc[] = {2, 1, 0};
        char attrDesc[] = {1, 0};

        executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                       sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
        ASSERT_NE(executor, nullptr);
        NnopbaseSetMatchArgsFlag(executor);
        if (i == 0) {
            ASSERT_EQ(NnopbaseAddInput(executor, tensor1, 0), 0);
            ASSERT_EQ(NnopbaseAddInput(executor, tensor1, 1), 0);
            ASSERT_EQ(NnopbaseAddFloatArrayInput(executor, floatArray, 2), OK);
            ASSERT_EQ(NnopbaseAddIntArrayInput(executor, intArray, 3), OK);
            ASSERT_EQ(NnopbaseAddBoolArrayInput(executor, boolArray, 4), OK);
            ASSERT_EQ(NnopbaseAddFloatArrayInput(executor, nullptr, 5), OK);
            ASSERT_EQ(NnopbaseAddIntArrayInput(executor, nullptr, 6), OK);
            ASSERT_EQ(NnopbaseAddBoolArrayInput(executor, nullptr, 7), OK);
            ASSERT_EQ(NnopbaseAddIgnoreContinuesInput(executor, tensor, 8), 0);
            ASSERT_EQ(NnopbaseAddScalarInput(executor, scalar, 9, -1, ge::DT_UNDEFINED), 0);
            ASSERT_EQ(NnopbaseAddScalarListInput(executor, scalarList, 10, -1, ge::DT_UNDEFINED), 0);
            ASSERT_EQ(NnopbaseAddDynamicOutput(executor, tensorList2, 0), 0);
            ASSERT_EQ(NnopbaseAddOutput(executor, tensor1, 1), 0);
        } else if (i == 2) {
            ASSERT_EQ(NnopbaseAddInput(executor, tensor, 0), 0);
            ASSERT_EQ(NnopbaseAddInput(executor, tensor, 1), 0);
            ASSERT_EQ(NnopbaseAddFloatArrayInput(executor, floatArray, 2), OK);
            ASSERT_EQ(NnopbaseAddIntArrayInput(executor, intArray, 3), OK);
            ASSERT_EQ(NnopbaseAddBoolArrayInput(executor, boolArray, 4), OK);
            ASSERT_EQ(NnopbaseAddFloatArrayInput(executor, nullptr, 5), OK);
            ASSERT_EQ(NnopbaseAddIntArrayInput(executor, nullptr, 6), OK);
            ASSERT_EQ(NnopbaseAddBoolArrayInput(executor, nullptr, 7), OK);
            ASSERT_EQ(NnopbaseAddIgnoreContinuesInput(executor, tensor, 8), 0);
            ASSERT_EQ(NnopbaseAddScalarInput(executor, scalar, 9, -1, ge::DT_UNDEFINED), 0);
            ASSERT_EQ(NnopbaseAddScalarListInput(executor, scalarList, 10, -1, ge::DT_UNDEFINED), 0);
            ASSERT_EQ(NnopbaseAddDynamicOutput(executor, tensorList1, 0), 0);
            ASSERT_EQ(NnopbaseAddOutput(executor, tensor, 1), 0);
        } else {
            ASSERT_EQ(NnopbaseAddInput(executor, tensor, 0), 0);
            ASSERT_EQ(NnopbaseAddInput(executor, tensor, 1), 0);
            ASSERT_EQ(NnopbaseAddFloatArrayInput(executor, floatArray, 2), OK);
            ASSERT_EQ(NnopbaseAddIntArrayInput(executor, intArray, 3), OK);
            ASSERT_EQ(NnopbaseAddBoolArrayInput(executor, boolArray, 4), OK);
            ASSERT_EQ(NnopbaseAddFloatArrayInput(executor, nullptr, 5), OK);
            ASSERT_EQ(NnopbaseAddIntArrayInput(executor, nullptr, 6), OK);
            ASSERT_EQ(NnopbaseAddBoolArrayInput(executor, nullptr, 7), OK);
            ASSERT_EQ(NnopbaseAddIgnoreContinuesInput(executor, tensor, 8), 0);
            ASSERT_EQ(NnopbaseAddScalarInput(executor, scalar, 9, -1, ge::DT_UNDEFINED), 0);
            ASSERT_EQ(NnopbaseAddScalarListInput(executor, scalarList, 10, -1, ge::DT_UNDEFINED), 0);
            ASSERT_EQ(NnopbaseAddDynamicOutput(executor, tensorList3, 0), 0);
            ASSERT_EQ(NnopbaseAddOutput(executor, tensor, 1), 0);
        }
        ASSERT_EQ(NnopbaseAddOutput(executor, nullptr, 2), 0);
        ASSERT_EQ(NnopbaseAddAttrWithDtype(executor, static_cast<void*>(&a), sizeof(int64_t), 0, kNnopbaseInt), 0);
        ASSERT_EQ(NnopbaseAddAttrWithDtype(executor, static_cast<void*>(str), strlen(str) + 1, 1, kNnopbaseString), OK);
        bool flag = NnopbaseMatchArgs(executor, &workspaceLen);
        if ((i == 2) || (i == 3)) {
            ASSERT_EQ(flag, true);
        } else {
            ASSERT_EQ(flag, false);
        }

        if (!flag) {
            ASSERT_EQ(NnopbaseRunForWorkspace(executor, &workspaceLen), OK);
        }
        if (i == 3) {
            ASSERT_EQ(((NnopbaseExecutor*)executor)->args->outputs.extTensors[0].rt2Tensor.GetAddr(), nullptr);
            ASSERT_EQ(((NnopbaseExecutor*)executor)->args->outputs.extTensors[1].rt2Tensor.GetAddr(), addr);
        }
        if ((workspace == nullptr) && (workspaceLen > 0U)) {
            workspace = (void*)malloc(workspaceLen);
        }
        ASSERT_EQ(NnopbaseRunWithWorkspace(executor, stream, workspace, workspaceLen), OK);
    }
    if (workspaceLen > 0U) {
        free(workspace);
    }
    delete ptr;
    aclDestroyScalarList(scalarList);
    aclDestroyTensorList(tensorList1);
    aclDestroyTensorList(tensorList2);
    aclDestroyTensorList(tensorList3);
    aclDestroyFloatArray(floatArray);
    aclDestroyIntArray(intArray);
    aclDestroyBoolArray(boolArray);
    NnopbaseExecutorGcSpace(executorSpace);
    NnopbaseUnsetEnvAndClearFolder();
    op::internal::opProfilingSwitch.recordOpArgFlag = true;
}

TEST_F(NnopbaseExtUnitTest, NnopBaseRunSuccessWithDynamic)
{
    op::internal::opProfilingSwitch.recordOpArgFlag = false;
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    std::vector<int64_t> shape2 = {1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    std::vector<const aclTensor*> tensor_list1;
    for (size_t i = 0U; i < 2; i++) {
        aclTensor* tmp = aclCreateTensor(shape2.data(), shape2.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                         aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
        tensor_list1.push_back(tmp);
    }
    tensor_list1.push_back(tensor);
    for (size_t i = 0U; i < 2; i++) {
        aclTensor* tmp = aclCreateTensor(shape2.data(), shape2.size(), aclDataType::ACL_FLOAT16, nullptr, 0,
                                         aclFormat::ACL_FORMAT_ND, shape2.data(), shape2.size(), nullptr);
        tensor_list1.push_back(tmp);
    }
    for (size_t i = 0U; i < 2; i++) {
        aclTensor* tmp = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                         aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
        tensor_list1.push_back(tmp);
    }
    aclTensorList* tensorList1 = aclCreateTensorList(tensor_list1.data(), tensor_list1.size());

    int* ptr = new int;
    void* addr = reinterpret_cast<void*>(ptr);
    aclTensor* tensor1 = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                         aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), addr);
    std::vector<const aclTensor*> tensor_list2;
    for (size_t i = 0U; i < 2; i++) {
        aclTensor* tmp = aclCreateTensor(shape2.data(), shape2.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                         aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), addr);
        tensor_list2.push_back(tmp);
    }
    tensor_list2.push_back(tensor1);
    for (size_t i = 0U; i < 2; i++) {
        aclTensor* tmp = aclCreateTensor(shape2.data(), shape2.size(), aclDataType::ACL_FLOAT16, nullptr, 0,
                                         aclFormat::ACL_FORMAT_ND, shape2.data(), shape2.size(), addr);
        tensor_list2.push_back(tmp);
    }
    for (size_t i = 0U; i < 2; i++) {
        aclTensor* tmp = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                         aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), addr);
        tensor_list2.push_back(tmp);
    }
    aclTensorList* tensorList2 = aclCreateTensorList(tensor_list2.data(), tensor_list2.size());

    size_t workspaceLen = 0U;
    void* workspace = nullptr;
    void* stream = nullptr;
    static void* executorSpace = nullptr;
    void* executor = nullptr;

    for (size_t i = 0; i < 3; i++) {
        ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

        const char* opType = "bninference_d_kernel";
        char inputDesc[] = {1, 2, 1};
        char outputDesc[] = {1};
        char attrDesc[] = {};

        executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                       sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
        ASSERT_NE(executor, nullptr);
        NnopbaseSetMatchArgsFlag(executor);
        if (i == 0) {
            ASSERT_EQ(NnopbaseAddInput(executor, tensor1, 0), 0);
            ASSERT_EQ(NnopbaseAddDynamicInput(executor, tensorList2, 1), 0);
            ASSERT_EQ(NnopbaseAddInput(executor, tensor1, 2), 0);
            ASSERT_EQ(NnopbaseAddOutput(executor, tensor1, 0), 0);
        } else {
            ASSERT_EQ(NnopbaseAddInput(executor, tensor, 0), 0);
            ASSERT_EQ(NnopbaseAddDynamicInput(executor, tensorList1, 1), 0);
            ASSERT_EQ(NnopbaseAddInput(executor, tensor, 2), 0);
            ASSERT_EQ(NnopbaseAddOutput(executor, tensor, 0), 0);
        }
        bool flag = NnopbaseMatchArgs(executor, &workspaceLen);
        if (i == 0) {
            ASSERT_EQ(flag, false);
        } else {
            ASSERT_EQ(flag, true);
        }
        if (!flag) {
            ASSERT_EQ(NnopbaseRunForWorkspace(executor, &workspaceLen), OK);
        }
        if ((workspace == nullptr) && (workspaceLen > 0U)) {
            workspace = (void*)malloc(workspaceLen);
        }

        if (i == 0) {
            ASSERT_EQ(((NnopbaseExecutor*)executor)->args->inputs.extTensors[0].rt2Tensor.GetAddr(), addr);
            ASSERT_EQ(((NnopbaseExecutor*)executor)->args->inputs.extTensors[1].rt2Tensor.GetAddr(), addr);
            ASSERT_EQ(((NnopbaseExecutor*)executor)->args->outputs.extTensors[0].rt2Tensor.GetAddr(), addr);
        } else {
            ASSERT_EQ(((NnopbaseExecutor*)executor)->args->inputs.extTensors[0].rt2Tensor.GetAddr(), nullptr);
            ASSERT_EQ(((NnopbaseExecutor*)executor)->args->inputs.extTensors[1].rt2Tensor.GetAddr(), nullptr);
            ASSERT_EQ(((NnopbaseExecutor*)executor)->args->outputs.extTensors[0].rt2Tensor.GetAddr(), nullptr);
        }
        ASSERT_EQ(NnopbaseRunWithWorkspace(executor, stream, workspace, workspaceLen), OK);
    }
    if (workspaceLen > 0U) {
        free(workspace);
    }
    delete ptr;
    aclDestroyTensorList(tensorList1);
    aclDestroyTensorList(tensorList2);
    NnopbaseExecutorGcSpace(executorSpace);
    NnopbaseUnsetEnvAndClearFolder();
    op::internal::opProfilingSwitch.recordOpArgFlag = true;
}

TEST_F(NnopbaseExtUnitTest, NnopBaseRunSuccessWithUnContiguousTensor)
{
    op::internal::opProfilingSwitch.recordOpArgFlag = false;
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    std::vector<int64_t> shape = {1, 3, 1, 1, 1};
    std::vector<int64_t> shape1 = {1, 2, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, shape1.data(), 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    int* ptr = new int;
    void* addr = reinterpret_cast<void*>(ptr);
    aclTensor* tensor1 = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, shape1.data(), 0,
                                         aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), addr);

    std::vector<const aclTensor*> tensor_list;
    for (size_t i = 0U; i < 2; i++) {
        aclTensor* tmp = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, shape1.data(), 0,
                                         aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
        tensor_list.push_back(tmp);
    }
    aclTensorList* tensorList = aclCreateTensorList(tensor_list.data(), tensor_list.size());

    std::vector<const aclTensor*> tensor_list1;
    for (size_t i = 0U; i < 2; i++) {
        aclTensor* tmp = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, shape1.data(), 0,
                                         aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), addr);
        tensor_list1.push_back(tmp);
    }
    aclTensorList* tensorList1 = aclCreateTensorList(tensor_list1.data(), tensor_list1.size());

    size_t workspaceLen = 0U;
    void* workspace = nullptr;
    void* stream = nullptr;
    static void* executorSpace = nullptr;
    void* executor = nullptr;

    const char* opType = "bninference_d_kernel";
    char inputDesc[] = {1, 1, 2};
    char outputDesc[] = {1};
    char attrDesc[] = {};

    for (size_t i = 0; i < 3; i++) {
        ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

        executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                       sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
        ASSERT_NE(executor, nullptr);
        NnopbaseSetMatchArgsFlag(executor);
        ASSERT_EQ(NnopbaseSetRef(executor, 0, 0), 0);
        if (i == 1) {
            ASSERT_EQ(NnopbaseAddInput(executor, tensor1, 0), 0);
            ASSERT_EQ(NnopbaseAddInput(executor, tensor1, 1), 0);
            ASSERT_EQ(NnopbaseAddDynamicInput(executor, tensorList1, 2), 0);
            ASSERT_EQ(NnopbaseAddOutput(executor, tensor1, 0), 0);
        } else {
            ASSERT_EQ(NnopbaseAddInput(executor, tensor, 0), 0);
            ASSERT_EQ(NnopbaseAddInput(executor, tensor, 1), 0);
            ASSERT_EQ(NnopbaseAddDynamicInput(executor, tensorList, 2), 0);
            ASSERT_EQ(NnopbaseAddOutput(executor, tensor, 0), 0);
        }
        bool flag = NnopbaseMatchArgs(executor, &workspaceLen);
        ASSERT_EQ(flag, false);
        uint64_t inContWorkspaceSize = 10U;
        const aclTensorList* inUnContTensors = nullptr;
        NnopbaseGetUnContiguousTensors(executor, &inUnContTensors);
        static aclOpExecutor aclInExecutor;
        if (inUnContTensors != nullptr) {
            // 执行非连续转连续一阶段
            aclInExecutor.SetWorkspaceOffsets({10, 10, 10, 10});
            ASSERT_EQ(NnopbaseSetUnContExecutor(executor, &aclInExecutor, inContWorkspaceSize), 0);
        }
        ASSERT_EQ(NnopbaseRunForWorkspace(executor, &workspaceLen), OK);
        aclOpExecutor* viewcopyExecutor = new aclOpExecutor;
        uint64_t viewcopyWsSize = 0U;
        if (inUnContTensors != nullptr) {
            const aclTensorList* unContTensors = nullptr;
            const aclTensorList* contTensors = nullptr;
            NnopbaseGetRefUnContiguousTensors(executor, &unContTensors, &contTensors);
            if (unContTensors != nullptr) {
                ASSERT_EQ(NnopbaseSetViewCopyExecutor(executor, viewcopyExecutor), OK);
            }
        }
        if (i == 1) {
            ASSERT_EQ(((NnopbaseExecutor*)executor)->args->inputs.extTensors[0].rt2Tensor.GetAddr(), addr);
            ASSERT_EQ(((NnopbaseExecutor*)executor)->args->inputs.extTensors[3].rt2Tensor.GetAddr(), addr);
            ASSERT_EQ(((NnopbaseExecutor*)executor)->args->outputs.extTensors[0].rt2Tensor.GetAddr(), addr);
        } else {
            ASSERT_EQ(((NnopbaseExecutor*)executor)->args->inputs.extTensors[0].rt2Tensor.GetAddr(), nullptr);
            ASSERT_EQ(((NnopbaseExecutor*)executor)->args->inputs.extTensors[3].rt2Tensor.GetAddr(), nullptr);
            ASSERT_EQ(((NnopbaseExecutor*)executor)->args->outputs.extTensors[0].rt2Tensor.GetAddr(), nullptr);
        }
        if ((workspace == nullptr) && (workspaceLen + inContWorkspaceSize > 0U)) {
            workspace = (void*)malloc(workspaceLen + inContWorkspaceSize);
        }
        viewcopyExecutor = nullptr;
        ASSERT_EQ(NnopbaseGetViewCopyExecutor(executor, &viewcopyExecutor), OK);
        const aclTensorList* viewcopyTensors = nullptr;
        ASSERT_EQ(NnopbaseReleaseRefContiguousTensors(executor, &viewcopyTensors), OK);
        ASSERT_EQ(NnopbaseRunWithWorkspace(executor, stream, workspace, workspaceLen), OK);
        if (viewcopyExecutor != nullptr) {
            ASSERT_EQ(aclDestroyTensorList(viewcopyTensors), OK);
            delete viewcopyExecutor;
        }
    }
    if (workspace != nullptr) {
        free(workspace);
    }
    delete ptr;

    aclDestroyTensor(tensor);
    aclDestroyTensor(tensor1);
    aclDestroyTensorList(tensorList);
    aclDestroyTensorList(tensorList1);
    NnopbaseExecutorGcSpace(executorSpace);
    NnopbaseUnsetEnvAndClearFolder();
    op::internal::opProfilingSwitch.recordOpArgFlag = true;
}

class NnopbaseLibWrapperUnitTest : public testing::Test {
protected:
    void SetUp() { setenv("ASCEND_C", "1", 1); }
    void TearDown()
    {
        unsetenv("ASCEND_C");
        nnopbase::IndvSoc::GetInstance().Reset();
        Adx::MmpaStub::GetInstance()->UnInstall();
    }
};

static int x = 2;

HcclResult HcclAllocComResourceException(HcclComm comm, void* stream, void* TilingData, void** commContext)
{
    return HCCL_E_PARA;
}

HcclResult HcclGetAicpuOpStreamAndNotifyException(HcclComm comm, void* opStream, uint8_t notifyCnt, void** aicpuNotify)
{
    return HCCL_E_PARA;
}

HcclResult HcomGetCommHandleByGroupException(const char* group, HcclComm* commHandle) { return HCCL_E_PARA; }

HcclResult HcclGetCcuTaskInfoException(HcclComm comm, void* fusionArgs, void* ccuTaskGroup) { return HCCL_E_PARA; }

HcclResult HcclGetRankIdException(HcclComm comm, uint32_t* rankId) { return HCCL_E_PARA; }

HcclResult HcclRankGraphGetLayersException(HcclComm comm, uint32_t** netLayers, uint32_t* netLayerNum)
{
    return HCCL_E_PARA;
}

HcclResult HcclRankGraphGetRankSizeByLayerException(HcclComm comm, uint32_t netLayer, uint32_t* rankNum)
{
    return HCCL_E_PARA;
}

HcclResult HcclRankGraphGetTopoTypeByLayerException(HcclComm comm, uint32_t netLayer, uint32_t* topoType)
{
    return HCCL_E_PARA;
}

HcclResult HcclGetHcclBufferException(HcclComm comm, void** buffer, uint64_t* size) { return HCCL_E_PARA; }

HcclResult HcclGetRankSizeException(HcclComm comm, uint32_t* rankSize) { return HCCL_E_PARA; }

HcclResult HcclGetCommNameException(HcclComm comm, char* commName) { return HCCL_E_PARA; }

HcclResult HcclEngineCtxGetException(HcclComm comm, const char* ctxTag, int32_t engine, void** ctx, uint64_t* size)
{
    return HCCL_E_PARA;
}

HcclResult HcclThreadResGetInfoException(HcclComm comm, uint64_t thread, int32_t resType, uint32_t infoLen, void** info)
{
    return HCCL_E_PARA;
}

HcclResult HcclThreadAcquireWithStreamException(HcclComm comm, int32_t engine, aclrtStream stream, uint32_t notifyNum,
                                                uint64_t* thread)
{
    return HCCL_E_PARA;
}

int32_t HcommThreadNotifyRecordOnThreadException(uint64_t thread, uint64_t dstThread, uint32_t dstNotifyIdx)
{
    return -1;
}

int32_t HcommThreadNotifyWaitOnThreadException(uint64_t thread, uint32_t notifyIdx, uint32_t timeOut) { return -1; }

HcclResult HcclThreadAcquireWithConfigException(HcclComm comm, int32_t engine, uint32_t threadNum, int32_t type,
                                                const void* config, uint64_t* threads)
{
    return HCCL_E_PARA;
}

HcclResult HcclEngineCtxCreateException(HcclComm comm, const char* ctxTag, int32_t engine, uint64_t size, void** ctx)
{
    return HCCL_E_PARA;
}

HcclResult HcclGetNotifyNumInThreadException(HcclComm comm, uint64_t thread, int32_t engine, uint32_t* notifyNum)
{
    return HCCL_E_PARA;
}

HcclResult HcclGetCcuTaskInfoNormal(HcclComm comm, void* fusionArgs, void* ccuTaskGroup) { return HCCL_SUCCESS; }

HcclResult HcclGetRankIdNormal(HcclComm comm, uint32_t* rankId)
{
    if (rankId != nullptr) {
        *rankId = 0U;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclAllocComResourceNormal(HcclComm comm, void* stream, void* TilingData, void** commContext)
{
    return HCCL_SUCCESS;
}

HcclResult HcclGetAicpuOpStreamAndNotifyNormal(HcclComm comm, aclrtStream* opStream, uint8_t notifyCnt,
                                               void** aicpuNotify)
{
    *opStream = &x;
    return HCCL_SUCCESS;
}

HcclResult HcomGetCommHandleByGroupNormal(const char* group, HcclComm* commHandle)
{
    *commHandle = (HcclComm*)(&x);
    return HCCL_SUCCESS;
}

HcclResult HcclRankGraphGetLayersNormal(HcclComm comm, uint32_t** netLayers, uint32_t* netLayerNum)
{
    return HCCL_SUCCESS;
}

HcclResult HcclRankGraphGetRankSizeByLayerNormal(HcclComm comm, uint32_t netLayer, uint32_t* rankNum)
{
    return HCCL_SUCCESS;
}

HcclResult HcclRankGraphGetTopoTypeByLayerNormal(HcclComm comm, uint32_t netLayer, uint32_t* topoType)
{
    return HCCL_SUCCESS;
}

HcclResult HcclGetHcclBufferNormal(HcclComm comm, void** buffer, uint64_t* size) { return HCCL_SUCCESS; }

HcclResult HcclGetRankSizeNormal(HcclComm comm, uint32_t* rankSize) { return HCCL_SUCCESS; }

HcclResult HcclGetCommNameNormal(HcclComm comm, char* commName)
{
    if (commName != nullptr) {
        commName[0] = '\0';
    }
    return HCCL_SUCCESS;
}

static uint64_t g_hcclUnfoldThreadHandle = 1U;

HcclResult HcclEngineCtxGetNormal(HcclComm comm, const char* ctxTag, int32_t engine, void** ctx, uint64_t* size)
{
    // 返回一个非空ctx，命中HcclGetUnfoldThread中"展开流线程已存在"的快速路径
    if (ctx != nullptr) {
        *ctx = &g_hcclUnfoldThreadHandle;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclThreadResGetInfoNormal(HcclComm comm, uint64_t thread, int32_t resType, uint32_t infoLen, void** info)
{
    return HCCL_SUCCESS;
}

HcclResult HcclThreadAcquireWithStreamNormal(HcclComm comm, int32_t engine, aclrtStream stream, uint32_t notifyNum,
                                             uint64_t* thread)
{
    return HCCL_SUCCESS;
}

int32_t HcommThreadNotifyRecordOnThreadNormal(uint64_t thread, uint64_t dstThread, uint32_t dstNotifyIdx) { return 0; }

int32_t HcommThreadNotifyWaitOnThreadNormal(uint64_t thread, uint32_t notifyIdx, uint32_t timeOut) { return 0; }

HcclResult HcclThreadAcquireWithConfigNormal(HcclComm comm, int32_t engine, uint32_t threadNum, int32_t type,
                                             const void* config, uint64_t* threads)
{
    return HCCL_SUCCESS;
}

HcclResult HcclEngineCtxCreateNormal(HcclComm comm, const char* ctxTag, int32_t engine, uint64_t size, void** ctx)
{
    return HCCL_SUCCESS;
}

HcclResult HcclGetNotifyNumInThreadNormal(HcclComm comm, uint64_t thread, int32_t engine, uint32_t* notifyNum)
{
    return HCCL_SUCCESS;
}

class MmpaExceptionStub : public Adx::MmpaStub {
public:
    void* mmDlsym(void* handle, const char* funcName)
    {
        if (strncmp(funcName, "HcclAllocComResourceByTiling", strlen("HcclAllocComResourceByTiling")) == 0) {
            return (void*)HcclAllocComResourceException;
        } else if (strncmp(funcName, "HcclGetAicpuOpStreamAndNotify", strlen("HcclGetAicpuOpStreamAndNotify")) == 0) {
            return (void*)HcclGetAicpuOpStreamAndNotifyException;
        } else if (strncmp(funcName, "HcomGetCommHandleByGroup", strlen("HcomGetCommHandleByGroup")) == 0) {
            return (void*)HcomGetCommHandleByGroupException;
        } else if (strncmp(funcName, "HcclGetRankId", strlen("HcclGetRankId")) == 0) {
            return (void*)HcclGetRankIdException;
        } else if (strncmp(funcName, "HcclGetCcuTaskInfo", strlen("HcclGetCcuTaskInfo")) == 0) {
            return (void*)HcclGetCcuTaskInfoException;
        } else if (strncmp(funcName, "HcclRankGraphGetRankSizeByLayer", strlen("HcclRankGraphGetRankSizeByLayer")) ==
                   0) {
            return (void*)HcclRankGraphGetRankSizeByLayerException;
        } else if (strncmp(funcName, "HcclRankGraphGetLayers", strlen("HcclRankGraphGetLayers")) == 0) {
            return (void*)HcclRankGraphGetLayersException;
        } else if (strncmp(funcName, "HcclRankGraphGetTopoTypeByLayer", strlen("HcclRankGraphGetTopoTypeByLayer")) ==
                   0) {
            return (void*)HcclRankGraphGetTopoTypeByLayerException;
        } else if (strncmp(funcName, "HcclGetHcclBuffer", strlen("HcclGetHcclBuffer")) == 0) {
            return (void*)HcclGetHcclBufferException;
        } else if (strncmp(funcName, "HcclGetRankSize", strlen("HcclGetRankSize")) == 0) {
            return (void*)HcclGetRankSizeException;
        } else if (strncmp(funcName, "HcclGetCommName", strlen("HcclGetCommName")) == 0) {
            return (void*)HcclGetCommNameException;
        } else if (strncmp(funcName, "HcclEngineCtxGet", strlen("HcclEngineCtxGet")) == 0) {
            return (void*)HcclEngineCtxGetException;
        } else if (strncmp(funcName, "HcclEngineCtxCreate", strlen("HcclEngineCtxCreate")) == 0) {
            return (void*)HcclEngineCtxCreateException;
        } else if (strncmp(funcName, "HcclThreadResGetInfo", strlen("HcclThreadResGetInfo")) == 0) {
            return (void*)HcclThreadResGetInfoException;
        } else if (strncmp(funcName, "HcclThreadAcquireWithStream", strlen("HcclThreadAcquireWithStream")) == 0) {
            return (void*)HcclThreadAcquireWithStreamException;
        } else if (strncmp(funcName, "HcclThreadAcquireWithConfig", strlen("HcclThreadAcquireWithConfig")) == 0) {
            return (void*)HcclThreadAcquireWithConfigException;
        } else if (strncmp(funcName, "HcommThreadNotifyRecordOnThread", strlen("HcommThreadNotifyRecordOnThread")) ==
                   0) {
            return (void*)HcommThreadNotifyRecordOnThreadException;
        } else if (strncmp(funcName, "HcommThreadNotifyWaitOnThread", strlen("HcommThreadNotifyWaitOnThread")) == 0) {
            return (void*)HcommThreadNotifyWaitOnThreadException;
        } else if (strncmp(funcName, "HcclGetNotifyNumInThread", strlen("HcclGetNotifyNumInThread")) == 0) {
            return (void*)HcclGetNotifyNumInThreadException;
        } else {
            return nullptr;
        }
    }
};

class MmpaExceptionDlopenlStub : public Adx::MmpaStub {
public:
    void* mmDlopen(const char* fileName, int32_t mode) { return nullptr; }
};

class MmpaNormalStub : public Adx::MmpaStub {
public:
    void* mmDlsym(void* handle, const char* funcName)
    {
        if (strncmp(funcName, "HcclAllocComResourceByTiling", strlen("HcclAllocComResourceByTiling")) == 0) {
            return (void*)HcclAllocComResourceNormal;
        } else if (strncmp(funcName, "HcclGetAicpuOpStreamAndNotify", strlen("HcclGetAicpuOpStreamAndNotify")) == 0) {
            return (void*)HcclGetAicpuOpStreamAndNotifyNormal;
        } else if (strncmp(funcName, "HcomGetCommHandleByGroup", strlen("HcomGetCommHandleByGroup")) == 0) {
            return (void*)HcomGetCommHandleByGroupNormal;
        } else if (strncmp(funcName, "HcclGetRankId", strlen("HcclGetRankId")) == 0) {
            return (void*)HcclGetRankIdNormal;
        } else if (strncmp(funcName, "HcclGetCcuTaskInfo", strlen("HcclGetCcuTaskInfo")) == 0) {
            return (void*)HcclGetCcuTaskInfoNormal;
        } else if (strncmp(funcName, "HcclRankGraphGetRankSizeByLayer", strlen("HcclRankGraphGetRankSizeByLayer")) ==
                   0) {
            return (void*)HcclRankGraphGetRankSizeByLayerNormal;
        } else if (strncmp(funcName, "HcclRankGraphGetLayers", strlen("HcclRankGraphGetLayers")) == 0) {
            return (void*)HcclRankGraphGetLayersNormal;
        } else if (strncmp(funcName, "HcclRankGraphGetTopoTypeByLayer", strlen("HcclRankGraphGetTopoTypeByLayer")) ==
                   0) {
            return (void*)HcclRankGraphGetTopoTypeByLayerNormal;
        } else if (strncmp(funcName, "HcclGetRankSize", strlen("HcclGetRankSize")) == 0) {
            return (void*)HcclGetRankSizeNormal;
        } else if (strncmp(funcName, "HcclGetHcclBuffer", strlen("HcclGetHcclBuffer")) == 0) {
            return (void*)HcclGetHcclBufferNormal;
        } else if (strncmp(funcName, "HcclGetCommName", strlen("HcclGetCommName")) == 0) {
            return (void*)HcclGetCommNameNormal;
        } else if (strncmp(funcName, "HcclEngineCtxGet", strlen("HcclEngineCtxGet")) == 0) {
            return (void*)HcclEngineCtxGetNormal;
        } else if (strncmp(funcName, "HcclEngineCtxCreate", strlen("HcclEngineCtxCreate")) == 0) {
            return (void*)HcclEngineCtxCreateNormal;
        } else if (strncmp(funcName, "HcclThreadResGetInfo", strlen("HcclThreadResGetInfo")) == 0) {
            return (void*)HcclThreadResGetInfoNormal;
        } else if (strncmp(funcName, "HcclThreadAcquireWithStream", strlen("HcclThreadAcquireWithStream")) == 0) {
            return (void*)HcclThreadAcquireWithStreamNormal;
        } else if (strncmp(funcName, "HcclThreadAcquireWithConfig", strlen("HcclThreadAcquireWithConfig")) == 0) {
            return (void*)HcclThreadAcquireWithConfigNormal;
        } else if (strncmp(funcName, "HcommThreadNotifyRecordOnThread", strlen("HcommThreadNotifyRecordOnThread")) ==
                   0) {
            return (void*)HcommThreadNotifyRecordOnThreadNormal;
        } else if (strncmp(funcName, "HcommThreadNotifyWaitOnThread", strlen("HcommThreadNotifyWaitOnThread")) == 0) {
            return (void*)HcommThreadNotifyWaitOnThreadNormal;
        } else if (strncmp(funcName, "HcclGetNotifyNumInThread", strlen("HcclGetNotifyNumInThread")) == 0) {
            return (void*)HcclGetNotifyNumInThreadNormal;
        } else {
            return nullptr;
        }
    }
};

MmpaNormalStub mmpaNormallStub;

uint32_t CcuKernelLaunchNormal(HcclComm comm, void* opResCtx) { return 0U; }

class Mc2ClientStub : public Adx::MmpaStub {
public:
    void* mmDlsym(void* handle, const char* funcName)
    {
        if (std::string(funcName) == "HcclAllocComResourceByTiling") {
            return reinterpret_cast<void*>(HcclAllocComResourceNormal);
        }
        if (std::string(funcName) == "CcuKernelLaunch") {
            ccuKernelLaunchRequested = true;
            return reinterpret_cast<void*>(CcuKernelLaunchNormal);
        }
        return nullptr;
    }

    bool ccuKernelLaunchRequested = false;
};

TEST_F(NnopbaseLibWrapperUnitTest, CcuKernelLaunchSymbolIsLoaded)
{
    Mc2ClientStub mmpaStub;
    Adx::MmpaStub::GetInstance()->Install(&mmpaStub);

    EXPECT_EQ(nnopbase::IndvMc2ClientWrapper::GetInstance().IndvMc2ClientWrapperInit("libmc2_client.so"), OK);
    EXPECT_TRUE(mmpaStub.ccuKernelLaunchRequested);
    EXPECT_EQ(nnopbase::IndvMc2ClientWrapper::GetInstance().CcuKernelLaunch(nullptr, nullptr), OK);
}

TEST_F(NnopbaseLibWrapperUnitTest, NnopbaseHcclLibException)
{
    ASSERT_EQ(nnopbase::IndvHcclWrapper::GetInstance().IndvHcclWrapperInit(nullptr, true), ACLNN_ERR_PARAM_NULLPTR);
    ASSERT_EQ(nnopbase::IndvHcclWrapper::GetInstance().IndvHcclWrapperInit("libhccl.so", true),
              ACLNN_ERR_PARAM_NULLPTR);

    ASSERT_EQ(nnopbase::IndvHcclWrapper::GetInstance().HcclAllocComResourceByTiling(nullptr, nullptr, nullptr, nullptr),
              ACLNN_ERR_PARAM_NULLPTR);
    ASSERT_EQ(nnopbase::IndvHcclWrapper::GetInstance().HcclGetAicpuOpStreamAndNotify(nullptr, nullptr, 2, nullptr),
              ACLNN_ERR_PARAM_NULLPTR);
    ASSERT_EQ(nnopbase::IndvHcclWrapper::GetInstance().HcomGetCommHandleByGroup(nullptr, nullptr),
              ACLNN_ERR_PARAM_NULLPTR);
    ASSERT_EQ(nnopbase::IndvHcclWrapper::GetInstance().HcclGetCcuTaskInfo(nullptr, nullptr, nullptr),
              ACLNN_ERR_PARAM_NULLPTR);

    MmpaExceptionStub mmpaStub;
    Adx::MmpaStub::GetInstance()->Install(&mmpaStub);
    ASSERT_EQ(nnopbase::IndvHcclWrapper::GetInstance().IndvHcclWrapperInit("libhccl.so", true), OK);
    ASSERT_EQ(nnopbase::IndvHcclWrapper::GetInstance().HcclAllocComResourceByTiling(nullptr, nullptr, nullptr, nullptr),
              ACLNN_ERR_INNER);
    ASSERT_EQ(nnopbase::IndvHcclWrapper::GetInstance().HcclGetAicpuOpStreamAndNotify(nullptr, nullptr, 2, nullptr),
              ACLNN_ERR_INNER);
    ASSERT_EQ(nnopbase::IndvHcclWrapper::GetInstance().HcomGetCommHandleByGroup(nullptr, nullptr), ACLNN_ERR_INNER);
    ASSERT_EQ(nnopbase::IndvHcclWrapper::GetInstance().HcclGetRankId(nullptr, nullptr), ACLNN_ERR_INNER);
    ASSERT_EQ(nnopbase::IndvHcclWrapper::GetInstance().HcclGetCcuTaskInfo(nullptr, nullptr, nullptr), ACLNN_ERR_INNER);

    Adx::MmpaStub::GetInstance()->UnInstall();

    MmpaExceptionDlopenlStub mmpaDlopenStub;
    Adx::MmpaStub::GetInstance()->Install(&mmpaDlopenStub);
    ASSERT_EQ(nnopbase::IndvHcclWrapper::GetInstance().IndvHcclWrapperInit("libhccl.so", true),
              ACLNN_ERR_PARAM_INVALID);
    Adx::MmpaStub::GetInstance()->UnInstall();
}

TEST_F(NnopbaseLibWrapperUnitTest, NnopbaseHcclLibSuccess)
{
    Adx::MmpaStub::GetInstance()->Install((Adx::MmpaStub*)&mmpaNormallStub);

    ASSERT_EQ(nnopbase::IndvHcclWrapper::GetInstance().IndvHcclWrapperInit("libhccl.so", true), OK);
    ASSERT_EQ(nnopbase::IndvHcclWrapper::GetInstance().HcclAllocComResourceByTiling(nullptr, nullptr, nullptr, nullptr),
              OK);
    aclrtStream aicpuStream = nullptr;
    ASSERT_EQ(nnopbase::IndvHcclWrapper::GetInstance().HcclGetAicpuOpStreamAndNotify(nullptr, &aicpuStream, 2, nullptr),
              OK);
    HcclComm commHandle = nullptr;
    ASSERT_EQ(nnopbase::IndvHcclWrapper::GetInstance().HcomGetCommHandleByGroup(nullptr, &commHandle), OK);
    ASSERT_EQ(nnopbase::IndvHcclWrapper::GetInstance().HcclGetRankId(nullptr, nullptr), OK);
    ASSERT_EQ(nnopbase::IndvHcclWrapper::GetInstance().HcclGetCcuTaskInfo(nullptr, nullptr, nullptr), OK);

    Adx::MmpaStub::GetInstance()->UnInstall();
}

TEST_F(NnopbaseExtUnitTest, NnopBaseMC2RunSuccessForDavidWithServerTypeAICPU)
{
    Adx::MmpaStub::GetInstance()->Install((Adx::MmpaStub*)&mmpaNormallStub);
    TestHcclServerType([](void* executor) { NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_AICPU); },
                       nnopbase::OPS_SUBPATH_ASCEND950);
    Adx::MmpaStub::GetInstance()->UnInstall();
}
