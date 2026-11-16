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
#include <future>
#include "mockcpp/mockcpp.hpp"
#include "executor/indv_bininfo.h"
#include "executor/indv_collector.h"
#include "executor/indv_executor.h"
#include "executor/indv_compute_node_info.h"
#include "executor/indv_non_finite_check_op.h"
#include "executor/indv_tilingcontext_builder.h"
#include "executor/indv_cache_key_builder.h"
#include "utils/file_faker.h"
#include "utils/indv_soc.h"
#include "utils/thread_var_container.h"
#include "individual_op_api.h"
#include "depends/op/op_stub.h"
#include "depends/profiler/profiler_stub.h"
#include "depends/dump/dump_stub.h"
#include "depends/mmpa/mmpa_stub.h"
#include "depends/runtime/runtime_stub.h"
#include "depends/platform/platform_stub.h"
#include "depends/acl/aclrt_stub.h"
#include "op_cache_internal.h"
#include "depends/op/op_stub.h"
#include "depends/op/aclnn_bninference_d_kernel_stub.h"
#include "depends/op/aclnn_custom_op_stub.h"
#include "utils/indv_types.h"
#include "executor/indv_mc2_aicpu.h"
#define private public
#include "executor/indv_args_pool.h"
#undef private

namespace {
void SetSocVersion(const std::string& version)
{
    auto& soc = nnopbase::IndvSoc::GetInstance();
    soc.socVersion = version;
    soc.isInit = true;
}

// 框架传给算子genSimplifiedKey的key缓冲大小，桩gen写入时以此为上限
constexpr size_t GEN_SIMPLIFIED_KEY_BUF_LEN = NNOPBASE_CUS_KEY_LEN;

// verbKey前缀中determinLevel与precision各占1字节，见NnopbaseAppendDynamicKeyHead
constexpr size_t VERBKEY_PREFIX_LEN = 2U;

// 动态键中每个张量贡献dtype、format各1字节
constexpr size_t DYNKEY_BYTES_PER_TENSOR = 2U;

// GetCustomizedExecutor构造的executor固定为3输入1输出
constexpr size_t CUSTOMIZED_STUB_INPUT_NUM = 3U;
constexpr size_t CUSTOMIZED_STUB_OUTPUT_NUM = 1U;
} // namespace

class NnopbaseExecutorUnitTest : public testing::Test {
protected:
    void SetUp() { setenv("ASCEND_C", "1", 1); }
    void TearDown() { unsetenv("ASCEND_C"); }
};

TEST_F(NnopbaseExecutorUnitTest, NnopbaseBuildAndRunMemsetTilingParse)
{
    NnopbaseExecutor* executor = new NnopbaseExecutor;
    NnopbaseExecutorArgs* args = new NnopbaseExecutorArgs;
    NnopbaseBinInfo* binInfo = new NnopbaseBinInfo;
    ASSERT_NE(executor, nullptr);
    ASSERT_NE(args, nullptr);
    ASSERT_NE(binInfo, nullptr);
    executor->args = args;
    executor->args->binInfo = binInfo;
    binInfo->memsetInfo = std::make_shared<MemsetOpInfo>();
    binInfo->memsetInfo->binInfo = std::make_shared<MemsetOpBinInfo>();
    binInfo->memsetInfo->binInfo->coreType = kAicore;
    struct SpaceRegistryGuard {
        ~SpaceRegistryGuard()
        {
            auto spaceRegistry = std::make_shared<gert::OpImplSpaceRegistryV2>();
            gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(spaceRegistry);
        }
    } spaceRegistryGuard;
    gert::DefaultOpImplSpaceRegistryV2::GetInstance().ClearSpaceRegistry();
    auto ret = NnopbaseBuildAndRunMemsetTilingParse(executor->args->binInfo->memsetInfo);
    ASSERT_EQ(ret, ACLNN_ERR_PARAM_NULLPTR);
    delete binInfo;
    delete args;
    delete executor;
}

TEST_F(NnopbaseExecutorUnitTest, ExecutorInitWithoutAttr)
{
    NnopbaseExecutor* executor = new NnopbaseExecutor;
    ASSERT_NE(executor, nullptr);

    char inputDesc[] = {1, 1, 1};
    char outputDesc[] = {1};
    char attrDesc[] = {};
    auto ret = NnopbaseExecutorInit(
        executor, {inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc, sizeof(outputDesc) / sizeof(char), attrDesc,
                   sizeof(attrDesc) / sizeof(char)});
    ASSERT_EQ(ret, OK);
    NnopbaseExecutorDeInit(executor);
    delete executor;
}

TEST_F(NnopbaseExecutorUnitTest, ExecutorInitWithAttr)
{
    NnopbaseExecutor* executor = new NnopbaseExecutor;
    ASSERT_NE(executor, nullptr);

    char inputDesc[] = {1, 1, 1};
    char outputDesc[] = {1};
    char attrDesc[] = {1};
    auto ret = NnopbaseExecutorInit(
        executor, {inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc, sizeof(outputDesc) / sizeof(char), attrDesc,
                   sizeof(attrDesc) / sizeof(char)});
    ASSERT_EQ(ret, OK);
    NnopbaseExecutorDeInit(executor);
    delete executor;
}

TEST_F(NnopbaseExecutorUnitTest, ExecutorInitDynamicInput)
{
    NnopbaseExecutor* executor = new NnopbaseExecutor;
    ASSERT_NE(executor, nullptr);

    char inputDesc[] = {2, 2, 2};
    char outputDesc[] = {2};
    char attrDesc[] = {};
    auto ret = NnopbaseExecutorInit(
        executor, {inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc, sizeof(outputDesc) / sizeof(char), attrDesc,
                   sizeof(attrDesc) / sizeof(char)});
    ASSERT_EQ(ret, OK);

    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    std::vector<const aclTensor*> tensor_list_a;
    tensor_list_a.push_back(tensor);
    aclTensorList* aclTensorTestList = aclCreateTensorList(tensor_list_a.data(), tensor_list_a.size());
    (void)NnopbaseExecutorAddDynamicTensors(executor, aclTensorTestList, 0, true);
    (void)NnopbaseExecutorAddDynamicTensors(executor, aclTensorTestList, 1, true);
    (void)NnopbaseExecutorAddDynamicTensors(executor, aclTensorTestList, 2, true);
    (void)NnopbaseExecutorAddDynamicTensors(executor, aclTensorTestList, 0, false);
    NnopbaseExecutorDeInit(executor);
    delete executor;
    aclDestroyTensorList((const aclTensorList*)aclTensorTestList);
}

void GetExecutor(NnopbaseExecutor*& executor, const char* opType = "bninference_d_kernel",
                 std::vector<int64_t> shape = {1, 1, 1, 1, 1})
{
    static NnopbaseBinCollector* gBinCollector = nullptr;
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    if (gBinCollector == nullptr) {
        gBinCollector = new NnopbaseBinCollector;
        ASSERT_NE(gBinCollector, nullptr);
        auto ret = NnopbaseCollectorInit(gBinCollector);
        ASSERT_EQ(ret, OK);
        ret = NnopbaseCollectorWork(gBinCollector);
        ASSERT_EQ(ret, OK);
    }

    executor = new NnopbaseExecutor;
    ASSERT_NE(executor, nullptr);
    char inputDesc[] = {1, 1, 1};
    char outputDesc[] = {1};
    char attrDesc[] = {};
    auto ret = NnopbaseExecutorInit(
        executor, {inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc, sizeof(outputDesc) / sizeof(char), attrDesc,
                   sizeof(attrDesc) / sizeof(char)});
    ASSERT_EQ(ret, OK);
    executor->space = new NnopbaseExecutorSpace();

    NnopbaseExecutorSetCollector(executor, gBinCollector);
    if (opType != "bninference_d_kernel") {
        return;
    }
    ret = NnopbaseExecutorSetRegInfo(executor, opType);
    ASSERT_EQ(ret, OK);

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

// customizedKey相关用例专用：与GetExecutor流程一致，但支持指定任意已注册的opType。
// 构造3输入1输出、ACL_FLOAT(dtype 0)、ND(format 2)，与桩json的"0,2"段对齐；
// 关闭args缓存，否则keyLen为0会使CreateArgs内的memcpy_s失败。
void GetCustomizedExecutor(NnopbaseExecutor*& executor, const char* opType,
                           std::vector<int64_t> shape = {1, 1, 1, 1, 1})
{
    static NnopbaseBinCollector* gCustBinCollector = nullptr;
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    if (gCustBinCollector == nullptr) {
        gCustBinCollector = new NnopbaseBinCollector;
        ASSERT_NE(gCustBinCollector, nullptr);
        ASSERT_EQ(NnopbaseCollectorInit(gCustBinCollector), OK);
        ASSERT_EQ(NnopbaseCollectorWork(gCustBinCollector), OK);
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
    NnopbaseExecutorSetCollector(executor, gCustBinCollector);
    ASSERT_EQ(NnopbaseExecutorSetRegInfo(executor, opType), OK);

    aclTensor* tensor = aclCreateTensor(&shape[0], shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, &shape[0], shape.size(), nullptr);
    for (uint32_t i = 0U; i < static_cast<uint32_t>(CUSTOMIZED_STUB_INPUT_NUM); i++) {
        ASSERT_EQ(NnopbaseExecutorUpdateTensorsIndex(&(executor->ownArgs.inputs), i), OK);
        ASSERT_EQ(NnopbaseExecutorAddTensor(executor, tensor, i, true, false), OK);
    }
    ASSERT_EQ(NnopbaseExecutorUpdateTensorsIndex(&(executor->ownArgs.outputs), 0), OK);
    ASSERT_EQ(NnopbaseExecutorAddTensor(executor, tensor, 0, false, false), OK);
    aclDestroyTensor(tensor);

    executor->ownArgs.enableCache = false;
    ASSERT_EQ(nnopbase::ArgsPool::GetInstance().CreateArgs(executor), OK);
    executor->deterministic = false;
    executor->tiling.contextExt.hasBuilt = false;
}

void FreeCustomizedExecutor(NnopbaseExecutor* executor)
{
    NnopbaseExecutorGcSpace((void*)executor->space);
    NnopbaseExecutorDeInit(executor);
    delete executor;
    NnopbaseUnsetEnvAndClearFolder();
}

void GetExecutorWithAttr(NnopbaseExecutor*& executor, const char* opType = "bninference_d_kernel",
                         std::vector<int64_t> shape = {1, 1, 1, 1, 1})
{
    static NnopbaseBinCollector* gBinCollector = nullptr;
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    if (gBinCollector == nullptr) {
        gBinCollector = new NnopbaseBinCollector;
        ASSERT_NE(gBinCollector, nullptr);
        auto ret = NnopbaseCollectorInit(gBinCollector);
        ASSERT_EQ(ret, OK);
        ret = NnopbaseCollectorWork(gBinCollector);
        ASSERT_EQ(ret, OK);
    }

    executor = new NnopbaseExecutor;
    ASSERT_NE(executor, nullptr);
    char inputDesc[] = {1, 1, 1};
    char outputDesc[] = {1};
    char attrDesc[] = {1};
    auto ret = NnopbaseExecutorInit(
        executor, {inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc, sizeof(outputDesc) / sizeof(char), attrDesc,
                   sizeof(attrDesc) / sizeof(char)});
    ASSERT_EQ(ret, OK);

    NnopbaseExecutorSetCollector(executor, gBinCollector);
    NnopbaseExecutorSpace space;
    executor->space = &space;
    if (opType != "bninference_d_kernel") {
        return;
    }
    ret = NnopbaseExecutorSetRegInfo(executor, opType);
    ASSERT_EQ(ret, OK);

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
    size_t bias1_size = 8;
    if (bias1) {
        NnopbaseAddAttrWithDtype(executor, bias1, bias1_size, 0, kNnopbaseInt);
    } else {
        static int64_t bias1_def[] = {1, 2};
        static size_t bias1_size_def = 2 * sizeof(int64_t);
        NnopbaseAddAttrWithDtype(executor, bias1_def, bias1_size_def, 0, kNnopbaseInt);
    }
}

void RunExecutorSuccess(NnopbaseExecutor* executor)
{
    size_t workspaceLen = 0U;
    ASSERT_EQ(NnopbaseExecutorRunForWorkspace(executor, &workspaceLen), OK);
    void* stream = (void*)0x12345;
    void* workspace = nullptr;
    if (workspaceLen > 0U) {
        workspace = (void*)malloc(workspaceLen);
    }
    ASSERT_EQ(NnopbaseExecutorRunWithWorkspace(executor, stream, workspace, workspaceLen), OK);
    if (workspaceLen > 0U) {
        free(workspace);
    }
}

TEST_F(NnopbaseExecutorUnitTest, ExecutorBinKey)
{
    nnopbase::ArgsPool::GetInstance().Finalize();
    NnopbaseExecutor* executor = nullptr;
    GetExecutor(executor);
    RunExecutorSuccess(executor);

    ASSERT_EQ(executor->binInfoKey.len, 30);
    std::string key0 = (char*)(&executor->binInfoKey.verbose[0]);
    ASSERT_EQ(key0, "bninference_d_kernel");
    ASSERT_EQ((int32_t)(executor->binInfoKey.verbose)[20], 0);
    ASSERT_EQ((int32_t)(executor->binInfoKey.verbose)[21], 0);
    ASSERT_EQ((int32_t)(executor->binInfoKey.verbose)[22], 0);
    ASSERT_EQ((int32_t)(executor->binInfoKey.verbose)[23], 2);
    ASSERT_EQ((int32_t)(executor->binInfoKey.verbose)[24], 0);
    ASSERT_EQ((int32_t)(executor->binInfoKey.verbose)[25], 2);
    ASSERT_EQ((int32_t)(executor->binInfoKey.verbose)[26], 0);
    ASSERT_EQ((int32_t)(executor->binInfoKey.verbose)[27], 2);
    ASSERT_EQ((int32_t)(executor->binInfoKey.verbose)[28], 0);

    NnopbaseExecutorGcSpace((void*)executor->space);
    NnopbaseExecutorDeInit(executor);
    delete executor;
    NnopbaseUnsetEnvAndClearFolder();
}

void NnopbaseExecutorUnitTestCheckArgs(NnopbaseExecutor* executor)
{
    const size_t tilingDataSize = ((NnopbaseTilingData*)(executor->args->tilingInfo.tilingData))->GetDataSize();
    const size_t alignTilingDataSize = ((tilingDataSize % 8U) != 0) ? (tilingDataSize / 8U + 1U) * 8U :
                                                                      tilingDataSize; // 8byte对齐
    uint64_t* ptr = (uint64_t*)((NnopbaseUChar*)((NnopbaseTilingData*)(executor->args->tilingInfo.tilingData))
                                    ->GetData() +
                                alignTilingDataSize);
    // check io size
    size_t expIOSize[] = {64, 64, 120, 120, 0};
#if 0 // comment out for ge decoupling
    for (size_t i = 0; i < 5; i++) {
        ASSERT_EQ(ptr[i], expIOSize[i]);
    }
#endif

    rtHostInputInfo_t* hostInputInfo = (rtHostInputInfo_t*)((NnopbaseUChar*)ptr +
                                                            (executor->ownArgs.inputs.paramDescs.count +
                                                             executor->ownArgs.outputs.paramDescs.count + 2) *
                                                                sizeof(uint64_t)); // 2 is workspace and atomicIndex
#if 0                                                                              // comment out for ge decoupling
    size_t k = 0U;
    uint16_t expAddrOffset[] = {8, 16, 24, 32};
    uint16_t expDataOffset[] = {208, 272, 336, 456};
    // check dynamic input or output
    for (size_t i = 0; i < executor->ownArgs.inputs.paramDescs.count; i++) {
        if (executor->ownArgs.inputs.paramDescs.instances[i].isDynamic) {
                ASSERT_EQ(expAddrOffset[k], hostInputInfo[k].addrOffset);
                ASSERT_EQ(expDataOffset[k], hostInputInfo[k].dataOffset);
                k++;
            }
    }
    for (size_t i = 0; i < executor->ownArgs.outputs.paramDescs.count; i++) {
        if (executor->ownArgs.outputs.paramDescs.instances[i].isDynamic) {
            ASSERT_EQ(expAddrOffset[k], hostInputInfo[k].addrOffset);
            ASSERT_EQ(expDataOffset[k], hostInputInfo[k].dataOffset);
            k++;
        }
    }
#endif
}

TEST_F(NnopbaseExecutorUnitTest, ExecutorDynamicInput)
{
    nnopbase::ArgsPool::GetInstance().Finalize();
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);
    static NnopbaseBinCollector* gBinCollector = nullptr;
    if (gBinCollector == nullptr) {
        gBinCollector = new NnopbaseBinCollector;
        ASSERT_NE(gBinCollector, nullptr);
        auto ret = NnopbaseCollectorInit(gBinCollector);
        ASSERT_EQ(ret, OK);
        ret = NnopbaseCollectorWork(gBinCollector);
        ASSERT_EQ(ret, OK);
    }

    NnopbaseExecutor* executor = new NnopbaseExecutor;
    ASSERT_NE(executor, nullptr);
    char inputDesc[] = {2, 2};
    char outputDesc[] = {2, 2};
    char attrDesc[] = {};
    auto ret = NnopbaseExecutorInit(
        executor, {inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc, sizeof(outputDesc) / sizeof(char), attrDesc,
                   sizeof(attrDesc) / sizeof(char)});
    ASSERT_EQ(ret, OK);

    NnopbaseExecutorSetCollector(executor, gBinCollector);
    NnopbaseExecutorSpace space;
    executor->space = &space;
    const char* opType = "bninference_d_kernel";
    if (opType != "bninference_d_kernel") {
        return;
    }
    ret = NnopbaseExecutorSetRegInfo(executor, opType);
    ASSERT_EQ(ret, OK);

    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);

    std::vector<const aclTensor*> tensor_list_a;
    tensor_list_a.push_back(tensor);
    aclTensorList* aclTensorTestList = aclCreateTensorList(tensor_list_a.data(), tensor_list_a.size());
    std::vector<const aclTensor*> tensor_list_b;
    for (int i = 0; i < 2; i++) {
        aclTensor* tensor2 = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                             aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
        tensor_list_b.push_back(tensor2);
    }
    aclTensorList* aclTensorTestList2 = aclCreateTensorList(tensor_list_b.data(), tensor_list_b.size());
    (void)NnopbaseAddDynamicInput(executor, aclTensorTestList, 0);
    (void)NnopbaseAddDynamicInput(executor, aclTensorTestList, 1);
    (void)NnopbaseAddDynamicOutput(executor, aclTensorTestList2, 0);
    (void)NnopbaseAddDynamicOutput(executor, aclTensorTestList2, 1);

    RunExecutorSuccess(executor);
    NnopbaseExecutorUnitTestCheckArgs(executor);
    NnopbaseExecutorDeInit(executor);
    delete executor;
    aclDestroyTensorList((const aclTensorList*)aclTensorTestList2);
    aclDestroyTensorList((const aclTensorList*)aclTensorTestList);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseExecutorUnitTest, ExecutorDynamicOutput1)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);
    static NnopbaseBinCollector* gBinCollector = nullptr;
    if (gBinCollector == nullptr) {
        gBinCollector = new NnopbaseBinCollector;
        ASSERT_NE(gBinCollector, nullptr);
        auto ret = NnopbaseCollectorInit(gBinCollector);
        ASSERT_EQ(ret, OK);
        ret = NnopbaseCollectorWork(gBinCollector);
        ASSERT_EQ(ret, OK);
    }

    NnopbaseExecutor* executor = new NnopbaseExecutor;
    ASSERT_NE(executor, nullptr);
    char inputDesc[] = {2, 2};
    char outputDesc[] = {2, 1};
    char attrDesc[] = {};
    auto ret = NnopbaseExecutorInit(
        executor, {inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc, sizeof(outputDesc) / sizeof(char), attrDesc,
                   sizeof(attrDesc) / sizeof(char)});
    ASSERT_EQ(ret, OK);

    NnopbaseExecutorSetCollector(executor, gBinCollector);
    NnopbaseExecutorSpace space;
    executor->space = &space;
    const char* opType = "custom_op1";
    if (opType != "custom_op1") {
        return;
    }
    ret = NnopbaseExecutorSetRegInfo(executor, opType);
    ASSERT_EQ(ret, OK);

    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);

    std::vector<const aclTensor*> tensor_list_a;
    tensor_list_a.push_back(tensor);
    aclTensorList* aclTensorTestList = aclCreateTensorList(tensor_list_a.data(), tensor_list_a.size());
    aclTensor* tensor1 = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                         aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);

    (void)NnopbaseExecutorAddDynamicTensors(executor, aclTensorTestList, 0, true);
    (void)NnopbaseExecutorAddDynamicTensors(executor, aclTensorTestList, 1, true);
    (void)NnopbaseExecutorAddDynamicTensors(executor, aclTensorTestList, 0, false);
    ASSERT_EQ(NnopbaseExecutorUpdateTensorsIndex(&(executor->ownArgs.outputs), 1), OK);
    ASSERT_EQ(NnopbaseExecutorAddTensor(executor, tensor, 1, false, false), OK);

    RunExecutorSuccess(executor);
    NnopbaseExecutorDeInit(executor);
    delete executor;
    aclDestroyTensorList((const aclTensorList*)aclTensorTestList);
    aclDestroyTensor(tensor1);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseExecutorUnitTest, ExecutorDynamicOutput2)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);
    static NnopbaseBinCollector* gBinCollector = nullptr;
    if (gBinCollector == nullptr) {
        gBinCollector = new NnopbaseBinCollector;
        ASSERT_NE(gBinCollector, nullptr);
        auto ret = NnopbaseCollectorInit(gBinCollector);
        ASSERT_EQ(ret, OK);
        ret = NnopbaseCollectorWork(gBinCollector);
        ASSERT_EQ(ret, OK);
    }

    NnopbaseExecutor* executor = new NnopbaseExecutor;
    ASSERT_NE(executor, nullptr);
    char inputDesc[] = {2, 2};
    char outputDesc[] = {1, 2};
    char attrDesc[] = {};
    auto ret = NnopbaseExecutorInit(
        executor, {inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc, sizeof(outputDesc) / sizeof(char), attrDesc,
                   sizeof(attrDesc) / sizeof(char)});
    ASSERT_EQ(ret, OK);

    NnopbaseExecutorSetCollector(executor, gBinCollector);
    NnopbaseExecutorSpace space;
    executor->space = &space;
    const char* opType = "custom_op2";
    if (opType != "custom_op2") {
        return;
    }
    ret = NnopbaseExecutorSetRegInfo(executor, opType);
    ASSERT_EQ(ret, OK);

    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);

    std::vector<const aclTensor*> tensor_list_a;
    tensor_list_a.push_back(tensor);
    aclTensorList* aclTensorTestList = aclCreateTensorList(tensor_list_a.data(), tensor_list_a.size());
    aclTensor* tensor1 = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                         aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);

    (void)NnopbaseExecutorAddDynamicTensors(executor, aclTensorTestList, 0, true);
    (void)NnopbaseExecutorAddDynamicTensors(executor, aclTensorTestList, 1, true);
    (void)NnopbaseExecutorUpdateTensorsIndex(&executor->ownArgs.outputs, 0);
    (void)NnopbaseExecutorAddTensor(executor, tensor1, 0, false, false);
    (void)NnopbaseExecutorAddDynamicTensors(executor, aclTensorTestList, 1, false);

    RunExecutorSuccess(executor);
    NnopbaseExecutorDeInit(executor);
    delete executor;
    aclDestroyTensorList((const aclTensorList*)aclTensorTestList);
    aclDestroyTensor(tensor1);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseExecutorUnitTest, ExecutorOptionalInput1)
{
    op::internal::opProfilingSwitch.reportFlag = true;
    op::internal::opProfilingSwitch.additionInfoFlag = true;

    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);
    static NnopbaseBinCollector* gBinCollector = nullptr;
    if (gBinCollector == nullptr) {
        gBinCollector = new NnopbaseBinCollector;
        ASSERT_NE(gBinCollector, nullptr);
        auto ret = NnopbaseCollectorInit(gBinCollector);
        ASSERT_EQ(ret, OK);
        ret = NnopbaseCollectorWork(gBinCollector);
        ASSERT_EQ(ret, OK);
    }

    NnopbaseExecutor* executor = new NnopbaseExecutor;
    ASSERT_NE(executor, nullptr);
    char inputDesc[] = {1, 1, 1, 0, 0, 0};
    char outputDesc[] = {1, 0};
    char attrDesc[] = {};
    auto ret = NnopbaseExecutorInit(
        executor, {inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc, sizeof(outputDesc) / sizeof(char), attrDesc,
                   sizeof(attrDesc) / sizeof(char)});
    ASSERT_EQ(ret, OK);

    NnopbaseExecutorSetCollector(executor, gBinCollector);
    NnopbaseExecutorSpace space;
    executor->space = &space;
    const char* opType = "bninference_d_kernel";
    if (opType != "bninference_d_kernel") {
        return;
    }
    ret = NnopbaseExecutorSetRegInfo(executor, opType);
    ASSERT_EQ(ret, OK);

    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);

    ASSERT_EQ(NnopbaseExecutorUpdateTensorsIndex(&(executor->ownArgs.inputs), 0), OK);
    ASSERT_EQ(NnopbaseExecutorAddTensor(executor, tensor, 0, true, false), OK);
    ASSERT_EQ(NnopbaseExecutorUpdateTensorsIndex(&(executor->ownArgs.inputs), 1), OK);
    ASSERT_EQ(NnopbaseExecutorAddTensor(executor, tensor, 1, true, false), OK);
    ASSERT_EQ(NnopbaseExecutorUpdateTensorsIndex(&(executor->ownArgs.inputs), 2), OK);
    ASSERT_EQ(NnopbaseExecutorAddTensor(executor, tensor, 2, true, false), OK);
    ASSERT_EQ(NnopbaseExecutorUpdateTensorsIndex(&(executor->ownArgs.inputs), 3), OK);
    ASSERT_EQ(NnopbaseExecutorAddTensor(executor, tensor, 3, true, false), OK);
    ASSERT_EQ(NnopbaseExecutorUpdateTensorsIndex(&(executor->ownArgs.inputs), 4), OK);
    ASSERT_EQ(NnopbaseExecutorAddTensor(executor, tensor, 4, true, false), OK);
    // for optional input null
    ASSERT_EQ(NnopbaseExecutorUpdateTensorsIndex(&(executor->ownArgs.inputs), 5), OK);
    ASSERT_EQ(NnopbaseExecutorAddTensor(executor, nullptr, 5, true, false), OK);

    ASSERT_EQ(NnopbaseExecutorUpdateTensorsIndex(&(executor->ownArgs.outputs), 0), OK);
    ASSERT_EQ(NnopbaseExecutorAddTensor(executor, tensor, 0, false, false), OK);
    ASSERT_EQ(NnopbaseExecutorUpdateTensorsIndex(&(executor->ownArgs.outputs), 1), OK);
    ASSERT_EQ(NnopbaseExecutorAddTensor(executor, tensor, 1, false, false), OK);

    RunExecutorSuccess(executor);
    const size_t tilingDataSize = ((NnopbaseTilingData*)(((NnopbaseExecutor*)executor)->args->tilingInfo.tilingData))
                                      ->GetDataSize();
    const size_t alignTilingDataSize = ((tilingDataSize % 8U) != 0) ? (tilingDataSize / 8U + 1U) * 8U :
                                                                      tilingDataSize; // 8byte对齐
    uint64_t* ptr = (uint64_t*)((NnopbaseUChar*)((NnopbaseTilingData*)(((NnopbaseExecutor*)executor)
                                                                           ->args->tilingInfo.tilingData))
                                    ->GetData() +
                                alignTilingDataSize);
    size_t expIOSize[] = {4, 4, 4, 4};
    for (size_t i = 0; i < 4; i++) {
        ASSERT_EQ(ptr[i], expIOSize[i]);
    }
    NnopbaseExecutorDeInit(executor);
    delete executor;
    aclDestroyTensor(tensor);
    NnopbaseUnsetEnvAndClearFolder();
    op::internal::opProfilingSwitch.reportFlag = false;
    op::internal::opProfilingSwitch.additionInfoFlag = false;
}

TEST_F(NnopbaseExecutorUnitTest, ExecutorOptionalInput2)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);
    static NnopbaseBinCollector* gBinCollector = nullptr;
    if (gBinCollector == nullptr) {
        gBinCollector = new NnopbaseBinCollector;
        ASSERT_NE(gBinCollector, nullptr);
        auto ret = NnopbaseCollectorInit(gBinCollector);
        ASSERT_EQ(ret, OK);
        ret = NnopbaseCollectorWork(gBinCollector);
        ASSERT_EQ(ret, OK);
    }

    NnopbaseExecutor* executor = new NnopbaseExecutor;
    ASSERT_NE(executor, nullptr);
    char inputDesc[] = {1, 1, 1, 0, 0, 0, 0};
    char outputDesc[] = {1, 0, 0};
    char attrDesc[] = {};
    auto ret = NnopbaseExecutorInit(
        executor, {inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc, sizeof(outputDesc) / sizeof(char), attrDesc,
                   sizeof(attrDesc) / sizeof(char)});
    ASSERT_EQ(ret, OK);

    NnopbaseExecutorSetCollector(executor, gBinCollector);
    NnopbaseExecutorSpace space;
    executor->space = &space;
    const char* opType = "bninference_d_kernel";
    if (opType != "bninference_d_kernel") {
        return;
    }
    ret = NnopbaseExecutorSetRegInfo(executor, opType);
    ASSERT_EQ(ret, OK);

    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    ASSERT_EQ(NnopbaseExecutorUpdateTensorsIndex(&(executor->ownArgs.inputs), 0), OK);
    ASSERT_EQ(NnopbaseExecutorAddTensor(executor, tensor, 0, true, false), OK);
    ASSERT_EQ(NnopbaseExecutorUpdateTensorsIndex(&(executor->ownArgs.inputs), 1), OK);
    ASSERT_EQ(NnopbaseExecutorAddTensor(executor, tensor, 1, true, false), OK);
    ASSERT_EQ(NnopbaseExecutorUpdateTensorsIndex(&(executor->ownArgs.inputs), 2), OK);
    ASSERT_EQ(NnopbaseExecutorAddTensor(executor, tensor, 2, true, false), OK);
    ASSERT_EQ(NnopbaseExecutorUpdateTensorsIndex(&(executor->ownArgs.inputs), 3), OK);
    ASSERT_EQ(NnopbaseExecutorAddTensor(executor, tensor, 3, true, false), OK);
    ASSERT_EQ(NnopbaseExecutorUpdateTensorsIndex(&(executor->ownArgs.inputs), 4), OK);
    ASSERT_EQ(NnopbaseExecutorAddTensor(executor, tensor, 4, true, false), OK);
    ASSERT_EQ(NnopbaseExecutorUpdateTensorsIndex(&(executor->ownArgs.inputs), 5), OK);
    ASSERT_EQ(NnopbaseExecutorAddTensor(executor, tensor, 5, true, false), OK);
    ASSERT_EQ(NnopbaseExecutorUpdateTensorsIndex(&(executor->ownArgs.inputs), 6), OK);
    ASSERT_EQ(NnopbaseExecutorAddTensor(executor, tensor, 6, true, false), OK);

    ASSERT_EQ(NnopbaseExecutorUpdateTensorsIndex(&(executor->ownArgs.outputs), 0), OK);
    ASSERT_EQ(NnopbaseExecutorAddTensor(executor, tensor, 0, false, false), OK);
    ASSERT_EQ(NnopbaseExecutorUpdateTensorsIndex(&(executor->ownArgs.outputs), 1), OK);
    ASSERT_EQ(NnopbaseExecutorAddTensor(executor, tensor, 1, false, false), OK);
    ASSERT_EQ(NnopbaseExecutorUpdateTensorsIndex(&(executor->ownArgs.outputs), 2), OK);
    ASSERT_EQ(NnopbaseExecutorAddTensor(executor, tensor, 2, false, false), OK);

    RunExecutorSuccess(executor);
    NnopbaseExecutorDeInit(executor);
    delete executor;
    aclDestroyTensor(tensor);
    NnopbaseUnsetEnvAndClearFolder();
}

aclnnStatus RunBnProfiling(std::vector<int64_t> shape = {1, 1, 1, 1, 1})
{
    op::internal::opProfilingSwitch.reportFlag = true;
    op::internal::opProfilingSwitch.additionInfoFlag = true;

    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    aclnnStatus ret = RunOp(aclnnBninferenceDKernel, tensor, tensor, tensor, tensor);
    aclDestroyTensor(tensor);

    op::internal::opProfilingSwitch.reportFlag = false;
    op::internal::opProfilingSwitch.additionInfoFlag = false;
    return ret;
}

class ApiProfiler : public ProfilerStub {
public:
    int32_t MsprofReportApi(uint32_t agingFlag, const MsprofApi* api)
    {
        if (api->type != MSPROF_REPORT_NODE_LAUNCH_TYPE) {
            auto iter = map_.find(api->type);
            EXPECT_NE(iter, map_.end());
            EXPECT_EQ(api->type >= 0x60000, true);
            EXPECT_EQ(api->type <= 0x6FFFF, true);
            auto name = map_[api->type];
            std::string s = name;
            EXPECT_EQ(s == "aclnnBninferenceDKernelTiling" || s == "aclnnBninferenceDKernelGetWorkspaceSize", true);
        }
        return 0;
    }

    int32_t MsprofRegTypeInfo(uint16_t level, uint32_t typeId, const char* typeName)
    {
        auto iter = map_.find(typeId);
        EXPECT_EQ(iter, map_.end());
        map_[typeId] = typeName;
        return 0;
    }

private:
    std::map<uint32_t, const char*> map_;
};

TEST_F(NnopbaseExecutorUnitTest, profilingApiSuccess)
{
    ApiProfiler apiProfiler;
    ProfilerStub::GetInstance()->Install(&apiProfiler);
    EXPECT_EQ(RunBnProfiling(), OK);

    ProfilerStub::GetInstance()->UnInstall();
}

TEST_F(NnopbaseExecutorUnitTest, profilingIdSuccess)
{
    auto timeStamp = NnopbaseMsprofSysTime();
    NnopbaseDfxId id1 = {0, "func1", false};
    NnopbaseDfxId id2 = {0, "func2", true};
    NnopbaseDfxId id3 = {0, "func3", false};
    op::internal::opProfilingSwitch.reportFlag = true;
    NnopbaseReportApiInfo(timeStamp, id1);
    auto id = id1.id;
    ASSERT_NE(id, 0);
    ASSERT_EQ(id1.hasReg, true);
    NnopbaseReportApiInfo(timeStamp, id1);
    ASSERT_EQ(id1.id, id);

    NnopbaseReportApiInfo(timeStamp, id2);
    ASSERT_EQ(id2.id, 0);

    NnopbaseReportApiInfo(timeStamp, id3);
    ASSERT_EQ(id3.id, id + 1);
    op::internal::opProfilingSwitch.reportFlag = false;
}

class AdditionalProfiler : public ProfilerStub {
public:
    int32_t MsprofReportAdditionalInfo(uint32_t agingFlag, const VOID_PTR data, uint32_t length)
    {
        MsprofAdditionalInfo* report_data = (MsprofAdditionalInfo*)data;
        if (report_data->type == MSPROF_REPORT_NODE_TENSOR_INFO_TYPE) {
            EXPECT_EQ(report_data->level, MSPROF_REPORT_NODE_LEVEL);
            auto profTensorData = reinterpret_cast<MsprofTensorInfo*>(report_data->data);
            EXPECT_EQ(profTensorData->tensorNum, 4);
            for (int32_t i = 0; i < profTensorData->tensorNum; ++i) {
                auto& tensor_data = profTensorData->tensorData[i];
                EXPECT_EQ(tensor_data.shape[0], 1);
                EXPECT_EQ(tensor_data.shape[1], 1);
                EXPECT_EQ(tensor_data.shape[2], 1);
                EXPECT_EQ(tensor_data.shape[3], 1);
                EXPECT_EQ(tensor_data.shape[4], 1);
                EXPECT_EQ(tensor_data.shape[5], 0);
            }
        }
        return 0;
    }
};

TEST_F(NnopbaseExecutorUnitTest, profilingAdditionalSuccess)
{
    AdditionalProfiler additionalProfiler;
    ProfilerStub::GetInstance()->Install(&additionalProfiler);

    EXPECT_EQ(RunBnProfiling(), OK);

    ProfilerStub::GetInstance()->UnInstall();
}

class MIXAICProfiler : public ProfilerStub {
public:
    int32_t MsprofReportCompactInfo(uint32_t agingFlag, const VOID_PTR data, uint32_t length)
    {
        MsprofCompactInfo* report_data = (MsprofCompactInfo*)data;
        auto& profNodeBasicInfo = report_data->data.nodeBasicInfo;
        EXPECT_EQ(report_data->level, MSPROF_REPORT_NODE_LEVEL);
        EXPECT_EQ(report_data->type, MSPROF_REPORT_NODE_BASIC_INFO_TYPE);
        EXPECT_EQ(profNodeBasicInfo.taskType, MSPROF_GE_TASK_TYPE_MIX_AIC);
        return 0;
    }
};

TEST_F(NnopbaseExecutorUnitTest, profiling1971basicInfoSuccess)
{
    MIXAICProfiler MixaicProfiler;
    ProfilerStub::GetInstance()->Install(&MixaicProfiler);

    EXPECT_EQ(RunBnProfiling(), OK);

    ProfilerStub::GetInstance()->UnInstall();
}

class AivProfiler : public ProfilerStub {
public:
    int32_t MsprofReportCompactInfo(uint32_t agingFlag, const VOID_PTR data, uint32_t length)
    {
        MsprofCompactInfo* report_data = (MsprofCompactInfo*)data;
        auto& profNodeBasicInfo = report_data->data.nodeBasicInfo;
        EXPECT_EQ(report_data->level, MSPROF_REPORT_NODE_LEVEL);
        EXPECT_EQ(report_data->type, MSPROF_REPORT_NODE_BASIC_INFO_TYPE);
        EXPECT_EQ(profNodeBasicInfo.taskType, MSPROF_GE_TASK_TYPE_AIV);
        return 0;
    }
};

TEST_F(NnopbaseExecutorUnitTest, profilingAivOp)
{
    AivProfiler aivProfiler;
    ProfilerStub::GetInstance()->Install(&aivProfiler);
    op::internal::opProfilingSwitch.reportFlag = true;
    op::internal::opProfilingSwitch.additionInfoFlag = true;

    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    ASSERT_EQ(RunOp(aclnnCustomOp, tensor, tensor, tensor, tensor), OK);
    aclDestroyTensor(tensor);

    op::internal::opProfilingSwitch.reportFlag = false;
    op::internal::opProfilingSwitch.additionInfoFlag = false;
    ProfilerStub::GetInstance()->UnInstall();
}

struct CacheLastTaskErrStub : public AclrtStub {
    aclError aclrtCacheLastTaskOpInfo(const void* const infoPtr, size_t infoSize) override
    {
        return static_cast<aclError>(1);
    }
};

struct StreamAttrErrStub : public AclrtStub {
    aclError aclrtGetStreamAttribute(aclrtStream stream, aclrtStreamAttr stmAttrType,
                                     aclrtStreamAttrValue* value) override
    {
        return static_cast<aclError>(1);
    }
};

TEST_F(NnopbaseExecutorUnitTest, CacheOpShapeError)
{
    CacheLastTaskErrStub cacheErrStub;
    AclrtStub::GetInstance()->Install(&cacheErrStub);
    EXPECT_EQ(RunBnProfiling(), OK);
    AclrtStub::GetInstance()->UnInstall();

    StreamAttrErrStub streamAttrErrStub;
    AclrtStub::GetInstance()->Install(&streamAttrErrStub);
    EXPECT_EQ(RunBnProfiling(), OK);
    AclrtStub::GetInstance()->UnInstall();
}

TEST_F(NnopbaseExecutorUnitTest, NnopTestInfNanOverflow)
{
    aclrtSetDeviceSatMode(ACL_RT_OVERFLOW_MODE_INFNAN);
    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    ASSERT_EQ(RunOp(aclnnBninferenceDKernel, tensor, tensor, tensor, tensor), OK);
    aclDestroyTensor(tensor);
    aclrtSetDeviceSatMode(ACL_RT_OVERFLOW_MODE_SATURATION);
}

class OverflowRuntimeStub : public RuntimeStub {
public:
    rtError_t rtsNpuGetFloatOverFlowDebugStatus(void* outputAddrPtr, uint64_t outputSize, uint32_t checkMode,
                                                aclrtStream stm)
    {
        ((uint8_t*)outputAddrPtr)[0U] = 1;
        return RT_ERROR_NONE;
    }
    rtError_t rtMemcpy(void* dst, uint64_t destMax, const void* src, uint64_t count, rtMemcpyKind_t kind)
    {
        memcpy(dst, src, count);
        return RT_ERROR_NONE;
    }
};

TEST_F(NnopbaseExecutorUnitTest, NnopTestOverflowSaturation)
{
    OverflowRuntimeStub stub;
    RuntimeStub::GetInstance()->Install(&stub);

    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    ASSERT_EQ(RunOp(aclnnBninferenceDKernel, tensor, tensor, tensor, tensor), OK);
    aclDestroyTensor(tensor);

    RuntimeStub::GetInstance()->UnInstall();
}

TEST_F(NnopbaseExecutorUnitTest, ExecutorCache)
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

    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    (void)NnopbaseAddInput(executor, tensor, 0);
    (void)NnopbaseAddInput(executor, tensor, 1);
    (void)NnopbaseAddInput(executor, tensor, 2);
    (void)NnopbaseAddOutput(executor, tensor, 0);

    RunExecutorSuccess((NnopbaseExecutor*)executor);

    void* exe2 = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                     sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    ASSERT_NE(exe2, nullptr);
    ASSERT_EQ(exe2, executor);

    NnopbaseExecutorGcSpace(executorSpace);
    aclDestroyTensor(tensor);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseExecutorUnitTest, ExecutorTilingWithoutWorkspace)
{
    NnopbaseExecutor* executor = nullptr;
    GetExecutor(executor);
    executor->regInfo->tiling = [](gert::TilingContext* context_) -> unsigned int {
        NnopbaseKernelRunContext* context = reinterpret_cast<NnopbaseKernelRunContext*>(context_);
        NnopbaseWorkspaceSizes* ws = NnopbaseTilingGetWorkspaceSizesPointer(context);
        ws->SetSize(0U);
        return 0;
    };

    size_t workspaceLen = 0U;
    ASSERT_EQ(NnopbaseExecutorRunForWorkspace(executor, &workspaceLen), OK);

#if 0 // comment out for ge decoupling
    ASSERT_EQ(executor->workspaces.num, 0);
#endif

    NnopbaseExecutorGcSpace((void*)executor->space);
    NnopbaseExecutorDeInit(executor);
    delete executor;
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseExecutorUnitTest, ExecutorTilingWithWorkspace)
{
    NnopbaseExecutor* executor = nullptr;
    GetExecutor(executor);
    executor->regInfo->tiling = [](gert::TilingContext* context_) -> unsigned int {
        NnopbaseKernelRunContext* context = reinterpret_cast<NnopbaseKernelRunContext*>(context_);
        NnopbaseWorkspaceSizes* ws = NnopbaseTilingGetWorkspaceSizesPointer(context);
        ws->SetSize(1U);
        return 0;
    };

    size_t workspaceLen = 0U;
    ASSERT_EQ(NnopbaseExecutorRunForWorkspace(executor, &workspaceLen), OK);

    NnopbaseExecutorGcSpace((void*)executor->space);
    NnopbaseExecutorDeInit(executor);
    delete executor;
    NnopbaseUnsetEnvAndClearFolder();
}

// ================= customizedKey 模式动态 bin 查找 (TC02 ~ TC07) =================

// TC02 [场景: tilingContext 重复构建防护]
// customizedKey 路径在查找 bin 阶段已构造过 tilingContext 并置位 hasBuilt，
// DoTiling 阶段再次调用 NnopbaseTilingContextBuild 时应在入口直接返回 OK，不重新绑定槽位、不重置输出。
TEST_F(NnopbaseExecutorUnitTest, ExecutorCustomizedTilingContextBuildOnce)
{
    NnopbaseExecutor* executor = nullptr;
    GetCustomizedExecutor(executor, "StubCustomOp");
    ASSERT_NE(executor->regInfo, nullptr);
    ASSERT_TRUE(executor->regInfo->customizedSimplifiedKey);
    ASSERT_NE(executor->regInfo->genSimplifiedKey, nullptr);
    ASSERT_FALSE(executor->tiling.contextExt.hasBuilt);

    // 首次构造: 由 GenCustomizedDynamicKey 内部触发
    ASSERT_EQ(NnopbaseExecutorGenCustomizedDynamicKey(executor), OK);
    ASSERT_TRUE(executor->tiling.contextExt.hasBuilt);
    ASSERT_NE(executor->tiling.tilingKey, nullptr);
    ASSERT_NE(executor->tiling.numBlocks, nullptr);
    ASSERT_EQ(*(executor->tiling.tilingKey), 0UL);
    ASSERT_EQ(*(executor->tiling.numBlocks), 0U);

    // 写入哨兵值，第二次构造若未短路则会被清零
    const uint64_t* const savedTilingKeyAddr = executor->tiling.tilingKey;
    const uint32_t* const savedNumBlocksAddr = executor->tiling.numBlocks;
    *(executor->tiling.tilingKey) = 0x5A5AUL;
    *(executor->tiling.numBlocks) = 32U;

    ASSERT_EQ(NnopbaseTilingContextBuild(executor), OK);
    ASSERT_TRUE(executor->tiling.contextExt.hasBuilt);
    ASSERT_EQ(executor->tiling.tilingKey, savedTilingKeyAddr);
    ASSERT_EQ(executor->tiling.numBlocks, savedNumBlocksAddr);
    ASSERT_EQ(*(executor->tiling.tilingKey), 0x5A5AUL);
    ASSERT_EQ(*(executor->tiling.numBlocks), 32U);

    FreeCustomizedExecutor(executor);
}

// TC03 [场景: genSimplifiedKey 返回失败 + 入口四处判空]
// gen 返回 GRAPH_FAILED 时应告警返回 ACLNN_ERR_PARAM_INVALID 而非 OP_LOGE，因为可回落动态查找。
// 入口 regInfo/args/opType 判空返回 ACLNN_ERR_PARAM_NULLPTR，genSimplifiedKey 判空返回可回落错误码。
TEST_F(NnopbaseExecutorUnitTest, ExecutorCustomizedGenKeyReturnFail)
{
    NnopbaseExecutor* executor = nullptr;
    GetCustomizedExecutor(executor, "StubCustomOp");
    ASSERT_NE(executor->regInfo, nullptr);

    GenSimplifiedKeyFun savedGen = executor->regInfo->genSimplifiedKey;
    executor->regInfo->genSimplifiedKey = [](gert::TilingContext*, ge::char_t*) -> ge::graphStatus {
        return ge::GRAPH_FAILED;
    };

    ASSERT_EQ(NnopbaseExecutorGenCustomizedDynamicKey(executor), ACLNN_ERR_PARAM_INVALID);
    // gen 失败处即返回，未拼后缀
    ASSERT_NE(executor->binInfoKey.len, strlen("StubCustomOp") + VERBKEY_PREFIX_LEN + strlen("diy,99"));

    // regInfo 判空：读 genSimplifiedKey 需先解引用它
    NnopbaseRegInfo* savedRegInfo = executor->regInfo;
    executor->regInfo = nullptr;
    ASSERT_EQ(NnopbaseExecutorGenCustomizedDynamicKey(executor), ACLNN_ERR_PARAM_NULLPTR);
    executor->regInfo = savedRegInfo;

    // args 判空：须在进入 TilingContextBuild 前拦截，否则解引用 args->inputs 崩溃
    NnopbaseExecutorArgs* savedArgs = executor->args;
    executor->args = nullptr;
    ASSERT_NE(NnopbaseExecutorGenCustomizedDynamicKey(executor), OK);
    executor->args = savedArgs;

    // opType 判空：拼接前缀需按字节读取它
    NnopbaseChar* savedOpType = executor->opType;
    executor->opType = nullptr;
    ASSERT_NE(NnopbaseExecutorGenCustomizedDynamicKey(executor), OK);
    executor->opType = savedOpType;

    // genSimplifiedKey 判空：未注册时仅告警，返回可回落错误码而非 NULLPTR
    executor->regInfo->genSimplifiedKey = nullptr;
    ASSERT_EQ(NnopbaseExecutorGenCustomizedDynamicKey(executor), ACLNN_ERR_PARAM_INVALID);

    executor->regInfo->genSimplifiedKey = savedGen;
    FreeCustomizedExecutor(executor);
}

// TC04 [场景: GenCustomizedDynamicKey 成功但 CollectorFindBinInfo 查找失败]
// gen 产出 "diy,98"，与 json 注册的 "diy,99" 仅末位差一字节，说明后缀参与了 hashKey 计算，
// TC05 的命中不是碰巧。
TEST_F(NnopbaseExecutorUnitTest, ExecutorCustomizedBinNotFound)
{
    NnopbaseExecutor* executor = nullptr;
    GetCustomizedExecutor(executor, "StubCustomOp");
    ASSERT_NE(executor->regInfo, nullptr);

    GenSimplifiedKeyFun savedGen = executor->regInfo->genSimplifiedKey;
    executor->regInfo->genSimplifiedKey = [](gert::TilingContext*, ge::char_t* key) -> ge::graphStatus {
        strcpy_s(key, GEN_SIMPLIFIED_KEY_BUF_LEN, "diy,98"); // 与 json 的 "diy,99" 差一字节
        return ge::GRAPH_SUCCESS;
    };

    ASSERT_EQ(NnopbaseExecutorGenCustomizedDynamicKey(executor), OK);
    ASSERT_EQ(executor->binInfoKey.len, strlen("StubCustomOp") + VERBKEY_PREFIX_LEN + strlen("diy,98"));
    auto binInfo = NnopbaseCollectorFindBinInfo(executor->regInfo, executor->binInfoKey.hashKey,
                                                &(executor->binInfoKey.verbose[0U]), executor->binInfoKey.len);
    ASSERT_EQ(binInfo, nullptr);

    executor->regInfo->genSimplifiedKey = savedGen;
    FreeCustomizedExecutor(executor);
}

// TC05 [场景: GenCustomizedDynamicKey 成功且 CollectorFindBinInfo 查找成功]
// 端到端闭环: 加载期 ConvertCustomizedVerbKey 对 json 原串建的索引，与运行期拼出的 verbKey
// 须逐字节相同、hashKey 相等，这是能命中 bin 的前提。
TEST_F(NnopbaseExecutorUnitTest, ExecutorCustomizedFindBinOk)
{
    NnopbaseExecutor* executor = nullptr;
    GetCustomizedExecutor(executor, "StubCustomOp");
    ASSERT_NE(executor->regInfo, nullptr);
    ASSERT_EQ(g_nnopbaseSysCfgParams.precision, 0); // 前缀两端一致的前提

    // 加载期: 用 json 原串算出 verbKey 与 hashKey
    const char* strKey = "StubCustomOp/d=0,p=0/diy,99";
    NnopbaseUChar loadKey[NNOPBASE_VEB_KEY_LEN] = {0};
    uint32_t loadSize = 0U;
    ASSERT_EQ(NnopbaseCollectorConvertCustomizedVerbKey(strKey, loadKey, &loadSize), OK);
    const size_t loadHash = NnopbaseHashBinary(loadKey, loadSize) % NNOPBASE_NORM_MAX_BIN_BUCKETS;

    // 运行期: 生成 verbKey 并查表
    ASSERT_TRUE(NnopbaseExecutorGetDynamicBinInfo(executor));
    ASSERT_NE(executor->args->binInfo, nullptr);

    // 两端逐字节闭环
    ASSERT_EQ(executor->binInfoKey.len, loadSize);
    ASSERT_EQ(executor->binInfoKey.len, strlen("StubCustomOp") + VERBKEY_PREFIX_LEN + strlen("diy,99"));
    ASSERT_EQ(memcmp(&(executor->binInfoKey.verbose[0U]), loadKey, loadSize), 0);
    ASSERT_EQ(executor->binInfoKey.hashKey, loadHash);

    FreeCustomizedExecutor(executor);
}

// TC06 [场景: FindBinInfoByCustomizedKey 失败 → GenDynamicKey 成功 → FindBinInfo 失败]
// StubCustomOpNoGen 为 customizedKey 模式但未注册 gen，自定义键路径在 genSimplifiedKey 判空处告警返回；
// 回落后按张量描述生成动态键，但该算子 binTbl 内的 bin 均按 customizedKey 规则建索引，
// 动态键必然查不到，故两级均未命中返回 false。
TEST_F(NnopbaseExecutorUnitTest, ExecutorCustomizedFallbackFindBinFail)
{
    NnopbaseExecutor* executor = nullptr;
    GetCustomizedExecutor(executor, "StubCustomOpNoGen");
    ASSERT_NE(executor->regInfo, nullptr);
    ASSERT_TRUE(executor->regInfo->customizedSimplifiedKey);
    ASSERT_EQ(executor->regInfo->genSimplifiedKey, nullptr); // 由 TC01 的加载期行为保证

    ASSERT_FALSE(NnopbaseExecutorGetDynamicBinInfo(executor));
    ASSERT_EQ(executor->args->binInfo, nullptr);

    // len 为动态键长度而非 customizedKey 长度，证明回落路径确实执行过
    const uint32_t dynLen = static_cast<uint32_t>(strlen("StubCustomOpNoGen") + VERBKEY_PREFIX_LEN +
                                                  (CUSTOMIZED_STUB_INPUT_NUM + CUSTOMIZED_STUB_OUTPUT_NUM) *
                                                      DYNKEY_BYTES_PER_TENSOR);
    ASSERT_EQ(executor->binInfoKey.len, dynLen);
    ASSERT_NE(executor->binInfoKey.len, strlen("StubCustomOpNoGen") + VERBKEY_PREFIX_LEN + strlen("diy,99"));
    // 回落前整体置换了 verbose，bufLen 随之复位而非沿用旧容量
    ASSERT_EQ(executor->binInfoKey.bufLen, executor->binInfoKey.verbose.size());
    ASSERT_GE(executor->binInfoKey.bufLen, dynLen);

    FreeCustomizedExecutor(executor);
}

// TC07 [场景: FindBinInfoByCustomizedKey 失败 → GenDynamicKey 成功 → FindBinInfo 成功]
// StubMixKeyOp 在自定义包声明 mode=0、内置包声明 mode=2，加载期按此顺序两次调 AddRepoInfo，
// 于是同一 binTbl 内动态键与 customizedKey 两种索引共存，标志位被后加载的内置包置为 true。
// 桩 gen 产出 "diy,99" 与内置包 json 的 "diy,zzz" 不匹配，故回落是加载期数据天然导致的，
// 全程不篡改 regInfo。
TEST_F(NnopbaseExecutorUnitTest, ExecutorCustomizedFallbackFindBinOk)
{
    NnopbaseExecutor* executor = nullptr;
    GetCustomizedExecutor(executor, "StubMixKeyOp");
    ASSERT_NE(executor->regInfo, nullptr);
    // 标志位由最后加载的内置包(mode=2)置位
    ASSERT_TRUE(executor->regInfo->customizedSimplifiedKey);
    ASSERT_NE(executor->regInfo->genSimplifiedKey, nullptr);

    // customizedKey 路生成的 verbKey 与 json 的 "diy,zzz" 不符，必然未命中
    ASSERT_EQ(NnopbaseExecutorGenCustomizedDynamicKey(executor), OK);
    ASSERT_EQ(NnopbaseCollectorFindBinInfo(executor->regInfo, executor->binInfoKey.hashKey,
                                           &(executor->binInfoKey.verbose[0U]), executor->binInfoKey.len),
              nullptr);

    // 走完整分派: customizedKey 未命中 → 复位 → 动态键 → 命中自定义包那条 bin
    NnopbaseExecutor* full = nullptr;
    GetCustomizedExecutor(full, "StubMixKeyOp");
    ASSERT_TRUE(NnopbaseExecutorGetDynamicBinInfo(full));
    ASSERT_NE(full->args->binInfo, nullptr);
    const uint32_t fallbackLen = full->binInfoKey.len;
    const size_t fallbackHash = full->binInfoKey.hashKey;
    std::vector<NnopbaseUChar> fallbackKey(full->binInfoKey.verbose.begin(),
                                           full->binInfoKey.verbose.begin() + fallbackLen);

    // 对照组: 同算子临时关掉标志，直接走动态键路径
    NnopbaseExecutor* direct = nullptr;
    GetCustomizedExecutor(direct, "StubMixKeyOp");
    const bool savedFlag = direct->regInfo->customizedSimplifiedKey;
    direct->regInfo->customizedSimplifiedKey = false;
    ASSERT_TRUE(NnopbaseExecutorGetDynamicBinInfo(direct));
    ASSERT_NE(direct->args->binInfo, nullptr);
    direct->regInfo->customizedSimplifiedKey = savedFlag; // 立即恢复，regInfo 按 opType 共享

    // 回落产出的动态键与直接生成的动态键完全一致，且命中同一条 bin
    ASSERT_EQ(full->args->binInfo, direct->args->binInfo);
    ASSERT_EQ(fallbackLen, direct->binInfoKey.len);
    ASSERT_EQ(fallbackHash, direct->binInfoKey.hashKey);
    ASSERT_EQ(memcmp(&fallbackKey[0U], &(direct->binInfoKey.verbose[0U]), fallbackLen), 0);

    FreeCustomizedExecutor(direct);
    FreeCustomizedExecutor(full);
    FreeCustomizedExecutor(executor);
}

TEST_F(NnopbaseExecutorUnitTest, ExecutorOptypeFailed)
{
    NnopbaseExecutor* executor = nullptr;
    GetExecutor(executor, "test");

    auto ret = NnopbaseExecutorSetRegInfo(executor, "test");
    ASSERT_EQ(ret, ACLNN_ERR_PARAM_NULLPTR);

    NnopbaseExecutorGcSpace((void*)executor->space);
    NnopbaseExecutorDeInit(executor);
    delete executor;
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseExecutorUnitTest, ExecutorSetOverflowAddrNotNull1)
{
    aclrtSetDeviceSatMode(ACL_RT_OVERFLOW_MODE_SATURATION);
    g_nnopbaseSysCfgParams.overflowAddr = nullptr;
    NnopbaseExecutorSetGlobalConfig();
    ASSERT_NE(g_nnopbaseSysCfgParams.overflowAddr, nullptr);
}

TEST_F(NnopbaseExecutorUnitTest, ExecutorSetOverflowAddrNotNull2)
{
    aclrtSetDeviceSatMode(ACL_RT_OVERFLOW_MODE_INFNAN);
    g_nnopbaseSysCfgParams.overflowAddr = nullptr;
    NnopbaseExecutorSetGlobalConfig();
    ASSERT_NE(g_nnopbaseSysCfgParams.overflowAddr, nullptr);
    aclrtSetDeviceSatMode(ACL_RT_OVERFLOW_MODE_SATURATION);
}

TEST_F(NnopbaseExecutorUnitTest, ExecutorSetOverflowAddrUndef)
{
    aclrtSetDeviceSatMode(ACL_RT_OVERFLOW_MODE_UNDEF);
    g_nnopbaseSysCfgParams.overflowAddr = nullptr;
    NnopbaseExecutorSetGlobalConfig();
    ASSERT_EQ(g_nnopbaseSysCfgParams.overflowAddr, nullptr);
}

TEST_F(NnopbaseExecutorUnitTest, ExecutorSetDeterministicClose)
{
    const int64_t configVal = 0;
    aclrtSetSysParamOpt(ACL_OPT_DETERMINISTIC, configVal);
    NnopbaseExecutorSetGlobalConfig();
    ASSERT_EQ(g_nnopbaseSysCfgParams.deterministicLevel, 0U);
}

TEST_F(NnopbaseExecutorUnitTest, ExecutorSetDeterministicOpen)
{
    int64_t configVal = 1;
    aclrtSetSysParamOpt(ACL_OPT_DETERMINISTIC, configVal);
    aclrtSetSysParamOpt(ACL_OPT_STRONG_CONSISTENCY, 0);
    NnopbaseExecutorSetGlobalConfig();
    ASSERT_EQ(g_nnopbaseSysCfgParams.deterministicLevel, 1U);

    configVal = 0;
    aclrtSetSysParamOpt(ACL_OPT_DETERMINISTIC, configVal);
    NnopbaseExecutorSetGlobalConfig();
    ASSERT_EQ(g_nnopbaseSysCfgParams.deterministicLevel, 0U);
}

TEST_F(NnopbaseExecutorUnitTest, ExecutorSetDeterministicStrongConsistency)
{
    aclrtSetSysParamOpt(ACL_OPT_DETERMINISTIC, 1);
    aclrtSetSysParamOpt(ACL_OPT_STRONG_CONSISTENCY, 1);
    NnopbaseExecutorSetGlobalConfig();
    ASSERT_EQ(g_nnopbaseSysCfgParams.deterministicLevel, 2U);

    aclrtSetSysParamOpt(ACL_OPT_STRONG_CONSISTENCY, 0);
    aclrtSetSysParamOpt(ACL_OPT_DETERMINISTIC, 0);
}

TEST_F(NnopbaseExecutorUnitTest, ExecutorSetImplMode)
{
    g_nnopbaseSysCfgParams.precision = 1;
    NnopbaseExecutorSetGlobalConfig();
    ASSERT_EQ(g_nnopbaseSysCfgParams.implMode, nnopbase::IMPL_MODE_HIGH_PERFORMANCE);

    g_nnopbaseSysCfgParams.precision = 2;
    NnopbaseExecutorSetGlobalConfig();
    ASSERT_EQ(g_nnopbaseSysCfgParams.implMode, nnopbase::IMPL_MODE_HIGH_PRECISION);

    g_nnopbaseSysCfgParams.precision = 0;
    NnopbaseExecutorSetGlobalConfig();
    ASSERT_EQ(g_nnopbaseSysCfgParams.implMode, "");
}

TEST_F(NnopbaseExecutorUnitTest, ExecutorSetDebugKernel)
{
    g_nnopbaseSysCfgParams.enableDebugKernel = true;
    aclrtCtxSetSysParamOpt(ACL_OPT_ENABLE_DEBUG_KERNEL, 0);
    NnopbaseExecutorSetGlobalConfig();
    ASSERT_EQ(g_nnopbaseSysCfgParams.enableDebugKernel, false);

    aclrtCtxSetSysParamOpt(ACL_OPT_ENABLE_DEBUG_KERNEL, 1);
    NnopbaseExecutorSetGlobalConfig();
    ASSERT_EQ(g_nnopbaseSysCfgParams.enableDebugKernel, true);

    aclrtCtxSetSysParamOpt(ACL_OPT_ENABLE_DEBUG_KERNEL, 0);
    NnopbaseExecutorSetGlobalConfig();
    ASSERT_EQ(g_nnopbaseSysCfgParams.enableDebugKernel, false);
}

TEST_F(NnopbaseExecutorUnitTest, NnopBaseComputeNodeInfo)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char* opType = "bninference_d_kernel";
    char inputDesc[] = {1, 1, 1};
    char outputDesc[] = {1};
    char attrDesc[] = {1};
    void* executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                         sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    ASSERT_NE(executor, nullptr);

    const int64_t configVal = 0;
    aclrtCtxSetSysParamOpt(ACL_OPT_DETERMINISTIC, configVal);

    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    (void)NnopbaseAddInput(executor, tensor, 0);
    (void)NnopbaseAddInput(executor, tensor, 1);
    (void)NnopbaseAddInput(executor, tensor, 2);
    (void)NnopbaseAddOutput(executor, tensor, 0);

    int64_t bias1[600] = {1};
    size_t bias1_size = 4800;
    if (bias1) {
        NnopbaseAddAttrWithDtype(executor, bias1, bias1_size, 0, kNnopbaseInt);
    } else {
        static int64_t bias1_def[] = {1, 2};
        static size_t bias1_size_def = 2 * sizeof(int64_t);
        NnopbaseAddAttrWithDtype(executor, bias1_def, bias1_size_def, 0, kNnopbaseInt);
    }

    RunExecutorSuccess((NnopbaseExecutor*)executor);

    NnopbaseExecutorGcSpace(executorSpace);
    aclDestroyTensor(tensor);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseExecutorUnitTest, NnopbaseParamCheck)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char* opType = "bninference_d_kernel";
    char inputDesc[] = {0, 1, 1};
    char outputDesc[] = {1};
    char attrDesc[] = {};

    void* executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                         sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    ASSERT_NE(executor, nullptr);

    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);

    (void)NnopbaseAddInput(executor, nullptr, 0);
    (void)NnopbaseAddInput(executor, tensor, 1);
    (void)NnopbaseAddInput(executor, tensor, 2);
    (void)NnopbaseAddOutput(executor, tensor, 0);

    TensorDesc inputDesc0[3] = {
        {ge::DT_FLOAT, ge::FORMAT_ND}, {ge::DT_FLOAT, ge::FORMAT_ND}, {ge::DT_FLOAT, ge::FORMAT_ND}};
    TensorDesc outputDesc0[1] = {{ge::DT_FLOAT, ge::FORMAT_ND}};
    SupportInfo list1 = {inputDesc0, 3, outputDesc0, 1};
    SupportInfo supportInfo0[1] = {list1};
    SupportInfo supportInfo1[1] = {list1};
    OpSocSupportInfo socSupportInfo0 = {supportInfo0, 1};
    OpSocSupportInfo socSupportInfo1 = {supportInfo1, 1};
    OpSocSupportInfo opSocSupportList[2] = {socSupportInfo0, socSupportInfo1};
    OpSupportList supportList = {opSocSupportList, 2};
    uint32_t socSupportList[] = {nnopbase::SOC_VERSION_ASCEND910A, nnopbase::SOC_VERSION_ASCEND910B};
    ASSERT_EQ(NnopbaseAddSupportList(executor, &supportList, socSupportList, 2), OK);

    size_t workspaceLen = 0U;
    (void)NnopbaseRunForWorkspace(executor, &workspaceLen);
    EXPECT_EQ(CheckSocVersionAndParam((NnopbaseExecutor*)executor, NnopbaseParamCheckMode::kCheckRequiredIo), OK);

    NnopbaseExecutorGcSpace(executorSpace);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseExecutorUnitTest, NnopbaseParamCheckNoSupportSoc)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char* opType = "bninference_d_kernel";
    char inputDesc[] = {0, 1, 1};
    char outputDesc[] = {1};
    char attrDesc[] = {};

    void* executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                         sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    ASSERT_NE(executor, nullptr);

    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);

    (void)NnopbaseAddInput(executor, nullptr, 0);
    (void)NnopbaseAddInput(executor, tensor, 1);
    (void)NnopbaseAddInput(executor, tensor, 2);
    (void)NnopbaseAddOutput(executor, tensor, 0);

    TensorDesc inputDesc0[3] = {
        {ge::DT_FLOAT, ge::FORMAT_ND}, {ge::DT_FLOAT, ge::FORMAT_ND}, {ge::DT_FLOAT, ge::FORMAT_ND}};
    TensorDesc outputDesc0[1] = {{ge::DT_FLOAT, ge::FORMAT_ND}};
    SupportInfo list1 = {inputDesc0, 3, outputDesc0, 1};
    SupportInfo supportInfo0[1] = {list1};
    OpSocSupportInfo socSupportInfo0 = {supportInfo0, 1};
    OpSocSupportInfo opSocSupportList[1] = {socSupportInfo0};
    OpSupportList supportList = {opSocSupportList, 1};
    uint32_t socSupportList[] = {nnopbase::SOC_VERSION_ASCEND910B};
    ASSERT_EQ(NnopbaseAddSupportList(executor, &supportList, socSupportList, 1), OK);

    size_t workspaceLen = 0U;
    (void)NnopbaseRunForWorkspace(executor, &workspaceLen);
    EXPECT_EQ(CheckSocVersionAndParam((NnopbaseExecutor*)executor, NnopbaseParamCheckMode::kCheckRequiredIo),
              ACLNN_ERR_INNER_FIND_KERNEL_ERROR);

    NnopbaseExecutorGcSpace(executorSpace);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseExecutorUnitTest, TestExecutorInit)
{
    NnopbaseExecutor* executor = nullptr;
    GetExecutor(executor);
    ASSERT_NE(executor, nullptr);
    ASSERT_EQ(executor->args, nullptr);
    ASSERT_EQ(executor->tiling.tilingKey, nullptr);
    ASSERT_EQ(executor->tiling.numBlocks, nullptr);
    ASSERT_EQ(executor->tiling.aicpuNumBlocks, nullptr);
    ASSERT_EQ(executor->workspaces.num, 0);
    ASSERT_EQ(executor->binInfoKey.len, 0);
    ASSERT_EQ(executor->binInfoKey.bufLen, 0);
    ASSERT_EQ(executor->hasTiling, true);
    ASSERT_EQ(executor->isWork, false);
    ASSERT_EQ(executor->dfx.tilingId, nullptr);
    NnopbaseExecutorGcSpace((void*)executor->space);
    NnopbaseExecutorDeInit(executor);
    delete executor;
}

class UtDump : public Adx::DumpStub {
public:
    int32_t AdumpDumpTensorV2(const std::string& opType, const std::string& opName,
                              const std::vector<Adx::TensorInfoV2>& tensors, aclrtStream stream)
    {
        if (opType.find("L2DfxAbsent_") != string::npos) {
            for (auto& info : tensors) {
                EXPECT_EQ(info.shape.size(), 5);
                EXPECT_EQ(info.dataType, aclDataType::ACL_FLOAT);
                EXPECT_EQ(info.tensorSize, 4);
                EXPECT_EQ(info.addrType, Adx::AddressType::TRADITIONAL);
            }
        }
        EXPECT_EQ(opName, std::string("_L0bninference_d_kernel"));
        return 0;
    }

    bool AdumpIsDumpEnable(Adx::DumpType type) override { return true; }
};

TEST_F(NnopbaseExecutorUnitTest, NnopbaseDumpTensorWorkspaceCheck)
{
    UtDump dumpStub;
    Adx::DumpStub::GetInstance()->Install(&dumpStub);

    NnopbaseExecutor* executor = nullptr;
    GetExecutor(executor);
    ASSERT_NE(executor, nullptr);

    size_t workspaceLen = 0U;
    ASSERT_EQ(NnopbaseExecutorRunForWorkspace(executor, &workspaceLen), OK);
    std::vector<uint8_t> vec(workspaceLen);
    ASSERT_EQ(workspaceLen, 0U);
    void* workspace = (void*)vec.data();
    ASSERT_EQ(NnopbaseExecutorRunWithWorkspace(executor, nullptr, workspace, workspaceLen), OK);

    NnopbaseExecutorGcSpace((void*)executor->space);
    NnopbaseExecutorDeInit(executor);
    delete executor;
    NnopbaseUnsetEnvAndClearFolder();
    Adx::DumpStub::GetInstance()->UnInstall();
}

void* exceptionDumpWorkspaceAddr = nullptr;
class UtExceptionDump : public Adx::DumpStub {
public:
    int32_t AdumpAddExceptionOperatorInfoV2(const Adx::OperatorInfoV2& opInfo)
    {
        if (opInfo.opType == "bninference_d_kernel") {
            EXPECT_EQ(opInfo.additionalInfo.at("workspace_bytes"), std::string("0 "));
            std::stringstream workspaceAddrStr;
            workspaceAddrStr << std::hex << exceptionDumpWorkspaceAddr << " ";
            EXPECT_EQ(opInfo.additionalInfo.at("workspace_addrs"), workspaceAddrStr.str());
            EXPECT_EQ(opInfo.additionalInfo.at("tvm_magic"), std::string("ACL_RT_BINARY_MAGIC_ELF_AICORE"));
            EXPECT_EQ(opInfo.additionalInfo.at("block_dim"), std::string("0"));
            EXPECT_EQ(opInfo.additionalInfo.at("tiling_key"), std::string("0"));
            for (auto& info : opInfo.tensorInfos) {
                if (info.type == Adx::TensorType::WORKSPACE) {
                    EXPECT_EQ(info.shape.size(), 1);
                    EXPECT_EQ(info.dataType, aclDataType::ACL_UINT8);
                    EXPECT_EQ(info.tensorSize, 0);
                    EXPECT_EQ(info.addrType, Adx::AddressType::TRADITIONAL);
                } else {
                    EXPECT_EQ(info.shape.size(), 5);
                    EXPECT_EQ(info.dataType, aclDataType::ACL_FLOAT);
                    EXPECT_EQ(info.tensorSize, 4);
                    EXPECT_EQ(info.addrType, Adx::AddressType::TRADITIONAL);
                }
            }
        }
        return 0;
    }
};

TEST_F(NnopbaseExecutorUnitTest, TestAclnnExceptionDump)
{
    nnopbase::ArgsPool::GetInstance().Finalize();
    UtExceptionDump dumpStub;
    Adx::DumpStub::GetInstance()->Install(&dumpStub);

    NnopbaseExecutor* executor = nullptr;
    GetExecutor(executor);
    ASSERT_NE(executor, nullptr);

    size_t workspaceLen = 0U;
    ASSERT_EQ(NnopbaseExecutorRunForWorkspace(executor, &workspaceLen), OK);
    std::vector<uint8_t> vec(workspaceLen);
    ASSERT_EQ(workspaceLen, 0);
    void* workspace = (void*)vec.data();
    exceptionDumpWorkspaceAddr = workspace;
    ASSERT_EQ(NnopbaseExecutorRunWithWorkspace(executor, nullptr, workspace, workspaceLen), OK);

    NnopbaseExecutorGcSpace((void*)executor->space);
    NnopbaseExecutorDeInit(executor);
    delete executor;
    NnopbaseUnsetEnvAndClearFolder();
    Adx::DumpStub::GetInstance()->UnInstall();
}

TEST_F(NnopbaseExecutorUnitTest, NnopbaseSetShapeAndAddrSuccess)
{
    NnopbaseExecutor* executor = nullptr;
    GetExecutor(executor);
    ASSERT_NE(executor, nullptr);

    ASSERT_EQ(executor->ownArgs.inputs.num, 3);
    ASSERT_EQ(executor->ownArgs.inputs.extTensors[0].rt2Tensor.GetStorageShape(), gert::Shape({1, 1, 1, 1, 1}));
    ASSERT_EQ(executor->ownArgs.inputs.extTensors[0].rt2Tensor.GetAddr(), nullptr);
    ASSERT_EQ(executor->ownArgs.inputs.extTensors[1].rt2Tensor.GetStorageShape(), gert::Shape({1, 1, 1, 1, 1}));
    ASSERT_EQ(executor->ownArgs.inputs.extTensors[1].rt2Tensor.GetAddr(), nullptr);
    ASSERT_EQ(executor->ownArgs.inputs.extTensors[2].rt2Tensor.GetStorageShape(), gert::Shape({1, 1, 1, 1, 1}));
    ASSERT_EQ(executor->ownArgs.inputs.extTensors[2].rt2Tensor.GetAddr(), nullptr);
    ASSERT_EQ(executor->ownArgs.outputs.num, 1);
    ASSERT_EQ(executor->ownArgs.outputs.extTensors[0].rt2Tensor.GetStorageShape(), gert::Shape({1, 1, 1, 1, 1}));
    ASSERT_EQ(executor->ownArgs.outputs.extTensors[0].rt2Tensor.GetAddr(), nullptr);

    NnopbaseExecutorGcSpace((void*)executor->space);
    NnopbaseExecutorDeInit(executor);
    delete executor;
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseExecutorUnitTest, NnopbaseMultiThreadForMatchCache)
{
    NnopbaseExecutor* executor = nullptr;
    GetExecutor(executor);
    ASSERT_NE(executor, nullptr);

    nnopbase::ArgsPool::GetInstance().MatchArgs(executor);
    std::vector<std::future<void>> futures(10);
    for (size_t i = 0; i < 5; ++i) {
        futures[2 * i] = std::async(std::launch::async,
                                    [&]() { nnopbase::ArgsPool::GetInstance().MatchArgs(executor); });
        futures[2 * i + 1] = std::async(std::launch::async,
                                        [&]() { nnopbase::ArgsPool::GetInstance().CreateArgs(executor); });
    }
    for (size_t i = 0; i < 10; ++i) {
        futures[i].get();
    }

    NnopbaseExecutorGcSpace((void*)executor->space);
    NnopbaseExecutorDeInit(executor);
    delete executor;
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseExecutorUnitTest, NnopbaseCacheMatchArgs)
{
    NnopbaseExecutor* executor = nullptr;
    GetExecutorWithAttr(executor);
    ASSERT_NE(executor, nullptr);
    executor->space = new NnopbaseExecutorSpace();

    ASSERT_EQ(nnopbase::ArgsPool::GetInstance().MatchArgs(executor), false);
    ASSERT_EQ(nnopbase::ArgsPool::GetInstance().CreateArgs(executor), OK);
    executor->args->isVisit = false;
    NnopbaseBinInfo binInfo;
    binInfo.isStaticShape = false;
    executor->args->binInfo = &binInfo;
    ASSERT_EQ(nnopbase::ArgsPool::GetInstance().MatchArgs(executor), true);

    NnopbaseExecutorGcSpace((void*)executor->space);
    NnopbaseExecutorDeInit(executor);
    delete executor;
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseExecutorUnitTest, NnopbaseCacheNoMatchArgs)
{
    NnopbaseExecutor* executor1 = nullptr;
    NnopbaseExecutor* executor2 = nullptr;
    GetExecutorWithAttr(executor1);
    GetExecutor(executor2);
    ASSERT_NE(executor1, nullptr);
    ASSERT_NE(executor2, nullptr);
    auto space = new NnopbaseExecutorSpace();
    executor1->space = space;

    NnopbaseExecutorGcSpace((void*)executor2->space);
    executor2->space = space;

    ASSERT_EQ(nnopbase::ArgsPool::GetInstance().MatchArgs(executor1), false);
    ASSERT_EQ(nnopbase::ArgsPool::GetInstance().CreateArgs(executor1), OK);
    executor1->args->isVisit = false;
    ASSERT_EQ(nnopbase::ArgsPool::GetInstance().MatchArgs(executor2), false);

    NnopbaseExecutorGcSpace((void*)space);
    NnopbaseExecutorDeInit(executor1);
    NnopbaseExecutorDeInit(executor2);
    delete executor1;
    delete executor2;
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseExecutorUnitTest, NnopbaseExceededMaxNum)
{
    NnopbaseExecutor* executor = nullptr;
    GetExecutor(executor);
    ASSERT_NE(executor, nullptr);

    nnopbase::ArgsPool::maxCacheNum = 100;
    nnopbase::ArgsPool::GetInstance().Finalize();
    ASSERT_EQ(nnopbase::ArgsPool::GetInstance().MatchArgs(executor), false);
    for (size_t i = 0; i < 200U; ++i) {
        ASSERT_EQ(nnopbase::ArgsPool::GetInstance().CreateArgs(executor), OK);
    }
    for (auto& iter : nnopbase::ArgsPool::GetInstance().argsMap) {
        for (auto& args : iter.second) {
            if (args != nullptr) {
                args->isVisit = false;
            }
        }
    }
    for (size_t i = 0; i < 200U; ++i) {
        ASSERT_EQ(nnopbase::ArgsPool::GetInstance().CreateArgs(executor), OK);
    }
    ASSERT_EQ(nnopbase::ArgsPool::GetInstance().MatchArgs(executor), false);
    size_t workspaceLen = 0U;
    ASSERT_EQ(NnopbaseExecutorRunForWorkspace(executor, &workspaceLen), OK);

    NnopbaseExecutorGcSpace((void*)executor->space);
    NnopbaseExecutorDeInit(executor);
    delete executor;
    NnopbaseUnsetEnvAndClearFolder();
    nnopbase::ArgsPool::GetInstance().Finalize();
    nnopbase::ArgsPool::maxCacheNum = 10000;
}

TEST_F(NnopbaseExecutorUnitTest, NnopbaseOneCacheNum)
{
    NnopbaseExecutor* executor1 = nullptr;
    NnopbaseExecutor* executor2 = nullptr;
    GetExecutorWithAttr(executor1);
    GetExecutor(executor2);
    ASSERT_NE(executor1, nullptr);
    ASSERT_NE(executor2, nullptr);
    auto space = new NnopbaseExecutorSpace();
    executor1->space = space;
    NnopbaseExecutorGcSpace((void*)executor2->space);
    executor2->space = space;

    nnopbase::ArgsPool::maxCacheNum = 1;
    nnopbase::ArgsPool::GetInstance().Finalize();
    ASSERT_EQ(nnopbase::ArgsPool::GetInstance().MatchArgs(executor1), false);
    ASSERT_EQ(nnopbase::ArgsPool::GetInstance().CreateArgs(executor1), OK);
    executor1->args->isVisit = false;
    ASSERT_EQ(nnopbase::ArgsPool::GetInstance().MatchArgs(executor2), OK);
    ASSERT_EQ(nnopbase::ArgsPool::GetInstance().CreateArgs(executor2), OK);

    NnopbaseExecutorGcSpace((void*)space);
    NnopbaseExecutorDeInit(executor1);
    NnopbaseExecutorDeInit(executor2);
    delete executor1;
    delete executor2;
    NnopbaseUnsetEnvAndClearFolder();
    nnopbase::ArgsPool::GetInstance().Finalize();
    nnopbase::ArgsPool::maxCacheNum = 10000;
}

TEST_F(NnopbaseExecutorUnitTest, NnopbaseZeroCacheNum)
{
    NnopbaseExecutor* executor = nullptr;
    GetExecutor(executor);
    ASSERT_NE(executor, nullptr);

    nnopbase::ArgsPool::maxCacheNum = 0;
    nnopbase::ArgsPool::GetInstance().Finalize();
    ASSERT_EQ(nnopbase::ArgsPool::GetInstance().MatchArgs(executor), false);
    ASSERT_EQ(executor->ownArgs.enableCache, false);
    ASSERT_EQ(nnopbase::ArgsPool::GetInstance().CreateArgs(executor), OK);

    NnopbaseExecutorGcSpace((void*)executor->space);
    NnopbaseExecutorDeInit(executor);
    delete executor;
    NnopbaseUnsetEnvAndClearFolder();
    nnopbase::ArgsPool::GetInstance().Finalize();
    nnopbase::ArgsPool::maxCacheNum = 10000;
}

TEST_F(NnopbaseExecutorUnitTest, ExecutorInitEmptyOutput)
{
    NnopbaseExecutor* executor = new NnopbaseExecutor;
    ASSERT_NE(executor, nullptr);

    char inputDesc[] = {1, 1, 1};
    char outputDesc[] = {1};
    char attrDesc[] = {};
    auto ret = NnopbaseExecutorInit(
        executor, {inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc, sizeof(outputDesc) / sizeof(char), attrDesc,
                   sizeof(attrDesc) / sizeof(char)});
    ASSERT_EQ(ret, OK);

    std::vector<int64_t> shape = {0};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    ASSERT_EQ(NnopbaseExecutorUpdateTensorsIndex(&(executor->ownArgs.inputs), 0), OK);
    ASSERT_EQ(NnopbaseExecutorAddTensor(executor, tensor, 0, true, false), OK);
    ASSERT_EQ(NnopbaseExecutorUpdateTensorsIndex(&(executor->ownArgs.inputs), 1), OK);
    ASSERT_EQ(NnopbaseExecutorAddTensor(executor, tensor, 1, true, false), OK);
    ASSERT_EQ(NnopbaseExecutorUpdateTensorsIndex(&(executor->ownArgs.inputs), 2), OK);
    ASSERT_EQ(NnopbaseExecutorAddTensor(executor, tensor, 2, true, false), OK);

    ASSERT_EQ(NnopbaseExecutorUpdateTensorsIndex(&(executor->ownArgs.outputs), 0), OK);
    ASSERT_EQ(NnopbaseExecutorAddTensor(executor, tensor, 0, false, false), OK);
    ASSERT_EQ(executor->ownArgs.outputs.paramDescs.emptyNum, 1U);
    aclDestroyTensor(tensor);
    NnopbaseExecutorDeInit(executor);
    delete executor;
}

TEST_F(NnopbaseExecutorUnitTest, ExecutorInitEmptyDynamicOutput)
{
    NnopbaseExecutor* executor = new NnopbaseExecutor;
    ASSERT_NE(executor, nullptr);

    char inputDesc[] = {2, 2, 2};
    char outputDesc[] = {2};
    char attrDesc[] = {};
    auto ret = NnopbaseExecutorInit(
        executor, {inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc, sizeof(outputDesc) / sizeof(char), attrDesc,
                   sizeof(attrDesc) / sizeof(char)});
    ASSERT_EQ(ret, OK);

    std::vector<int64_t> shape = {0};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    std::vector<const aclTensor*> tensor_list_a;
    tensor_list_a.push_back(tensor);
    aclTensorList* aclTensorTestList = aclCreateTensorList(tensor_list_a.data(), tensor_list_a.size());
    (void)NnopbaseExecutorAddDynamicTensors(executor, aclTensorTestList, 0, true);
    (void)NnopbaseExecutorAddDynamicTensors(executor, aclTensorTestList, 1, true);
    (void)NnopbaseExecutorAddDynamicTensors(executor, aclTensorTestList, 2, true);
    (void)NnopbaseExecutorAddDynamicTensors(executor, aclTensorTestList, 0, false);
    ASSERT_EQ(executor->ownArgs.outputs.paramDescs.emptyNum, 1U);
    NnopbaseExecutorDeInit(executor);
    delete executor;
    aclDestroyTensorList((const aclTensorList*)aclTensorTestList);
}

TEST_F(NnopbaseExecutorUnitTest, ExecutorProfilingWithEmptyTensorSuccess)
{
    std::vector<int64_t> shape = {0};
    EXPECT_EQ(RunBnProfiling(shape), OK);
}

TEST_F(NnopbaseExecutorUnitTest, NnopbaseDumpTensorWithEmptyTensorSuccess)
{
    NnopbaseExecutor* executor = nullptr;
    std::vector<int64_t> shape = {0};
    GetExecutor(executor, "bninference_d_kernel", shape);
    ASSERT_NE(executor, nullptr);

    size_t workspaceLen = 0U;
    ASSERT_EQ(NnopbaseExecutorRunForWorkspace(executor, &workspaceLen), OK);
    std::vector<uint8_t> vec(workspaceLen);
    ASSERT_EQ(workspaceLen, 0U);
    void* workspace = (void*)vec.data();
    ASSERT_EQ(NnopbaseExecutorRunWithWorkspace(executor, nullptr, workspace, workspaceLen), OK);

    NnopbaseExecutorGcSpace((void*)executor->space);
    NnopbaseExecutorDeInit(executor);
    delete executor;
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseExecutorUnitTest, NnopbaseSupportScalarConvertDtype)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char* opType = "bninference_d_kernel";
    char inputDesc[] = {1, 1, 1};
    char outputDesc[] = {};
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
    (void)NnopbaseAddScalarInput(executor, scalar, 2, -1, ge::DT_FLOAT16);
    auto inputs = &(((NnopbaseExecutor*)executor)->ownArgs.inputs);
    auto tensors = &inputs->extTensors[2];
    gert::TensorV2* rt2Tensor = &tensors->rt2Tensor;
    ASSERT_EQ(rt2Tensor->GetDataType(), ge::DT_FLOAT16);
    inputs->expectIndex = 2;
    (void)NnopbaseAddScalarInput(executor, scalar, 2, -1, ge::DT_BF16);
    ASSERT_EQ(rt2Tensor->GetDataType(), ge::DT_BF16);
    inputs->expectIndex = 2;
    (void)NnopbaseAddScalarInput(executor, scalar, 2, -1, ge::DT_INT8);
    ASSERT_EQ(rt2Tensor->GetDataType(), ge::DT_INT8);
    inputs->expectIndex = 2;
    (void)NnopbaseAddScalarInput(executor, scalar, 2, -1, ge::DT_INT16);
    ASSERT_EQ(rt2Tensor->GetDataType(), ge::DT_INT16);
    inputs->expectIndex = 2;
    (void)NnopbaseAddScalarInput(executor, scalar, 2, -1, ge::DT_UINT16);
    ASSERT_EQ(rt2Tensor->GetDataType(), ge::DT_UINT16);
    inputs->expectIndex = 2;
    (void)NnopbaseAddScalarInput(executor, scalar, 2, -1, ge::DT_UINT8);
    ASSERT_EQ(rt2Tensor->GetDataType(), ge::DT_UINT8);
    inputs->expectIndex = 2;
    float float_value = 5;
    auto* float_scalar = aclCreateScalar(&float_value, aclDataType::ACL_FLOAT);
    (void)NnopbaseAddScalarInput(executor, float_scalar, 2, -1, ge::DT_INT32);
    ASSERT_EQ(rt2Tensor->GetDataType(), ge::DT_INT32);
    inputs->expectIndex = 2;
    (void)NnopbaseAddScalarInput(executor, scalar, 2, -1, ge::DT_INT64);
    ASSERT_EQ(rt2Tensor->GetDataType(), ge::DT_INT64);
    inputs->expectIndex = 2;
    (void)NnopbaseAddScalarInput(executor, scalar, 2, -1, ge::DT_UINT32);
    ASSERT_EQ(rt2Tensor->GetDataType(), ge::DT_UINT32);
    inputs->expectIndex = 2;
    (void)NnopbaseAddScalarInput(executor, scalar, 2, -1, ge::DT_UINT64);
    ASSERT_EQ(rt2Tensor->GetDataType(), ge::DT_UINT64);
    inputs->expectIndex = 2;
    (void)NnopbaseAddScalarInput(executor, scalar, 2, -1, ge::DT_BOOL);
    ASSERT_EQ(rt2Tensor->GetDataType(), ge::DT_BOOL);
    inputs->expectIndex = 2;
    (void)NnopbaseAddScalarInput(executor, scalar, 2, -1, ge::DT_DOUBLE);
    ASSERT_EQ(rt2Tensor->GetDataType(), ge::DT_DOUBLE);
    inputs->expectIndex = 2;
    (void)NnopbaseAddScalarInput(executor, scalar, 2, -1, ge::DT_COMPLEX64);
    ASSERT_EQ(rt2Tensor->GetDataType(), ge::DT_COMPLEX64);
    inputs->expectIndex = 2;
    (void)NnopbaseAddScalarInput(executor, scalar, 2, -1, ge::DT_COMPLEX128);
    ASSERT_EQ(rt2Tensor->GetDataType(), ge::DT_COMPLEX128);
    inputs->expectIndex = 2;
    ASSERT_EQ(NnopbaseAddScalarInput(executor, scalar, 2, -1, ge::DT_STRING), ACLNN_ERR_PARAM_INVALID);
    inputs->expectIndex = 2;
    ASSERT_EQ(NnopbaseAddScalarInput(executor, nullptr, 2, -1, ge::DT_FLOAT16), OK);

    aclDestroyTensor(tensor);
    aclDestroyScalar(scalar);
    aclDestroyScalar(float_scalar);
    NnopbaseExecutorGcSpace(executorSpace);
    NnopbaseUnsetEnvAndClearFolder();
}

void* addr[100];
class UtNnopbaseExceptionDump : public Adx::DumpStub {
public:
    void* AdumpGetSizeInfoAddr(uint32_t space, uint32_t& atomicIndex)
    {
        auto mc2Env = std::getenv("MC2_TEST");
        if (mc2Env != nullptr) {
            if (std::atoi(mc2Env) == 1) {
                EXPECT_EQ(space, 8U);
            }
        } else {
            EXPECT_EQ(space, 7U);
        }
        atomicIndex = 2;
        return addr;
    }
};

TEST_F(NnopbaseExecutorUnitTest, NnopbaseMC2ExceptionArgsUt)
{
    setenv("MC2_TEST", "1", 1);
    UtNnopbaseExceptionDump dumpStub;
    Adx::DumpStub::GetInstance()->Install(&dumpStub);

    NnopbaseExecutor* executor = nullptr;
    GetExecutor(executor);
    ASSERT_NE(executor, nullptr);

    size_t workspaceLen = 0U;
    ASSERT_EQ(NnopbaseExecutorRunForWorkspace(executor, &workspaceLen), OK);
    ASSERT_EQ(NnopbaseSetMc2(executor), OK);
    char* group = "123";
    ASSERT_EQ(NnopbaseSetHcomGroup(executor, group), OK);
    std::vector<uint8_t> vec(workspaceLen);
    void* workspace = (void*)vec.data();
    exceptionDumpWorkspaceAddr = workspace;
    ASSERT_EQ(NnopbaseExecutorRunWithWorkspace(executor, nullptr, workspace, workspaceLen), OK);

    NnopbaseExecutorGcSpace((void*)executor->space);
    NnopbaseExecutorDeInit(executor);
    delete executor;
    NnopbaseUnsetEnvAndClearFolder();
    Adx::DumpStub::GetInstance()->UnInstall();
    unsetenv("MC2_TEST");
}

TEST_F(NnopbaseExecutorUnitTest, NnopbaseExceptionArgsUt)
{
    UtNnopbaseExceptionDump dumpStub;
    Adx::DumpStub::GetInstance()->Install(&dumpStub);

    NnopbaseExecutor* executor = nullptr;
    GetExecutor(executor);
    ASSERT_NE(executor, nullptr);

    size_t workspaceLen = 0U;
    ASSERT_EQ(NnopbaseExecutorRunForWorkspace(executor, &workspaceLen), OK);
    std::vector<uint8_t> vec(workspaceLen);
    void* workspace = (void*)vec.data();
    exceptionDumpWorkspaceAddr = workspace;
    ASSERT_EQ(NnopbaseExecutorRunWithWorkspace(executor, nullptr, workspace, workspaceLen), OK);

    NnopbaseExecutorGcSpace((void*)executor->space);
    NnopbaseExecutorDeInit(executor);
    delete executor;
    NnopbaseUnsetEnvAndClearFolder();
    Adx::DumpStub::GetInstance()->UnInstall();
}

class UtNnopbaseDynamicExceptionDump : public Adx::DumpStub {
public:
    void* AdumpGetSizeInfoAddr(uint32_t space, uint32_t& atomicIndex)
    {
        EXPECT_EQ(space, 11U);
        atomicIndex = 2;
        return addr;
    }
};

TEST_F(NnopbaseExecutorUnitTest, NnopbaseDynamicInputExceptionArgsUt)
{
    UtNnopbaseDynamicExceptionDump dumpStub;
    Adx::DumpStub::GetInstance()->Install(&dumpStub);

    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);
    static NnopbaseBinCollector* gBinCollector = nullptr;
    if (gBinCollector == nullptr) {
        gBinCollector = new NnopbaseBinCollector;
        ASSERT_NE(gBinCollector, nullptr);
        auto ret = NnopbaseCollectorInit(gBinCollector);
        ASSERT_EQ(ret, OK);
        ret = NnopbaseCollectorWork(gBinCollector);
        ASSERT_EQ(ret, OK);
    }

    NnopbaseExecutor* executor = new NnopbaseExecutor;
    ASSERT_NE(executor, nullptr);
    char inputDesc[] = {2, 2};
    char outputDesc[] = {2, 1};
    char attrDesc[] = {};
    auto ret = NnopbaseExecutorInit(
        executor, {inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc, sizeof(outputDesc) / sizeof(char), attrDesc,
                   sizeof(attrDesc) / sizeof(char)});
    ASSERT_EQ(ret, OK);

    NnopbaseExecutorSetCollector(executor, gBinCollector);
    NnopbaseExecutorSpace space;
    executor->space = &space;
    const char* opType = "custom_op1";
    if (opType != "custom_op1") {
        return;
    }
    ret = NnopbaseExecutorSetRegInfo(executor, opType);
    ASSERT_EQ(ret, OK);

    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);

    std::vector<const aclTensor*> tensor_list_a;
    tensor_list_a.push_back(tensor);
    aclTensorList* aclTensorTestList = aclCreateTensorList(tensor_list_a.data(), tensor_list_a.size());
    aclTensor* tensor1 = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                         aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);

    (void)NnopbaseExecutorAddDynamicTensors(executor, aclTensorTestList, 0, true);
    (void)NnopbaseExecutorAddDynamicTensors(executor, aclTensorTestList, 1, true);
    (void)NnopbaseExecutorAddDynamicTensors(executor, aclTensorTestList, 0, false);
    ASSERT_EQ(NnopbaseExecutorUpdateTensorsIndex(&(executor->ownArgs.outputs), 1), OK);
    ASSERT_EQ(NnopbaseExecutorAddTensor(executor, tensor, 1, false, false), OK);

    RunExecutorSuccess(executor);
    NnopbaseExecutorDeInit(executor);
    delete executor;
    aclDestroyTensorList((const aclTensorList*)aclTensorTestList);
    aclDestroyTensor(tensor1);
    NnopbaseUnsetEnvAndClearFolder();
    Adx::DumpStub::GetInstance()->UnInstall();
}

TEST_F(NnopbaseExecutorUnitTest, ReportAicpuAdditionInfo)
{
    auto timeStamp = NnopbaseMsprofSysTime();
    op::internal::opProfilingSwitch.reportFlag = true;
    op::internal::opProfilingSwitch.additionInfoFlag = true;
    NnopbaseReportLaunchInfo(timeStamp, "Aicpu");
    ASSERT_EQ(NnopbaseReportAicpuAdditionInfo(timeStamp, "Aicpu"), OK);
}

TEST_F(NnopbaseExecutorUnitTest, TestEnvACLNN_CACHE_LIMIT)
{
    setenv("ACLNN_CACHE_LIMIT", "10", 1);
    ASSERT_EQ(nnopbase::ArgsPool::GetCacheSizeLimit(), 10);
    setenv("ACLNN_CACHE_LIMIT", "xxxx", 1);
    ASSERT_EQ(nnopbase::ArgsPool::GetCacheSizeLimit(), 10000);
    unsetenv("ACLNN_CACHE_LIMIT");
}

TEST_F(NnopbaseExecutorUnitTest, TestEnvDISABLE_L2_CACHE)
{
    ASSERT_EQ(nnopbase::EnableNnopbaseArgsCache(), true);
    setenv("DISABLE_L2_CACHE", "1", 1);
    ASSERT_EQ(nnopbase::EnableNnopbaseArgsCache(), false);
    setenv("DISABLE_L2_CACHE", "2", 1);
    ASSERT_EQ(nnopbase::EnableNnopbaseArgsCache(), true);
    unsetenv("DISABLE_L2_CACHE");
}

void* GetIntArrayExecutor(void* executorSpace, int64_t* intValues, size_t intLen)
{
    const char* opType = "AddTik2";
    char inputDesc[] = {1, 1, 1};
    char outputDesc[] = {0};
    char attrDesc[] = {};
    void* executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                         sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));

    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);

    auto* intArray = aclCreateIntArray(intValues, intLen);
    NnopbaseAddInput(executor, tensor, 0);
    NnopbaseAddInput(executor, tensor, 1);
    NnopbaseAddIntArrayInput(executor, intArray, 2);
    NnopbaseAddOutput(executor, nullptr, 0); // 可选输出
    aclDestroyTensor(tensor);
    aclDestroyIntArray(intArray);
    return executor;
}

TEST_F(NnopbaseExecutorUnitTest, TestCacheIntArrayDependkey)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);
    int64_t intValues[] = {3, 4, 5};
    void* executor = GetIntArrayExecutor(executorSpace, intValues, 3);

    NnopbaseExecutor* opExecutor = (NnopbaseExecutor*)executor;
    nnopbase::ArgsPool::GetInstance().MatchArgs(opExecutor);
    ASSERT_EQ(opExecutor->ownArgs.enableCache, true);

    NnopbaseExecutorGcSpace(executorSpace);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseExecutorUnitTest, TestCacheWithLargerLength)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);
    std::vector<int64_t> intValues(2000);
    void* executor = GetIntArrayExecutor(executorSpace, intValues.data(), intValues.size());

    NnopbaseExecutor* opExecutor = (NnopbaseExecutor*)executor;
    ASSERT_EQ(nnopbase::ArgsPool::GetInstance().MatchArgs(opExecutor), false);
    ASSERT_EQ(opExecutor->ownArgs.enableCache, true);
    ASSERT_EQ(nnopbase::ArgsPool::GetInstance().CreateArgs(opExecutor), OK);

    NnopbaseExecutorGcSpace(executorSpace);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseExecutorUnitTest, TestInvalidStaticShape)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);
    std::vector<int64_t> intValues(10);
    void* executor = GetIntArrayExecutor(executorSpace, intValues.data(), intValues.size());

    NnopbaseExecutor* opExecutor = (NnopbaseExecutor*)executor;
    ASSERT_EQ(NnopbaseExecutorGetStaticBinInfo(opExecutor), false);

    intValues = std::vector<int64_t>(100000);
    executor = GetIntArrayExecutor(executorSpace, intValues.data(), intValues.size());
    opExecutor = (NnopbaseExecutor*)executor;
    ASSERT_EQ(NnopbaseExecutorGenStaticKey(opExecutor, false), ACLNN_ERR_PARAM_INVALID);
    ASSERT_EQ(NnopbaseExecutorGetStaticBinInfo(opExecutor), false);

    NnopbaseExecutorGcSpace(executorSpace);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseExecutorUnitTest, TestNonFiniteCheck)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    NnopbaseTensors tensors;
    tensors.extTensors = std::vector<NnopbaseTensor>(25);
    tensors.num = 25;
    for (auto& tensor : tensors.extTensors) {
        tensor.isNull = false;
        tensor.rt2Tensor.MutableOriginShape() = {1};
        tensor.rt2Tensor.MutableStorageShape() = {1};
        tensor.rt2Tensor.SetDataType(ge::DataType::DT_FLOAT);
        tensor.rt2Tensor.SetOriginFormat(ge::FORMAT_ND);
        tensor.rt2Tensor.SetStorageFormat(ge::FORMAT_ND);
        tensor.rt2Tensor.MutableTensorData().SetPlacement(gert::kOnDeviceHbm);
        tensor.rt2Tensor.SetSize(sizeof(float));
    }
    tensors.extTensors[0U].isNull = true;

    bool isOverflow = false;
    ASSERT_EQ(NnopbaseRunNonFiniteCheckOp(tensors, nullptr, isOverflow), OK);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseExecutorUnitTest, NnopbaseSupportScalarConvertDtypeWithInput)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);

    const char* opType = "bninference_d_kernel";
    char inputDesc[] = {1, 1};
    char outputDesc[] = {};
    char attrDesc[] = {};
    void* executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                         sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    ASSERT_NE(executor, nullptr);
    auto inputs = &(((NnopbaseExecutor*)executor)->ownArgs.inputs);
    auto tensors = &inputs->extTensors[0];
    std::vector<int64_t> shape = {1, 1, 1, 1, 1};

    static std::map<aclDataType, std::vector<aclDataType>> convertMap{
        {ACL_INT8,
         {ACL_INT16, ACL_INT32, ACL_INT64, ACL_UINT8, ACL_UINT16, ACL_UINT32, ACL_UINT64, ACL_FLOAT, ACL_FLOAT16,
          ACL_BF16, ACL_BOOL, ACL_DOUBLE}},
        {ACL_INT16,
         {ACL_INT8, ACL_INT32, ACL_INT64, ACL_UINT8, ACL_UINT16, ACL_UINT32, ACL_UINT64, ACL_FLOAT, ACL_FLOAT16,
          ACL_BF16, ACL_BOOL, ACL_DOUBLE}},
        {ACL_INT32,
         {ACL_INT8, ACL_INT16, ACL_INT64, ACL_UINT8, ACL_UINT16, ACL_UINT32, ACL_UINT64, ACL_FLOAT, ACL_FLOAT16,
          ACL_BF16, ACL_BOOL, ACL_DOUBLE}},
        {ACL_INT64,
         {ACL_INT8, ACL_INT16, ACL_INT32, ACL_UINT8, ACL_UINT16, ACL_UINT32, ACL_UINT64, ACL_FLOAT, ACL_FLOAT16,
          ACL_BF16, ACL_BOOL, ACL_DOUBLE}},
        {ACL_UINT8,
         {ACL_INT8, ACL_INT16, ACL_INT32, ACL_INT64, ACL_UINT16, ACL_UINT32, ACL_UINT64, ACL_FLOAT, ACL_FLOAT16,
          ACL_BF16, ACL_BOOL, ACL_DOUBLE}},
        {ACL_UINT16,
         {ACL_INT8, ACL_INT16, ACL_INT32, ACL_INT64, ACL_UINT8, ACL_UINT32, ACL_UINT64, ACL_FLOAT, ACL_FLOAT16,
          ACL_BF16, ACL_BOOL, ACL_DOUBLE}},
        {ACL_UINT32,
         {ACL_INT8, ACL_INT16, ACL_INT32, ACL_INT64, ACL_UINT8, ACL_UINT16, ACL_UINT64, ACL_FLOAT, ACL_FLOAT16,
          ACL_BF16, ACL_BOOL, ACL_DOUBLE}},
        {ACL_UINT64,
         {ACL_INT8, ACL_INT16, ACL_INT32, ACL_INT64, ACL_UINT8, ACL_UINT16, ACL_UINT32, ACL_FLOAT, ACL_FLOAT16,
          ACL_BF16, ACL_BOOL, ACL_DOUBLE}},
        {ACL_FLOAT,
         {ACL_INT8, ACL_INT16, ACL_INT32, ACL_INT64, ACL_UINT8, ACL_UINT16, ACL_UINT32, ACL_UINT64, ACL_FLOAT16,
          ACL_BF16, ACL_BOOL, ACL_DOUBLE}},
        {ACL_FLOAT16,
         {ACL_INT8, ACL_INT16, ACL_INT32, ACL_INT64, ACL_UINT8, ACL_UINT16, ACL_UINT32, ACL_UINT64, ACL_FLOAT, ACL_BF16,
          ACL_BOOL, ACL_DOUBLE}},
        {ACL_BF16,
         {ACL_INT8, ACL_INT16, ACL_INT32, ACL_INT64, ACL_UINT8, ACL_UINT16, ACL_UINT32, ACL_UINT64, ACL_FLOAT,
          ACL_FLOAT16, ACL_BOOL, ACL_DOUBLE}},
        {ACL_BOOL,
         {ACL_INT8, ACL_INT16, ACL_INT32, ACL_INT64, ACL_UINT8, ACL_UINT16, ACL_UINT32, ACL_UINT64, ACL_FLOAT,
          ACL_FLOAT16, ACL_BF16, ACL_DOUBLE}},
        {ACL_DOUBLE,
         {ACL_INT8, ACL_INT16, ACL_INT32, ACL_INT64, ACL_UINT8, ACL_UINT16, ACL_UINT32, ACL_UINT64, ACL_FLOAT,
          ACL_FLOAT16, ACL_BF16, ACL_BOOL}},
        {ACL_COMPLEX64, {ACL_COMPLEX128}},
        {ACL_COMPLEX128, {ACL_COMPLEX64}}};

    aclTensor* tmpTensor = nullptr;
    aclScalar* scalar = nullptr;
    void* scalar_value = malloc(16);
    ASSERT_NE(scalar_value, nullptr);
    (void)memset(scalar_value, 0, 16);
    for (auto& map : convertMap) {
        tmpTensor = aclCreateTensor(shape.data(), shape.size(), map.first, nullptr, 0, aclFormat::ACL_FORMAT_ND,
                                    shape.data(), shape.size(), nullptr);
        std::vector<aclDataType>& vec = map.second;
        for (size_t i = 0U; i < vec.size(); i++) {
            scalar = aclCreateScalar(scalar_value, vec[i]);
            inputs->expectIndex = 0;
            (void)NnopbaseAddScalarInput(executor, scalar, 0, 1, static_cast<ge::DataType>(vec[i]));
            (void)NnopbaseAddInput(executor, tmpTensor, 1);
            gert::TensorV2* rt2Tensor = &tensors->rt2Tensor;
            ASSERT_EQ(rt2Tensor->GetDataType(), map.first);
        }
    }
    free(scalar_value);
    if (scalar != nullptr) {
        aclDestroyScalar(scalar);
    }
    if (tmpTensor != nullptr) {
        aclDestroyTensor(tmpTensor);
    }
    NnopbaseExecutorGcSpace(executorSpace);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseExecutorUnitTest, LoadTilingSoFailed)
{
    std::vector<std::pair<std::string, gert::OppImplVersionTag>> basePath;
    basePath.push_back(std::make_pair("/usr/local", gert::OppImplVersionTag::kOpp));
    setenv("ASCEND_HOME_PATH", OP_API_COMMON_UT_SRC_DIR, 1);
    setenv("ASCEND_OPP_PATH", OP_API_COMMON_UT_SRC_DIR, 1);
    ASSERT_NE(NnopbaseLoadTilingSo(basePath), OK);
    NnopbaseUnsetEnvAndClearFolder();
}

TEST_F(NnopbaseExecutorUnitTest, NnopbaseStreamCallbackFuncForStream)
{
    extern std::unordered_map<void*, NnopbaseStreamForCombineExecution> g_nnopbaseStreamMap;
    g_nnopbaseStreamMap.clear();

    aclrtStream mainStream;
    EXPECT_EQ(aclrtCreateStreamWithConfig(&mainStream, 0, 0), ACL_SUCCESS);
    EXPECT_NE(mainStream, nullptr);
    NnopbaseStreamForCombineExecution nnopbaseStream;
    EXPECT_EQ(aclrtCreateStreamWithConfig(&nnopbaseStream.stream, 0, 0), ACL_SUCCESS);
    EXPECT_EQ(aclrtCreateStreamWithConfig(&nnopbaseStream.eventA, 0, ACL_EVENT_SYNC), ACL_SUCCESS);
    EXPECT_EQ(aclrtCreateStreamWithConfig(&nnopbaseStream.eventB, 0, ACL_EVENT_SYNC), ACL_SUCCESS);
    g_nnopbaseStreamMap[mainStream] = nnopbaseStream;

    NnopbaseDestroyStreamCallback(mainStream, false);
    EXPECT_EQ(g_nnopbaseStreamMap.size(), 0U);
    rtStreamDestroy(mainStream);
}

TEST_F(NnopbaseExecutorUnitTest, NnopbaseStreamCallbackFuncForContextWithDefaultStream)
{
    extern std::unordered_map<void*, NnopbaseStreamForCombineExecution> g_nnopbaseStreamMap;
    g_nnopbaseStreamMap.clear();

    NnopbaseStreamForCombineExecution nnopbaseStream;
    EXPECT_EQ(aclrtCreateStreamWithConfig(&nnopbaseStream.stream, 0, 0), ACL_SUCCESS);
    EXPECT_EQ(aclrtCreateStreamWithConfig(&nnopbaseStream.eventA, 0, ACL_EVENT_SYNC), ACL_SUCCESS);
    EXPECT_EQ(aclrtCreateStreamWithConfig(&nnopbaseStream.eventB, 0, ACL_EVENT_SYNC), ACL_SUCCESS);
    g_nnopbaseStreamMap[nullptr] = nnopbaseStream;

    NnopbaseDestroyStreamCallback(nullptr, false);
    EXPECT_EQ(g_nnopbaseStreamMap.size(), 0U);
}

TEST_F(NnopbaseExecutorUnitTest, NnopbaseGet8ByteSize)
{
    uint32_t len = 0;
    NnopbaseExecutorGet8ByteSize(8, &len);
    EXPECT_EQ(len, 8);
    NnopbaseExecutorGet8ByteSize(1, &len);
    EXPECT_EQ(len, 16);
}

TEST_F(NnopbaseExecutorUnitTest, NnopbaseInvalidAttr)
{
    EXPECT_EQ(nnopbase::ToStr(kNnopbaseAttrEnd), "4");

    int64_t i = 2;
    NnopbaseAttr attr;
    NnopbaseAttrs attrs;
    attr.dtype = (NnopbaseAttrDtype)4;
    attr.addr.addr = &i;
    attr.addr.size = 8;
    attrs.attrs = &attr;
    attrs.num = 1;
    EXPECT_EQ(nnopbase::ToStr(attrs), "[0x02 0x00 0x00 0x00 0x00 0x00 0x00 0x00 ]");
}

TEST_F(NnopbaseExecutorUnitTest, NnopbaseNotCacheMatchArgs)
{
    NnopbaseExecutor* executor = nullptr;
    GetExecutorWithAttr(executor);
    ASSERT_NE(executor, nullptr);
    executor->space = new NnopbaseExecutorSpace();

    ASSERT_EQ(nnopbase::ArgsPool::GetInstance().MatchArgs(executor), false);
    ASSERT_EQ(nnopbase::ArgsPool::GetInstance().CreateArgs(executor), OK);
    auto& args = executor->ownArgs;
    executor->args->isVisit = false;
    NnopbaseBinInfo binInfo;
    binInfo.isStaticShape = false;
    executor->args->binInfo = &binInfo;

    executor->ownArgs.inputs.num = 2;
    ASSERT_EQ(nnopbase::ArgsPool::GetInstance().MatchArgs(executor), false);
    nnopbase::ArgsPool::GetInstance().argsMap[executor->ownArgs.seed].clear();

    executor->ownArgs.inputs.num = 3;
    executor->ownArgs.inputs.extTensors[2].rt2Tensor.MutableStorageShape() = {2, 2, 2, 2, 2};
    ASSERT_EQ(nnopbase::ArgsPool::GetInstance().MatchArgs(executor), false);
    nnopbase::ArgsPool::GetInstance().argsMap[executor->ownArgs.seed].clear();

    NnopbaseExecutorGcSpace((void*)executor->space);
    NnopbaseExecutorDeInit(executor);
    delete executor;
    NnopbaseUnsetEnvAndClearFolder();

    NnopbaseGuard guard2([]() { throw "error"; });
}

TEST_F(NnopbaseExecutorUnitTest, TestNullArray)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);

    void* executorSpace = nullptr;
    ASSERT_EQ(NnopbaseCreateExecutorSpace(&executorSpace), OK);
    const char* opType = "AddTik2";
    char inputDesc[] = {1, 1, 0};
    char outputDesc[] = {0};
    char attrDesc[] = {};
    void* executor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                         sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));

    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    NnopbaseAddInput(executor, tensor, 0);
    NnopbaseAddInput(executor, tensor, 1);
    NnopbaseAddIntArrayInput(executor, nullptr, 2);
    NnopbaseAddOutput(executor, nullptr, 0); // 可选输出
    aclDestroyTensor(tensor);

    NnopbaseExecutor* opExecutor = (NnopbaseExecutor*)executor;
    ASSERT_EQ(opExecutor->ownArgs.inputs.paramDescs.instances[2].num, 0U);
    ASSERT_EQ(opExecutor->ownArgs.inputs.paramDescs.emptyNum, 1U);

    NnopbaseExecutorGcSpace(executorSpace);
    NnopbaseUnsetEnvAndClearFolder();
}

#if 0 // comment out for ge decoupling
TEST_F(NnopbaseExecutorUnitTest, TestOutputAutomicCleanSuccessFor1971)
{
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

    auto oriSocVersion = nnopbase::IndvSoc::GetInstance().GetCurSocVersion();
    SetSocVersion(nnopbase::OPS_SUBPATH_ASCEND910B);
    ((NnopbaseExecutor *)executor)->args->binInfo->initValues.clear();
    NnopbaseInitValueInfo info = {2, op::DataType::DT_FLOAT, 0.0, 0, 0};
    ((NnopbaseExecutor *)executor)->args->binInfo->initValues.push_back(info);
    ASSERT_EQ(SetTensorDataSizeToInitValue((NnopbaseExecutor *)executor), OK);
    ASSERT_EQ(((NnopbaseExecutor *)executor)->args->binInfo->initValues[0].tensorDataSize, 4U);
    SetSocVersion(oriSocVersion);

    NnopbaseExecutorGcSpace(executorSpace);
    aclDestroyTensor(tensor);
    NnopbaseUnsetEnvAndClearFolder();
}
#endif

TEST_F(NnopbaseExecutorUnitTest, TestLoadMemSetJsonFailed)
{
    std::string oppPath = std::string(OP_API_COMMON_UT_SRC_DIR);
    std::shared_ptr<MemsetOpInfo> memsetInfo = std::make_shared<MemsetOpInfo>();
    auto binInfo = std::make_shared<MemsetOpBinInfo>();
    memsetInfo->binInfo = std::move(binInfo);
    memsetInfo->memSetJsonPath = oppPath + "/built-in/op_impl/ai_core/tbe/kernel/ascend910/mem_set/"
                                           "MemSet_2e0edc6c5154edf0badbec9c57f44db6_high_performance_error.json";
    ASSERT_EQ(NnopbaseLoadMemsetJson(memsetInfo), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(NnopbaseExecutorUnitTest, TestGetMemsetBinInfoFailed)
{
    std::string jsonPath = "/built-in/op_impl/ai_core/tbe/kernel/ascend910/mem_set/";
    std::string binPath;
    ASSERT_EQ(NnopbaseGetBinPath(jsonPath, binPath), ACLNN_ERR_PARAM_INVALID);
    ASSERT_EQ(NnopbaseGetKernelJsonPath(jsonPath, binPath), ACLNN_ERR_PARAM_INVALID);

    std::string oppPath = std::string(OP_API_COMMON_UT_SRC_DIR) +
                          "/built-in/op_impl/ai_core/tbe/kernel/config/ascend910";
    std::shared_ptr<MemsetOpInfo> memsetInfo = std::make_shared<MemsetOpInfo>();
    memsetInfo->memSetJsonPath = oppPath +
                                 "/mem_set/MemSet_2e0edc6c5154edf0badbec9c57f44db6_high_performance_error.json";
    ASSERT_EQ(NnopbaseGetMemsetBinInfo(memsetInfo), ACLNN_ERR_PARAM_INVALID);

    const std::string memsetDirPath = oppPath + "/mem_set.json";
    nlohmann::json memsetJsonInfo;
    ASSERT_EQ(NnopbaseReadJsonConfig(memsetDirPath, memsetJsonInfo), OK);
    ASSERT_EQ(NnopbaseReadMemsetJsonInfo(oppPath, memsetJsonInfo, memsetInfo, 3U, false), ACLNN_ERR_PARAM_INVALID);

    ASSERT_EQ(NnopbaseReadMemsetJsonInfo(oppPath, memsetJsonInfo, memsetInfo, 20U, false), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(NnopbaseExecutorUnitTest, NnopbaseGenPlaceHolderKey)
{
    NnopbaseExecutorArgs args;
    args.remainKeyLen = 0;
    Indv::CacheKeyBuilder::AppendPlaceHolder(&args, args.inputKey.data());
    std::vector<int64_t> shape = {1, 3, 1, 1, 1};
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
    std::vector<const aclTensor*> tensor_list;
    tensor_list.push_back(tensor);
    tensor_list.push_back(nullptr);
    aclTensorList* tensorList = aclCreateTensorList(tensor_list.data(), tensor_list.size());

    args.remainKeyLen = 0;
    Indv::CacheKeyBuilder::AppendShapeInfo(&args, tensor);
    Indv::CacheKeyBuilder::AppendTensorList(&args, tensorList);
    aclDestroyTensorList(tensorList);
}

TEST_F(NnopbaseExecutorUnitTest, NnopbaseAppendCoreNum)
{
    NnopbaseExecutorArgs args;
    NnopbaseCoreNum coreNum = {1, 2};
    args.remainKeyLen = 1;
    args.keyLen = 0;
    NnopbaseUChar* key = Indv::CacheKeyBuilder::AppendCoreNum(&args, &coreNum);
    ASSERT_EQ(sizeof(NnopbaseCoreNum) + 1, args.keyLen);
}

TEST_F(NnopbaseExecutorUnitTest, AddRankIdToKeySuccess)
{
    NnopbaseExecutorArgs args;
    uint32_t rankId = 0U;
    args.remainKeyLen = 1;
    args.keyLen = 0;
    NnopbaseUChar* key = Indv::CacheKeyBuilder::AppendMc2RankId(&args, &rankId);
    ASSERT_EQ(sizeof(uint32_t) + 1, args.keyLen);
}

TEST_F(NnopbaseExecutorUnitTest, NnopbasePrepareMC2ParamsSuccess)
{
    NnopbaseExecutor* executor = nullptr;
    GetExecutor(executor);
    ASSERT_NE(executor, nullptr);

    executor->opType = strdup("test_op");
    executor->mc2.serverType = NNOPBASE_HCCL_SERVER_TYPE_AICPU;
    executor->mc2.enabled = true;
    executor->collector = new NnopbaseBinCollector;
    executor->collector->isMc2FusionLaunch = false;
    executor->args = new NnopbaseExecutorArgs;
    executor->argsExt.hostInputInfoNum = 1U;
    executor->mc2.aicpuArgs.hostInputInfoNum = 1U;
    NnopbaseHcclCommParamDesc paramDesc = {0, 0, 0, 0, 0};
    NnopbaseExecutorArgsAddr argsAddr = {nullptr, nullptr, nullptr, nullptr, &paramDesc};
    argsAddr.ptr = executor->args->argsBuf.data() + 2 * sizeof(void*);
    argsAddr.hostInputData = argsAddr.ptr + 2 * sizeof(aclrtPlaceHolderInfo);
    argsAddr.aicpuHostInputInfo = op::internal::PtrCastTo<aclrtPlaceHolderInfo>(argsAddr.ptr +
                                                                                sizeof(aclrtPlaceHolderInfo));
    argsAddr.hostInputInfo = op::internal::PtrCastTo<aclrtPlaceHolderInfo>(argsAddr.ptr);
    executor->mc2.aicpuArgs.args = argsAddr.ptr;
    executor->argsExt.args = op::internal::PtrCastTo<void>(argsAddr.ptr + sizeof(void*));

    NnopbaseUChar* soNamePtr = argsAddr.hostInputData;
    ASSERT_EQ(NnopbasePrepareMC2Params(executor, &argsAddr), OK);
    std::string soName(reinterpret_cast<char*>(soNamePtr), NNOPBASE_AICPU_PARAM_LEN);
    const std::string expected_soName = "libccl_kernel.so" +
                                        std::string(NNOPBASE_AICPU_PARAM_LEN - std::strlen("libccl_kernel.so"), '\0');
    ASSERT_EQ(soName, expected_soName);
    std::string kernelName(reinterpret_cast<char*>(soNamePtr + NNOPBASE_AICPU_PARAM_LEN), NNOPBASE_AICPU_PARAM_LEN);
    const std::string expected_kernelName = "RunAicpuKfcSrvLaunch" +
                                            std::string(NNOPBASE_AICPU_PARAM_LEN - std::strlen("RunAicpuKfcSrvLaunch"),
                                                        '\0');
    ASSERT_EQ(kernelName, expected_kernelName);

    delete executor->args;
    delete executor->collector;
    free(executor->opType);
    delete executor;
}

TEST_F(NnopbaseExecutorUnitTest, NnopbasePrepareMC2ParamsSuccessForascend950WithAiCPU)
{
    NnopbaseExecutor* executor = nullptr;
    GetExecutor(executor);
    ASSERT_NE(executor, nullptr);

    executor->opType = strdup("test_op");
    executor->mc2.serverType = NNOPBASE_HCCL_SERVER_TYPE_AICPU;
    executor->mc2.enabled = true;
    executor->collector = new NnopbaseBinCollector;
    executor->collector->isMc2FusionLaunch = false;
    executor->args = new NnopbaseExecutorArgs;
    executor->argsExt.hostInputInfoNum = 1U;
    executor->mc2.aicpuArgs.hostInputInfoNum = 1U;
    NnopbaseHcclCommParamDesc paramDesc = {0, 0, 0, 0, 0};
    NnopbaseExecutorArgsAddr argsAddr = {nullptr, nullptr, nullptr, nullptr, &paramDesc};
    argsAddr.ptr = executor->args->argsBuf.data() + 2 * sizeof(void*);
    argsAddr.hostInputData = argsAddr.ptr + 2 * sizeof(aclrtPlaceHolderInfo);
    argsAddr.aicpuHostInputInfo = op::internal::PtrCastTo<aclrtPlaceHolderInfo>(argsAddr.ptr +
                                                                                sizeof(aclrtPlaceHolderInfo));
    argsAddr.hostInputInfo = op::internal::PtrCastTo<aclrtPlaceHolderInfo>(argsAddr.ptr);
    executor->mc2.aicpuArgs.args = argsAddr.ptr;
    executor->argsExt.args = op::internal::PtrCastTo<void>(argsAddr.ptr + sizeof(void*));
    auto oriSocVersion = nnopbase::IndvSoc::GetInstance().GetCurSocVersion();
    struct SocVersionGuard {
        std::string originalVersion;
        ~SocVersionGuard() { SetSocVersion(originalVersion); }
    } socGuard{oriSocVersion};
    SetSocVersion(nnopbase::OPS_SUBPATH_ASCEND950);
    ASSERT_EQ(nnopbase::IndvSoc::GetInstance().GetCurSocVersion(), nnopbase::OPS_SUBPATH_ASCEND950);
    NnopbaseUChar* soNamePtr = argsAddr.hostInputData;
    ASSERT_EQ(NnopbasePrepareMC2Params(executor, &argsAddr), OK);
    std::string soName(reinterpret_cast<char*>(soNamePtr), NNOPBASE_AICPU_PARAM_LEN);
    const std::string expected_soName = "libmc2_server.so" +
                                        std::string(NNOPBASE_AICPU_PARAM_LEN - std::strlen("libmc2_server.so"), '\0');
    ASSERT_EQ(soName, expected_soName);
    std::string kernelName(reinterpret_cast<char*>(soNamePtr + NNOPBASE_AICPU_PARAM_LEN), NNOPBASE_AICPU_PARAM_LEN);
    const std::string expected_kernelName = "Mc2ServerKernel" +
                                            std::string(NNOPBASE_AICPU_PARAM_LEN - std::strlen("Mc2ServerKernel"),
                                                        '\0');
    ASSERT_EQ(kernelName, expected_kernelName);
    // SOC version will be restored by SocVersionGuard destructor

    delete executor->args;
    delete executor->collector;
    free(executor->opType);
    delete executor;
}

TEST_F(NnopbaseExecutorUnitTest, NnopbasePrepareMC2ParamsReturnsErrorWhenArgsBufIsInsufficient)
{
    NnopbaseExecutor* executor = nullptr;
    GetExecutor(executor);
    ASSERT_NE(executor, nullptr);

    executor->opType = strdup("test_op");
    executor->mc2.serverType = NNOPBASE_HCCL_SERVER_TYPE_AICPU;
    executor->mc2.enabled = true;
    executor->args = new NnopbaseExecutorArgs;
    executor->args->argsBuf.resize(128U);
    NnopbaseHcclCommParamDesc paramDesc = {0, 0, 0, 0, 0};
    NnopbaseExecutorArgsAddr argsAddr = {nullptr, nullptr, nullptr, nullptr, &paramDesc};
    argsAddr.ptr = executor->args->argsBuf.data();
    argsAddr.hostInputData = executor->args->argsBuf.data() + executor->args->argsBuf.size() - 1U;
    executor->mc2.aicpuArgs.args = executor->args->argsBuf.data();
    executor->argsExt.args = op::internal::PtrCastTo<void>(executor->args->argsBuf.data() + sizeof(void*));

    NnopbaseUChar* const hostInputData = argsAddr.hostInputData;
    ASSERT_EQ(NnopbasePrepareMC2Params(executor, &argsAddr), ACLNN_ERR_PARAM_INVALID);
    ASSERT_EQ(argsAddr.hostInputData, hostInputData);

    delete executor->args;
    free(executor->opType);
    delete executor;
}
