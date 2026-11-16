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
#include <array>
#include <iostream>
#include <memory>
#include <stdlib.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <filesystem>
#include <thread>

#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnn/aclnn_base.h"
#include "kernel_mgr.h"
#include "memset_ctx_holder.h"
#include "kernel_context_holder.h"
#include "op_kernel.h"
#include "op_ctx_def.h"
#include "op_run_context.h"
#include "rts_arg.h"
#include "opdev/op_def.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_cache.h"
#include "opdev/op_errno.h"
#include "individual_op_internal.h"
#include "thread_local_context.h"
#include "op_cache_internal.h"
#include "kernel_launcher.h"
#include "kernel_workspace.h"
#include "op_kernel_lib.h"
#include "depends/dump/dump_stub.h"
#include "depends/acl/aclrt_stub.h"
#include "test_comp_op_common.h"

using Json = nlohmann::json;
using namespace op::internal;
using namespace op::internal::test;

extern "C" void InitPTACacheThreadLocal();
extern inline uint32_t SortOpTypeId();
extern inline uint32_t AxpyOpTypeId();
extern inline uint32_t AddNOpTypeId();

class OpKernelUT : public testing::Test {
protected:
    struct ViewOffsetTilingSizeResult {
        bool ok{false};
        uint64_t tensorSize{0};
    };

    static void SetUpTestCase()
    {
        // aclInit(nullptr)
        setenv("ASCEND_OPP_PATH", OP_API_COMMON_UT_SRC_DIR, 1); // does overwrite
        GetThreadLocalContext().cacheHasFull_ = true;
        SortOpTypeId();
        AxpyOpTypeId();
        AddNOpTypeId();
    }

    static void TearDownTestCase() {}

    std::unique_ptr<OpKernelBin> CreateFakeOpKernelBin(bool genPlaceholder, bool hasDevPtrArg)
    {
        uint32_t opType = op::OpTypeDict::ToOpType("QuantBatchMatmulV3");
        const char* p = std::getenv("ASCEND_OPP_PATH");
        EXPECT_NE(p, nullptr);
        KeyAndDetail key;
        key.key = "hahaha";
        size_t hashKey = 123;
        char jsonPath[1024];
        char binPath[1024];
        snprintf_s(jsonPath, sizeof(jsonPath), sizeof(jsonPath),
                   "%s/built-in/op_impl/ai_core/tbe/kernel/ascend910/quant_batch_matmul_v3/"
                   "QuantBatchMatmulV3_ND_ND_int8_int8_bf16_high_performance.json",
                   p);
        snprintf_s(binPath, sizeof(binPath), sizeof(binPath),
                   "%s/built-in/op_impl/ai_core/tbe/kernel/ascend910/add/"
                   "Add_41dadce325b0f810d03359af2a38990b_high_performance.o",
                   p);

        std::unique_ptr<OpKernelBin> kernelBin = std::make_unique<OpKernelBin>(
            opType, jsonPath, jsonPath, binPath, key, hashKey, BinType::DYNAMIC_BIN, genPlaceholder, hasDevPtrArg);
        return std::move(kernelBin);
    }

    std::unique_ptr<OpKernelBin> CreateFailOpKernelBin(bool genPlaceholder, bool hasDevPtrArg)
    {
        uint32_t opType = op::OpTypeDict::ToOpType("QuantBatchMatmulV3");
        const char* p = std::getenv("ASCEND_OPP_PATH");
        EXPECT_NE(p, nullptr);
        KeyAndDetail key;
        key.key = "hahaha";
        size_t hashKey = 123;
        char jsonPath[1024];
        char binPath[1024];
        snprintf_s(
            jsonPath, sizeof(jsonPath), sizeof(jsonPath),
            "%s/built-in/op_impl/ai_core/tbe/kernel/ascend910/quant_batch_matmul_v3/QuantBatchMatmulV3_fail.json", p);
        snprintf_s(binPath, sizeof(binPath), sizeof(binPath),
                   "%s/built-in/op_impl/ai_core/tbe/kernel/ascend910/add/"
                   "Add_41dadce325b0f810d03359af2a38990b_high_performance.o",
                   p);

        std::unique_ptr<OpKernelBin> kernelBin = std::make_unique<OpKernelBin>(
            opType, jsonPath, jsonPath, binPath, key, hashKey, BinType::DYNAMIC_BIN, genPlaceholder, hasDevPtrArg);
        return std::move(kernelBin);
    }

    static ViewOffsetTilingSizeResult CollectViewOffsetTensorSize(const char* npuArch)
    {
        struct NpuArchEnvGuard {
            ~NpuArchEnvGuard() { unsetenv("ASCEND_NPU_ARCH"); }
        } guard;

        if (npuArch == nullptr) {
            unsetenv("ASCEND_NPU_ARCH");
        } else {
            setenv("ASCEND_NPU_ARCH", npuArch, 1);
        }
        uint32_t opType = op::OpTypeDict::ToOpType("QuantBatchMatmulV3");
        const char* p = std::getenv("ASCEND_OPP_PATH");
        if (p == nullptr) {
            return {};
        }
        KeyAndDetail key;
        key.key = "hahaha";
        size_t hashKey = 123;
        char jsonPath[1024];
        char binPath[1024];
        snprintf_s(jsonPath, sizeof(jsonPath), sizeof(jsonPath),
                   "%s/built-in/op_impl/ai_core/tbe/kernel/ascend910/quant_batch_matmul_v3/"
                   "QuantBatchMatmulV3_ND_ND_int8_int8_bf16_high_performance.json",
                   p);
        snprintf_s(binPath, sizeof(binPath), sizeof(binPath),
                   "%s/built-in/op_impl/ai_core/tbe/kernel/ascend910/add/"
                   "Add_41dadce325b0f810d03359af2a38990b_high_performance.o",
                   p);
        OpKernelBin kernelBin(opType, jsonPath, jsonPath, binPath, key, hashKey, BinType::DYNAMIC_BIN, false, false);
        if (kernelBin.JsonLoad() != ACLNN_SUCCESS) {
            return {};
        }

        std::array<uint8_t, 64> tensorData{};
        aclTensor base(op::Shape({18}), op::DataType::DT_FLOAT16, op::Format::FORMAT_ND, tensorData.data());
        aclOpExecutor exe;
        aclTensor* view = exe.CreateView(&base, op::Shape({17}), op::Shape({18}), base.GetViewStrides(), 1);
        if (view == nullptr) {
            return {};
        }

        auto ctx = op::MakeOpArgContext(OP_INPUT(view));
        if (ctx == nullptr) {
            return {};
        }
        op::internal::ExpandableRtsArgBuffer buffer;
        buffer.Init(TEST_LAUNCH_ARG_INIT_CAP, TEST_TILING_HOST_DATA_INIT_CAP);
        op::internal::TilingData* tilingData = buffer.GetTilingDataPtr();
        tilingData->data_ = buffer.GetTilingDataAddr();
        tilingData->data_size_ = 0;
        tilingData->capacity_ = TEST_TILING_HOST_DATA_INIT_CAP;

        ViewOffsetTilingSizeResult result;
        result.ok = kernelBin.AppendTilingOOMInfo(&buffer, ctx) == ACLNN_SUCCESS;
        if (result.ok) {
            result.tensorSize = *reinterpret_cast<uint64_t*>(buffer.GetTilingDataAddr());
        }
        op::DestroyOpArgContext(ctx);
        return result;
    }

    static ViewOffsetTilingSizeResult CollectViewOffsetTensorSizeInNewThread(const char* npuArch)
    {
        ViewOffsetTilingSizeResult result;
        std::thread thread([&result, npuArch]() { result = CollectViewOffsetTensorSize(npuArch); });
        thread.join();
        return result;
    }
};

TEST_F(OpKernelUT, OpKernelUTCase1)
{
    op::Shape selfShape{33, 15, 1, 48};
    op::Shape otherShape{33, 15, 14, 48};
    op::Shape outShape{33, 15, 14, 48};

    auto self = std::make_unique<aclTensor>(selfShape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND, nullptr);
    auto other = std::make_unique<aclTensor>(otherShape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND, nullptr);
    float alpha = 13.37;
    auto out = std::make_unique<aclTensor>(outShape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND, nullptr);

    uint32_t opType = op::OpTypeDict::ToOpType("Axpy");
    auto input = OP_INPUT(self.get(), other.get());
    auto output = OP_OUTPUT(out.get());
    auto attr = OP_ATTR(alpha);
    auto ws = OP_WORKSPACE(out.get());
    auto ctx = op::MakeOpArgContext(input, output, attr, ws);
    int dummyStream = 0;
    void* stream = &dummyStream;
    GetLauncherCtx().Reset();
    auto rc = gKernelMgr.Run(opType, stream, ctx);
    op::DestroyOpArgContext(ctx);
    EXPECT_EQ(rc, ACL_SUCCESS);

    auto tilingRes = OpRunContextMgr::opRunCtx_.tilingCtx_.GetTilingResult();

    const char* p = std::getenv("ASCEND_OPP_PATH");
    EXPECT_NE(p, nullptr);
    KeyAndDetail key;
    key.key = "hahaha";
    size_t hashKey = 123;
    char jsonPath[1024];
    char binPath[1024];
    snprintf_s(jsonPath, sizeof(jsonPath), sizeof(jsonPath),
               "%s/built-in/op_impl/ai_core/tbe/kernel/ascend910/axpy/"
               "Axpy_233851a3505389e43928a8bba133a74d_high_performance.json",
               p);
    snprintf_s(binPath, sizeof(binPath), sizeof(binPath),
               "%s/built-in/op_impl/ai_core/tbe/kernel/ascend910/axpy/"
               "Axpy_233851a3505389e43928a8bba133a74d_high_performance.o",
               p);

    OpKernelBin kernelBin(opType, jsonPath, jsonPath, binPath, key, hashKey, BinType::DYNAMIC_BIN, false, false);

    aclnnStatus rec = kernelBin.JsonLoad();
    EXPECT_EQ(rec, ACLNN_SUCCESS);

    auto taskType = kernelBin.GetTaskInfo(*(tilingRes->tilingKey_));
    EXPECT_NE(taskType.type, MSPROF_GE_TASK_TYPE_INVALID);

    // contain taskRation
    snprintf_s(jsonPath, sizeof(jsonPath), sizeof(jsonPath),
               "%s/built-in/op_impl/ai_core/tbe/kernel/ascend910/dummy/dummy_1.json", p);
    snprintf_s(binPath, sizeof(binPath), sizeof(binPath),
               "%s/built-in/op_impl/ai_core/tbe/kernel/ascend910/dummy/dummy_1.o", p);
    OpKernelBin kernelBin1(9999, jsonPath, jsonPath, binPath, key, hashKey, BinType::DYNAMIC_BIN, false, false);
    aclnnStatus rec1 = kernelBin1.JsonLoad();
    EXPECT_EQ(rec1, ACLNN_SUCCESS);
    auto taskType1 = kernelBin1.GetTaskInfo(*(tilingRes->tilingKey_));
    EXPECT_EQ(taskType1.type, MSPROF_GE_TASK_TYPE_MIX_AIC);
    auto taskType2 = kernelBin1.GetTaskInfo(9999999);
    EXPECT_EQ(taskType1.type, MSPROF_GE_TASK_TYPE_MIX_AIC);

    // not contains any bin list.
    OpKernel* kernel = gKernelMgr.GetKernel(opType);
    std::string debugDynBinAndJsonDir;
    auto res = kernel->AppendDynBin(jsonPath, debugDynBinAndJsonDir, false);
    EXPECT_EQ(res, ACLNN_ERR_INNER);

    // gen placeholder.
    OpKernel* kernelAxpy = gKernelMgr.GetKernel(op::OpTypeDict::ToOpType("Axpy"));
    snprintf_s(jsonPath, sizeof(jsonPath), sizeof(jsonPath),
               "%s/built-in/op_impl/ai_core/tbe/kernel/config/ascend910/axpy.json", p);
    auto resAxpy = kernelAxpy->AppendDynBin(jsonPath, debugDynBinAndJsonDir, false);
    EXPECT_EQ(resAxpy, ACLNN_SUCCESS);
}

TEST_F(OpKernelUT, GetTaskTypeSingleBinMmultiKernelType)
{
    uint32_t opType = op::OpTypeDict::ToOpType("QuantBatchMatmulV3");
    const char* p = std::getenv("ASCEND_OPP_PATH");
    EXPECT_NE(p, nullptr);
    KeyAndDetail key;
    key.key = "hahaha";
    size_t hashKey = 123;
    char jsonPath[1024];
    char binPath[1024];
    snprintf_s(jsonPath, sizeof(jsonPath), sizeof(jsonPath),
               "%s/built-in/op_impl/ai_core/tbe/kernel/ascend910/quant_batch_matmul_v3/"
               "QuantBatchMatmulV3_ND_ND_int8_int8_bf16_high_performance.json",
               p);
    snprintf_s(
        binPath, sizeof(binPath), sizeof(binPath),
        "%s/built-in/op_impl/ai_core/tbe/kernel/ascend910/add/Add_41dadce325b0f810d03359af2a38990b_high_performance.o",
        p);

    OpKernelBin kernelBin(opType, jsonPath, jsonPath, binPath, key, hashKey, BinType::DYNAMIC_BIN, false, false);

    aclnnStatus rec = kernelBin.JsonLoad();
    EXPECT_EQ(rec, ACLNN_SUCCESS);

    auto taskType = kernelBin.GetTaskInfo(0);
    EXPECT_EQ(taskType.type, MSPROF_GE_TASK_TYPE_MIX_AIC);
    EXPECT_EQ(taskType.ration, 1);
    taskType = kernelBin.GetTaskInfo(10);
    EXPECT_EQ(taskType.type, MSPROF_GE_TASK_TYPE_AI_CORE);
    EXPECT_EQ(taskType.ration, 0);
    taskType = kernelBin.GetTaskInfo(11);
    EXPECT_EQ(taskType.type, MSPROF_GE_TASK_TYPE_MIX_AIC);
    EXPECT_EQ(taskType.ration, 1);
    MsprofCompactInfo compactInfo;
    PrepareBasicInfo(compactInfo, taskType, 0, 0);

    taskType = kernelBin.GetTaskInfo(10000);
    EXPECT_EQ(taskType.type, MSPROF_GE_TASK_TYPE_AI_CORE);
    EXPECT_EQ(taskType.ration, 0);
    taskType = kernelBin.GetTaskInfo(10001);
    EXPECT_EQ(taskType.type, MSPROF_GE_TASK_TYPE_MIX_AIC);
    EXPECT_EQ(taskType.ration, 0);

    taskType = kernelBin.GetTaskInfo(1001);
    EXPECT_EQ(taskType.type, MSPROF_GE_TASK_TYPE_MIX_AIC);
    EXPECT_EQ(taskType.ration, 2);
    ReportAdditionInfo(taskType, 0, 0);

    taskType = kernelBin.GetTaskInfo(100011);
    EXPECT_EQ(taskType.type, MSPROF_GE_TASK_TYPE_AI_CORE);
    EXPECT_EQ(taskType.ration, 0);
    ReportAdditionInfo(taskType, 0, 0);
}

TEST_F(OpKernelUT, TestKernelListCoreTypeMIX)
{
    uint32_t opType = op::OpTypeDict::ToOpType("QuantBatchMatmulV3");
    const char* p = std::getenv("ASCEND_OPP_PATH");
    EXPECT_NE(p, nullptr);
    KeyAndDetail key;
    key.key = "hahaha";
    size_t hashKey = 123;
    char jsonPath[1024];
    char binPath[1024];
    snprintf_s(jsonPath, sizeof(jsonPath), sizeof(jsonPath),
               "%s/built-in/op_impl/ai_core/tbe/kernel/ascend910/mat_mul/MatMulV2_kernelList_coreType_MIX.json", p);
    snprintf_s(
        binPath, sizeof(binPath), sizeof(binPath),
        "%s/built-in/op_impl/ai_core/tbe/kernel/ascend910/add/Add_41dadce325b0f810d03359af2a38990b_high_performance.o",
        p);

    OpKernelBin kernelBin(opType, jsonPath, jsonPath, binPath, key, hashKey, BinType::DYNAMIC_BIN, false, false);

    aclnnStatus rec = kernelBin.JsonLoad();
    EXPECT_EQ(rec, ACLNN_SUCCESS);
}

TEST_F(OpKernelUT, opProfilingSwitchSet)
{
    setenv("GE_PROFILING_TO_STD_OUT", "1", 1);
    opProfilingSwitch.reportFlag = true;
    opProfilingSwitch.kernelLaunchFlag = true;
    opProfilingSwitch.additionInfoFlag = true;
    opProfilingSwitch.level2ProfilingFlag = true;

    op::Shape selfShape{33, 15, 1, 48};
    op::Shape otherShape{33, 15, 14, 48};
    op::Shape outShape{33, 15, 14, 48};

    auto self = std::make_unique<aclTensor>(selfShape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND, nullptr);
    auto other = std::make_unique<aclTensor>(otherShape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND, nullptr);
    float alpha = 13.37;
    auto out = std::make_unique<aclTensor>(outShape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND, nullptr);

    uint32_t opType = op::OpTypeDict::ToOpType("Axpy");
    auto input = OP_INPUT(self.get(), other.get());
    auto output = OP_OUTPUT(out.get());
    auto attr = OP_ATTR(alpha);
    auto ws = OP_WORKSPACE(out.get());

    auto uniqueExecutor = CREATE_EXECUTOR();
    aclOpExecutor* executor = uniqueExecutor.get();
    thread_local uint64_t kernelLaunchIdDefinedInL0Dfx = GenKernelLauncherId("Axpy");
    ProfilingInfoId profilingInfoId(0, kernelLaunchIdDefinedInL0Dfx, 0);

    auto ctx = op::MakeOpArgContext(OP_INPUT(self.get(), other.get()), OP_OUTPUT(out.get()), OP_ATTR(alpha),
                                    OP_WORKSPACE(out.get()));
    auto* launcher = new op::AiCoreKernelLauncher{opType, op::AI_CORE, profilingInfoId, executor, ctx};

    auto rc = launcher->Launch();

    EXPECT_EQ(rc, ACL_SUCCESS);
    delete launcher;
    InitPTACacheThreadLocal();
}

TEST_F(OpKernelUT, LaunchArgCacheTest)
{
    op::Shape selfShape{33, 15, 64};
    op::Shape outShape{33, 15, 64};
    op::Shape idxShape{33, 15, 64};
    op::Shape wsShape{32};

    int64_t dim = 0;
    bool descending = true;

    auto self = std::make_unique<aclTensor>(selfShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND, nullptr);
    auto out = std::make_unique<aclTensor>(outShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND, nullptr);
    auto idx = std::make_unique<aclTensor>(idxShape, op::DataType::DT_INT32, op::Format::FORMAT_ND, nullptr);

    uint32_t opType = op::OpTypeDict::ToOpType("Sort");
    aclTensorList* tensorList = nullptr;
    aclTensor* tensorNull = nullptr;
    auto input = OP_INPUT(self.get(), tensorList, tensorNull);
    auto output = OP_OUTPUT(out.get(), idx.get(), static_cast<aclTensor*>(nullptr),
                            static_cast<aclTensorList*>(nullptr));
    auto attr = OP_ATTR(dim, descending);

    GetLauncherCtx().Reset();
    aclrtStream stream = 0;

    // creat OpKernelBin
    const char* p = std::getenv("ASCEND_OPP_PATH");
    EXPECT_NE(p, nullptr);
    KeyAndDetail key;
    key.key = "hahaha";
    size_t hashKey = 123;
    char jsonPath[1024];
    char binPath[1024];
    snprintf_s(jsonPath, sizeof(jsonPath), sizeof(jsonPath),
               "%s/built-in/op_impl/ai_core/tbe/kernel/ascend910/axpy/"
               "Axpy_233851a3505389e43928a8bba133a74d_high_performance.json",
               p);
    snprintf_s(binPath, sizeof(binPath), sizeof(binPath),
               "%s/built-in/op_impl/ai_core/tbe/kernel/ascend910/axpy/"
               "Axpy_233851a3505389e43928a8bba133a74d_high_performance.o",
               p);

    OpKernelBin kernelBin(opType, jsonPath, jsonPath, binPath, key, hashKey, BinType::DYNAMIC_BIN, false, false);

    aclnnStatus rec = kernelBin.InitTilingParseCtx();
    EXPECT_EQ(rec, ACLNN_SUCCESS);

    rec = kernelBin.JsonLoad();
    EXPECT_EQ(rec, ACLNN_SUCCESS);

    op::internal::ExpandableRtsArgBuffer buffer;
    buffer.Init(TEST_LAUNCH_ARG_INIT_CAP, TEST_TILING_HOST_DATA_INIT_CAP);
    op::internal::TilingData* tilingData = buffer.GetTilingDataPtr();
    constexpr size_t TEST_TILING_DATA_LEN = 32;
    tilingData->data_ = buffer.GetTilingDataAddr();
    memset_s(tilingData->data_, TEST_TILING_HOST_DATA_INIT_CAP, 0, TEST_TILING_DATA_LEN);
    tilingData->data_size_ = TEST_TILING_DATA_LEN;
    tilingData->capacity_ = TEST_TILING_HOST_DATA_INIT_CAP;

    auto ctx = op::MakeOpArgContext(input, output);
    LaunchArgInfo argInfo(false, false, ctx);
    RtsArg arg(true, argInfo, &buffer);
    arg.FillArgs();
    PrintRtArg(arg.GetRtsArg());

    GetThreadLocalContext().hashKey_ = 0;
    GetThreadLocalContext().cacheHashKey_ = (uint8_t*)"hello";
    GetThreadLocalContext().cacheHashKeyLen_ = 5;
    auto opExecCache = new OpExecCache();
    opExecCache->SetCacheBuf(GetCacheBuf());
    GetOpCacheContext().SetOpCache(opExecCache);
    auto cacheCtx = arg.DumpToCache();
    EXPECT_NE(cacheCtx, nullptr);

    // get TilingCtxOutput
    const TilingCtxOutput* res = nullptr;
    auto ctx2 = op::MakeOpArgContext(input, output, attr);
    uint32_t aic = op::internal::GetThreadLocalContext().opConfigInfo_.aicNum_;
    uint32_t aiv = op::internal::GetThreadLocalContext().opConfigInfo_.aivNum_;
    // OpRunContextMgr opRunContextMgr
    res = OpRunContextMgr::Tiling(opType, kernelBin.tilingParseCtxHolder_[op::internal::ThreadCoreNum(aic, aiv)].get(),
                                  *ctx2->GetOpArg(op::OpArgDef::OP_INPUT_ARG),
                                  *ctx2->GetOpArg(op::OpArgDef::OP_OUTPUT_ARG),
                                  *ctx2->GetOpArg(op::OpArgDef::OP_ATTR_ARG));
    EXPECT_NE(res, nullptr);

    std::vector<int32_t> tensorOffset;
    auto rc = kernelBin.DoLaunch(res, stream, false, ctx2, tensorOffset);
    EXPECT_EQ(rc, ACLNN_SUCCESS);
    InitPTACacheThreadLocal();
    delete opExecCache;
    op::DestroyOpArgContext(ctx);
    GetOpCacheContext().SetOpCache(nullptr);

    opProfilingSwitch.recordOpArgFlag = true;
    auto resTiling = OpRunContextMgr::Tiling(
        opType, kernelBin.tilingParseCtxHolder_[op::internal::ThreadCoreNum(aic, aiv)].get(),
        *ctx2->GetOpArg(op::OpArgDef::OP_INPUT_ARG), *ctx2->GetOpArg(op::OpArgDef::OP_OUTPUT_ARG),
        *ctx2->GetOpArg(op::OpArgDef::OP_ATTR_ARG));
    EXPECT_NE(resTiling, nullptr);
    for (auto& [key, value] : kernelBin.tilingParseCtxHolder_) {
        if (value.get()) {
            value.get()->ReleaseTilingParse();
        }
    }
    op::DestroyOpArgContext(ctx2);
    // 恢复全局开关，避免污染后续用例
    opProfilingSwitch.recordOpArgFlag = false;
}

TEST_F(OpKernelUT, ReportOpAttrInfoTest)
{
    setenv("GE_PROFILING_TO_STD_OUT", "1", 1);
    op::Shape outShape{3};
    op::Shape idxShape{3};
    op::Shape wsShape{2};
    int64_t dim = 0;
    bool descending = true;

    aclOpExecutor executor;

    vector<int64_t> vec{1, 2, 3};
    auto self = executor.AllocHostTensor(vec.data(), vec.size(), op::DataType::DT_INT64);
    EXPECT_NE(self, nullptr);
    vector<float> vec1{1.0, 2.0, 3.0};
    auto self1 = executor.AllocHostTensor(vec1.data(), vec1.size(), op::DataType::DT_FLOAT);
    EXPECT_NE(self1, nullptr);
    bool boolValue = true;
    auto self2 = executor.AllocHostTensor(&boolValue, 1, op::DataType::DT_BOOL);
    EXPECT_NE(self2, nullptr);
    std::vector<op::fp16_t> values3(3, 3.0);
    auto self3 = executor.AllocHostTensor(values3.data(), values3.size(), op::DataType::DT_FLOAT16);
    EXPECT_NE(self3, nullptr);
    vector<bfloat16> vec16{1.0, 2.0, 3.0};
    auto self4 = executor.AllocHostTensor(vec16.data(), vec16.size(), op::DataType::DT_BF16);
    EXPECT_NE(self4, nullptr);

    vector<int32_t> vec5{1, 2, 3};
    auto self5 = executor.AllocHostTensor(vec5.data(), vec5.size(), op::DataType::DT_INT32);
    EXPECT_NE(self5, nullptr);
    vector<uint32_t> vec6{1, 2, 3};
    auto self6 = executor.AllocHostTensor(vec6.data(), vec6.size(), op::DataType::DT_UINT32);
    EXPECT_NE(self6, nullptr);
    vector<uint64_t> vec7{1, 2, 3};
    auto self7 = executor.AllocHostTensor(vec7.data(), vec7.size(), op::DataType::DT_UINT64);
    EXPECT_NE(self7, nullptr);

    vector<double> vec8{1.0, 2.0, 3.0};
    auto self8 = executor.AllocHostTensor(vec8.data(), vec8.size(), op::DataType::DT_DOUBLE);
    EXPECT_NE(self8, nullptr);

    vector<int8_t> vec9{1, 2, 3};
    auto self9 = executor.AllocHostTensor(vec9.data(), vec9.size(), op::DataType::DT_INT8);
    EXPECT_NE(self9, nullptr);

    vector<uint8_t> vec10{1, 2, 3};
    auto self10 = executor.AllocHostTensor(vec10.data(), vec10.size(), op::DataType::DT_UINT8);
    EXPECT_NE(self10, nullptr);

    vector<int16_t> vec11{1, 2, 3};
    auto self11 = executor.AllocHostTensor(vec11.data(), vec11.size(), op::DataType::DT_INT16);
    EXPECT_NE(self11, nullptr);

    vector<uint16_t> vec13{1, 2, 3};
    auto self13 = executor.AllocHostTensor(vec13.data(), vec13.size(), op::DataType::DT_UINT16);
    EXPECT_NE(self13, nullptr);

    auto out = std::make_unique<aclTensor>(outShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND, nullptr);
    auto idx = std::make_unique<aclTensor>(idxShape, op::DataType::DT_INT32, op::Format::FORMAT_ND, nullptr);

    aclTensor* tensorNull = nullptr;

    auto ws1 = std::make_unique<aclTensor>(wsShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND, nullptr);
    auto ws2 = std::make_unique<aclTensor>(wsShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND, nullptr);
    auto ws3 = std::make_unique<aclTensor>(wsShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND, nullptr);
    const aclTensor* wsArr[] = {ws1.get(), ws2.get(), ws3.get()};
    aclTensorList* wsList = aclCreateTensorList(wsArr, 3);
    const aclTensor* wsArr1[] = {self9, self10, self11};
    aclTensorList* wsList1 = aclCreateTensorList(wsArr1, 3);

    auto self12 = std::make_unique<aclTensor>(idxShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND, nullptr);
    auto input = OP_INPUT(self, self1, self2, self3, self4, self5, self6, self7, self8, self12.get(), self13, wsList,
                          wsList1, tensorNull, static_cast<aclTensorList*>(nullptr));
    auto output = OP_OUTPUT(out.get(), idx.get(), static_cast<aclTensor*>(nullptr),
                            static_cast<aclTensorList*>(nullptr));
    auto attr = OP_ATTR(dim, descending);

    uint32_t opType = op::OpTypeDict::ToOpType("Sort");
    GetLauncherCtx().Reset();
    aclrtStream stream = 0;

    // creat OpKernelBin
    const char* p = std::getenv("ASCEND_OPP_PATH");
    EXPECT_NE(p, nullptr);
    KeyAndDetail key;
    key.key = "hahaha";
    size_t hashKey = 123;
    char jsonPath[1024];
    char binPath[1024];
    snprintf_s(jsonPath, sizeof(jsonPath), sizeof(jsonPath),
               "%s/built-in/op_impl/ai_core/tbe/kernel/ascend910/axpy/"
               "Axpy_233851a3505389e43928a8bba133a74d_high_performance.json",
               p);
    snprintf_s(binPath, sizeof(binPath), sizeof(binPath),
               "%s/built-in/op_impl/ai_core/tbe/kernel/ascend910/axpy/"
               "Axpy_233851a3505389e43928a8bba133a74d_high_performance.o",
               p);
    int64_t i = 1;
    OpKernelBin kernelBin(opType, jsonPath, jsonPath, binPath, key, hashKey, BinType::DYNAMIC_BIN, false, false,
                          static_cast<void*>(&i));

    auto ctx = op::MakeOpArgContext(input, output);

    op::internal::opProfilingSwitch.level2ProfilingFlag = true;
    kernelBin.ReportOpAttrInfo(ctx, 0);

    op::DestroyOpArgContext(ctx);

    delete wsList;
    delete wsList1;
    unsetenv("GE_PROFILING_TO_STD_OUT");
}

TEST_F(OpKernelUT, AppendAttrCharTest0)
{
    size_t attrSizeInProto = 1;
    size_t idx = 0;
    char* value = nullptr;
    op::FVector<NnopbaseAttrAddr*> attrsVec;

    uint32_t opType = op::OpTypeDict::ToOpType("AddN");

    auto opKernel = gKernelMgr.GetKernel(opType);
    EXPECT_NE(opKernel, nullptr);
    op::OpArg arg;
    arg.type = op::OpArgType::OPARG_STRING;
    arg->pointer = value;
    aclnnStatus rec = opKernel->AppendAttr(attrSizeInProto, idx, arg, attrsVec);
    EXPECT_EQ(rec, ACLNN_SUCCESS);
}

TEST_F(OpKernelUT, AppendAttrCharTest1)
{
    size_t attrSizeInProto = 1;
    size_t idx = 0;
    char* value = "12345";
    op::FVector<NnopbaseAttrAddr*> attrsVec;

    uint32_t opType = op::OpTypeDict::ToOpType("AddN");

    auto opKernel = gKernelMgr.GetKernel(opType);
    EXPECT_NE(opKernel, nullptr);
    op::OpArg arg;
    arg.type = op::OpArgType::OPARG_STRING;
    arg->pointer = value;
    aclnnStatus rec = opKernel->AppendAttr(attrSizeInProto, idx, arg, attrsVec);
    EXPECT_EQ(rec, ACLNN_SUCCESS);
}

TEST_F(OpKernelUT, aclIntArrayTest1)
{
    size_t attrSizeInProto = 1;
    size_t idx = 0;
    aclIntArray* value = nullptr;
    op::FVector<NnopbaseAttrAddr*> attrsVec;

    uint32_t opType = op::OpTypeDict::ToOpType("AddN");

    auto opKernel = gKernelMgr.GetKernel(opType);
    EXPECT_NE(opKernel, nullptr);
    op::OpArg arg;
    arg.type = op::OpArgType::OPARG_INT_LIST;
    arg->pointer = value;
    aclnnStatus rec = opKernel->AppendAttr(attrSizeInProto, idx, arg, attrsVec);
    EXPECT_EQ(rec, ACLNN_SUCCESS);
}

TEST_F(OpKernelUT, tilingCacheNotNull)
{
    op::Shape selfShape{33, 15, 64};
    op::Shape outShape{33, 15, 64};
    op::Shape idxShape{33, 15, 64};
    op::Shape wsShape{32};

    int64_t dim = 0;
    bool descending = true;

    auto self = std::make_unique<aclTensor>(selfShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND, nullptr);
    auto out = std::make_unique<aclTensor>(outShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND, nullptr);
    auto idx = std::make_unique<aclTensor>(idxShape, op::DataType::DT_INT32, op::Format::FORMAT_ND, nullptr);

    uint32_t opType = op::OpTypeDict::ToOpType("Sort");
    auto input = OP_INPUT(self.get());
    auto output = OP_OUTPUT(out.get(), idx.get(), static_cast<aclTensor*>(nullptr),
                            static_cast<aclTensorList*>(nullptr));
    auto attr = OP_ATTR(dim, descending);

    GetLauncherCtx().Reset();

    auto uniqueExecutor = CREATE_EXECUTOR();
    aclOpExecutor* executor = uniqueExecutor.get();
    aclTensorList* workspace1 = nullptr;
    auto ctx = op::MakeOpArgContext(input, output, attr);
    GetWorkspace(opType, &workspace1, executor, *ctx->GetOpArg(op::OpArgDef::OP_INPUT_ARG),
                 *ctx->GetOpArg(op::OpArgDef::OP_OUTPUT_ARG), *ctx->GetOpArg(op::OpArgDef::OP_ATTR_ARG));

    const TilingResCache* tilingCache = nullptr;
    tilingCache = GetLauncherCtx().GetTilingResCache();
    EXPECT_NE(tilingCache, nullptr);

    int dummyStream = 0;
    void* stream = &dummyStream;
    GetThreadLocalContext().hashKey_ = 0;
    GetThreadLocalContext().cacheHashKey_ = (uint8_t*)"hello";
    GetThreadLocalContext().cacheHashKeyLen_ = 5;
    auto opExecCache = new OpExecCache();
    opExecCache->SetCacheBuf(GetCacheBuf());
    GetOpCacheContext().SetOpCache(opExecCache);
    ctx->AppendOpWorkspaceArg(workspace1);
    auto rc = gKernelMgr.Run(opType, stream, ctx);
    EXPECT_EQ(rc, ACL_SUCCESS);

    InitPTACacheThreadLocal();

    delete opExecCache;
    aclDestroyTensorList((const aclTensorList*)workspace1);
    uniqueExecutor.ReleaseTo(&executor);
    GetOpCacheContext().SetOpCache(nullptr);
    op::DestroyOpArgContext(ctx);
}

TEST_F(OpKernelUT, invalidTilingArg)
{
    op::Shape selfShape{33, 15, 64};
    op::Shape outShape{33, 15, 64};
    op::Shape idxShape{33, 15, 64};
    op::Shape wsShape{32};

    int64_t dim = 0;
    bool descending = true;

    auto self = std::make_unique<aclTensor>(selfShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND, nullptr);
    auto out = std::make_unique<aclTensor>(outShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND, nullptr);
    auto idx = std::make_unique<aclTensor>(idxShape, op::DataType::DT_INT32, op::Format::FORMAT_ND, nullptr);
    int xx = 1;

    uint32_t opType = op::OpTypeDict::ToOpType("Sort");
    auto input = OP_INPUT(self.get());
    auto output = OP_OUTPUT(out.get(), idx.get(), static_cast<aclTensor*>(nullptr),
                            static_cast<aclTensorList*>(nullptr), xx);
    auto attr = OP_ATTR(dim, descending);

    GetLauncherCtx().Reset();

    auto uniqueExecutor = CREATE_EXECUTOR();
    aclOpExecutor* executor = uniqueExecutor.get();
    aclTensorList* workspace1 = nullptr;
    auto ctx = op::MakeOpArgContext(input, output, attr);
    aclnnStatus ret = GetWorkspace(opType, &workspace1, executor, *ctx->GetOpArg(op::OpArgDef::OP_INPUT_ARG),
                                   *ctx->GetOpArg(op::OpArgDef::OP_OUTPUT_ARG),
                                   *ctx->GetOpArg(op::OpArgDef::OP_ATTR_ARG));
    EXPECT_EQ(ret, ACLNN_ERR_INNER);
    aclDestroyTensorList((const aclTensorList*)workspace1);
    op::internal::OpRunContextMgr::opRunCtx_.kernelCtx_.UpdateOutputArgIr(*ctx->GetOpArg(op::OpArgDef::OP_OUTPUT_ARG));
    uniqueExecutor.ReleaseTo(&executor);
    op::DestroyOpArgContext(ctx);
}

TEST_F(OpKernelUT, TestDoLaunchWithSplitAicAndAiv)
{
    op::Shape selfShape{33, 15, 64};
    op::Shape outShape{33, 15, 64};
    op::Shape idxShape{33, 15, 64};
    op::Shape wsShape{32};

    int64_t dim = 0;
    bool descending = true;

    auto self = std::make_unique<aclTensor>(selfShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND, nullptr);
    auto out = std::make_unique<aclTensor>(outShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND, nullptr);
    auto idx = std::make_unique<aclTensor>(idxShape, op::DataType::DT_INT32, op::Format::FORMAT_ND, nullptr);

    uint32_t opType = op::OpTypeDict::ToOpType("Sort");
    aclTensorList* tensorList = nullptr;
    aclTensor* tensorNull = nullptr;
    auto input = OP_INPUT(self.get(), tensorList, tensorNull);
    auto output = OP_OUTPUT(out.get(), idx.get(), static_cast<aclTensor*>(nullptr),
                            static_cast<aclTensorList*>(nullptr));
    auto attr = OP_ATTR(dim, descending);

    GetLauncherCtx().Reset();
    aclrtStream stream = 0;

    // creat OpKernelBin
    const char* p = std::getenv("ASCEND_OPP_PATH");
    EXPECT_NE(p, nullptr);
    KeyAndDetail key;
    key.key = "hahaha";
    size_t hashKey = 123;
    char jsonPath[1024];
    char binPath[1024];
    snprintf_s(jsonPath, sizeof(jsonPath), sizeof(jsonPath),
               "%s/built-in/op_impl/ai_core/tbe/kernel/ascend910/axpy/"
               "Axpy_233851a3505389e43928a8bba133a74d_high_performance.json",
               p);
    snprintf_s(binPath, sizeof(binPath), sizeof(binPath),
               "%s/built-in/op_impl/ai_core/tbe/kernel/ascend910/axpy/"
               "Axpy_233851a3505389e43928a8bba133a74d_high_performance.o",
               p);

    OpKernelBin kernelBin(opType, jsonPath, jsonPath, binPath, key, hashKey, BinType::DYNAMIC_BIN, false, false);

    aclnnStatus rec = kernelBin.InitTilingParseCtx();
    EXPECT_EQ(rec, ACLNN_SUCCESS);

    rec = kernelBin.JsonLoad();
    EXPECT_EQ(rec, ACLNN_SUCCESS);

    op::internal::ExpandableRtsArgBuffer buffer;
    buffer.Init(TEST_LAUNCH_ARG_INIT_CAP, TEST_TILING_HOST_DATA_INIT_CAP);
    op::internal::TilingData* tilingData = buffer.GetTilingDataPtr();
    constexpr size_t TEST_TILING_DATA_LEN = 32;
    tilingData->data_ = buffer.GetTilingDataAddr();
    memset_s(tilingData->data_, TEST_TILING_HOST_DATA_INIT_CAP, 0, TEST_TILING_DATA_LEN);
    tilingData->data_size_ = TEST_TILING_DATA_LEN;
    tilingData->capacity_ = TEST_TILING_HOST_DATA_INIT_CAP;

    auto ctx = op::MakeOpArgContext(input, output);
    LaunchArgInfo argInfo(false, false, ctx);
    RtsArg arg(true, argInfo, &buffer);
    arg.FillArgs();
    PrintRtArg(arg.GetRtsArg());

    GetThreadLocalContext().hashKey_ = 0;
    GetThreadLocalContext().cacheHashKey_ = (uint8_t*)"hello";
    GetThreadLocalContext().cacheHashKeyLen_ = 5;
    auto opExecCache = new OpExecCache();
    opExecCache->SetCacheBuf(GetCacheBuf());
    GetOpCacheContext().SetOpCache(opExecCache);
    auto cacheCtx = arg.DumpToCache();
    EXPECT_NE(cacheCtx, nullptr);

    // get TilingCtxOutput
    const TilingCtxOutput* res = nullptr;
    auto ctx2 = op::MakeOpArgContext(input, output, attr);
    // OpRunContextMgr opRunContextMgr
    uint32_t aic = op::internal::GetThreadLocalContext().opConfigInfo_.aicNum_;
    uint32_t aiv = op::internal::GetThreadLocalContext().opConfigInfo_.aivNum_;
    res = OpRunContextMgr::Tiling(opType, kernelBin.tilingParseCtxHolder_[op::internal::ThreadCoreNum(aic, aiv)].get(),
                                  *ctx2->GetOpArg(op::OpArgDef::OP_INPUT_ARG),
                                  *ctx2->GetOpArg(op::OpArgDef::OP_OUTPUT_ARG),
                                  *ctx2->GetOpArg(op::OpArgDef::OP_ATTR_ARG));
    EXPECT_NE(res, nullptr);

    auto rc = kernelBin.DoLaunchWithSplitAicAndAiv(res, stream, ctx2);
    EXPECT_EQ(rc, ACLNN_SUCCESS);
    InitPTACacheThreadLocal();
    delete opExecCache;
    op::DestroyOpArgContext(ctx);
    GetOpCacheContext().SetOpCache(nullptr);

    opProfilingSwitch.recordOpArgFlag = true;
    auto resTiling = OpRunContextMgr::Tiling(
        opType, kernelBin.tilingParseCtxHolder_[op::internal::ThreadCoreNum(aic, aiv)].get(),
        *ctx2->GetOpArg(op::OpArgDef::OP_INPUT_ARG), *ctx2->GetOpArg(op::OpArgDef::OP_OUTPUT_ARG),
        *ctx2->GetOpArg(op::OpArgDef::OP_ATTR_ARG));
    EXPECT_NE(resTiling, nullptr);
    for (auto& [key, value] : kernelBin.tilingParseCtxHolder_) {
        if (value.get()) {
            value.get()->ReleaseTilingParse();
        }
    }
    op::DestroyOpArgContext(ctx2);
    // 恢复全局开关，避免污染后续用例
    opProfilingSwitch.recordOpArgFlag = false;
}

TEST_F(OpKernelUT, NoKernelBinTest)
{
    op::Shape selfShape{33, 15, 64};
    op::Shape outShape{33, 15, 64};
    op::Shape idxShape{33, 15, 64};
    int alpha = 1337;

    auto self = std::make_unique<aclTensor>(selfShape, op::DataType::DT_INT8, op::Format::FORMAT_ND, nullptr);
    auto other = std::make_unique<aclTensor>(outShape, op::DataType::DT_INT8, op::Format::FORMAT_ND, nullptr);
    auto out = std::make_unique<aclTensor>(idxShape, op::DataType::DT_INT8, op::Format::FORMAT_ND, nullptr);

    const aclTensor* inputTensor[2] = {self.get(), other.get()};
    aclTensorList* inputList = aclCreateTensorList(inputTensor, 2);

    auto input = OP_INPUT(inputList);
    auto output = OP_OUTPUT(out.get());
    auto attr = OP_ATTR(alpha);
    auto ws = OP_WORKSPACE(out.get());

    uint32_t opType = op::OpTypeDict::ToOpType("AddN");
    auto opKernel = gKernelMgr.GetKernel(opType);
    EXPECT_NE(opKernel, nullptr);
    auto actualOpType = opKernel->GetOpType();
    EXPECT_EQ(actualOpType, opType);
    int dummyStream = 0;
    void* stream = &dummyStream;
    auto ctx = op::MakeOpArgContext(input, output, attr, ws);
    auto rc = opKernel->Run(stream, ctx);

    EXPECT_EQ(rc, ACLNN_ERR_INNER);

    uint32_t opType1 = op::OpTypeDict::ToOpType("AddNs");
    auto opKernel1 = gKernelMgr.GetKernel(opType1);
    EXPECT_EQ(opKernel1, nullptr);
    delete inputList;
    op::DestroyOpArgContext(ctx);
}

TEST_F(OpKernelUT, OpKernelBinFuncTest)
{
    op::Shape selfShape{33, 15, 1, 48};
    op::Shape otherShape{33, 15, 14, 48};
    op::Shape outShape{33, 15, 14, 48};

    auto self = std::make_unique<aclTensor>(selfShape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND, nullptr);
    auto other = std::make_unique<aclTensor>(otherShape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND, nullptr);
    float alpha = 13.37;
    auto out = std::make_unique<aclTensor>(outShape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND, nullptr);

    uint32_t opType = op::OpTypeDict::ToOpType("Axpy");
    auto input = OP_INPUT(self.get(), other.get());
    auto output = OP_OUTPUT(out.get());
    auto attr = OP_ATTR(alpha);
    auto ws = OP_WORKSPACE(out.get());
    int dummyStream = 0;
    void* stream = &dummyStream;

    const char* p = std::getenv("ASCEND_OPP_PATH");
    EXPECT_NE(p, nullptr);
    KeyAndDetail key;
    key.key = "hahaha";
    size_t hashKey = 123;
    char jsonPath[1024];
    char binPath[1024];
    snprintf_s(jsonPath, sizeof(jsonPath), sizeof(jsonPath),
               "%s/built-in/op_impl/ai_core/tbe/kernel/ascend910/sort/"
               "Sort_90da8e603a0431c8b18a0c23fdc8d8c9_high_performance.json",
               p);
    snprintf_s(binPath, sizeof(binPath), sizeof(binPath),
               "%s/built-in/op_impl/ai_core/tbe/kernel/ascend910/sort/"
               "Sort_90da8e603a0431c8b18a0c23fdc8d8c9_high_performance.o",
               p);

    OpKernelBin kernelBin(opType, jsonPath, jsonPath, binPath, key, hashKey, BinType::DYNAMIC_BIN, false, false);

    std::ifstream f(jsonPath);
    nlohmann::json jsonObj = nlohmann::json::parse(f);

    // ParseStaticBlockdim: Blockdim<0
    auto ret = kernelBin.ParseStaticBlockdim(jsonObj);
    EXPECT_EQ(ret, ACLNN_ERR_INNER_STATIC_BLOCK_DIM_INVALID);
    // no Blockdim
    jsonObj.erase("numBlocks");
    ret = kernelBin.ParseStaticBlockdim(jsonObj);
    EXPECT_EQ(ret, ACLNN_ERR_INNER_STATIC_BLOCK_DIM_INVALID);

    // ParseStaticImplMode: no implMode
    kernelBin.ParseStaticImplMode(jsonObj);

    // ParseStaticWorkSpace
    auto workspaceJson = jsonObj.find("workspace");
    ret = kernelBin.ParseStaticWorkSpace(*workspaceJson);
    EXPECT_EQ(ret, ACLNN_SUCCESS);
    // workspace no size
    ret = kernelBin.ParseStaticWorkSpace(jsonObj);
    EXPECT_EQ(ret, ACLNN_ERR_INNER_STATIC_WORKSPACE_INVALID);
}

TEST_F(OpKernelUT, GetHF32)
{
    uint32_t opType = op::OpTypeDict::ToOpType("QuantBatchMatmulV3");
    const char* p = std::getenv("ASCEND_OPP_PATH");
    EXPECT_NE(p, nullptr);
    KeyAndDetail key;
    key.key = "hahaha";
    size_t hashKey = 123;
    char jsonPath[1024];
    char binPath[1024];
    snprintf_s(jsonPath, sizeof(jsonPath), sizeof(jsonPath),
               "%s/built-in/op_impl/ai_core/tbe/kernel/ascend910/quant_batch_matmul_v3/"
               "QuantBatchMatmulV3_ND_ND_int8_int8_bf16_high_performance.json",
               p);
    snprintf_s(
        binPath, sizeof(binPath), sizeof(binPath),
        "%s/built-in/op_impl/ai_core/tbe/kernel/ascend910/add/Add_41dadce325b0f810d03359af2a38990b_high_performance.o",
        p);

    OpKernelBin kernelBin(opType, jsonPath, jsonPath, binPath, key, hashKey, BinType::DYNAMIC_BIN, false, false);

    aclnnStatus rec = kernelBin.JsonLoad();
    EXPECT_EQ(rec, ACLNN_SUCCESS);

    op::Shape selfShape{33, 15, 64};
    op::Shape outShape{33, 15, 64};
    op::Shape idxShape{33, 15, 64};
    int alpha = 1337;

    auto self = std::make_unique<aclTensor>(selfShape, op::DataType::DT_INT8, op::Format::FORMAT_ND, nullptr);
    auto other = std::make_unique<aclTensor>(outShape, op::DataType::DT_INT8, op::Format::FORMAT_ND, nullptr);
    auto out = std::make_unique<aclTensor>(idxShape, op::DataType::DT_INT8, op::Format::FORMAT_ND, nullptr);

    const aclTensor* inputTensor[2] = {self.get(), other.get()};
    aclTensorList* inputList = aclCreateTensorList(inputTensor, 2);

    auto input = OP_INPUT(inputList);
    auto output = OP_OUTPUT(out.get());
    auto attr = OP_ATTR(alpha);
    auto ws = OP_WORKSPACE(out.get());
    auto mode = OP_MODE(static_cast<uint32_t>(op::OpExecMode::OP_EXEC_MODE_HF32));
    auto ctx = op::MakeOpArgContext(input, output, attr, ws, mode);

    auto taskType = kernelBin.GetTaskInfo(0, ctx);
    EXPECT_EQ(taskType.execMode, op::OpExecMode::OP_EXEC_MODE_HF32);

    auto ctx1 = op::MakeOpArgContext(input, output, attr, ws);
    taskType = kernelBin.GetTaskInfo(0, ctx1);
    EXPECT_EQ(taskType.execMode, op::OpExecMode::OP_EXEC_MODE_DEFAULT);

    auto mode2 = OP_MODE(static_cast<uint32_t>(op::OpExecMode::OP_EXEC_MODE_HF32),
                         static_cast<uint32_t>(op::OpExecMode::OP_EXEC_MODE_HF32));
    auto ctx2 = op::MakeOpArgContext(input, output, attr, ws, mode2);
    taskType = kernelBin.GetTaskInfo(0, ctx2);
    EXPECT_EQ(taskType.execMode, op::OpExecMode::OP_EXEC_MODE_DEFAULT);

    uint64_t invalidMode = 1;
    auto mode3 = OP_MODE(invalidMode);
    auto ctx3 = op::MakeOpArgContext(input, output, attr, ws, mode3);
    taskType = kernelBin.GetTaskInfo(0, ctx3);
    EXPECT_EQ(taskType.execMode, op::OpExecMode::OP_EXEC_MODE_DEFAULT);

    nlohmann::json opJson;
    opJson["coreType"] = 1;
    op::internal::TaskInfo info;
    kernelBin.GetTaskRationForSingleBinMutilKernel(info, opJson, 0, MSPROF_GE_TASK_TYPE_AI_CORE);
    EXPECT_EQ(info.type, MSPROF_GE_TASK_TYPE_AI_CORE);
    EXPECT_EQ(info.ration, 0);

    delete inputList;
}

TEST_F(OpKernelUT, ParseKernelDfxConfigTest)
{
    uint32_t opType = op::OpTypeDict::ToOpType("QuantBatchMatmulV3");
    const char* p = std::getenv("ASCEND_OPP_PATH");
    EXPECT_NE(p, nullptr);
    KeyAndDetail key;
    key.key = "hahaha";
    size_t hashKey = 123;
    char jsonPath[1024];
    char binPath[1024];
    snprintf_s(jsonPath, sizeof(jsonPath), sizeof(jsonPath),
               "%s/built-in/op_impl/ai_core/tbe/kernel/ascend910/quant_batch_matmul_v3/"
               "QuantBatchMatmulV3_ND_ND_int8_int8_bf16_high_performance.json",
               p);
    snprintf_s(
        binPath, sizeof(binPath), sizeof(binPath),
        "%s/built-in/op_impl/ai_core/tbe/kernel/ascend910/add/Add_41dadce325b0f810d03359af2a38990b_high_performance.o",
        p);

    OpKernelBin kernelBin(opType, jsonPath, jsonPath, binPath, key, hashKey, BinType::DYNAMIC_BIN, false, false);
    nlohmann::json opJson;
    opJson["debugOptions"] = "assert,printf";
    opJson["debugBufSize"] = 75 * 1024;
    kernelBin.ParseKernelDfxConfig(opJson);
    EXPECT_EQ(kernelBin.IsAssertEnable(), true);
    EXPECT_EQ(kernelBin.IsPrintFEnable(), true);
    EXPECT_EQ(kernelBin.GetKernelDfxBufSize(), 75 * 1024);

    kernelBin.kernelDfxType_ = 0;
    kernelBin.kernelDfxBufSize_ = 0;

    nlohmann::json opJson1;
    opJson1["debugOptions"] = "assert,printf";
    kernelBin.ParseKernelDfxConfig(opJson1);
    EXPECT_EQ(kernelBin.IsAssertEnable(), false);
    EXPECT_EQ(kernelBin.IsPrintFEnable(), false);
    EXPECT_EQ(kernelBin.GetKernelDfxBufSize(), 0);

    kernelBin.kernelDfxType_ = 0;
    kernelBin.kernelDfxBufSize_ = 0;

    nlohmann::json opJson2;
    opJson2["debugOptions"] = "printf";
    opJson2["debugBufSize"] = 75;
    kernelBin.ParseKernelDfxConfig(opJson2);
    EXPECT_EQ(kernelBin.IsAssertEnable(), false);
    EXPECT_EQ(kernelBin.IsPrintFEnable(), true);
    EXPECT_EQ(kernelBin.GetKernelDfxBufSize(), 75);

    kernelBin.kernelDfxType_ = 0;
    kernelBin.kernelDfxBufSize_ = 0;

    nlohmann::json opJson3;
    kernelBin.ParseKernelDfxConfig(opJson3);
    EXPECT_EQ(kernelBin.IsAssertEnable(), false);
    EXPECT_EQ(kernelBin.IsPrintFEnable(), false);
    EXPECT_EQ(kernelBin.GetKernelDfxBufSize(), 0);
}

TEST_F(OpKernelUT, ParseStaticKernelDevPtrTest)
{
    uint32_t opType = op::OpTypeDict::ToOpType("QuantBatchMatmulV3");
    const char* p = std::getenv("ASCEND_OPP_PATH");
    EXPECT_NE(p, nullptr);
    KeyAndDetail key;
    key.key = "hahaha";
    size_t hashKey = 123;
    char jsonPath[1024];
    char binPath[1024];
    snprintf_s(jsonPath, sizeof(jsonPath), sizeof(jsonPath),
               "%s/built-in/op_impl/ai_core/tbe/kernel/ascend910/quant_batch_matmul_v3/"
               "QuantBatchMatmulV3_ND_ND_int8_int8_bf16_high_performance.json",
               p);
    snprintf_s(
        binPath, sizeof(binPath), sizeof(binPath),
        "%s/built-in/op_impl/ai_core/tbe/kernel/ascend910/add/Add_41dadce325b0f810d03359af2a38990b_high_performance.o",
        p);

    OpKernelBin kernelBin(opType, jsonPath, jsonPath, binPath, key, hashKey, BinType::DYNAMIC_BIN, false, false);
    nlohmann::json opJson;
    opJson["dynamicParamMode"] = "folded_with_desc";
    kernelBin.ParseStaticDevPtrMode(opJson);
    EXPECT_EQ(kernelBin.GetHasDevPtrArg(), true);
}

class PrintFDumpStub : public Adx::DumpStub {
public:
    void AdumpPrintWorkSpace(const void* workSpaceAddr, const size_t dumpWorkSpaceSize, aclrtStream stream,
                             const char* opType)
    {
        EXPECT_NE(workSpaceAddr, nullptr);
        EXPECT_EQ(dumpWorkSpaceSize, 75 * 1024);
        EXPECT_EQ(stream, nullptr);
        std::string str(opType);
        EXPECT_EQ(str, "QuantBatchMatmulV3");
        return;
    }
};

OP_TYPE_REGISTER(QuantBatchMatmulV3)

TEST_F(OpKernelUT, DumpWorkspaceData)
{
    QuantBatchMatmulV3OpTypeId();
    PrintFDumpStub dumpStub;
    Adx::DumpStub::GetInstance()->Install(&dumpStub);
    uint32_t opType = op::OpTypeDict::ToOpType("QuantBatchMatmulV3");
    const char* p = std::getenv("ASCEND_OPP_PATH");
    EXPECT_NE(p, nullptr);
    KeyAndDetail key;
    key.key = "hahaha";
    size_t hashKey = 123;
    char jsonPath[1024];
    char binPath[1024];
    snprintf_s(jsonPath, sizeof(jsonPath), sizeof(jsonPath),
               "%s/built-in/op_impl/ai_core/tbe/kernel/ascend910/quant_batch_matmul_v3/"
               "QuantBatchMatmulV3_ND_ND_int8_int8_bf16_high_performance.json",
               p);
    snprintf_s(
        binPath, sizeof(binPath), sizeof(binPath),
        "%s/built-in/op_impl/ai_core/tbe/kernel/ascend910/add/Add_41dadce325b0f810d03359af2a38990b_high_performance.o",
        p);

    OpKernelBin kernelBin(opType, jsonPath, jsonPath, binPath, key, hashKey, BinType::DYNAMIC_BIN, false, false);
    nlohmann::json opJson;
    opJson["debugOptions"] = "assert,printf";
    opJson["debugBufSize"] = 75 * 1024;
    kernelBin.ParseKernelDfxConfig(opJson);
    EXPECT_EQ(kernelBin.IsAssertEnable(), true);
    EXPECT_EQ(kernelBin.IsPrintFEnable(), true);
    EXPECT_EQ(kernelBin.GetKernelDfxBufSize(), 75 * 1024);

    op::Shape selfShape{33, 15, 64};
    op::Shape outShape{33, 15, 64};
    op::Shape idxShape{33, 15, 64};
    int alpha = 1337;

    int* ptr = new int;
    void* addr = reinterpret_cast<void*>(ptr);

    auto self = std::make_unique<aclTensor>(selfShape, op::DataType::DT_INT8, op::Format::FORMAT_ND, addr);
    auto other = std::make_unique<aclTensor>(outShape, op::DataType::DT_INT8, op::Format::FORMAT_ND, addr);
    auto out = std::make_unique<aclTensor>(idxShape, op::DataType::DT_INT8, op::Format::FORMAT_ND, addr);

    const aclTensor* inputTensor[2] = {self.get(), other.get()};
    aclTensorList* inputList = aclCreateTensorList(inputTensor, 2);

    auto input = OP_INPUT(inputList);
    auto output = OP_OUTPUT(out.get());
    auto attr = OP_ATTR(alpha);
    auto mode = OP_MODE(static_cast<uint32_t>(op::OpExecMode::OP_EXEC_MODE_HF32));
    auto ctx = op::MakeOpArgContext(input, output, attr, mode);
    ctx->AppendOpWorkspaceArg(inputList);

    kernelBin.DumpWorkspaceData(nullptr, ctx);
    kernelBin.DumpWorkspaceData(nullptr, nullptr);
    Adx::DumpStub::GetInstance()->UnInstall();
    delete ptr;
    delete inputList;
}

class AssertExceptionStub : public Adx::DumpStub {
public:
    void* AdumpGetSizeInfoAddr(uint32_t space, uint32_t& atomicIndex)
    {
        space_ = space;
        atomicIndex = 1;
        return (void*)exceptionMem_;
    }

    void CheckResult()
    {
        EXPECT_EQ(space_, 7);
        EXPECT_EQ(exceptionMem_[0], 1);
        EXPECT_EQ(exceptionMem_[1], 4294967301);
        EXPECT_EQ(exceptionMem_[2], 31680);
        EXPECT_EQ(exceptionMem_[3], 31680);
        EXPECT_EQ(exceptionMem_[4], 31680);
        EXPECT_EQ(exceptionMem_[5], 288230376151743424);
        EXPECT_EQ(exceptionMem_[6], 31680);
    }

    uint32_t space_{0};
    int64_t exceptionMem_[1000] = {0};
};

TEST_F(OpKernelUT, AssertExceptionDump)
{
    AssertExceptionStub dumpStub;
    Adx::DumpStub::GetInstance()->Install(&dumpStub);

    op::Shape selfShape{33, 15, 64};
    op::Shape outShape{33, 15, 64};
    op::Shape idxShape{33, 15, 64};
    int alpha = 1337;

    int* ptr = new int;
    void* addr = reinterpret_cast<void*>(ptr);

    auto self = std::make_unique<aclTensor>(selfShape, op::DataType::DT_INT8, op::Format::FORMAT_ND, addr);
    auto other = std::make_unique<aclTensor>(outShape, op::DataType::DT_INT8, op::Format::FORMAT_ND, addr);
    auto out = std::make_unique<aclTensor>(idxShape, op::DataType::DT_INT8, op::Format::FORMAT_ND, addr);

    const aclTensor* inputTensor[2] = {self.get(), other.get()};
    aclTensorList* inputList = aclCreateTensorList(inputTensor, 2);

    auto input = OP_INPUT(inputList);
    auto output = OP_OUTPUT(out.get());
    auto attr = OP_ATTR(alpha);
    auto mode = OP_MODE(static_cast<uint32_t>(op::OpExecMode::OP_EXEC_MODE_HF32));
    auto ctx = op::MakeOpArgContext(input, output, attr, mode);
    ctx->AppendOpWorkspaceArg(inputList);

    op::internal::ExpandableRtsArgBuffer buffer;
    buffer.Init(TEST_LAUNCH_ARG_INIT_CAP, TEST_TILING_HOST_DATA_INIT_CAP);
    op::internal::TilingData* tilingData = buffer.GetTilingDataPtr();
    constexpr size_t TEST_TILING_DATA_LEN = 32;
    tilingData->data_ = buffer.GetTilingDataAddr();
    memset_s(tilingData->data_, TEST_TILING_HOST_DATA_INIT_CAP, 0, TEST_TILING_DATA_LEN);
    tilingData->data_size_ = TEST_TILING_DATA_LEN;
    tilingData->capacity_ = TEST_TILING_HOST_DATA_INIT_CAP;

    op::internal::LaunchArgInfo argInfo(false, false, ctx);
    op::internal::RtsArg arg(true, argInfo, &buffer);
    arg.FillArgs(true);
    dumpStub.CheckResult();
    Adx::DumpStub::GetInstance()->UnInstall();
    delete ptr;
    delete inputList;
}

TEST_F(OpKernelUT, ParseImplModeByJsonTest)
{
    nlohmann::json jsonObj1 = {{"name", "John"},
                               {"age", 30},
                               {"address", {{"street", "123 Walnut St"}, {"city", "New York"}, {"zip", 10001}}}};
    op::FVector<op::OpImplMode> implModes;
    op::internal::ParseImplModeByJson(jsonObj1, "xxx.json", implModes);
    EXPECT_EQ(implModes[0], op::OpImplMode::IMPL_MODE_HIGH_PRECISION);

    nlohmann::json jsonObj2 = {{"implMode", "John"},
                               {"age", 30},
                               {"address", {{"street", "123 Walnut St"}, {"city", "New York"}, {"zip", 10001}}}};
    ParseImplModeByJson(jsonObj2, "xxx.json", implModes);
    EXPECT_EQ(implModes[1], op::OpImplMode::IMPL_MODE_HIGH_PRECISION);
}

TEST_F(OpKernelUT, testGetConfigImplPath)
{
    setenv("ASCEND_OPP_PATH", "./fake_path/", 1);
    OpKernelLib opKnlLib;
    const std::vector<std::string>& implPath = opKnlLib.GetConfigImplPath();
    EXPECT_EQ(implPath.size(), 0);
    setenv("ASCEND_OPP_PATH", OP_API_COMMON_UT_SRC_DIR, 1);
}

TEST_F(OpKernelUT, Initialize_failed)
{
    setenv("ASCEND_OPP_PATH", "./", 1);
    namespace fs = std::filesystem;
    fs::create_directories("built-in/op_impl/ai_core/tbe/config/ascend910/");
    nlohmann::json jsonObj = {"]"};
    std::ofstream jsonfile("./built-in/op_impl/ai_core/tbe/config/ascend910/aic-ascend910b-ops-info1.json.json");
    jsonfile << jsonObj.dump(4);
    OpKernelLib opKnlLib;
    aclnnStatus ret = opKnlLib.Initialize();
    EXPECT_NE(ret, 0);
    setenv("ASCEND_OPP_PATH", OP_API_COMMON_UT_SRC_DIR, 1);
}

TEST_F(OpKernelUT, Initialize_Custom)
{
    const std::string oppPath = "./custom_opp";
    setenv("ASCEND_OPP_PATH", oppPath.c_str(), 1);
    namespace fs = std::filesystem;
    std::string pathVendors = oppPath + "/vendors";
    std::string pathConfig = pathVendors + "/config.ini";
    system(("mkdir -p " + pathVendors).c_str());
    system(("echo 'load_priority=custom_nn_one,custom_nn_two' > " + pathConfig).c_str());

    fs::create_directories(oppPath + "/vendors/custom_nn_one/op_impl/ai_core/tbe/config/ascend910/");
    std::ofstream jsonfileOne(
        oppPath + "/vendors/custom_nn_one/op_impl/ai_core/tbe/config/ascend910/aic-ascend910b-ops-info.json");
    std::string jsonObjOne = "{\"AvgPool2D\": {\"opFile\": {\"value\": \"avg_pool2_d\"}}}";
    jsonfileOne << jsonObjOne;
    jsonfileOne.close();

    fs::create_directories(oppPath + "/vendors/custom_nn_two/op_impl/ai_core/tbe/config/ascend910/");
    std::ofstream jsonfileTwo(
        oppPath + "/vendors/custom_nn_two/op_impl/ai_core/tbe/config/ascend910/aic-ascend910b-ops-info.json");
    std::string jsonObjTwo = "{\"AvgPool3D\": {\"opFile\": {\"value\": \"avg_pool3_d\"}}}";
    jsonfileTwo << jsonObjTwo;
    jsonfileTwo.close();

    OpKernelLib opKnlLib;
    uint32_t opType = 1;
    aclnnStatus resultOfInit = opKnlLib.Initialize();
    EXPECT_EQ(resultOfInit, 0);

    fs::remove_all(oppPath + "/vendors/custom_nn_one/op_impl/ai_core/tbe/config/ascend910/");
    fs::remove_all(oppPath + "/vendors/custom_nn_two/op_impl/ai_core/tbe/config/ascend910/");
    setenv("ASCEND_OPP_PATH", OP_API_COMMON_UT_SRC_DIR, 1);
}

static void WriteOpInfoJson(const std::string& dir, const std::string& fileName, const std::string& opKey,
                            const std::string& opFileValue)
{
    namespace fs = std::filesystem;
    fs::create_directories(dir);
    std::ofstream f(dir + "/" + fileName);
    f << "{\"" << opKey << "\": {\"opFile\": {\"value\": \"" << opFileValue << "\"}}}";
}

static const std::string UT_SOC_DIR = "ascend910/";

TEST_F(OpKernelUT, Initialize_MergePriority_CustomWinsOverBuiltin)
{
    const std::string oppPath = "./test_merge_custom_vs_builtin";
    setenv("ASCEND_OPP_PATH", oppPath.c_str(), 1);
    namespace fs = std::filesystem;

    system(("mkdir -p " + oppPath + "/vendors").c_str());
    system(("echo 'priority=custom_vnd' > " + oppPath + "/vendors/config.ini").c_str());

    WriteOpInfoJson(oppPath + "/vendors/custom_vnd/op_impl/ai_core/tbe/config/" + UT_SOC_DIR, "custom.json",
                    "AvgPool2D", "from_custom");
    WriteOpInfoJson(oppPath + "/built-in/op_impl/ai_core/tbe/config/" + UT_SOC_DIR, "builtin.json", "AvgPool2D",
                    "from_builtin");

    OpKernelLib opKnlLib;
    ASSERT_EQ(opKnlLib.Initialize(), 0);
    EXPECT_EQ(opKnlLib.allKernelsJson_["AvgPool2D"]["opFile"]["value"].get<std::string>(), "from_custom");

    fs::remove_all(oppPath);
    setenv("ASCEND_OPP_PATH", OP_API_COMMON_UT_SRC_DIR, 1);
}

TEST_F(OpKernelUT, Initialize_MergePriority_FirstVendorWins)
{
    const std::string oppPath = "./test_merge_vendor_order";
    setenv("ASCEND_OPP_PATH", oppPath.c_str(), 1);
    namespace fs = std::filesystem;

    system(("mkdir -p " + oppPath + "/vendors").c_str());
    system(("echo 'priority=vnd_a,vnd_b' > " + oppPath + "/vendors/config.ini").c_str());

    WriteOpInfoJson(oppPath + "/vendors/vnd_a/op_impl/ai_core/tbe/config/" + UT_SOC_DIR, "a.json", "AvgPool2D",
                    "from_vnd_a");
    WriteOpInfoJson(oppPath + "/vendors/vnd_b/op_impl/ai_core/tbe/config/" + UT_SOC_DIR, "b.json", "AvgPool2D",
                    "from_vnd_b");

    OpKernelLib opKnlLib;
    ASSERT_EQ(opKnlLib.Initialize(), 0);
    EXPECT_EQ(opKnlLib.allKernelsJson_["AvgPool2D"]["opFile"]["value"].get<std::string>(), "from_vnd_a");

    fs::remove_all(oppPath);
    setenv("ASCEND_OPP_PATH", OP_API_COMMON_UT_SRC_DIR, 1);
}

TEST_F(OpKernelUT, Initialize_MergePriority_NonOverlappingKeys)
{
    const std::string oppPath = "./test_merge_nonoverlap";
    setenv("ASCEND_OPP_PATH", oppPath.c_str(), 1);
    namespace fs = std::filesystem;

    system(("mkdir -p " + oppPath + "/vendors").c_str());
    system(("echo 'priority=custom_vnd' > " + oppPath + "/vendors/config.ini").c_str());

    WriteOpInfoJson(oppPath + "/vendors/custom_vnd/op_impl/ai_core/tbe/config/" + UT_SOC_DIR, "custom.json",
                    "AvgPool2D", "custom_only");
    WriteOpInfoJson(oppPath + "/built-in/op_impl/ai_core/tbe/config/" + UT_SOC_DIR, "builtin.json", "Add",
                    "builtin_only");

    OpKernelLib opKnlLib;
    ASSERT_EQ(opKnlLib.Initialize(), 0);
    EXPECT_EQ(opKnlLib.allKernelsJson_["AvgPool2D"]["opFile"]["value"].get<std::string>(), "custom_only");
    EXPECT_EQ(opKnlLib.allKernelsJson_["Add"]["opFile"]["value"].get<std::string>(), "builtin_only");

    fs::remove_all(oppPath);
    setenv("ASCEND_OPP_PATH", OP_API_COMMON_UT_SRC_DIR, 1);
}

TEST_F(OpKernelUT, testGetConfigImplPath2)
{
    const std::string validVendorsEnv = std::string(OP_API_COMMON_UT_SRC_DIR) + "/valid_vendors/";
    setenv("ASCEND_OPP_PATH", validVendorsEnv.c_str(), 1);
    OpKernelLib opKnlLib;
    const std::vector<std::string>& implPath = opKnlLib.GetConfigImplPath();
    EXPECT_EQ(implPath.size(), 1);
    const std::string expectImplPath = validVendorsEnv + "vendors/test_path/op_impl/ai_core/tbe";
    EXPECT_EQ(implPath[0], expectImplPath);
    setenv("ASCEND_OPP_PATH", OP_API_COMMON_UT_SRC_DIR, 1);
}

TEST_F(OpKernelUT, TestGetSocVersion)
{
    OpKernelLib opKnlLib;
    EXPECT_STREQ(opKnlLib.GetSocPath().c_str(), "ascend910/");

    setenv("ASCEND_C", "1", 1);
    OpKernelLib opKnlLib1;
    EXPECT_STREQ(opKnlLib1.GetSocPath().c_str(), "ascend910b/");

    setenv("ASCEND_C", "2", 1);
    OpKernelLib opKnlLib3;
    EXPECT_STREQ(opKnlLib3.GetSocPath().c_str(), "ascend950/");

    setenv("ASCEND_C", "3", 1);
    OpKernelLib opKnlLib4;
    EXPECT_STREQ(opKnlLib4.GetSocPath().c_str(), "ascend310p/");

    setenv("ASCEND_C", "999", 1);
    OpKernelLib opKnlLib5;
    EXPECT_STREQ(opKnlLib5.GetSocPath().c_str(), "");

    unsetenv("ASCEND_C");
}

TEST_F(OpKernelUT, TestTilingOOMInfoViewOffsetUsesStorageSizeFromAscend950)
{
    ViewOffsetTilingSizeResult ascend910bResult = CollectViewOffsetTensorSizeInNewThread("2201");
    ViewOffsetTilingSizeResult ascend950Result = CollectViewOffsetTensorSizeInNewThread("3510");
    ViewOffsetTilingSizeResult laterArchResult = CollectViewOffsetTensorSizeInNewThread("3801");
    ViewOffsetTilingSizeResult reservedArchResult = CollectViewOffsetTensorSizeInNewThread(nullptr);

    ASSERT_TRUE(ascend910bResult.ok);
    ASSERT_TRUE(ascend950Result.ok);
    ASSERT_TRUE(laterArchResult.ok);
    ASSERT_TRUE(reservedArchResult.ok);
    EXPECT_EQ(ascend910bResult.tensorSize, 64);
    EXPECT_EQ(ascend950Result.tensorSize, 34);
    EXPECT_EQ(laterArchResult.tensorSize, 34);
    EXPECT_EQ(reservedArchResult.tensorSize, 64);
}

TEST_F(OpKernelUT, TestTilingOOMInfo1)
{
    bool genPlaceholder = false, hasDevPtrArg = false;
    auto kernelBin = CreateFakeOpKernelBin(genPlaceholder, hasDevPtrArg);
    aclnnStatus rec = kernelBin->JsonLoad();
    EXPECT_EQ(rec, ACLNN_SUCCESS);

    // create shape
    op::Shape selfShape{33, 15, 64};

    // input
    int* ptr1 = new int;
    aclTensor tensor1(selfShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND, ptr1);
    tensor1.SetFromWorkspace(false);
    aclTensor* input1 = &tensor1;

    int* ptr2 = new int;
    auto tensor2Ptr = std::make_unique<aclTensor>(selfShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND, ptr2);
    aclTensor* tensor2 = tensor2Ptr.release();
    tensor2->SetFromWorkspace(false);
    int* ptr3 = new int;
    auto tensor3Ptr = std::make_unique<aclTensor>(selfShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND, ptr3);
    aclTensor* tensor3 = tensor3Ptr.release();
    tensor3->SetFromWorkspace(false);
    aclTensor* tensor4 = nullptr;
    const aclTensor* inputTensors[3] = {tensor2, tensor3, tensor4};
    aclTensorList* input2 = aclCreateTensorList(inputTensors, 3);

    aclTensor* input3 = tensor4;
    aclTensorList* input4 = nullptr;

    std::vector<int64_t> hostValue(10, 1);
    aclIntArray intArray(hostValue.data(), hostValue.size());
    aclTensor hostTensor(&intArray, op::DataType::DT_INT64);
    aclTensor* input5 = &hostTensor;

    float fpValue = 3.2;
    aclScalar fpScalar(&fpValue, op::DataType::DT_FLOAT);
    aclTensor scalarTensor(&fpScalar, op::DataType::DT_FLOAT);
    aclTensor* input6 = &scalarTensor;

    // output
    aclTensor* output1 = input1;
    aclTensorList* output2 = input2;
    aclTensor* output3 = input3;
    aclTensorList* output4 = input4;

    // workspace
    int* workspacePtr = new int;
    auto tensor5 = std::make_unique<aclTensor>(selfShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND,
                                               workspacePtr);
    aclTensor* tensor5Ptr = tensor5.release();
    tensor5Ptr->SetFromWorkspace(true);
    tensor5Ptr->SetWorkspaceOffset(0);
    auto tensor6 = std::make_unique<aclTensor>(selfShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND,
                                               workspacePtr);
    aclTensor* tensor6Ptr = tensor6.release();
    tensor6Ptr->SetFromWorkspace(true);
    tensor6Ptr->SetWorkspaceOffset(512);
    const aclTensor* workspaceTensors[2] = {tensor5Ptr, tensor6Ptr};
    aclTensorList* workspace1 = aclCreateTensorList(workspaceTensors, 2);

    // outshape
    aclTensor* outshapeTensor = input1;

    auto input = OP_INPUT(input1, input2, input3, input4, input5, input6);
    auto output = OP_OUTPUT(output1, output2, output3, output4);
    auto outshape = OP_OUTSHAPE(outshapeTensor, 0);

    auto ctx = op::MakeOpArgContext(input, output, outshape);
    ctx->AppendOpWorkspaceArg(workspace1);

    // init tiling data
    constexpr size_t TEST_TILING_HOST_DATA_INIT_SMALL_CAP = 180;
    op::internal::ExpandableRtsArgBuffer buffer;
    buffer.Init(TEST_LAUNCH_ARG_INIT_CAP, TEST_TILING_HOST_DATA_INIT_SMALL_CAP);
    int64_t* tilingDataData = static_cast<int64_t*>(buffer.GetTilingDataAddr());
    for (size_t i = 0; i < 21; i++) {
        tilingDataData[i] = i;
    }
    op::internal::TilingData* tilingData = buffer.GetTilingDataPtr();
    tilingData->data_ = buffer.GetTilingDataAddr();
    tilingData->data_size_ = sizeof(size_t) * 21;
    tilingData->capacity_ = TEST_TILING_HOST_DATA_INIT_SMALL_CAP;

    kernelBin->AppendTilingOOMInfo(&buffer, ctx);
    tilingData = buffer.GetTilingDataPtr();
    EXPECT_EQ(tilingData->data_size_, 8 * (21 + 11));
    EXPECT_EQ(tilingData->capacity_, TEST_TILING_HOST_DATA_INIT_SMALL_CAP * 2);

    int64_t* tilingVal = reinterpret_cast<int64_t*>(buffer.GetTilingDataAddr());
    for (int64_t i = 0; i < 21; i++) {
        EXPECT_EQ(tilingVal[i], i);
    }
    EXPECT_EQ(tilingVal[21], 63360);
    EXPECT_EQ(tilingVal[22], 63360);
    EXPECT_EQ(tilingVal[23], 63360);
    EXPECT_EQ(tilingVal[24], 96);
    EXPECT_EQ(tilingVal[25], 32);
    EXPECT_EQ(tilingVal[26], 63360);
    EXPECT_EQ(tilingVal[27], 63360);
    EXPECT_EQ(tilingVal[28], 63360);
    EXPECT_EQ(tilingVal[29], 63360);
    EXPECT_EQ(tilingVal[30], 63360);
    EXPECT_EQ(tilingVal[31], 63360);

    // tear down
    op::DestroyOpArgContext(ctx);
    delete ptr1;
    delete ptr2;
    delete ptr3;
    delete workspacePtr;
    aclDestroyTensorList(input2);
    aclDestroyTensorList(workspace1);
}

TEST_F(OpKernelUT, TestTilingOOMInfo2)
{
    bool genPlaceholder = true, hasDevPtrArg = false;
    auto kernelBin = CreateFakeOpKernelBin(genPlaceholder, hasDevPtrArg);
    aclnnStatus rec = kernelBin->JsonLoad();
    EXPECT_EQ(rec, ACLNN_SUCCESS);

    // create shape
    op::Shape selfShape{33, 15, 64};

    // input
    int* ptr1 = new int;
    aclTensor tensor1(selfShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND, ptr1);
    tensor1.SetFromWorkspace(false);
    aclTensor* input1 = &tensor1;

    int* ptr2 = new int;
    auto tensor2Ptr = std::make_unique<aclTensor>(selfShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND, ptr2);
    aclTensor* tensor2 = tensor2Ptr.release();
    tensor2->SetFromWorkspace(false);
    int* ptr3 = new int;
    auto tensor3Ptr = std::make_unique<aclTensor>(selfShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND, ptr3);
    aclTensor* tensor3 = tensor3Ptr.release();
    tensor3->SetFromWorkspace(false);
    aclTensor* tensor4 = nullptr;
    const aclTensor* inputTensors[3] = {tensor2, tensor3, tensor4};
    aclTensorList* input2 = aclCreateTensorList(inputTensors, 3);

    aclTensor* input3 = tensor4;
    aclTensorList* input4 = nullptr;

    std::vector<int64_t> hostValue(10, 1);
    aclIntArray intArray(hostValue.data(), hostValue.size());
    aclTensor hostTensor(&intArray, op::DataType::DT_INT64);
    aclTensor* input5 = &hostTensor;

    float fpValue = 3.2;
    aclScalar fpScalar(&fpValue, op::DataType::DT_FLOAT);
    aclTensor scalarTensor(&fpScalar, op::DataType::DT_FLOAT);
    aclTensor* input6 = &scalarTensor;

    // output
    aclTensor* output1 = input1;
    aclTensorList* output2 = input2;
    aclTensor* output3 = input3;
    aclTensorList* output4 = input4;

    // workspace
    int* workspacePtr = new int;
    auto tensor5 = std::make_unique<aclTensor>(selfShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND,
                                               workspacePtr);
    aclTensor* tensor5Ptr = tensor5.release();
    tensor5Ptr->SetFromWorkspace(true);
    tensor5Ptr->SetWorkspaceOffset(0);
    auto tensor6 = std::make_unique<aclTensor>(selfShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND,
                                               workspacePtr);
    aclTensor* tensor6Ptr = tensor6.release();
    tensor6Ptr->SetFromWorkspace(true);
    tensor6Ptr->SetWorkspaceOffset(512);
    const aclTensor* workspaceTensors[2] = {tensor5Ptr, tensor6Ptr};
    aclTensorList* workspace1 = aclCreateTensorList(workspaceTensors, 2);

    // outshape
    aclTensor* outshapeTensor = input1;

    auto input = OP_INPUT(input1, input2, input3, input4, input5, input6);
    auto output = OP_OUTPUT(output1, output2, output3, output4);
    auto outshape = OP_OUTSHAPE(outshapeTensor, 0);

    auto ctx = op::MakeOpArgContext(input, output, outshape);
    ctx->AppendOpWorkspaceArg(workspace1);

    // init tiling data
    op::internal::ExpandableRtsArgBuffer buffer;
    buffer.Init(TEST_LAUNCH_ARG_INIT_CAP, TEST_TILING_HOST_DATA_INIT_CAP);
    int64_t* tilingDataData = static_cast<int64_t*>(buffer.GetTilingDataAddr());
    for (size_t i = 0; i < 21; i++) {
        tilingDataData[i] = i;
    }
    op::internal::TilingData* tilingData = buffer.GetTilingDataPtr();
    tilingData->data_ = buffer.GetTilingDataAddr();
    tilingData->data_size_ = sizeof(size_t) * 21;
    tilingData->capacity_ = TEST_TILING_HOST_DATA_INIT_CAP;

    kernelBin->AppendTilingOOMInfo(&buffer, ctx);
    EXPECT_EQ(buffer.GetTilingDataPtr()->data_size_, 8 * (21 + 13));

    int64_t* tilingVal = reinterpret_cast<int64_t*>(buffer.GetTilingDataAddr());
    for (int64_t i = 0; i < 21; i++) {
        EXPECT_EQ(tilingVal[i], i);
    }
    EXPECT_EQ(tilingVal[21], 63360);
    EXPECT_EQ(tilingVal[22], 63360);
    EXPECT_EQ(tilingVal[23], 63360);
    EXPECT_EQ(tilingVal[24], 0);
    EXPECT_EQ(tilingVal[25], 0);
    EXPECT_EQ(tilingVal[26], 96);
    EXPECT_EQ(tilingVal[27], 32);
    EXPECT_EQ(tilingVal[28], 63360);
    EXPECT_EQ(tilingVal[29], 63360);
    EXPECT_EQ(tilingVal[30], 63360);
    EXPECT_EQ(tilingVal[31], 63360);
    EXPECT_EQ(tilingVal[32], 63360);
    EXPECT_EQ(tilingVal[33], 63360);

    // tear down
    op::DestroyOpArgContext(ctx);
    delete ptr1;
    delete ptr2;
    delete ptr3;
    delete workspacePtr;
    aclDestroyTensorList(input2);
    aclDestroyTensorList(workspace1);
}

TEST_F(OpKernelUT, TestTilingOOMInfo3)
{
    bool genPlaceholder = false, hasDevPtrArg = true;
    auto kernelBin = CreateFakeOpKernelBin(genPlaceholder, hasDevPtrArg);
    aclnnStatus rec = kernelBin->JsonLoad();
    EXPECT_EQ(rec, ACLNN_SUCCESS);

    // create shape
    op::Shape selfShape{33, 15, 64};

    // input
    int* ptr1 = new int;
    aclTensor tensor1(selfShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND, ptr1);
    tensor1.SetFromWorkspace(false);
    aclTensor* input1 = &tensor1;

    int* ptr2 = new int;
    auto tensor2Ptr = std::make_unique<aclTensor>(selfShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND, ptr2);
    aclTensor* tensor2 = tensor2Ptr.release();
    tensor2->SetFromWorkspace(false);
    int* ptr3 = new int;
    auto tensor3Ptr = std::make_unique<aclTensor>(selfShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND, ptr3);
    aclTensor* tensor3 = tensor3Ptr.release();
    tensor3->SetFromWorkspace(false);
    aclTensor* tensor4 = nullptr;
    const aclTensor* inputTensors[3] = {tensor2, tensor3, tensor4};
    aclTensorList* input2 = aclCreateTensorList(inputTensors, 3);

    aclTensor* input3 = tensor4;
    aclTensorList* input4 = nullptr;

    std::vector<int64_t> hostValue(10, 1);
    aclIntArray intArray(hostValue.data(), hostValue.size());
    aclTensor hostTensor(&intArray, op::DataType::DT_INT64);
    aclTensor* input5 = &hostTensor;

    float fpValue = 3.2;
    aclScalar fpScalar(&fpValue, op::DataType::DT_FLOAT);
    aclTensor scalarTensor(&fpScalar, op::DataType::DT_FLOAT);
    aclTensor* input6 = &scalarTensor;

    // output
    aclTensor* output1 = input1;
    aclTensorList* output2 = input2;
    aclTensor* output3 = input3;
    aclTensorList* output4 = input4;

    // workspace
    int* workspacePtr = new int;
    auto tensor5 = std::make_unique<aclTensor>(selfShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND,
                                               workspacePtr);
    aclTensor* tensor5Ptr = tensor5.release();
    tensor5Ptr->SetFromWorkspace(true);
    tensor5Ptr->SetWorkspaceOffset(0);
    auto tensor6 = std::make_unique<aclTensor>(selfShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND,
                                               workspacePtr);
    aclTensor* tensor6Ptr = tensor6.release();
    tensor6Ptr->SetFromWorkspace(true);
    tensor6Ptr->SetWorkspaceOffset(512);
    const aclTensor* workspaceTensors[2] = {tensor5Ptr, tensor6Ptr};
    aclTensorList* workspace1 = aclCreateTensorList(workspaceTensors, 2);

    // outshape
    aclTensor* outshapeTensor = input1;

    auto input = OP_INPUT(input1, input2, input3, input4, input5, input6);
    auto output = OP_OUTPUT(output1, output2, output3, output4);
    auto outshape = OP_OUTSHAPE(outshapeTensor, 0);

    auto ctx = op::MakeOpArgContext(input, output, outshape);
    ctx->AppendOpWorkspaceArg(workspace1);

    // init tiling data
    constexpr size_t TEST_TILING_HOST_DATA_INIT_SMALL_CAP = 180;
    op::internal::ExpandableRtsArgBuffer buffer;
    buffer.Init(TEST_LAUNCH_ARG_INIT_CAP, TEST_TILING_HOST_DATA_INIT_SMALL_CAP);
    int64_t* tilingDataData = static_cast<int64_t*>(buffer.GetTilingDataAddr());
    for (size_t i = 0; i < 21; i++) {
        tilingDataData[i] = i;
    }
    op::internal::TilingData* tilingData = buffer.GetTilingDataPtr();
    tilingData->data_ = buffer.GetTilingDataAddr();
    tilingData->data_size_ = sizeof(size_t) * 21;
    tilingData->capacity_ = TEST_TILING_HOST_DATA_INIT_SMALL_CAP;

    kernelBin->AppendTilingOOMInfo(&buffer, ctx);
    tilingData = buffer.GetTilingDataPtr();
    EXPECT_EQ(tilingData->data_size_, 8 * (21 + 9));
    EXPECT_EQ(tilingData->capacity_, TEST_TILING_HOST_DATA_INIT_SMALL_CAP * 2);
    int64_t* tilingVal = reinterpret_cast<int64_t*>(buffer.GetTilingDataAddr());
    for (int64_t i = 0; i < 21; i++) {
        EXPECT_EQ(tilingVal[i], i);
    }
    EXPECT_EQ(tilingVal[21], 63360);
    EXPECT_EQ(tilingVal[22], 80);
    EXPECT_EQ(tilingVal[23], 96);
    EXPECT_EQ(tilingVal[24], 32);
    EXPECT_EQ(tilingVal[25], 63360);
    EXPECT_EQ(tilingVal[26], 80);
    EXPECT_EQ(tilingVal[27], 63360);
    EXPECT_EQ(tilingVal[28], 63360);
    EXPECT_EQ(tilingVal[29], 63360);

    // tear down
    op::DestroyOpArgContext(ctx);
    delete ptr1;
    delete ptr2;
    delete ptr3;
    delete workspacePtr;
    aclDestroyTensorList(input2);
    aclDestroyTensorList(workspace1);
}

TEST_F(OpKernelUT, TestTilingOOMInfo4)
{
    bool genPlaceholder = true, hasDevPtrArg = true;
    auto kernelBin = CreateFakeOpKernelBin(genPlaceholder, hasDevPtrArg);
    aclnnStatus rec = kernelBin->JsonLoad();
    EXPECT_EQ(rec, ACLNN_SUCCESS);

    // create shape
    op::Shape selfShape{33, 15, 64};

    // input
    int* ptr1 = new int;
    aclTensor tensor1(selfShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND, ptr1);
    tensor1.SetFromWorkspace(false);
    aclTensor* input1 = &tensor1;

    int* ptr2 = new int;
    auto tensor2Ptr = std::make_unique<aclTensor>(selfShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND, ptr2);
    aclTensor* tensor2 = tensor2Ptr.release();
    tensor2->SetFromWorkspace(false);
    int* ptr3 = new int;
    auto tensor3Ptr = std::make_unique<aclTensor>(selfShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND, ptr3);
    aclTensor* tensor3 = tensor3Ptr.release();
    tensor3->SetFromWorkspace(false);
    aclTensor* tensor4 = nullptr;
    const aclTensor* inputTensors[3] = {tensor2, tensor3, tensor4};
    aclTensorList* input2 = aclCreateTensorList(inputTensors, 3);

    aclTensor* input3 = tensor4;
    aclTensorList* input4 = nullptr;

    std::vector<int64_t> hostValue(10, 1);
    aclIntArray intArray(hostValue.data(), hostValue.size());
    aclTensor hostTensor(&intArray, op::DataType::DT_INT64);
    aclTensor* input5 = &hostTensor;

    float fpValue = 3.2;
    aclScalar fpScalar(&fpValue, op::DataType::DT_FLOAT);
    aclTensor scalarTensor(&fpScalar, op::DataType::DT_FLOAT);
    aclTensor* input6 = &scalarTensor;

    // output
    aclTensor* output1 = input1;
    aclTensorList* output2 = input2;
    aclTensor* output3 = input3;
    aclTensorList* output4 = input4;

    // workspace
    int* workspacePtr = new int;
    auto tensor5 = std::make_unique<aclTensor>(selfShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND,
                                               workspacePtr);
    aclTensor* tensor5Ptr = tensor5.release();
    tensor5Ptr->SetFromWorkspace(true);
    tensor5Ptr->SetWorkspaceOffset(0);
    auto tensor6 = std::make_unique<aclTensor>(selfShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND,
                                               workspacePtr);
    aclTensor* tensor6Ptr = tensor6.release();
    tensor6Ptr->SetFromWorkspace(true);
    tensor6Ptr->SetWorkspaceOffset(512);
    const aclTensor* workspaceTensors[2] = {tensor5Ptr, tensor6Ptr};
    aclTensorList* workspace1 = aclCreateTensorList(workspaceTensors, 2);

    // outshape
    aclTensor* outshapeTensor = input1;

    auto input = OP_INPUT(input1, input2, input3, input4, input5, input6);
    auto output = OP_OUTPUT(output1, output2, output3, output4);
    auto outshape = OP_OUTSHAPE(outshapeTensor, 0);

    auto ctx = op::MakeOpArgContext(input, output, outshape);
    ctx->AppendOpWorkspaceArg(workspace1);

    // init tiling data
    op::internal::ExpandableRtsArgBuffer buffer;
    buffer.Init(TEST_LAUNCH_ARG_INIT_CAP, TEST_TILING_HOST_DATA_INIT_CAP);
    int64_t* tilingDataData = static_cast<int64_t*>(buffer.GetTilingDataAddr());
    for (size_t i = 0; i < 21; i++) {
        tilingDataData[i] = i;
    }
    op::internal::TilingData* tilingData = buffer.GetTilingDataPtr();
    tilingData->data_ = buffer.GetTilingDataAddr();
    tilingData->data_size_ = sizeof(size_t) * 21;
    tilingData->capacity_ = TEST_TILING_HOST_DATA_INIT_CAP;

    kernelBin->AppendTilingOOMInfo(&buffer, ctx);
    EXPECT_EQ(buffer.GetTilingDataPtr()->data_size_, 8 * (21 + 10));
    int64_t* tilingVal = reinterpret_cast<int64_t*>(buffer.GetTilingDataAddr());
    for (int64_t i = 0; i < 21; i++) {
        EXPECT_EQ(tilingVal[i], i);
    }
    EXPECT_EQ(tilingVal[21], 63360);
    EXPECT_EQ(tilingVal[22], 80);
    EXPECT_EQ(tilingVal[23], 0);
    EXPECT_EQ(tilingVal[24], 96);
    EXPECT_EQ(tilingVal[25], 32);
    EXPECT_EQ(tilingVal[26], 63360);
    EXPECT_EQ(tilingVal[27], 80);
    EXPECT_EQ(tilingVal[28], 63360);
    EXPECT_EQ(tilingVal[29], 63360);
    EXPECT_EQ(tilingVal[30], 63360);

    // tear down
    op::DestroyOpArgContext(ctx);
    delete ptr1;
    delete ptr2;
    delete ptr3;
    delete workspacePtr;
    aclDestroyTensorList(input2);
    aclDestroyTensorList(workspace1);
}

TEST_F(OpKernelUT, SetExceptionDumpInfoTest)
{
    CreateFakeOpKernelBin(false, false)->SetExceptionDumpInfo(0, 0, nullptr, 0);
}

TEST_F(OpKernelUT, LoadJsonFail)
{
    auto kernelBin = CreateFailOpKernelBin(false, false);
    Json opJson;
    aclnnStatus rec = kernelBin->GetBinJson(opJson);
    EXPECT_NE(rec, ACLNN_SUCCESS);
}

TEST_F(OpKernelUT, TilingBufferEnlarge)
{
    ExpandableRtsArgBuffer buffer;
    buffer.Init(TEST_LAUNCH_ARG_EXPAND_CAP, TEST_TILING_HOST_DATA_EXPAND_CAP);
    int data = 1;
    buffer.AppendTilingHostData(&data, sizeof(int));
    buffer.AppendTilingHostData(&data, sizeof(int));
    EXPECT_EQ(buffer.GetTilingHostDataSize(), 8);
}

// ============== LocalMemorySize Parse Tests ==============

std::unique_ptr<OpKernelBin> CreateStaticOpKernelBin(const std::string& jsonFileName)
{
    uint32_t opType = op::OpTypeDict::ToOpType("Add");
    const char* p = std::getenv("ASCEND_OPP_PATH");
    EXPECT_NE(p, nullptr);
    KeyAndDetail key;
    key.key = "test_kernel";
    size_t hashKey = 123;
    char jsonPath[1024];
    char binPath[1024];
    snprintf_s(jsonPath, sizeof(jsonPath), sizeof(jsonPath), "%s/../mock/static_kernel/test_json/%s", p,
               jsonFileName.c_str());
    snprintf_s(binPath, sizeof(binPath), sizeof(binPath),
               "%s/built-in/op_impl/ai_core/tbe/kernel/ascend910/axpy/"
               "Axpy_233851a3505389e43928a8bba133a74d_high_performance.o",
               p);

    std::unique_ptr<OpKernelBin> kernelBin = std::make_unique<OpKernelBin>(opType, jsonPath, jsonPath, binPath, key,
                                                                           hashKey, BinType::STATIC_BIN, false, false);
    return kernelBin;
}

TEST_F(OpKernelUT, ParseStaticDynUBufSize_NormalValue)
{
    auto kernelBin = CreateStaticOpKernelBin("Test_normal_value.json");
    ASSERT_NE(kernelBin, nullptr);

    aclnnStatus ret = kernelBin->JsonLoad();
    EXPECT_EQ(ret, ACLNN_SUCCESS);
    EXPECT_EQ(kernelBin->GetStaticKernelDynUBufSize(), 4096);
}

TEST_F(OpKernelUT, ParseStaticDynUBufSize_ZeroValue)
{
    auto kernelBin = CreateStaticOpKernelBin("Test_zero_value.json");
    ASSERT_NE(kernelBin, nullptr);

    aclnnStatus ret = kernelBin->JsonLoad();
    EXPECT_EQ(ret, ACLNN_SUCCESS);
    EXPECT_EQ(kernelBin->GetStaticKernelDynUBufSize(), 0);
}

TEST_F(OpKernelUT, ParseStaticDynUBufSize_MaxValue)
{
    auto kernelBin = CreateStaticOpKernelBin("Test_max_uint32.json");
    ASSERT_NE(kernelBin, nullptr);

    aclnnStatus ret = kernelBin->JsonLoad();
    EXPECT_EQ(ret, ACLNN_SUCCESS);
    EXPECT_EQ(kernelBin->GetStaticKernelDynUBufSize(), 4294967295);
}

TEST_F(OpKernelUT, ParseStaticDynUBufSize_MissingField)
{
    auto kernelBin = CreateStaticOpKernelBin("Test_missing_field.json");
    ASSERT_NE(kernelBin, nullptr);

    aclnnStatus ret = kernelBin->JsonLoad();
    EXPECT_EQ(ret, ACLNN_SUCCESS);
    EXPECT_EQ(kernelBin->GetStaticKernelDynUBufSize(), 0);
}

TEST_F(OpKernelUT, ParseStaticDynUBufSize_NullValue)
{
    auto kernelBin = CreateStaticOpKernelBin("Test_null_value.json");
    ASSERT_NE(kernelBin, nullptr);

    aclnnStatus ret = kernelBin->JsonLoad();
    EXPECT_EQ(ret, ACLNN_SUCCESS);
    EXPECT_EQ(kernelBin->GetStaticKernelDynUBufSize(), 0);
}

TEST_F(OpKernelUT, ParseStaticDynUBufSize_InvalidTypeString)
{
    auto kernelBin = CreateStaticOpKernelBin("Test_invalid_type_string.json");
    ASSERT_NE(kernelBin, nullptr);

    aclnnStatus ret = kernelBin->JsonLoad();
    EXPECT_EQ(ret, ACLNN_SUCCESS);
    EXPECT_EQ(kernelBin->GetStaticKernelDynUBufSize(), 0);
}

TEST_F(OpKernelUT, ParseStaticDynUBufSize_InvalidTypeBoolean)
{
    auto kernelBin = CreateStaticOpKernelBin("Test_invalid_type_bool.json");
    ASSERT_NE(kernelBin, nullptr);

    aclnnStatus ret = kernelBin->JsonLoad();
    EXPECT_EQ(ret, ACLNN_SUCCESS);
    // nlohmann::json supports implicit conversion from bool to uint32_t (true -> 1, false -> 0)
    EXPECT_EQ(kernelBin->GetStaticKernelDynUBufSize(), 1);
}

TEST_F(OpKernelUT, ParseStaticDynUBufSize_OverflowValue)
{
    auto kernelBin = CreateStaticOpKernelBin("Test_overflow_value.json");
    ASSERT_NE(kernelBin, nullptr);

    aclnnStatus ret = kernelBin->JsonLoad();
    EXPECT_EQ(ret, ACLNN_SUCCESS);
    EXPECT_EQ(kernelBin->GetStaticKernelDynUBufSize(), 0);
}

TEST_F(OpKernelUT, testGetSocPathConsistentAcrossCalls)
{
    OpKernelLib opKnlLib;
    const std::string first = opKnlLib.GetSocPath();
    const std::string second = opKnlLib.GetSocPath();
    const std::string third = opKnlLib.GetSocPath();
    EXPECT_EQ(first, second);
    EXPECT_EQ(first, third);
}

TEST_F(OpKernelUT, testGetCustomOppFilePaths)
{
    const std::string mockDir = OP_API_COMMON_UT_SRC_DIR;
    const std::string defaultCustomPath = mockDir + "/custom";

    // 分支 1: env 未设置
    unsetenv("ASCEND_CUSTOM_OPP_PATH");
    {
        OpKernelLib opKnlLib;
        const std::vector<std::string> result = opKnlLib.GetCustomOppFilePaths();
        EXPECT_EQ(result.size(), 0U);
    }

    // 分支 2: 默认（mock/custom 提供 1 个 json）
    setenv("ASCEND_CUSTOM_OPP_PATH", defaultCustomPath.c_str(), 1);
    {
        OpKernelLib opKnlLib;
        const std::vector<std::string> result = opKnlLib.GetCustomOppFilePaths();
        EXPECT_GE(result.size(), 1U);
        bool found = false;
        for (const auto& path : result) {
            if (path.find("aic-custom-ops-info.json") != std::string::npos) {
                found = true;
                break;
            }
        }
        EXPECT_TRUE(found);
    }

    // 分支 3: 多路径（mock/custom + mock/merge_priority/custom_opp）
    const std::string altCustomPath = mockDir + "/merge_priority/custom_opp";
    const std::string multiPath = defaultCustomPath + ":" + altCustomPath;
    setenv("ASCEND_CUSTOM_OPP_PATH", multiPath.c_str(), 1);
    {
        OpKernelLib opKnlLib;
        const std::vector<std::string> result = opKnlLib.GetCustomOppFilePaths();
        EXPECT_GE(result.size(), 2U);
    }

    // 分支 4: dir 不存在
    setenv("ASCEND_CUSTOM_OPP_PATH", "./nonexistent_custom_opp_path", 1);
    {
        OpKernelLib opKnlLib;
        const std::vector<std::string> result = opKnlLib.GetCustomOppFilePaths();
        EXPECT_EQ(result.size(), 0U);
    }

    // 还原 main.cpp 默认 env
    setenv("ASCEND_CUSTOM_OPP_PATH", defaultCustomPath.c_str(), 1);
}

TEST_F(OpKernelUT, Initialize_MergePriority_CustomOppWinsOverVendorsAndBuiltin)
{
    const std::string mockDir = OP_API_COMMON_UT_SRC_DIR;
    const std::string oppPath = mockDir + "/merge_priority/opp";
    const std::string customPath = mockDir + "/merge_priority/custom_opp";
    setenv("ASCEND_OPP_PATH", oppPath.c_str(), 1);
    setenv("ASCEND_CUSTOM_OPP_PATH", customPath.c_str(), 1);

    OpKernelLib opKnlLib;
    ASSERT_EQ(opKnlLib.Initialize(), 0);
    EXPECT_EQ(opKnlLib.allKernelsJson_["AvgPool2D"]["opFile"]["value"].get<std::string>(), "from_custom_opp");

    // 还原 main.cpp 默认 env
    setenv("ASCEND_OPP_PATH", mockDir.c_str(), 1);
    setenv("ASCEND_CUSTOM_OPP_PATH", (mockDir + "/custom").c_str(), 1);
}
