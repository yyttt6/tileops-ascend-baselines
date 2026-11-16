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
#include "opdev/op_dfx.h"
#include "op_dfx_internal.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "thread_local_context.h"
#include "launcher_ctx.h"
#include "kernel_launcher.h"
#include "opdev/make_op_executor.h"

using namespace op;
using namespace std;
using namespace op::internal;

class OpExecutorTest : public testing::Test {
protected:
    void TearDown() override
    {
        aclrtSetSysParamOpt(ACL_OPT_DETERMINISTIC, 0);
        aclrtSetSysParamOpt(ACL_OPT_STRONG_CONSISTENCY, 0);
        op::internal::GetThreadLocalContext().opConfigInfo_.deterministicLevel_ = 0;
        op::internal::GetThreadLocalContext().opConfigInfo_.isDeterministicOn_ = false;
    }
};

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
    auto tensor10 = executor.AllocHostTensor({100}, DataType::DT_DOUBLE, Format::FORMAT_ND);
    EXPECT_NE(tensor10, nullptr);
    auto tensor31 = executor.ConvertToTensor(scalar1, op::DataType::DT_INT4);
    EXPECT_EQ(tensor31, nullptr);

    uint64_t size = 1;
    const int64_t int64Value = 3;
    aclTensor* tensor21 = executor.AllocHostTensor(&int64Value, size, op::DataType::DT_INT64);
    EXPECT_NE(tensor21, nullptr);
    int32_t intValue1 = 3;
    aclTensor* tensor22 = executor.AllocHostTensor(&intValue1, size, op::DataType::DT_INT32);
    EXPECT_NE(tensor22, nullptr);
    double doubleValue = 3.2;
    aclTensor* tensor23 = executor.AllocHostTensor(&doubleValue, size, op::DataType::DT_DOUBLE);
    EXPECT_NE(tensor23, nullptr);
    const char* Value = "qqq";
    aclTensor* tensor24 = executor.AllocHostTensor(Value, size, op::DataType::DT_STRING);
    EXPECT_EQ(tensor24, nullptr);
    const uint64_t uint64Value = 3;
    aclTensor* tensor25 = executor.AllocHostTensor(&uint64Value, size, op::DataType::DT_UINT64);
    EXPECT_NE(tensor25, nullptr);

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

TEST_F(OpExecutorTest, CommonOpExecutorRunNullptr)
{
    EXPECT_EQ(CommonOpExecutorRun(nullptr, 0, nullptr, nullptr), ACLNN_ERR_PARAM_NULLPTR);
}
TEST_F(OpExecutorTest, AddToKernelLauncherListDvpp)
{
    op::Shape selfShape{33, 15, 14, 48};
    op::Shape outShape{33, 15, 14, 48};
    op::Shape idxShape{33, 15, 14, 48};
    auto self = std::make_unique<aclTensor>(selfShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND, nullptr);
    auto out = std::make_unique<aclTensor>(outShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND, nullptr);
    auto idx = std::make_unique<aclTensor>(idxShape, op::DataType::DT_INT32, op::Format::FORMAT_ND, nullptr);

    const aclTensor* wsArr[] = {out.get(), idx.get()};

    uint32_t opType = op::OpTypeDict::ToOpType("Sort");

    int64_t dim = 0;
    bool descending = true;

    std::string apiName = "aclnnArgsort";
    auto input = OP_INPUT(self.get());
    auto output = OP_OUTPUT(out.get(), idx.get());
    auto attr = OP_ATTR(dim, descending);

    auto uniqueExecutor = CREATE_EXECUTOR();
    aclOpExecutor* executor = uniqueExecutor.get();
    thread_local uint64_t kernelLaunchIdDefinedInL0Dfx = op::internal::GenKernelLauncherId("Sort");
    op::internal::ProfilingInfoId profilingInfoId(0, kernelLaunchIdDefinedInL0Dfx, 0);

    op::internal::GetLauncherCtx().ClearTilingCache();

    aclTensorList* workspace = nullptr;
    auto ctx = op::MakeOpArgContext(input, output, attr);
    auto* launcher = new op::AiCoreKernelLauncher{opType, op::AI_CORE, profilingInfoId, executor, ctx};
    launcher->SaveLaunchCtx(std::move(op::internal::GetLauncherCtx()));
    auto ret = executor->AddToKernelLauncherListDvpp(opType, launcher, ctx);
    EXPECT_EQ(ret, ACL_SUCCESS);
}
TEST_F(OpExecutorTest, AddToKernelLauncherListAiCpu)
{
    op::Shape selfShape{33, 15, 14, 48};
    op::Shape outShape{33, 15, 14, 48};
    op::Shape idxShape{33, 15, 14, 48};
    auto self = std::make_unique<aclTensor>(selfShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND, nullptr);
    auto out = std::make_unique<aclTensor>(outShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND, nullptr);
    auto idx = std::make_unique<aclTensor>(idxShape, op::DataType::DT_INT32, op::Format::FORMAT_ND, nullptr);

    const aclTensor* wsArr[] = {out.get(), idx.get()};

    uint32_t opType = op::OpTypeDict::ToOpType("Sort");

    int64_t dim = 0;
    bool descending = true;

    std::string apiName = "aclnnArgsort";
    auto input = OP_INPUT(self.get());
    auto output = OP_OUTPUT(out.get(), idx.get());
    auto attr = OP_ATTR(dim, descending);

    auto uniqueExecutor = CREATE_EXECUTOR();
    aclOpExecutor* executor = uniqueExecutor.get();
    thread_local uint64_t kernelLaunchIdDefinedInL0Dfx = op::internal::GenKernelLauncherId("Sort");
    op::internal::ProfilingInfoId profilingInfoId(0, kernelLaunchIdDefinedInL0Dfx, 0);

    op::internal::GetLauncherCtx().ClearTilingCache();

    aclTensorList* workspace = nullptr;
    auto ctx = op::MakeOpArgContext(input, output, attr);
    auto* launcher = new op::AiCoreKernelLauncher{opType, op::AI_CORE, profilingInfoId, executor, ctx};
    launcher->SaveLaunchCtx(std::move(op::internal::GetLauncherCtx()));
    auto ret = executor->AddToKernelLauncherListAiCpu(opType, launcher, ctx);
    EXPECT_EQ(ret, ACL_SUCCESS);
}
TEST_F(OpExecutorTest, AddToKernelLauncherListCopyTask)
{
    op::Shape selfShape{33, 15, 14, 48};
    op::Shape outShape{33, 15, 14, 48};
    op::Shape idxShape{33, 15, 14, 48};
    auto self = std::make_unique<aclTensor>(selfShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND, nullptr);
    auto out = std::make_unique<aclTensor>(outShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND, nullptr);
    auto idx = std::make_unique<aclTensor>(idxShape, op::DataType::DT_INT32, op::Format::FORMAT_ND, nullptr);

    const aclTensor* wsArr[] = {out.get(), idx.get()};

    uint32_t opType = op::OpTypeDict::ToOpType("Sort");

    int64_t dim = 0;
    bool descending = true;

    std::string apiName = "aclnnArgsort";
    auto input = OP_INPUT(self.get());
    auto output = OP_OUTPUT(out.get(), idx.get());
    auto attr = OP_ATTR(dim, descending);

    auto uniqueExecutor = CREATE_EXECUTOR();
    aclOpExecutor* executor = uniqueExecutor.get();
    thread_local uint64_t kernelLaunchIdDefinedInL0Dfx = op::internal::GenKernelLauncherId("Sort");
    op::internal::ProfilingInfoId profilingInfoId(0, kernelLaunchIdDefinedInL0Dfx, 0);

    op::internal::GetLauncherCtx().ClearTilingCache();

    aclTensorList* workspace = nullptr;
    auto ctx = op::MakeOpArgContext(input, output, attr);
    auto* launcher = new op::AiCoreKernelLauncher{opType, op::AI_CORE, profilingInfoId, executor, ctx};
    launcher->SaveLaunchCtx(std::move(op::internal::GetLauncherCtx()));
    auto ret1 = executor->AddToKernelLauncherListCopyTask(opType, launcher, *ctx->GetOpArg(op::OpArgDef::OP_OPTION_ARG),
                                                          *ctx->GetOpArg(op::OpArgDef::OP_OUTPUT_ARG),
                                                          *ctx->GetOpArg(op::OpArgDef::OP_ATTR_ARG));
    EXPECT_EQ(ret1, ACL_SUCCESS);
}

TEST_F(OpExecutorTest, apiTest)
{
    op::Shape selfShape{33, 15, 14, 48};
    op::Shape outShape{33, 15, 14, 48};
    op::Shape idxShape{33, 15, 14, 48};
    auto self = std::make_unique<aclTensor>(selfShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND, nullptr);
    auto out = std::make_unique<aclTensor>(outShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND, nullptr);
    auto idx = std::make_unique<aclTensor>(idxShape, op::DataType::DT_INT32, op::Format::FORMAT_ND, nullptr);

    const aclTensor* wsArr[] = {out.get(), idx.get()};

    uint32_t opType = op::OpTypeDict::ToOpType("Sort");

    int64_t dim = 0;
    bool descending = true;

    std::string apiName = "aclnnArgsort";
    auto input = OP_INPUT(self.get());
    auto output = OP_OUTPUT(out.get(), idx.get());
    auto attr = OP_ATTR(dim, descending);

    auto uniqueExecutor = CREATE_EXECUTOR();
    aclOpExecutor* executor = uniqueExecutor.get();

    executor->AddTensorRelation(self.get(), out.get());

    op::fp16_t fp16Value = 3.2;
    aclScalar* scalar1 = executor->AllocScalar(fp16Value);
    EXPECT_NE(scalar1, nullptr);
    op::bfloat16 bfp16Value = 3.2;
    aclScalar* scalar2 = executor->AllocScalar(bfp16Value);
    EXPECT_NE(scalar2, nullptr);
    double doubleValue = 3.2;
    aclScalar* scalar3 = executor->AllocScalar(doubleValue);
    EXPECT_NE(scalar3, nullptr);
    int32_t intValue = 3;
    aclScalar* intScalar = executor->AllocScalar(intValue);
    int64_t int64Value = 3;
    aclScalar* int64Scalar = executor->AllocScalar(int64Value);
    int16_t int16Value = 3;
    aclScalar* int16Scalar = executor->AllocScalar(int16Value);
    int8_t int8Value = 3;
    aclScalar* int8Scalar = executor->AllocScalar(int8Value);
    uint32_t uint32Value = 3;
    aclScalar* uint32Scalar = executor->AllocScalar(uint32Value);
    uint64_t uint64Value = 3;
    aclScalar* uint64Scalar = executor->AllocScalar(uint64Value);
    uint16_t uint16Value = 3;
    aclScalar* uint16Scalar = executor->AllocScalar(uint16Value);
    uint8_t uint8Value = 3;
    aclScalar* uint8Scalar = executor->AllocScalar(uint8Value);
    bool boolValue = true;
    aclScalar* boolScalar = executor->AllocScalar(boolValue);
}
TEST_F(OpExecutorTest, apiOtherTest)
{
    char* apiName = "aclnnArgsort";
    auto uniqueExecutor = CREATE_EXECUTOR();
    aclOpExecutor* executor = uniqueExecutor.get();
    InitL2Phase1Context(apiName, &executor);
    InitL2Phase2Context(apiName, executor);

    op::Shape selfShape{33, 15, 1, 48};
    op::Shape otherShape{33, 15, 14, 48};
    op::Shape outShape{33, 15, 14, 48};

    auto self = new aclTensor(selfShape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND, nullptr);
    auto other = new aclTensor(otherShape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND, nullptr);
    float alpha = 13.37;
    auto out = new aclTensor(outShape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND, nullptr);

    auto input = OP_INPUT(self, other);
    auto output = OP_OUTPUT(out);
    auto attr = OP_ATTR(alpha);
    auto ws = OP_WORKSPACE(out);
    auto ctx = op::MakeOpArgContext(input, output, attr, ws);
    uint32_t opType = op::OpTypeDict::ToOpType("Axpy");
    auto ret = CreatAiCoreKernelLauncher("Axpy", opType, executor, ctx);
    EXPECT_EQ(ret, ACL_SUCCESS);

    op::OpArgContext* ctx2 = nullptr;
    CreatDSAKernelLauncher("Axpy", opType, DSARandomNormalTaskType, executor, ctx2);

    auto ctx1 = op::MakeOpArgContext(input, output, attr);
    ret = CreatAiCoreKernelLauncher("Axpy", opType, executor, ctx1);
    EXPECT_EQ(ret, ACL_SUCCESS);

    ret = InferShape(opType, *ctx->GetOpArg(op::OpArgDef::OP_OPTION_ARG), *ctx->GetOpArg(op::OpArgDef::OP_OUTPUT_ARG),
                     *ctx->GetOpArg(op::OpArgDef::OP_ATTR_ARG));
    EXPECT_EQ(ret, ACLNN_SUCCESS);
    // op::DestroyOpArgContext(ctx);
}

TEST_F(OpExecutorTest, DSAKernelLauncherUnNormalTaskType)
{
    char* apiName = "aclnnArgsort";
    auto uniqueExecutor = CREATE_EXECUTOR();
    aclOpExecutor* executor = uniqueExecutor.get();
    InitL2Phase1Context(apiName, &executor);
    InitL2Phase2Context(apiName, executor);

    op::Shape selfShape{33, 15, 1, 48};
    op::Shape otherShape{33, 15, 14, 48};
    op::Shape outShape{33, 15, 14, 48};

    auto self = new aclTensor(selfShape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND, nullptr);
    auto other = new aclTensor(otherShape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND, nullptr);
    float alpha = 13.37;
    auto out = new aclTensor(outShape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND, nullptr);

    auto input = OP_INPUT(self, other);
    auto output = OP_OUTPUT(out);
    auto attr = OP_ATTR(alpha);
    auto ws = OP_WORKSPACE(out);
    auto ctx = op::MakeOpArgContext(input, output, attr, ws);
    op::GenOpTypeId("Axpy"); // ensure OpTypeDict is initialized so ToOpType/ToString work in isolation
    uint32_t opType = op::OpTypeDict::ToOpType("Axpy");

    DSA_TASK_TYPE unNormalDsaTaskType = static_cast<DSA_TASK_TYPE>(100); // > enum DSA_TASK_TYPE
    CreatDSAKernelLauncher("Axpy", opType, unNormalDsaTaskType, executor, ctx);

    aclnnStatus ret = InferShape(opType, *ctx->GetOpArg(op::OpArgDef::OP_OPTION_ARG),
                                 *ctx->GetOpArg(op::OpArgDef::OP_OUTPUT_ARG),
                                 *ctx->GetOpArg(op::OpArgDef::OP_ATTR_ARG));
    EXPECT_EQ(ret, ACLNN_SUCCESS);
    aclDestroyTensor(self);
    aclDestroyTensor(other);
    aclDestroyTensor(out);
    op::DestroyOpArgContext(ctx);
}

TEST_F(OpExecutorTest, phase2ParamCheckTest)
{
    char* apiName = "aclnnArgsort";
    aclrtStream stream = nullptr;
    InitL2Phase2Context(apiName, nullptr);
    int* ptr = new int;
    auto ret = CommonOpExecutorRun(ptr, 500, nullptr, stream);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_NULLPTR);

    auto uniqueExecutor = CREATE_EXECUTOR();
    aclOpExecutor* executor = nullptr;
    uniqueExecutor.ReleaseTo(&executor);
    ret = CommonOpExecutorRun(nullptr, 500, executor, stream);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_NULLPTR);

    auto opExecCache = new OpExecCache();
    OpExecCacheWrap* cacheWrap = CreateCacheWrap(opExecCache);
    executor = reinterpret_cast<aclOpExecutor*>(cacheWrap);
    ret = CommonOpExecutorRun(nullptr, 500, executor, stream);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_NULLPTR);

    delete opExecCache;
    delete ptr;
}

TEST_F(OpExecutorTest, phase2ParamCheckSuccessTest)
{
    aclrtStream stream = nullptr;
    auto uniqueExecutor = CREATE_EXECUTOR();
    aclOpExecutor* executor = nullptr;
    uniqueExecutor.ReleaseTo(&executor);
    auto ret = CommonOpExecutorRun(nullptr, 0, executor, stream);
    EXPECT_EQ(ret, ACLNN_SUCCESS);
}

TEST_F(OpExecutorTest, phase1ParamCheckTest)
{
    uint64_t workspaceSize;
    aclOpExecutor** executor = nullptr;
    auto ret = CheckPhase1Params(executor, &workspaceSize);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_NULLPTR);

    int* ptr = new int;
    executor = reinterpret_cast<aclOpExecutor**>(ptr);
    ret = CheckPhase1Params(executor, nullptr);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_NULLPTR);

    ret = CheckPhase1Params(executor, &workspaceSize);
    EXPECT_EQ(ret, ACLNN_SUCCESS);

    delete ptr;
}

static void L2Phase2Func(const char* apiName, aclOpExecutor* executor, int32_t expectedLevel)
{
    std::cout << "phase2 thread id: " << std::this_thread::get_id() << std::endl;
    InitL2Phase2Context(apiName, executor);
    EXPECT_EQ(executor->GetOpConfigInfo().deterministicLevel_, expectedLevel);
    EXPECT_EQ(op::internal::GetThreadLocalContext().opConfigInfo_.deterministicLevel_, expectedLevel);
    EXPECT_EQ(executor->GetOpConfigInfo().isDeterministicOn_, (expectedLevel >= 1));
    EXPECT_EQ(op::internal::GetThreadLocalContext().opConfigInfo_.isDeterministicOn_, (expectedLevel >= 1));
}

TEST_F(OpExecutorTest, DeterministicTest)
{
    const char* apiName = "aclnnArgsort";

    // 子用例 1：level=0（关闭）
    {
        aclrtSetSysParamOpt(ACL_OPT_DETERMINISTIC, 0);
        aclrtSetSysParamOpt(ACL_OPT_STRONG_CONSISTENCY, 0);
        aclOpExecutor* executor = nullptr;
        InitL2Phase1Context(apiName, &executor);
        EXPECT_EQ(op::internal::GetThreadLocalContext().opConfigInfo_.deterministicLevel_, 0);
        EXPECT_EQ(op::internal::GetThreadLocalContext().opConfigInfo_.isDeterministicOn_, false);
        auto uniqueExecutor = CREATE_EXECUTOR();
        uniqueExecutor.ReleaseTo(&executor);
        EXPECT_EQ(executor->GetOpConfigInfo().deterministicLevel_, 0);
        EXPECT_EQ(executor->GetOpConfigInfo().isDeterministicOn_, false);
        std::thread t(L2Phase2Func, apiName, executor, 0);
        t.join();
        delete executor;
    }

    // 子用例 2：level=1（仅确定性，consistency=0 不升级）
    {
        aclrtSetSysParamOpt(ACL_OPT_DETERMINISTIC, 1);
        aclrtSetSysParamOpt(ACL_OPT_STRONG_CONSISTENCY, 0);
        aclOpExecutor* executor = nullptr;
        InitL2Phase1Context(apiName, &executor);
        EXPECT_EQ(op::internal::GetThreadLocalContext().opConfigInfo_.deterministicLevel_, 1);
        EXPECT_EQ(op::internal::GetThreadLocalContext().opConfigInfo_.isDeterministicOn_, true);
        auto uniqueExecutor = CREATE_EXECUTOR();
        uniqueExecutor.ReleaseTo(&executor);
        EXPECT_EQ(executor->GetOpConfigInfo().deterministicLevel_, 1);
        EXPECT_EQ(executor->GetOpConfigInfo().isDeterministicOn_, true);
        std::thread t(L2Phase2Func, apiName, executor, 1);
        t.join();
        delete executor;
    }

    // 子用例 3：level=2（强一致性，新 runtime 直接返回）
    {
        aclrtSetSysParamOpt(ACL_OPT_DETERMINISTIC, 2);
        aclOpExecutor* executor = nullptr;
        InitL2Phase1Context(apiName, &executor);
        EXPECT_EQ(op::internal::GetThreadLocalContext().opConfigInfo_.deterministicLevel_, 2);
        EXPECT_EQ(op::internal::GetThreadLocalContext().opConfigInfo_.isDeterministicOn_, true);
        auto uniqueExecutor = CREATE_EXECUTOR();
        uniqueExecutor.ReleaseTo(&executor);
        EXPECT_EQ(executor->GetOpConfigInfo().deterministicLevel_, 2);
        EXPECT_EQ(executor->GetOpConfigInfo().isDeterministicOn_, true);
        std::thread t(L2Phase2Func, apiName, executor, 2);
        t.join();
        delete executor;
    }

    // 子用例 4：老 runtime 回退路径（deterministic=1 + consistency=1 → 升级为 2）
    {
        aclrtSetSysParamOpt(ACL_OPT_DETERMINISTIC, 1);
        aclrtSetSysParamOpt(ACL_OPT_STRONG_CONSISTENCY, 1);
        aclOpExecutor* executor = nullptr;
        InitL2Phase1Context(apiName, &executor);
        EXPECT_EQ(op::internal::GetThreadLocalContext().opConfigInfo_.deterministicLevel_, 2);
        EXPECT_EQ(op::internal::GetThreadLocalContext().opConfigInfo_.isDeterministicOn_, true);
        auto uniqueExecutor = CREATE_EXECUTOR();
        uniqueExecutor.ReleaseTo(&executor);
        EXPECT_EQ(executor->GetOpConfigInfo().deterministicLevel_, 2);
        EXPECT_EQ(executor->GetOpConfigInfo().isDeterministicOn_, true);
        std::thread t(L2Phase2Func, apiName, executor, 2);
        t.join();
        delete executor;
    }

    // 子用例 5：level=3（batch 一致性，新 runtime 路径）
    {
        aclrtSetSysParamOpt(ACL_OPT_DETERMINISTIC, 3);
        aclOpExecutor* executor = nullptr;
        InitL2Phase1Context(apiName, &executor);
        EXPECT_EQ(op::internal::GetThreadLocalContext().opConfigInfo_.deterministicLevel_, 3);
        EXPECT_EQ(op::internal::GetThreadLocalContext().opConfigInfo_.isDeterministicOn_, true);
        auto uniqueExecutor = CREATE_EXECUTOR();
        uniqueExecutor.ReleaseTo(&executor);
        EXPECT_EQ(executor->GetOpConfigInfo().deterministicLevel_, 3);
        EXPECT_EQ(executor->GetOpConfigInfo().isDeterministicOn_, true);
        std::thread t(L2Phase2Func, apiName, executor, 3);
        t.join();
        delete executor;
    }
}

// ============================================================================
// 场景: float/double 类型的 aclScalar 通过 ConvertToTensor 转换为 float8/6/4 类型 tensor
// ============================================================================

// 1.1 Float scalar → Float8E4M3FN tensor (OCP FP8, bias=7, 最大值448, 无Infinity)
TEST_F(OpExecutorTest, ConvertFloatScalarToFloat8E4M3FN)
{
    op::internal::GetThreadLocalContext().cacheHasFull_ = true;
    aclOpExecutor executor;

    float fValue = 1.0f;
    auto scalar = executor.AllocScalar(fValue);
    ASSERT_NE(scalar, nullptr);

    auto tensor = executor.ConvertToTensor(scalar, op::DataType::DT_FLOAT8_E4M3FN);
    EXPECT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->GetDataType(), op::DataType::DT_FLOAT8_E4M3FN);

    // 验证: 1.0 → OCP E4M3FN 编码 0x38 (sign=0, exp=0111, man=000)
    auto* data = static_cast<uint8_t*>(tensor->GetStorageAddr());
    op::Float8E4M3FN result;
    result.value = *data;
    EXPECT_FLOAT_EQ(static_cast<float>(result), 1.0f);
}

// 1.2 Float scalar → Float8E5M2 tensor (OCP FP8, bias=15, 最大值57344, 支持NaN/Infinity)
TEST_F(OpExecutorTest, ConvertFloatScalarToFloat8E5M2)
{
    op::internal::GetThreadLocalContext().cacheHasFull_ = true;
    aclOpExecutor executor;

    float fValue = 1.0f;
    auto scalar = executor.AllocScalar(fValue);
    ASSERT_NE(scalar, nullptr);

    auto tensor = executor.ConvertToTensor(scalar, op::DataType::DT_FLOAT8_E5M2);
    EXPECT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->GetDataType(), op::DataType::DT_FLOAT8_E5M2);

    // 验证: 1.0 → OCP E5M2 编码 0x30 (sign=0, exp=01111, man=00)
    auto* data = static_cast<uint8_t*>(tensor->GetStorageAddr());
    op::Float8E5M2 result;
    result.value = *data;
    EXPECT_FLOAT_EQ(static_cast<float>(result), 1.0f);
}

// 1.3 Float scalar → Float8E8M0 tensor (MX缩放因子, 纯指数格式, value=2^(exp-127))
TEST_F(OpExecutorTest, ConvertFloatScalarToFloat8E8M0)
{
    op::internal::GetThreadLocalContext().cacheHasFull_ = true;
    aclOpExecutor executor;

    float fValue = 1.0f;
    auto scalar = executor.AllocScalar(fValue);
    ASSERT_NE(scalar, nullptr);

    auto tensor = executor.ConvertToTensor(scalar, op::DataType::DT_FLOAT8_E8M0);
    EXPECT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->GetDataType(), op::DataType::DT_FLOAT8_E8M0);

    // 验证: 1.0 = 2^(127-127) = 2^0 → 编码 0x7F (OCP E8M0)
    auto* data = static_cast<uint8_t*>(tensor->GetStorageAddr());
    op::Float8E8M0 result;
    result.value = *data;
    EXPECT_FLOAT_EQ(static_cast<float>(result), 1.0f);
}

// 2.1 Float scalar → Float6E3M2 tensor (MX FP6, bias=3, 最大值28.0)
TEST_F(OpExecutorTest, ConvertFloatScalarToFloat6E3M2)
{
    op::internal::GetThreadLocalContext().cacheHasFull_ = true;
    aclOpExecutor executor;

    float fValue = 1.0f;
    auto scalar = executor.AllocScalar(fValue);
    ASSERT_NE(scalar, nullptr);

    auto tensor = executor.ConvertToTensor(scalar, op::DataType::DT_FLOAT6_E3M2);
    EXPECT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->GetDataType(), op::DataType::DT_FLOAT6_E3M2);

    auto* data = static_cast<uint8_t*>(tensor->GetStorageAddr());
    op::Float6E3M2 result;
    result.value = *data;
    EXPECT_FLOAT_EQ(static_cast<float>(result), 1.0f);
}

// 2.2 Float scalar → Float6E2M3 tensor (MX FP6, bias=1, 最大值7.0)
TEST_F(OpExecutorTest, ConvertFloatScalarToFloat6E2M3)
{
    op::internal::GetThreadLocalContext().cacheHasFull_ = true;
    aclOpExecutor executor;

    float fValue = 1.0f;
    auto scalar = executor.AllocScalar(fValue);
    ASSERT_NE(scalar, nullptr);

    auto tensor = executor.ConvertToTensor(scalar, op::DataType::DT_FLOAT6_E2M3);
    EXPECT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->GetDataType(), op::DataType::DT_FLOAT6_E2M3);

    auto* data = static_cast<uint8_t*>(tensor->GetStorageAddr());
    op::Float6E2M3 result;
    result.value = *data;
    EXPECT_FLOAT_EQ(static_cast<float>(result), 1.0f);
}

// 3.1 Float scalar → Float4E2M1 tensor (MX FP4, bias=1, 最大值6.0)
TEST_F(OpExecutorTest, ConvertFloatScalarToFloat4E2M1)
{
    op::internal::GetThreadLocalContext().cacheHasFull_ = true;
    aclOpExecutor executor;

    float fValue = 1.0f;
    auto scalar = executor.AllocScalar(fValue);
    ASSERT_NE(scalar, nullptr);

    auto tensor = executor.ConvertToTensor(scalar, op::DataType::DT_FLOAT4_E2M1);
    EXPECT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->GetDataType(), op::DataType::DT_FLOAT4_E2M1);

    auto* data = static_cast<uint8_t*>(tensor->GetStorageAddr());
    op::Float4E2M1 result;
    result.value = *data;
    EXPECT_FLOAT_EQ(static_cast<float>(result), 1.0f);
}

// 3.2 Float scalar → Float4E1M2 tensor (bias=1, 最大值1.75)
TEST_F(OpExecutorTest, ConvertFloatScalarToFloat4E1M2)
{
    op::internal::GetThreadLocalContext().cacheHasFull_ = true;
    aclOpExecutor executor;

    float fValue = 1.0f;
    auto scalar = executor.AllocScalar(fValue);
    ASSERT_NE(scalar, nullptr);

    auto tensor = executor.ConvertToTensor(scalar, op::DataType::DT_FLOAT4_E1M2);
    EXPECT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->GetDataType(), op::DataType::DT_FLOAT4_E1M2);

    auto* data = static_cast<uint8_t*>(tensor->GetStorageAddr());
    op::Float4E1M2 result;
    result.value = *data;
    EXPECT_FLOAT_EQ(static_cast<float>(result), 1.0f);
}

// 4.1 Double scalar → Float8E4M3FN tensor
TEST_F(OpExecutorTest, ConvertDoubleScalarToFloat8E4M3FN)
{
    op::internal::GetThreadLocalContext().cacheHasFull_ = true;
    aclOpExecutor executor;

    double dValue = 1.0;
    auto scalar = executor.AllocScalar(dValue);
    ASSERT_NE(scalar, nullptr);

    auto tensor = executor.ConvertToTensor(scalar, op::DataType::DT_FLOAT8_E4M3FN);
    EXPECT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->GetDataType(), op::DataType::DT_FLOAT8_E4M3FN);

    auto* data = static_cast<uint8_t*>(tensor->GetStorageAddr());
    op::Float8E4M3FN result;
    result.value = *data;
    EXPECT_FLOAT_EQ(static_cast<float>(result), 1.0f);
}

// 4.2 Double scalar → Float8E5M2 tensor
TEST_F(OpExecutorTest, ConvertDoubleScalarToFloat8E5M2)
{
    op::internal::GetThreadLocalContext().cacheHasFull_ = true;
    aclOpExecutor executor;

    double dValue = 1.0;
    auto scalar = executor.AllocScalar(dValue);
    ASSERT_NE(scalar, nullptr);

    auto tensor = executor.ConvertToTensor(scalar, op::DataType::DT_FLOAT8_E5M2);
    EXPECT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->GetDataType(), op::DataType::DT_FLOAT8_E5M2);

    auto* data = static_cast<uint8_t*>(tensor->GetStorageAddr());
    op::Float8E5M2 result;
    result.value = *data;
    EXPECT_FLOAT_EQ(static_cast<float>(result), 1.0f);
}

// 4.3 Double scalar → Float8E8M0 tensor
TEST_F(OpExecutorTest, ConvertDoubleScalarToFloat8E8M0)
{
    op::internal::GetThreadLocalContext().cacheHasFull_ = true;
    aclOpExecutor executor;

    double dValue = 1.0;
    auto scalar = executor.AllocScalar(dValue);
    ASSERT_NE(scalar, nullptr);

    auto tensor = executor.ConvertToTensor(scalar, op::DataType::DT_FLOAT8_E8M0);
    EXPECT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->GetDataType(), op::DataType::DT_FLOAT8_E8M0);

    auto* data = static_cast<uint8_t*>(tensor->GetStorageAddr());
    op::Float8E8M0 result;
    result.value = *data;
    EXPECT_FLOAT_EQ(static_cast<float>(result), 1.0f);
}

// 4.4 Double scalar → Float6E3M2 tensor
TEST_F(OpExecutorTest, ConvertDoubleScalarToFloat6E3M2)
{
    op::internal::GetThreadLocalContext().cacheHasFull_ = true;
    aclOpExecutor executor;

    double dValue = 1.0;
    auto scalar = executor.AllocScalar(dValue);
    ASSERT_NE(scalar, nullptr);

    auto tensor = executor.ConvertToTensor(scalar, op::DataType::DT_FLOAT6_E3M2);
    EXPECT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->GetDataType(), op::DataType::DT_FLOAT6_E3M2);

    auto* data = static_cast<uint8_t*>(tensor->GetStorageAddr());
    op::Float6E3M2 result;
    result.value = *data;
    EXPECT_FLOAT_EQ(static_cast<float>(result), 1.0f);
}

// 4.5 Double scalar → Float6E2M3 tensor
TEST_F(OpExecutorTest, ConvertDoubleScalarToFloat6E2M3)
{
    op::internal::GetThreadLocalContext().cacheHasFull_ = true;
    aclOpExecutor executor;

    double dValue = 1.0;
    auto scalar = executor.AllocScalar(dValue);
    ASSERT_NE(scalar, nullptr);

    auto tensor = executor.ConvertToTensor(scalar, op::DataType::DT_FLOAT6_E2M3);
    EXPECT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->GetDataType(), op::DataType::DT_FLOAT6_E2M3);

    auto* data = static_cast<uint8_t*>(tensor->GetStorageAddr());
    op::Float6E2M3 result;
    result.value = *data;
    EXPECT_FLOAT_EQ(static_cast<float>(result), 1.0f);
}

// 4.6 Double scalar → Float4E2M1 tensor
TEST_F(OpExecutorTest, ConvertDoubleScalarToFloat4E2M1)
{
    op::internal::GetThreadLocalContext().cacheHasFull_ = true;
    aclOpExecutor executor;

    double dValue = 1.0;
    auto scalar = executor.AllocScalar(dValue);
    ASSERT_NE(scalar, nullptr);

    auto tensor = executor.ConvertToTensor(scalar, op::DataType::DT_FLOAT4_E2M1);
    EXPECT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->GetDataType(), op::DataType::DT_FLOAT4_E2M1);

    auto* data = static_cast<uint8_t*>(tensor->GetStorageAddr());
    op::Float4E2M1 result;
    result.value = *data;
    EXPECT_FLOAT_EQ(static_cast<float>(result), 1.0f);
}

// 4.7 Double scalar → Float4E1M2 tensor
TEST_F(OpExecutorTest, ConvertDoubleScalarToFloat4E1M2)
{
    op::internal::GetThreadLocalContext().cacheHasFull_ = true;
    aclOpExecutor executor;

    double dValue = 1.0;
    auto scalar = executor.AllocScalar(dValue);
    ASSERT_NE(scalar, nullptr);

    auto tensor = executor.ConvertToTensor(scalar, op::DataType::DT_FLOAT4_E1M2);
    EXPECT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->GetDataType(), op::DataType::DT_FLOAT4_E1M2);

    auto* data = static_cast<uint8_t*>(tensor->GetStorageAddr());
    op::Float4E1M2 result;
    result.value = *data;
    EXPECT_FLOAT_EQ(static_cast<float>(result), 1.0f);
}

// 5.1 Float scalar 负值 → Float8E4M3FN
TEST_F(OpExecutorTest, ConvertFloatScalarNegativeToFloat8E4M3FN)
{
    op::internal::GetThreadLocalContext().cacheHasFull_ = true;
    aclOpExecutor executor;

    float fValue = -1.0f;
    auto scalar = executor.AllocScalar(fValue);
    ASSERT_NE(scalar, nullptr);

    auto tensor = executor.ConvertToTensor(scalar, op::DataType::DT_FLOAT8_E4M3FN);
    EXPECT_NE(tensor, nullptr);

    auto* data = static_cast<uint8_t*>(tensor->GetStorageAddr());
    op::Float8E4M3FN result;
    result.value = *data;
    EXPECT_FLOAT_EQ(static_cast<float>(result), -1.0f);
}

// 5.2 Float scalar 零值 → Float8E5M2
TEST_F(OpExecutorTest, ConvertFloatScalarZeroToFloat8E5M2)
{
    op::internal::GetThreadLocalContext().cacheHasFull_ = true;
    aclOpExecutor executor;

    float fValue = 0.0f;
    auto scalar = executor.AllocScalar(fValue);
    ASSERT_NE(scalar, nullptr);

    auto tensor = executor.ConvertToTensor(scalar, op::DataType::DT_FLOAT8_E5M2);
    EXPECT_NE(tensor, nullptr);

    auto* data = static_cast<uint8_t*>(tensor->GetStorageAddr());
    op::Float8E5M2 result;
    result.value = *data;
    EXPECT_FLOAT_EQ(static_cast<float>(result), 0.0f);
}

// 5.3 Float scalar 溢出值 → Float4E1M2 (最大值1.75, 输入2.0应被钳制到1.75)
TEST_F(OpExecutorTest, ConvertFloatScalarOverflowToFloat4E1M2)
{
    op::internal::GetThreadLocalContext().cacheHasFull_ = true;
    aclOpExecutor executor;

    float fValue = 2.0f;
    auto scalar = executor.AllocScalar(fValue);
    ASSERT_NE(scalar, nullptr);

    auto tensor = executor.ConvertToTensor(scalar, op::DataType::DT_FLOAT4_E1M2);
    EXPECT_NE(tensor, nullptr);

    auto* data = static_cast<uint8_t*>(tensor->GetStorageAddr());
    op::Float4E1M2 result;
    result.value = *data;
    // Float4E1M2无Infinity, 溢出应钳制到最大值1.75
    EXPECT_FLOAT_EQ(static_cast<float>(result), 1.75f);
}

// 5.4 Double scalar 溢出值 → Float4E2M1 (最大值6.0, 输入10.0应被钳制到6.0)
TEST_F(OpExecutorTest, ConvertDoubleScalarOverflowToFloat4E2M1)
{
    op::internal::GetThreadLocalContext().cacheHasFull_ = true;
    aclOpExecutor executor;

    double dValue = 10.0;
    auto scalar = executor.AllocScalar(dValue);
    ASSERT_NE(scalar, nullptr);

    auto tensor = executor.ConvertToTensor(scalar, op::DataType::DT_FLOAT4_E2M1);
    EXPECT_NE(tensor, nullptr);

    auto* data = static_cast<uint8_t*>(tensor->GetStorageAddr());
    op::Float4E2M1 result;
    result.value = *data;
    // OCP规范: FP4无Infinity, 溢出应钳制到最大值6.0
    EXPECT_FLOAT_EQ(static_cast<float>(result), 6.0f);
}
