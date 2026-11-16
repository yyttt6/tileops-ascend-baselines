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

#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnn/aclnn_base.h"
#include "kernel_mgr.h"
#include "memset_ctx_holder.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "op_cache_internal.h"
#include "op_ctx_def.h"
#include "opdev/op_cache.h"
#include "opdev/op_errno.h"
#include "register/op_impl_registry.h"
#include "rts_arg.h"
#include "thread_local_context.h"
#include "runtime/rt.h"
#include "runtime/rts/rts_kernel.h"
#include "kernel_workspace.h"
#include "kernel_launcher.h"
#include "depends/platform/platform_stub.h"

using namespace op::internal;

extern inline uint32_t SortOpTypeId();
extern op::internal::OpExecCacheWrap* GetOpExecCacheFromExecutor(aclOpExecutor* executor);
extern "C" aclOpExecutor* PTAFindExecCache(uint8_t* buf, size_t len, uint64_t* workspaceSize);
extern "C" aclOpExecutor* PTAGetExecCache(uint64_t hash, uint64_t* workspaceSize);
extern "C" void InitPTACacheThreadLocal();
extern "C" void ResetCacheThreadLocal();
extern "C" bool CanUsePTACache(const char* api);

class OpCacheSt : public testing::Test {
protected:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}
};

TEST_F(OpCacheSt, CreateHashTest)
{
    op::Shape tShape{1, 2, 3};
    aclTensor t1(tShape, op::DataType::DT_INT32, ge::FORMAT_ND, nullptr);
    aclTensor t2(tShape, op::DataType::DT_INT32, ge::FORMAT_ND, nullptr);
    aclTensor t3(tShape, op::DataType::DT_INT32, ge::FORMAT_ND, nullptr);
    // aclTensor
    const aclTensor* tensorPtr1 = &t1;
    aclTensor* tensorPtr2 = &t2;
    // aclTensorList
    aclTensor* list2[] = {&t1, &t2, &t3, nullptr};
    aclTensorList tensorList(list2, 4);
    const aclTensorList* tensorListPtr = &tensorList;
    aclTensorList* tensorList2Ptr = &tensorList;
    // bool array
    bool boolValue[] = {true, false, true};
    aclBoolArray array1(boolValue, 3);
    const aclBoolArray* array1Ptr = &array1;
    // float array
    float floatValue[] = {1.1, 2.2, 3.3};
    aclFloatArray array2(floatValue, 3);
    const aclFloatArray* array2Ptr = &array2;
    // int64_t array
    int64_t intValue[] = {1, 2, 3};
    aclIntArray array3(intValue, 3);
    const aclIntArray* array3Ptr = &array3;
    // fp16 array
    float value = 3.1;
    std::vector<op::fp16_t> values(3, value);
    aclFp16Array array4(values.data(), 3);
    const aclFp16Array* array4Ptr = &array4;
    // scalar
    float fpValue = 3.2;
    float fpValue2 = 3.3;
    aclScalar scalar1(&fpValue, op::DataType::DT_FLOAT);
    aclScalar scalar2(&fpValue2, op::DataType::DT_FLOAT);
    aclScalar scalar3(&fpValue2, op::DataType::DT_FLOAT);
    const aclScalar* constScalarPtr = &scalar1;
    aclScalar* scalarPtr = &scalar1;
    // aclScalarList
    aclScalar* list3[] = {&scalar1, &scalar2, &scalar3, nullptr};
    aclScalarList scalarList(list3, 4);
    aclScalarList* scalarListPtr1 = &scalarList;
    const aclScalarList* scalarListPtr2 = &scalarList;
    // string
    std::string apiName = "aclnnClamp";
    const aclDataType dataType = aclDataType::ACL_FLOAT;
    // nullptr param
    const aclBoolArray* arrayNull1 = nullptr;
    const aclFloatArray* arrayNull2 = nullptr;
    const aclIntArray* arrayNull3 = nullptr;
    const aclFp16Array* arrayNull4 = nullptr;
    const aclTensorList* tensorListNull1 = nullptr;
    aclTensorList* tensorListNull2 = nullptr;
    const aclScalarList* scalarListNull1 = nullptr;
    aclScalarList* scalarListNull2 = nullptr;
    const aclScalar* scalarNull1 = nullptr;
    int dim = 0;
    bool keepdim = true;
    auto in0 = std::make_tuple(tensorPtr1, tensorListPtr, constScalarPtr, scalarPtr, scalarListPtr1, array1Ptr,
                               array2Ptr, tensorList2Ptr, arrayNull1, arrayNull2, arrayNull3, dim);
    auto out0 = std::make_tuple(array3Ptr, array4Ptr, tensorPtr2, scalarListPtr2, dataType, arrayNull4, tensorListNull1,
                                tensorListNull2, scalarListNull1, scalarListNull2, scalarNull1, keepdim);
    OpCacheKey key;
    // OpExecCache::GenerateOpCacheKey(key, apiName, in0, out0);
    UnInitPTACacheThreadLocal();
    InitExecutorCacheThreadLocal();
    GetThreadLocalContext().opConfigInfo_.aicNum_ = 0;
    GetThreadLocalContext().opConfigInfo_.aivNum_ = 0;
    AddParamToBuf(apiName);
    AddOpConfigInfoToBuf();
    CalculateHashKey(in0);
    CalculateHashKey(out0);
    AddSeperator();
    SetOpCacheKey(key);
    char* hashBuf =
        "0x61636c6e6e436c616d7000000000000000000100000000000000020000000000000003000000000000002c0600000000000000030000"
        "000000000001000000000000002c0100000000000000020000000000000003000000000000002c030000002c00000000000000002c0200"
        "00000100000000000000020000000000000003000000000000002c0600000000000000030000000000000001000000000000002c010000"
        "0000000000020000000000000003000000000000002c030000002c00000000000000002c02000000010000000000000002000000000000"
        "0003000000000000002c0600000000000000030000000000000001000000000000002c0100000000000000020000000000000003000000"
        "000000002c030000002c00000000000000002c020000000100000000000000020000000000000003000000000000002c06000000000000"
        "00030000000000000001000000000000002c0100000000000000020000000000000003000000000000002c030000002c00000000000000"
        "002c020000002c0400000000000000cdcc4c402ccdcc4c402ccdcc4c402c333353402c333353402c2c0400000000000000010001030000"
        "0000000000cdcc8c3fcdcc0c403333534003000000000000000100000000000000020000000000000003000000000000002c0600000000"
        "000000030000000000000001000000000000002c0100000000000000020000000000000003000000000000002c030000002c0000000000"
        "0000002c020000000100000000000000020000000000000003000000000000002c06000000000000000300000000000000010000000000"
        "00002c0100000000000000020000000000000003000000000000002c030000002c00000000000000002c02000000010000000000000002"
        "0000000000000003000000000000002c0600000000000000030000000000000001000000000000002c0100000000000000020000000000"
        "000003000000000000002c030000002c00000000000000002c020000002c04000000000000002c2c2c0000000001000000000000000200"
        "00000000000003000000000000000300000000000000334233423342030000000000000001000000000000000200000000000000030000"
        "00000000002c0600000000000000030000000000000001000000000000002c010000000000000002000000000000000300000000000000"
        "2c030000002c00000000000000002c02000000cdcc4c402c333353402c333353402c2c0400000000000000000000002c2c2c2c2c2c012c"
        "00000000000000000000000000000000010000000000000002000000000000000000000000000000010000000000000002000000000000"
        "000100000000000000";
    EXPECT_STREQ(key.ToString().GetString(), hashBuf);
    EXPECT_EQ(key.len, 998);
    auto cache = GetOpExecCache(key);
    EXPECT_EQ(cache, nullptr);
    auto cache1 = GetOpExecCache(11);
    EXPECT_EQ(cache1, nullptr);
    uint64_t workspaceSize;
    auto executor = PTAFindExecCache(key.buf, key.len, &workspaceSize);
    EXPECT_EQ(executor, nullptr);
    InitPTACacheThreadLocal();
}

TEST_F(OpCacheSt, RtsArgCacheTest)
{
    op::Shape outShape{100};
    int64_t inputData[8];
    aclIntArray self(inputData, 8);
    aclTensor out(outShape, op::DataType::DT_INT32, ge::FORMAT_ND, nullptr);
    aclTensor inputTensor(&self, op::DataType::DT_INT32);

    auto input_arg = OP_INPUT(&inputTensor);
    auto output_arg = OP_OUTPUT(&out);
    auto ctx = op::MakeOpArgContext(input_arg, output_arg);

    op::internal::ExpandableRtsArgBuffer buffer;
    buffer.Init(1000, 2000);
    op::internal::TilingData* tilingData = buffer.GetTilingDataPtr();
    tilingData->data_size_ = 100;

    op::internal::LaunchArgInfo argInfo(false, false, ctx);
    op::internal::RtsArg arg(true, argInfo, &buffer);
    arg.FillArgs();
    op::internal::PrintRtArg(arg.GetRtsArg());

    GetThreadLocalContext().hashKey_ = 0;
    GetThreadLocalContext().cacheHashKey_ = (uint8_t*)"hello";
    GetThreadLocalContext().cacheHashKeyLen_ = 5;
    auto opExecCache = new OpExecCache();
    opExecCache->SetCacheBuf(GetCacheBuf());
    GetOpCacheContext().SetOpCache(opExecCache);
    auto cacheCtx = arg.DumpToCache();
    EXPECT_NE(cacheCtx, nullptr);

    aclrtStream stream = 0;
    op::internal::LaunchArgCache::RunFromCache(stream, GetCacheBuf());
    auto opExecCacheWrap = CreateCacheWrap(opExecCache);
    auto wrap = GetOpExecCacheFromExecutor(reinterpret_cast<aclOpExecutor*>(opExecCacheWrap));
    EXPECT_NE(wrap, nullptr);
    bool succ = AddOpExecCache(opExecCache);
    EXPECT_EQ(succ, true);
    op::DestroyOpArgContext(ctx);
    delete opExecCacheWrap;
}

TEST_F(OpCacheSt, CacheShrink)
{
    setenv("ACLNN_CACHE_LIMIT", "1", 1);
    GetThreadLocalContext().hashKey_ = 0;
    GetThreadLocalContext().cacheHashKey_ = (uint8_t*)"hello1";
    GetThreadLocalContext().cacheHashKeyLen_ = 6;
    auto opExecCache = new OpExecCache();
    opExecCache->SetCacheBuf(GetCacheBuf());
    GetOpCacheContext().SetOpCache(opExecCache);

    aclrtStream stream = 0;
    op::internal::LaunchArgCache::RunFromCache(stream, GetCacheBuf());
    auto opExecCacheWrap = CreateCacheWrap(opExecCache);
    auto wrap = GetOpExecCacheFromExecutor(reinterpret_cast<aclOpExecutor*>(opExecCacheWrap));
    EXPECT_NE(wrap, nullptr);
    bool succ = AddOpExecCache(opExecCache);
    EXPECT_EQ(succ, true);

    opExecCache->OldCacheClear();
    delete opExecCacheWrap;
    InitPTACacheThreadLocal();

    GetThreadLocalContext().hashKey_ = 0;
    GetThreadLocalContext().cacheHashKey_ = (uint8_t*)"hello2";
    GetThreadLocalContext().cacheHashKeyLen_ = 6;
    auto opExecCache2 = new OpExecCache();
    opExecCache2->SetCacheBuf(GetCacheBuf());
    GetOpCacheContext().SetOpCache(opExecCache2);

    aclrtStream stream2 = 0;
    op::internal::LaunchArgCache::RunFromCache(stream2, GetCacheBuf());
    auto opExecCacheWrap2 = CreateCacheWrap(opExecCache2);
    auto wrap2 = GetOpExecCacheFromExecutor(reinterpret_cast<aclOpExecutor*>(opExecCacheWrap2));
    EXPECT_NE(wrap2, nullptr);
    succ = AddOpExecCache(opExecCache2);
    EXPECT_EQ(succ, true);

    opExecCache2->OldCacheClear();
    delete opExecCacheWrap2;
    InitPTACacheThreadLocal();

    unsetenv("ACLNN_CACHE_LIMIT");
}

TEST_F(OpCacheSt, CacheUseTest)
{
    OpExecCache opExecCache;
    auto opExecCacheWrap = CreateCacheWrap(&opExecCache);
    auto wrap = reinterpret_cast<aclOpExecutor*>(opExecCacheWrap);
    aclrtStream stream = 0;
    CommonOpExecutorRun(nullptr, 0, wrap, stream);
}

TEST_F(OpCacheSt, UsePTACache)
{
    std::string api = "aclnnAdd";
    bool use = CanUsePTACache(api.c_str());
    // EXPECT_EQ(use, true);
}

TEST_F(OpCacheSt, StaticRtsArgCacheTest)
{
    op::Shape outShape{100};
    int64_t inputData[8];
    aclIntArray self(inputData, 8);
    aclTensor out(outShape, op::DataType::DT_INT32, ge::FORMAT_ND, nullptr);
    aclTensor inputTensor(&self, op::DataType::DT_INT32);

    auto input_arg = OP_INPUT(&inputTensor);
    auto output_arg = OP_OUTPUT(&out);
    auto ctx = op::MakeOpArgContext(input_arg, output_arg);

    op::internal::ExpandableRtsArgBuffer buffer;
    buffer.Init(1000, 2000);
    op::internal::TilingData* tilingData = buffer.GetTilingDataPtr();
    tilingData->data_size_ = 100;

    op::internal::LaunchArgInfo argInfo(false, false, ctx);
    op::internal::RtsArg arg(true, argInfo, &buffer);
    arg.FillArgs();
    op::internal::PrintRtArg(arg.GetRtsArg());

    GetThreadLocalContext().cacheHashKey_ = (uint8_t*)"hello";
    GetThreadLocalContext().cacheHashKeyLen_ = 5;
    auto opExecCache = new OpExecCache();
    opExecCache->SetCacheBuf(GetCacheBuf());
    GetOpCacheContext().SetOpCache(opExecCache);
    auto cacheCtx = arg.DumpToCache();
    EXPECT_EQ(cacheCtx, nullptr);
    opExecCache->Finalize();
    opExecCache->SetUse();
    EXPECT_EQ(opExecCache->CanUse(), false);
    ResetCacheThreadLocal();
    op::DestroyOpArgContext(ctx);
    delete opExecCache;
}

TEST_F(OpCacheSt, CacheExceptionDumpTest)
{
    op::Shape outShape{100};
    int64_t inputData[8];
    aclIntArray self(inputData, 8);
    aclTensor out(outShape, op::DataType::DT_INT32, ge::FORMAT_ND, nullptr);
    aclTensor inputTensor(&self, op::DataType::DT_INT32);

    op::Tensor* inputOpTensor = inputTensor.GetTensor();
    op::Tensor* outputOpTensor = out.GetTensor();
    OpLogInfo opLogInfo;
    ExceptionDumpInfo exceptionDumpInfo;
    aclrtStream stream = (aclrtStream)0x1;
    PrepareExceptionDumpInfo({inputOpTensor}, {outputOpTensor}, opLogInfo, exceptionDumpInfo, stream);
}

TEST_F(OpCacheSt, CreateHashGetCacheSuccessTest1)
{
    GetThreadLocalContext().cacheHasFull_ = false;
    bool usePTAHash = GetThreadLocalContext().usePTAHash_;
    GetThreadLocalContext().usePTAHash_ = false;
    setenv("ACLNN_CACHE_LIMIT", "10000000", 1);

    op::Shape selfShape{33, 15, 14, 48};
    op::Shape otherShape{33, 15, 14, 48};
    op::Shape outShape{33, 15, 14, 48};
    op::Shape idxShape{33, 15, 14, 48};

    auto self = std::make_unique<aclTensor>(selfShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND, nullptr);

    auto out = std::make_unique<aclTensor>(outShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND, nullptr);
    auto idx = std::make_unique<aclTensor>(idxShape, op::DataType::DT_INT32, op::Format::FORMAT_ND, nullptr);

    const aclTensor* wsArr[] = {out.get(), idx.get()};

    SortOpTypeId();
    uint32_t opType = op::OpTypeDict::ToOpType("Sort");

    int64_t dim = 0;
    bool descending = true;

    std::string apiName = "aclnnArgsort";
    auto input = OP_INPUT(self.get());
    auto output = OP_OUTPUT(out.get(), idx.get());

    auto attr = OP_ATTR(dim, descending);

    auto in0 = std::make_tuple(self.get(), dim, descending);
    auto out0 = std::make_tuple(out.get(), idx.get());

    // 第一次获取cache,失败
    aclrtStream stream = 0;
    op::internal::GetThreadLocalContext().logInfo_.l2ApiName = "aclnnArgsort";
    op::internal::GetThreadLocalContext().logInfo_.l2SequenceCounter = OpGetLogSequence();
    GetThreadLocalContext().logInfo_.l0Name = "Sort";
    OpCacheKey key;
    // OpExecCache::GenerateOpCacheKey(key, apiName, in0, out0);
    AddParamToBuf(apiName);
    CalculateHashKey(in0);
    CalculateHashKey(out0);
    SetOpCacheKey(key);
    char* hashBuf =
        "0x61636c6e6e417267736f727421000000000000000f000000000000000e0000000000000030000000000000002c6027000000000000a0"
        "02000000000000300000000000000001000000000000002c21000000000000000f000000000000000e0000000000000030000000000000"
        "002c010000002c00000000000000002c0200000000000000000000000121000000000000000f000000000000000e000000000000003000"
        "0000000000002c6027000000000000a002000000000000300000000000000001000000000000002c21000000000000000f000000000000"
        "000e0000000000000030000000000000002c010000002c00000000000000002c0200000021000000000000000f000000000000000e0000"
        "000000000030000000000000002c6027000000000000a002000000000000300000000000000001000000000000002c2100000000000000"
        "0f000000000000000e0000000000000030000000000000002c030000002c00000000000000002c02000000000000000000000001000000"
        "000000000200000000000000";
    EXPECT_STREQ(key.ToString().GetString(), hashBuf);
    EXPECT_EQ(key.len, 396);
    auto cache = GetOpExecCache(key);
    EXPECT_EQ(cache, nullptr);

    auto uniqueExecutor = CREATE_EXECUTOR();
    aclOpExecutor* executor = uniqueExecutor.get();
    EXPECT_NE(executor->GetOpExecCache(), nullptr);
    thread_local uint64_t kernelLaunchIdDefinedInL0Dfx = GenKernelLauncherId("Sort");
    ProfilingInfoId profilingInfoId(0, kernelLaunchIdDefinedInL0Dfx, 0);

    op::internal::GetLauncherCtx().ClearTilingCache();

    aclTensorList* workspace = nullptr;
    auto ctx2 = op::MakeOpArgContext(input, output, attr);
    GetWorkspace(opType, &workspace, executor, *ctx2->GetOpArg(op::OpArgDef::OP_INPUT_ARG),
                 *ctx2->GetOpArg(op::OpArgDef::OP_OUTPUT_ARG), *ctx2->GetOpArg(op::OpArgDef::OP_ATTR_ARG));
    op::DestroyOpArgContext(ctx2);

    auto ctx = op::MakeOpArgContext(OP_WORKSPACE(workspace), OP_INPUT(self.get()), OP_OUTPUT(out.get(), idx.get()),
                                    OP_ATTR(dim, descending));
    auto* launcher = new op::AiCoreKernelLauncher{opType, op::AI_CORE, profilingInfoId, executor, ctx};
    launcher->SaveLaunchCtx(std::move(op::internal::GetLauncherCtx()));
    executor->AddToKernelLauncherList(launcher);

    uint64_t workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(&executor);

    auto* cachePtr = executor->GetOpExecCache();
    EXPECT_NE(cachePtr, nullptr);
    GetOpCacheContext().SetOpCache(cachePtr);

    executor->SetStream(stream);
    executor->UpdateTensorAddr(workspace, workspaceSize);
    auto rc = launcher->Launch();
    EXPECT_EQ(rc, ACL_SUCCESS);
    OpCacheKey opCacheKey = executor->GetOpExecCache()->GetOpCacheKey();
    delete executor;
    // 第二次获取到cache
    op::internal::GetThreadLocalContext().logInfo_.l2ApiName = "aclnnArgsort";
    auto cache1 = GetOpExecCache(opCacheKey);
    EXPECT_EQ(cache1, cachePtr);

    std::vector<void*> tensors;
    tensors.push_back(const_cast<void*>(reinterpret_cast<const void*>(self.get()->GetStorage())));
    op::internal::opProfilingSwitch.kernelLaunchFlag = true;
    op::internal::opProfilingSwitch.additionInfoFlag = true;
    cache1->Run(nullptr, stream, tensors);

    // Inplace
    //  第一次获取cache,失败
    op::internal::GetThreadLocalContext().logInfo_.l2ApiName = "aclnnArgsortInplace";

    auto uniqueExecutorInplace = CREATE_EXECUTOR();
    OpCacheKey keyInplace;
    // OpExecCache::GenerateOpCacheKey(keyInplace, apiName, in0, out0);
    AddParamToBuf(apiName);
    CalculateHashKey(in0);
    CalculateHashKey(out0);
    SetOpCacheKey(keyInplace);
    EXPECT_NE(keyInplace.buf, nullptr);
    EXPECT_NE(keyInplace.len, 0);
    auto cacheInplace = GetOpExecCache(keyInplace);
    EXPECT_EQ(cacheInplace, nullptr);

    // 开始执行
    aclOpExecutor* executorInplace = uniqueExecutorInplace.get();
    EXPECT_NE(executorInplace->GetOpExecCache(), nullptr);
    ProfilingInfoId profilingInfoIdInplace(0, kernelLaunchIdDefinedInL0Dfx, 0);

    op::internal::GetLauncherCtx().ClearTilingCache();

    aclTensorList* workspaceInplace = nullptr;
    auto ctx3 = op::MakeOpArgContext(input, output, attr);
    GetWorkspace(opType, &workspaceInplace, executorInplace, *ctx3->GetOpArg(op::OpArgDef::OP_INPUT_ARG),
                 *ctx3->GetOpArg(op::OpArgDef::OP_OUTPUT_ARG), *ctx3->GetOpArg(op::OpArgDef::OP_ATTR_ARG));
    op::DestroyOpArgContext(ctx3);

    auto ctx4 = op::MakeOpArgContext(OP_WORKSPACE(workspaceInplace), OP_INPUT(self.get()),
                                     OP_OUTPUT(out.get(), idx.get()), OP_ATTR(dim, descending));

    auto* launcherInplace = new op::AiCoreKernelLauncher{opType, op::AI_CORE, profilingInfoIdInplace, executorInplace,
                                                         ctx4};
    launcherInplace->SaveLaunchCtx(std::move(op::internal::GetLauncherCtx()));
    executorInplace->AddToKernelLauncherList(launcherInplace);

    aclrtStream streamInplace = 0;
    uint64_t workspaceSizeInplace = uniqueExecutorInplace->GetWorkspaceSize();
    uniqueExecutorInplace.ReleaseTo(&executorInplace);

    EXPECT_NE(executorInplace->GetOpExecCache(), nullptr);

    OpCacheKey opCacheKeyInplace = executorInplace->GetOpExecCache()->GetOpCacheKey();

    GetOpCacheContext().SetOpCache(executorInplace->GetOpExecCache());

    executorInplace->SetStream(streamInplace);
    executorInplace->UpdateTensorAddr(workspaceInplace, workspaceSizeInplace);

    auto rcInplace = launcherInplace->Launch();
    EXPECT_EQ(rcInplace, ACL_SUCCESS);

    delete executorInplace;

    // 第二次获取到cache
    op::internal::GetThreadLocalContext().logInfo_.l2ApiName = "aclnnArgsortInplace";
    auto cacheInplace1 = GetOpExecCache(opCacheKeyInplace);
    EXPECT_NE(cacheInplace1, nullptr);

    // 非Inplace和Inplace命中不同的cache
    EXPECT_NE(cacheInplace1, cache1);

    GetThreadLocalContext().usePTAHash_ = usePTAHash;

    GetOpCacheContext().SetOpCache(nullptr);
    setenv("ACLNN_CACHE_LIMIT", "1", 1);
}