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
// #include "mockcpp/mockcpp.hpp"
#include <array>
#include <atomic>
#include <chrono>
#include <iostream>
#include <memory>
#include <stdlib.h>
#include <thread>

#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnn/aclnn_base.h"
#include "kernel_mgr.h"
#include "memset_ctx_holder.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_cache.h"
#include "op_cache_internal.h"
#include "opdev/op_errno.h"
#include "register/op_impl_registry.h"
#include "rts_arg.h"
#include "opdev/aicpu/aicpu_task.h"
#include "thread_local_context.h"
#include "kernel_workspace.h"
#include "kernel_launcher.h"
#include "depends/platform/platform_stub.h"
#include "runtime/runtime/rts/rts_kernel.h"
#include "test_comp_op_common.h"

using namespace op::internal;
using namespace op::internal::test;

extern inline uint32_t SortOpTypeId();
extern inline uint32_t GeluOpTypeId();
extern op::internal::OpExecCacheWrap* GetOpExecCacheFromExecutor(aclOpExecutor* executor);
extern "C" aclOpExecutor* PTAFindExecCache(uint8_t* buf, size_t len, uint64_t* workspaceSize);
extern "C" aclOpExecutor* PTAGetExecCache(uint64_t hash, uint64_t* workspaceSize);
extern "C" void InitPTACacheThreadLocal();
extern "C" void UnInitPTACacheThreadLocal();
extern "C" void ResetCacheThreadLocal();
extern "C" bool CanUsePTACache(const char* api);
extern "C" void AddTensorAddrToCachedList(void* addr);
extern "C" void SetPTAHashKey(uint64_t hash);
extern "C" void SetPTACacheHashKey(uint8_t* key, size_t len);

class OpCacheUt : public testing::Test {
protected:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}
};

aclError aclrtBinaryLoadFromFileInvoked(const char_t* const binPath, const rtLoadBinaryConfig_t* const optionalCfg,
                                        rtBinHandle* handle)
{
    rtBinHandle handle_tmp = nullptr;
    *handle = &handle_tmp;
    return RT_ERROR_NONE;
}

aclError aclrtFuncGetByNameInvoked(const rtBinHandle binHandle, const char_t* kernelName, rtFuncHandle* funcHandle)
{
    rtFuncHandle funcHandle_tmp = nullptr;
    *funcHandle = &funcHandle_tmp;
    return RT_ERROR_NONE;
}

TEST_F(OpCacheUt, CreateHashTest)
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
    const aclScalar* scalarPtr = &scalar1;
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
    auto in0 = std::make_tuple(tensorPtr1, tensorListPtr, scalarPtr, scalarListPtr1, array1Ptr, array2Ptr,
                               tensorList2Ptr, arrayNull1, arrayNull2, arrayNull3, dim);
    auto out0 = std::make_tuple(array3Ptr, array4Ptr, tensorPtr2, scalarListPtr2, dataType, arrayNull4, tensorListNull1,
                                tensorListNull2, scalarListNull1, scalarListNull2, scalarNull1, keepdim);
    OpCacheKey key;
    // OpExecCache::GenerateOpCacheKey(key, apiName, in0, out0);
    UnInitPTACacheThreadLocal();
    InitExecutorCacheThreadLocal();
    AddParamToBuf(apiName);
    AddOpConfigInfoToBuf();
    CalculateHashKey(in0);
    CalculateHashKey(out0);
    AddSeperator();
    SetOpCacheKey(key);
    char* hashBuf =
        "0x61636c6e6e436c616d700000000000000000000100000000000000020000000000000003000000000000002c06000000000000000300"
        "00000000000001000000000000002c0100000000000000020000000000000003000000000000002c030000002c00000000000000002c02"
        "0000000100000000000000020000000000000003000000000000002c0600000000000000030000000000000001000000000000002c0100"
        "000000000000020000000000000003000000000000002c030000002c00000000000000002c020000000100000000000000020000000000"
        "000003000000000000002c0600000000000000030000000000000001000000000000002c01000000000000000200000000000000030000"
        "00000000002c030000002c00000000000000002c020000000100000000000000020000000000000003000000000000002c060000000000"
        "0000030000000000000001000000000000002c0100000000000000020000000000000003000000000000002c030000002c000000000000"
        "00002c020000002c0400000000000000cdcc4c402c000000002ccdcc4c402c000000002c333353402c000000002c333353402c00000000"
        "2c2c04000000000000000100010300000000000000cdcc8c3fcdcc0c403333534003000000000000000100000000000000020000000000"
        "000003000000000000002c0600000000000000030000000000000001000000000000002c01000000000000000200000000000000030000"
        "00000000002c030000002c00000000000000002c020000000100000000000000020000000000000003000000000000002c060000000000"
        "0000030000000000000001000000000000002c0100000000000000020000000000000003000000000000002c030000002c000000000000"
        "00002c020000000100000000000000020000000000000003000000000000002c0600000000000000030000000000000001000000000000"
        "002c0100000000000000020000000000000003000000000000002c030000002c00000000000000002c020000002c04000000000000002c"
        "2c2c0000000001000000000000000200000000000000030000000000000003000000000000003342334233420300000000000000010000"
        "0000000000020000000000000003000000000000002c0600000000000000030000000000000001000000000000002c0100000000000000"
        "020000000000000003000000000000002c030000002c00000000000000002c02000000cdcc4c402c000000002c333353402c000000002c"
        "333353402c000000002c2c0400000000000000000000002c2c2c2c2c2c012c000000000000000000000000000000000100000000000000"
        "02000000000000000000000000000000010000000000000002000000000000000100000000000000";

    EXPECT_STREQ(key.ToString().GetString(), hashBuf);
    EXPECT_EQ(key.len, 1029);
    auto cache = GetOpExecCache(key);
    EXPECT_EQ(cache, nullptr);
    auto cache1 = GetOpExecCache(11);
    EXPECT_EQ(cache1, nullptr);
    uint64_t workspaceSize;
    auto executor = PTAFindExecCache(key.buf, key.len, &workspaceSize);
    EXPECT_EQ(executor, nullptr);
    InitPTACacheThreadLocal();
}

TEST_F(OpCacheUt, RtsArgCacheTest)
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
    buffer.Init(TEST_LAUNCH_ARG_INIT_CAP, TEST_TILING_HOST_DATA_INIT_CAP);
    op::internal::TilingData* tilingData = buffer.GetTilingDataPtr();
    constexpr size_t TEST_TILING_DATA_LEN = 32;
    tilingData->data_ = buffer.GetTilingDataAddr();
    memset_s(tilingData->data_, TEST_TILING_HOST_DATA_INIT_CAP, 0, TEST_TILING_DATA_LEN);
    tilingData->data_size_ = TEST_TILING_DATA_LEN;
    tilingData->capacity_ = TEST_TILING_HOST_DATA_INIT_CAP;

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
    delete opExecCacheWrap;
    op::DestroyOpArgContext(ctx);
    InitPTACacheThreadLocal();
}

TEST_F(OpCacheUt, RtsArgCacheRepeatSetTest)
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
    buffer.Init(TEST_LAUNCH_ARG_INIT_CAP, TEST_TILING_HOST_DATA_INIT_CAP);
    op::internal::TilingData* tilingData = buffer.GetTilingDataPtr();
    constexpr size_t TEST_TILING_DATA_LEN = 60;
    tilingData->data_ = buffer.GetTilingDataAddr();
    memset_s(tilingData->data_, TEST_TILING_HOST_DATA_INIT_CAP, 0, TEST_TILING_DATA_LEN);
    tilingData->data_size_ = TEST_TILING_DATA_LEN;
    tilingData->capacity_ = TEST_TILING_HOST_DATA_INIT_CAP;

    op::internal::LaunchArgInfo argInfo(false, false, ctx);
    op::internal::RtsArg arg(true, argInfo, &buffer);
    arg.FillArgs();
    op::internal::PrintRtArg(arg.GetRtsArg());

    GetThreadLocalContext().hashKey_ = 0;
    GetThreadLocalContext().cacheHashKey_ = (uint8_t*)"hello";
    GetThreadLocalContext().cacheHashKeyLen_ = 5;
    auto opExecCache = new OpExecCache();
    EXPECT_EQ(opExecCache->IsOpCacheValid(), true);
    opExecCache->SetUse();
    EXPECT_EQ(opExecCache->CanUse(), true);
    GetOpCacheContext().SetOpCache(opExecCache);
    auto cacheCtx = arg.DumpToCache();
    EXPECT_EQ(cacheCtx, nullptr);
    opExecCache->MarkOpCacheInvalid();
    EXPECT_EQ(opExecCache->IsOpCacheValid(), true);

    op::DestroyOpArgContext(ctx);
    delete opExecCache;
}

TEST_F(OpCacheUt, ExecutorCacheRepeatSetTest)
{
    bool oriOption = GetThreadLocalContext().cacheHasFull_;
    GetThreadLocalContext().cacheHasFull_ = false;
    GetThreadLocalContext().hashKey_ = 0;
    GetThreadLocalContext().cacheHashKey_ = (uint8_t*)"ExecutorCacheRepeatSetTest";
    GetThreadLocalContext().cacheHashKeyLen_ = 27;
    auto uniqueExecutor = CREATE_EXECUTOR();
    aclOpExecutor* executor = uniqueExecutor.get();
    executor->SetRepeatable();
    EXPECT_EQ(executor->IsRepeatable(), true);
    auto opExecCache = executor->GetOpExecCache();

    int* intPtr = new int;
    void* ptr = reinterpret_cast<void*>(intPtr);
    opExecCache->SetCacheBuf(ptr);
    EXPECT_EQ(opExecCache->cacheBuf_, ptr);
    opExecCache->SetUse();
    EXPECT_EQ(opExecCache->CanUse(), true);

    executor->UpdateTensorAddr(nullptr, 0);
    EXPECT_EQ(opExecCache->cacheBuf_, ptr);

    executor->FinalizeCache();
    EXPECT_EQ(opExecCache->cacheBuf_, ptr);

    delete intPtr;
    GetThreadLocalContext().cacheHasFull_ = oriOption;
}

TEST_F(OpCacheUt, RtsArgCacheTestHostDataNullptr)
{
    op::Shape outShape{0};
    int64_t inputData[8];
    aclIntArray self(inputData, 0);
    aclTensor out(outShape, op::DataType::DT_INT32, ge::FORMAT_ND, nullptr);
    aclTensor inputTensor(&self, op::DataType::DT_INT32);

    auto input_arg = OP_INPUT(&inputTensor);
    auto output_arg = OP_OUTPUT(&out);
    auto ctx = op::MakeOpArgContext(input_arg, output_arg);

    op::internal::ExpandableRtsArgBuffer buffer;
    buffer.Init(TEST_LAUNCH_ARG_INIT_CAP, TEST_TILING_HOST_DATA_INIT_CAP);
    op::internal::TilingData* tilingData = buffer.GetTilingDataPtr();
    constexpr size_t TEST_TILING_DATA_LEN = 60;
    tilingData->data_ = buffer.GetTilingDataAddr();
    memset_s(tilingData->data_, TEST_TILING_HOST_DATA_INIT_CAP, 0, TEST_TILING_DATA_LEN);
    tilingData->data_size_ = TEST_TILING_DATA_LEN;
    tilingData->capacity_ = TEST_TILING_HOST_DATA_INIT_CAP;

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

    op::DestroyOpArgContext(ctx);
    InitPTACacheThreadLocal();
    delete opExecCache;
}

TEST_F(OpCacheUt, AbnormalRtsArgCacheTest)
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
    buffer.Init(TEST_LAUNCH_ARG_INIT_CAP, TEST_TILING_HOST_DATA_INIT_CAP);
    op::internal::TilingData* tilingData = buffer.GetTilingDataPtr();
    constexpr size_t TEST_TILING_DATA_LEN = 60;
    tilingData->data_ = buffer.GetTilingDataAddr();
    memset_s(tilingData->data_, TEST_TILING_HOST_DATA_INIT_CAP, 0, TEST_TILING_DATA_LEN);
    tilingData->data_size_ = TEST_TILING_DATA_LEN;
    tilingData->capacity_ = TEST_TILING_HOST_DATA_INIT_CAP;

    op::internal::LaunchArgInfo argInfo(false, false, ctx);
    op::internal::RtsArg arg(true, argInfo, &buffer);
    arg.FillArgs();

    GetThreadLocalContext().hashKey_ = 0;
    GetThreadLocalContext().cacheHashKey_ = (uint8_t*)"hello";
    GetThreadLocalContext().cacheHashKeyLen_ = 5;
    auto opExecCache = new OpExecCache();
    opExecCache->SetCacheBuf(GetCacheBuf());
    GetOpCacheContext().SetOpCache(opExecCache);
    auto cacheCtx = arg.DumpToCache();
    EXPECT_NE(cacheCtx, nullptr);

    op::internal::KernelLaunchConfig launchCfg;
    launchCfg.binHandle = (void*)0x12341234;
    launchCfg.funcHandle = (void*)0x12345678;
    launchCfg.tilingKey = 1234;
    launchCfg.numBlocks = 32;
    launchCfg.schemMode = 1;
    launchCfg.dynUBufSize = 0;
    launchCfg.blockDimOffset = 0;
    launchCfg.engineType = op::internal::LaunchKernelEngineType::VECTOR_CORE_ENGINE_AIC;
    launchCfg.isFatBin = false;
    std::string kernelName = "abs";
    cacheCtx->SetRunParam(launchCfg, kernelName);

    EXPECT_EQ(cacheCtx->launchCfg_.binHandle, launchCfg.binHandle);
    EXPECT_EQ(cacheCtx->launchCfg_.funcHandle, launchCfg.funcHandle);
    EXPECT_EQ(cacheCtx->launchCfg_.tilingKey, launchCfg.tilingKey);
    EXPECT_EQ(cacheCtx->launchCfg_.numBlocks, launchCfg.numBlocks);
    EXPECT_EQ(cacheCtx->launchCfg_.schemMode, launchCfg.schemMode);
    EXPECT_EQ(cacheCtx->launchCfg_.dynUBufSize, launchCfg.dynUBufSize);
    EXPECT_EQ(cacheCtx->launchCfg_.blockDimOffset, launchCfg.blockDimOffset);
    EXPECT_EQ(cacheCtx->launchCfg_.engineType, launchCfg.engineType);
    EXPECT_EQ(cacheCtx->launchCfg_.isFatBin, launchCfg.isFatBin);
    EXPECT_STREQ(cacheCtx->launchCfg_.kernelNameOfNoFatBin, kernelName.c_str());

    aclrtStream stream = 0;
    auto res = op::internal::LaunchArgCache::RunFromCache(stream, nullptr);
    EXPECT_EQ(res, ACLNN_ERR_INNER);

    res = op::internal::LaunchArgCache::RunFromCache(stream, GetCacheBuf());
    EXPECT_EQ(res, ACLNN_SUCCESS);
    op::DestroyOpArgContext(ctx);
    InitPTACacheThreadLocal();
    delete opExecCache;
}

TEST_F(OpCacheUt, CacheShrink)
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

TEST_F(OpCacheUt, CacheUseTest)
{
    OpExecCache opExecCache;
    auto opExecCacheWrap = CreateCacheWrap(&opExecCache);
    auto wrap = reinterpret_cast<aclOpExecutor*>(opExecCacheWrap);
    aclrtStream stream = 0;
    CommonOpExecutorRun(nullptr, 0, wrap, stream);
}

TEST_F(OpCacheUt, UsePTACache)
{
    std::string api = "aclnnAdd";
    bool use = CanUsePTACache(api.c_str());
    // EXPECT_EQ(use, true);
}

TEST_F(OpCacheUt, SetPTACache)
{
    InitPTACacheThreadLocal();
    op::internal::GetThreadLocalContext().cachedTensorList_.clear();
    constexpr size_t ARR_SIZE = 6;
    int arr[ARR_SIZE] = {0};
    for (int i = 0; i < ARR_SIZE; i++) {
        AddTensorAddrToCachedList(&arr[i]);
    }
    EXPECT_EQ(op::internal::GetThreadLocalContext().cachedTensorListSize_, ARR_SIZE);
    EXPECT_EQ(op::internal::GetThreadLocalContext().cachedTensorList_.size(), ARR_SIZE);

    InitPTACacheThreadLocal();
    int arr2[ARR_SIZE] = {0};
    AddTensorAddrToCachedList(&arr2[0]);
    AddTensorAddrToCachedList(&arr2[1]);
    EXPECT_EQ(op::internal::GetThreadLocalContext().cachedTensorListSize_, 2);
    EXPECT_EQ(op::internal::GetThreadLocalContext().cachedTensorList_.size(), ARR_SIZE);

    uint64_t hash = 0x12345678;
    SetPTAHashKey(hash);
    EXPECT_EQ(op::internal::GetThreadLocalContext().hashKey_, hash);

    char* hashKey = "aclnnMatmul12345678";
    SetPTACacheHashKey(reinterpret_cast<uint8_t*>(hashKey), strlen(hashKey));
    EXPECT_STREQ(reinterpret_cast<const char*>(op::internal::GetThreadLocalContext().cacheHashKey_), hashKey);
    EXPECT_EQ(op::internal::GetThreadLocalContext().cacheHashKeyLen_, strlen(hashKey));

    OpCacheKey key;
    SetOpCacheKey(key);
    EXPECT_STREQ(reinterpret_cast<char*>(key.buf), hashKey);
    EXPECT_EQ(key.len, strlen(hashKey));

    UnInitPTACacheThreadLocal();
}

TEST_F(OpCacheUt, PtrListCacheTest)
{
    op::Shape outShape{100};
    int64_t inputData[8];
    aclIntArray self(inputData, 8);
    aclIntArray self2(inputData, 8);
    aclTensor out(outShape, op::DataType::DT_INT32, ge::FORMAT_ND, nullptr);
    aclTensor inputTensor(&self, op::DataType::DT_INT32);
    aclTensor inputTensor2(&self2, op::DataType::DT_INT32);

    const aclTensor* inputArr[] = {&inputTensor, &inputTensor2};
    aclTensorList* inputTensors = aclCreateTensorList(inputArr, 2);

    auto input_arg = OP_INPUT(inputTensors);
    auto output_arg = OP_OUTPUT(&out);
    auto ctx = op::MakeOpArgContext(input_arg, output_arg);

    op::internal::ExpandableRtsArgBuffer buffer;
    buffer.Init(TEST_LAUNCH_ARG_INIT_CAP, TEST_TILING_HOST_DATA_INIT_CAP);
    op::internal::TilingData* tilingData = buffer.GetTilingDataPtr();
    constexpr size_t TEST_TILING_DATA_LEN = 60;
    tilingData->data_ = buffer.GetTilingDataAddr();
    memset_s(tilingData->data_, TEST_TILING_HOST_DATA_INIT_CAP, 0, TEST_TILING_DATA_LEN);
    tilingData->data_size_ = TEST_TILING_DATA_LEN;
    tilingData->capacity_ = TEST_TILING_HOST_DATA_INIT_CAP;

    op::internal::LaunchArgInfo argInfo(false, true, ctx);
    op::internal::RtsArg arg(true, argInfo, &buffer);
    arg.FillArgs();

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

    delete inputTensors;
    delete opExecCache;
    op::DestroyOpArgContext(ctx);
}

TEST_F(OpCacheUt, StaticRtsArgCacheTest)
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
    buffer.Init(TEST_LAUNCH_ARG_INIT_CAP, TEST_TILING_HOST_DATA_INIT_CAP);
    op::internal::TilingData* tilingData = buffer.GetTilingDataPtr();
    tilingData->data_size_ = 0;
    tilingData->capacity_ = TEST_TILING_HOST_DATA_INIT_CAP;

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
    delete opExecCache;
    InitPTACacheThreadLocal();
    GetOpCacheContext().SetOpCache(nullptr);
    op::DestroyOpArgContext(ctx);
}

TEST_F(OpCacheUt, CacheExceptionDumpTest)
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

TEST_F(OpCacheUt, CreateHashGetCacheSuccessTest1)
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

TEST_F(OpCacheUt, CreateHashAbnormalBufferOverflow)
{
    bool usePTAHash = GetThreadLocalContext().usePTAHash_;
    GetThreadLocalContext().usePTAHash_ = false;
    op::Shape tShape{1, 2, 3};
    aclTensor t1(tShape, op::DataType::DT_INT32, ge::FORMAT_ND, nullptr);
    aclTensor t2(tShape, op::DataType::DT_INT32, ge::FORMAT_ND, nullptr);
    aclTensor t3(tShape, op::DataType::DT_INT32, ge::FORMAT_ND, nullptr);
    aclTensor t4(tShape, op::DataType::DT_INT32, ge::FORMAT_ND, nullptr);
    aclTensor t5(tShape, op::DataType::DT_INT32, ge::FORMAT_ND, nullptr);
    aclTensor t6(tShape, op::DataType::DT_INT32, ge::FORMAT_ND, nullptr);
    aclTensor t7(tShape, op::DataType::DT_INT32, ge::FORMAT_ND, nullptr);
    aclTensor t8(tShape, op::DataType::DT_INT32, ge::FORMAT_ND, nullptr);
    aclTensor t9(tShape, op::DataType::DT_INT32, ge::FORMAT_ND, nullptr);
    aclTensor t10(tShape, op::DataType::DT_INT32, ge::FORMAT_ND, nullptr);
    aclTensor t11(tShape, op::DataType::DT_INT32, ge::FORMAT_ND, nullptr);
    aclTensor t12(tShape, op::DataType::DT_INT32, ge::FORMAT_ND, nullptr);
    aclTensor t13(tShape, op::DataType::DT_INT32, ge::FORMAT_ND, nullptr);
    aclTensor t14(tShape, op::DataType::DT_INT32, ge::FORMAT_ND, nullptr);
    aclTensor t15(tShape, op::DataType::DT_INT32, ge::FORMAT_ND, nullptr);
    aclTensor t16(tShape, op::DataType::DT_INT32, ge::FORMAT_ND, nullptr);
    aclTensor t17(tShape, op::DataType::DT_INT32, ge::FORMAT_ND, nullptr);
    aclTensor t18(tShape, op::DataType::DT_INT32, ge::FORMAT_ND, nullptr);
    aclTensor t19(tShape, op::DataType::DT_INT32, ge::FORMAT_ND, nullptr);
    aclTensor* listTensor[] = {&t1,  &t2,  &t3,  &t4,  &t5,  &t6,  &t7,  &t8,  &t9,  &t10,
                               &t11, &t12, &t13, &t14, &t15, &t16, &t17, &t18, &t19, nullptr};
    aclTensorList tensorList(listTensor, 20);
    const aclTensorList* tensorListPtr = &tensorList;
    aclTensorList* tensorList2Ptr = &tensorList;
    aclTensorList* tensorList3Ptr = &tensorList;
    aclTensorList* tensorList4Ptr = &tensorList;
    aclTensorList* tensorList5Ptr = &tensorList;
    // float array
    int64_t intValue[] = {1, 2, 3};
    aclIntArray array3(intValue, 3);
    const aclIntArray* array3Ptr = &array3;
    // scalar
    float fpValue = 3.2;
    aclScalar scalar(&fpValue, op::DataType::DT_FLOAT);
    const aclScalar* scalarPtr = &scalar;
    std::string apiName = "aclnnClamp";
    const aclDataType dataType = aclDataType::ACL_FLOAT;
    // nullptr param
    const aclFp16Array* arrayNull4 = nullptr;
    const aclTensorList* tensorListNull1 = nullptr;
    aclTensorList* tensorListNull2 = nullptr;
    int dim = 0;
    bool keepdim = true;
    auto in0 = std::make_tuple(tensorListPtr, tensorList2Ptr, tensorList3Ptr, tensorList4Ptr, tensorList5Ptr, dim,
                               keepdim);
    auto out0 = std::make_tuple(array3Ptr, scalarPtr, dataType, arrayNull4, tensorListNull1, tensorListNull2);
    OpCacheKey key;
    // OpExecCache::GenerateOpCacheKey(key, apiName, in0, out0);
    AddParamToBuf(apiName);
    CalculateHashKey(in0);
    CalculateHashKey(out0);
    AddSeperator();
    SetOpCacheKey(key);
    EXPECT_EQ(key.buf, nullptr);
    EXPECT_EQ(key.len, 0);
    auto cache = GetOpExecCache(key);
    EXPECT_EQ(cache, nullptr);
    uint64_t workspaceSize;
    auto executor = PTAFindExecCache(key.buf, key.len, &workspaceSize);
    EXPECT_EQ(executor, nullptr);
    uint64_t hash = 11;
    executor = PTAGetExecCache(hash, &workspaceSize);
    EXPECT_EQ(executor, nullptr);
    InitPTACacheThreadLocal();
    GetThreadLocalContext().usePTAHash_ = usePTAHash;
}

TEST_F(OpCacheUt, AbnormalIncludeOpOutShape)
{
    bool usePTAHash = GetThreadLocalContext().usePTAHash_;
    GetThreadLocalContext().usePTAHash_ = false;
    op::Shape tShape{1, 2, 3};
    auto self = std::make_unique<aclTensor>(tShape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND, nullptr);
    auto out = std::make_unique<aclTensor>(tShape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND, nullptr);
    auto outShape = std::make_unique<aclTensor>(tShape, op::DataType::DT_INT32, op::Format::FORMAT_ND, nullptr);
    int64_t dim = 0;
    bool descending = true;

    auto attr = OP_ATTR(dim, descending);
    auto input = OP_INPUT(self.get());
    auto output = OP_OUTPUT(out.get());

    auto in0 = std::make_tuple(self.get());
    auto out0 = std::make_tuple(out.get());
    std::string apiName = "aclnnGelu";
    uint32_t OpTypeId = op::OpTypeDict::ToOpType("Gelu");

    op::internal::GetThreadLocalContext().logInfo_.l2ApiName = "aclnnGelu";
    op::internal::GetThreadLocalContext().logInfo_.l2SequenceCounter = OpGetLogSequence();
    auto uniqueExecutor = CREATE_EXECUTOR();

    OpCacheKey key;
    // OpExecCache::GenerateOpCacheKey(key, apiName, in0, out0);
    AddParamToBuf(apiName);
    CalculateHashKey(in0);
    CalculateHashKey(out0);
    SetOpCacheKey(key);
    EXPECT_NE(key.buf, nullptr);
    EXPECT_NE(key.len, 0);
    auto cache = GetOpExecCache(key);
    EXPECT_EQ(cache, nullptr);

    aclOpExecutor* executor = uniqueExecutor.get();
    OpCacheKey opCacheKey = executor->GetOpExecCache()->GetOpCacheKey();
    EXPECT_NE(executor->GetOpExecCache(), nullptr);
    op::internal::GetLauncherCtx().ClearTilingCache();

    // 包含OP_OUTSHAPE_ARG参数，禁止cache
    ADD_TO_LAUNCHER_LIST_AICORE(Gelu, OP_INPUT(self.get()), OP_OUTPUT(out.get()), OP_OUTSHAPE(outShape.get(), 0));
    EXPECT_NE(executor->GetOpExecCache(), nullptr);
    EXPECT_EQ(opCacheKey.buf, nullptr);
    EXPECT_EQ(opCacheKey.len, 0);
    uniqueExecutor.ReleaseTo(&executor);
    GetThreadLocalContext().usePTAHash_ = usePTAHash;
    delete executor->GetOpExecCache();
    delete executor;
}

// TEMP_COMMENT 需要解决mockcpp安装问题
// TEST_F(OpCacheUt, AICPUNoCache) {
//     MOCKER_CPP(&aclrtBinaryLoadFromFile)
//         .stubs()
//         .will(invoke(aclrtBinaryLoadFromFileInvoked));
//     MOCKER_CPP(&aclrtFuncGetByName)
//         .stubs()
//         .will(invoke(aclrtFuncGetByNameInvoked));
//     bool usePTAHash = GetThreadLocalContext().usePTAHash_;
//     GetThreadLocalContext().usePTAHash_ = false;
//     op::Shape tShape{1, 2, 3};
//     auto self = std::make_unique<aclTensor>(tShape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND, nullptr);
//     auto out = std::make_unique<aclTensor>(tShape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND, nullptr);

//     auto in0 = std::make_tuple(self.get());
//     auto out0 = std::make_tuple(out.get());
//     std::string apiName = "aclnnGelu";
//     uint32_t OpTypeId = op::OpTypeDict::ToOpType("Gelu");

//     op::internal::GetThreadLocalContext().logInfo_.l2ApiName = "aclnnGelu";
//     op::internal::GetThreadLocalContext().logInfo_.l2SequenceCounter = OpGetLogSequence();
//     auto uniqueExecutor = CREATE_EXECUTOR();

//     OpCacheKey key;
//     // OpExecCache::GenerateOpCacheKey(key, apiName, in0, out0);
//     AddParamToBuf(apiName);
//     CalculateHashKey(in0);
//     CalculateHashKey(out0);
//     SetOpCacheKey(key);
//     EXPECT_NE(key.buf, nullptr);
//     EXPECT_NE(key.len, 0);
//     auto cache = GetOpExecCache(key);
//     EXPECT_EQ(cache, nullptr);

//     aclOpExecutor *executor = uniqueExecutor.get();
//     OpCacheKey opCacheKey = executor->GetOpExecCache()->GetOpCacheKey();
//     EXPECT_NE(opCacheKey.buf, nullptr);
//     EXPECT_NE(opCacheKey.len, 0);

//     op::internal::GetLauncherCtx().ClearTilingCache();

//     // AICPU 禁止cache
//     static op::internal::AicpuTaskSpace space("Gelu", ge::DEPEND_IN_SHAPE, false);

//     ADD_TO_LAUNCHER_LIST_AICPU(Gelu, OP_ATTR_NAMES(), OP_INPUT(self.get()), OP_OUTPUT(out.get()));
//     EXPECT_NE(executor->GetOpExecCache(), nullptr);
//     EXPECT_EQ(executor->GetOpExecCache()->GetHash(), 0);

//     EXPECT_EQ(executor->SetRepeatable(), 0);

//     EXPECT_EQ(executor->GetOpExecCache()->GetOpCacheKey().buf, nullptr);
//     EXPECT_EQ(executor->GetOpExecCache()->GetOpCacheKey().len, 0);
//     uniqueExecutor.ReleaseTo(&executor);
//     GetThreadLocalContext().usePTAHash_ = usePTAHash;
//     delete executor->GetOpExecCache();
//     delete executor;
// }

TEST_F(OpCacheUt, GetFromCache)
{
    bool usePTAHash = GetThreadLocalContext().usePTAHash_;
    GetThreadLocalContext().usePTAHash_ = false;
    op::Shape tShape{1, 2, 3};
    auto self = std::make_unique<aclTensor>(tShape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND, nullptr);
    auto out = std::make_unique<aclTensor>(tShape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND, nullptr);

    auto in0 = std::make_tuple(self.get());
    auto out0 = std::make_tuple(out.get());
    std::string apiName = "aclnnGelu";
    uint32_t OpTypeId = op::OpTypeDict::ToOpType("Gelu");

    op::internal::GetThreadLocalContext().logInfo_.l2ApiName = "aclnnGelu";
    op::internal::GetThreadLocalContext().logInfo_.l2SequenceCounter = OpGetLogSequence();
    auto uniqueExecutor = CREATE_EXECUTOR();

    aclOpExecutor* executor = uniqueExecutor.get();
    uint64_t* workspaceSize;
    auto ret = op::internal::GetFromCache(&executor, workspaceSize, "aclnnGelu", in0, out0);
    EXPECT_EQ(ret, false);
    GetThreadLocalContext().usePTAHash_ = usePTAHash;
    uniqueExecutor.ReleaseTo(&executor);
    delete executor;
}
TEST_F(OpCacheUt, AddLaunchTensor)
{
    float fpValue = 3.2;
    uint64_t size = 1;
    auto self = std::make_unique<aclTensor>(&fpValue, size, op::DataType::DT_FLOAT);
    const aclStorage* storage = self.get()->GetStorage();
    EXPECT_EQ(storage->IsFromWorkspace(), false);
    auto opExecCache = new OpExecCache();
    opExecCache->SetCacheBuf(GetCacheBuf());
    opExecCache->AddLaunchTensor(self.get(), sizeof(void*));
    EXPECT_EQ(opExecCache->AddLaunchData(65537), nullptr);
    delete opExecCache;
}
TEST_F(OpCacheUt, AddTensorRelation)
{
    op::Shape tShape{1, 2, 3};
    auto self = std::make_unique<aclTensor>(tShape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND, nullptr);
    auto out = std::make_unique<aclTensor>(tShape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND, nullptr);
    auto opExecCache1 = new OpExecCache();
    opExecCache1->SetCacheBuf(GetCacheBuf());

    opExecCache1->AddTensorRelation(self.get(), out.get());
    opExecCache1->Finalize();
    delete opExecCache1;
}
TEST_F(OpCacheUt, apiTest)
{
    auto opExecCache = new OpExecCache();
    auto workspaceSize = opExecCache->GetWorkspaceSize();
    EXPECT_EQ(workspaceSize, 0);

    bool cacheable = op::internal::CheckCacheable();
    EXPECT_EQ(cacheable, false);

    delete opExecCache;
}
TEST_F(OpCacheUt, AddParamToBufaclScalarList)
{
    bool usePTAHash = GetThreadLocalContext().usePTAHash_;
    GetThreadLocalContext().usePTAHash_ = false;
    double scalar_value = 5;
    auto* scalar = aclCreateScalar(&scalar_value, aclDataType::ACL_DOUBLE);
    auto scalarList = aclCreateScalarList(&scalar, 1);
    AddParamToBuf(scalarList);
    OpCacheKey key;
    SetOpCacheKey(key);
    EXPECT_NE(key.buf, nullptr);
    EXPECT_NE(key.len, 0);
    aclDestroyScalarList(scalarList);
    GetThreadLocalContext().usePTAHash_ = usePTAHash;
}
TEST_F(OpCacheUt, AddParamToBufChar)
{
    bool usePTAHash = GetThreadLocalContext().usePTAHash_;
    GetThreadLocalContext().usePTAHash_ = false;
    char* apiName = "aclnnGelu";
    AddParamToBuf(apiName);
    const char* nullptrCharPtr = nullptr;
    AddParamToBuf(nullptrCharPtr);
    char* nullptrCharPtrWithoutConst = nullptr;
    AddParamToBuf(nullptrCharPtrWithoutConst);
    OpCacheKey key;
    SetOpCacheKey(key);
    EXPECT_NE(key.buf, nullptr);
    EXPECT_NE(key.len, 0);
    GetThreadLocalContext().usePTAHash_ = usePTAHash;
}

TEST_F(OpCacheUt, OpExecCacheRemove)
{
    const char_t* const cacheLimit = std::getenv("ACLNN_CACHE_LIMIT");
    setenv("ACLNN_CACHE_LIMIT", "100000", 1);
    GetThreadLocalContext().hashKey_ = 0;
    GetThreadLocalContext().cacheHashKey_ = (uint8_t*)"hello1";
    GetThreadLocalContext().cacheHashKeyLen_ = 6;
    auto opExecCache = new OpExecCache();
    bool succ = AddOpExecCache(opExecCache);
    opExecCache->OldCacheClear();
    EXPECT_EQ(succ, true);
    RemoveExecCache(opExecCache);
    delete opExecCache;

    GetThreadLocalContext().hashKey_ = 10;
    GetThreadLocalContext().cacheHashKey_ = nullptr;
    GetThreadLocalContext().cacheHashKeyLen_ = 0;
    auto opExecCache1 = new OpExecCache();
    succ = AddOpExecCache(opExecCache1);
    opExecCache1->OldCacheClear();
    EXPECT_EQ(succ, true);
    RemoveExecCache(opExecCache1);
    delete opExecCache1;

    RemoveExecCache(nullptr);
    setenv("ACLNN_CACHE_LIMIT", cacheLimit, 1);
}

TEST_F(OpCacheUt, OpExecCacheDeleteTest)
{
    auto old = GetThreadLocalContext().cacheHasFull_;
    GetThreadLocalContext().cacheHasFull_ = false;
    auto uniqueExecutor = CREATE_EXECUTOR();
    GetThreadLocalContext().cacheHasFull_ = old;
}

TEST_F(OpCacheUt, CacheLaunch1982Test2)
{
    setenv("ENABLE_1982", "1", 1);
    PlatformInfoStub::GetInstance()->SetSoCVersion("Ascend910_93", "Ascend910_9391");
    op::Shape outShape{100};
    int64_t inputData[8];
    aclIntArray self(inputData, 8);
    aclTensor out(outShape, op::DataType::DT_INT32, ge::FORMAT_ND, nullptr);
    aclTensor inputTensor(&self, op::DataType::DT_INT32);

    auto input_arg = OP_INPUT(&inputTensor);
    auto output_arg = OP_OUTPUT(&out);
    auto ctx = op::MakeOpArgContext(input_arg, output_arg);

    op::internal::ExpandableRtsArgBuffer buffer;
    buffer.Init(TEST_LAUNCH_ARG_INIT_CAP, TEST_TILING_HOST_DATA_INIT_CAP);
    op::internal::TilingData* tilingData = buffer.GetTilingDataPtr();
    constexpr size_t TEST_TILING_DATA_LEN = 100;
    tilingData->data_ = buffer.GetTilingDataAddr();
    memset_s(tilingData->data_, TEST_TILING_HOST_DATA_INIT_CAP, 0, TEST_TILING_DATA_LEN);
    tilingData->data_size_ = TEST_TILING_DATA_LEN;
    tilingData->capacity_ = TEST_TILING_HOST_DATA_INIT_CAP;

    op::internal::LaunchArgInfo argInfo(false, false, ctx);
    op::internal::RtsArg arg(true, argInfo, &buffer);
    arg.FillArgs();

    GetThreadLocalContext().hashKey_ = 0;
    GetThreadLocalContext().cacheHashKey_ = (uint8_t*)"hello";
    GetThreadLocalContext().cacheHashKeyLen_ = 5;
    auto opExecCache = new OpExecCache();
    opExecCache->SetCacheBuf(GetCacheBuf());
    GetOpCacheContext().SetOpCache(opExecCache);
    auto cacheCtx = arg.DumpToCache();
    EXPECT_NE(cacheCtx, nullptr);

    op::internal::KernelLaunchConfig launchCfg;
    launchCfg.funcHandle = (void*)0x12345678;
    launchCfg.numBlocks = 32;
    launchCfg.schemMode = 1;
    launchCfg.dynUBufSize = 0;
    launchCfg.blockDimOffset = 0;
    launchCfg.engineType = op::internal::LaunchKernelEngineType::NO_VECTOR_CORE;
    launchCfg.isFatBin = false;
    std::string kernelName = "abs";
    cacheCtx->SetRunParam(launchCfg, kernelName);

    aclrtStream stream = 0;
    auto res = op::internal::LaunchArgCache::RunFromCache(stream, GetCacheBuf());
    EXPECT_EQ(res, ACLNN_SUCCESS);
    op::DestroyOpArgContext(ctx);
    InitPTACacheThreadLocal();
    GetOpCacheContext().SetOpCache(nullptr);
    delete opExecCache;
    PlatformInfoStub::GetInstance()->Reset();
    unsetenv("ENABLE_1982");
}

TEST_F(OpCacheUt, OpCacheManagerCount)
{
    void* manager = op::internal::GetOpExecCacheManager();
    EXPECT_NE(manager, nullptr);
    op::internal::ReleaseOpExecCacheManager(manager);
    op::internal::ReleaseOpExecCacheManager(nullptr);
}

TEST_F(OpCacheUt, MurmurHashRemain)
{
    const int len = 15;
    uint8_t buf[len];
    memset_s(buf, len, 0, len);
    op::internal::OpCacheKey key(buf, len);
    op::internal::OpCacheKeyHash hasher;
    std::size_t hash = hasher(key);
    std::size_t exepectedHash = {9371257502052110381};
    EXPECT_EQ(exepectedHash, hash);
}

namespace op {
namespace internal {
extern std::atomic<bool> g_enableOpCacheCount;
}
} // namespace op

static void OpCacheUseCountTestFunc()
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    EXPECT_EQ(g_enableOpCacheCount.load(), true);
    DisableOpCacheCount();
    EXPECT_EQ(g_enableOpCacheCount.load(), false);
}

TEST_F(OpCacheUt, DisableOpCacheUseCountTest)
{
    std::thread t(OpCacheUseCountTestFunc);
    t.join();
}

// 复现 OpCacheKey 多线程竞态：主线程反复 GetOpExecCache 查找（经 operator== 读 exec->key_），
// 线程 B 反复重建 key_ 后 MarkOpCacheInvalid（写 exec->key_）。
// patch 前：MarkOpCacheInvalid 的 delete[] key_.buf; key_.buf=nullptr 与 operator== 读 key_ 并发，
// 中间态 buf 为空但 len 非 0，memcmp 解引用空指针崩溃；patch 后不崩。
TEST_F(OpCacheUt, OpCacheKeyRaceConditionTest)
{
    constexpr int64_t KEY_LEN = 7;
    auto keyBuf = reinterpret_cast<uint8_t*>(const_cast<char*>("racekey"));

    InitPTACacheThreadLocal();
    SetPTACacheHashKey(keyBuf, KEY_LEN);
    auto cache = new OpExecCache();
    cache->SetCacheBuf(GetCacheBuf());
    // 不调 SetUse：保持 canUse_ 为 false，使 MarkOpCacheInvalid 内 CanUse() 返回 false 从而真正执行失效
    ASSERT_TRUE(AddOpExecCache(cache));

    OpCacheKey findKey(keyBuf, KEY_LEN);

    std::atomic<bool> stop{false};
    // 线程 B：循环重建 key_ 后失效，持续制造 buf 有效变空指针的中间态
    std::thread threadB([&]() {
        InitPTACacheThreadLocal();
        SetPTACacheHashKey(keyBuf, KEY_LEN);
        while (!stop.load()) {
            cache->InitOpCacheKey();
            cache->MarkOpCacheInvalid();
        }
    });

    // 主线程：循环查找，经 operator== -> GetOpCacheKey 读 exec->key_
    auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
    while (std::chrono::steady_clock::now() < end) {
        GetOpExecCache(findKey);
    }
    stop.store(true);
    threadB.join();

    RemoveExecCache(cache);
    UnInitPTACacheThreadLocal();
}

// 复现 AddOpExecCache 中 cache2_.find 调 operator== 的多线程竞态（实际 coredump 场景）：
// 线程 A 反复 AddOpExecCache，其 cache2_.find 遍历到容器内 targetCache 调 operator== 读 exec->key_；
// 线程 B 反复 InitOpCacheKey + MarkOpCacheInvalid（真实失效流程），持续制造 buf 有效变 nullptr 的中间态。
// patch 后：operator== 防御 buf 为空/len 为 0 返回 false 不崩，且 OP_LOGW 日志可观测到竞态真实发生
// （日志打印 key len 为 0，说明 find 在 MarkOpCacheInvalid 失效 targetCache 期间被调用）。
// patch 前：该路径因 AddOpExecCache 每次 new OpExecCache 导致 find 频率较低，未稳定复现 SIGSEGV；
// 同源竞态在 OpCacheKeyRaceConditionTest（GetOpExecCache 高频 find）中已稳定复现崩溃。
// 注意：UT 默认 cacheLimit_ 为 1，需在新线程构造 manager 前通过环境变量调大，否则 cache 满直接返回走不到 find。
TEST_F(OpCacheUt, AddOpExecCacheRaceConditionTest)
{
    constexpr int64_t KEY_LEN = 7;
    auto keyBuf = reinterpret_cast<uint8_t*>(const_cast<char*>("racekey"));

    // 保存并调大 cacheLimit，新线程首次访问 manager 时按新值构造
    const char* const originLimit = std::getenv("ACLNN_CACHE_LIMIT");
    setenv("ACLNN_CACHE_LIMIT", "100", 1);

    std::atomic<OpExecCache*> targetCachePtr{nullptr};
    std::atomic<bool> stop{false};

    // 线程 A：构造大 cacheLimit 的 manager，预入 targetCache（指针共享给线程 B），再反复 AddOpExecCache 触发 find
    std::thread threadA([&]() {
        InitPTACacheThreadLocal();
        SetPTACacheHashKey(keyBuf, KEY_LEN);
        auto targetCache = new OpExecCache();
        targetCache->SetCacheBuf(GetCacheBuf());
        if (AddOpExecCache(targetCache)) {
            targetCachePtr.store(targetCache);
        } else {
            delete targetCache;
        }

        while (!stop.load()) {
            auto newCache = new OpExecCache();
            // AddOpExecCache 内 cache2_.find 遍历到 targetCache 调 operator== 读 exec->key_
            // 注意：AddOpExecCache 返回 false 时内部已 delete exec，这里不可重复 delete
            (void)AddOpExecCache(newCache);
        }
        // 不主动 RemoveExecCache：避免与线程 B 的 InitOpCacheKey 并发释放导致 double free，
        // targetCache 留在本线程 manager，线程结束时由 manager 析构统一清理（此时线程 B 已 join）
        UnInitPTACacheThreadLocal();
    });

    // 线程 B：等 targetCache 就绪后，反复重建 key_ 后真实失效，持续制造 buf 有效变空指针的中间态
    std::thread threadB([&]() {
        InitPTACacheThreadLocal();
        SetPTACacheHashKey(keyBuf, KEY_LEN);
        OpExecCache* cache = nullptr;
        while (!stop.load()) {
            if (cache == nullptr) {
                cache = targetCachePtr.load();
                continue;
            }
            cache->InitOpCacheKey();
            cache->MarkOpCacheInvalid();
        }
        UnInitPTACacheThreadLocal();
    });

    auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
    while (std::chrono::steady_clock::now() < end) {
        std::this_thread::yield();
    }
    stop.store(true);
    threadA.join();
    threadB.join();

    if (originLimit != nullptr) {
        setenv("ACLNN_CACHE_LIMIT", originLimit, 1);
    } else {
        unsetenv("ACLNN_CACHE_LIMIT");
    }
}
