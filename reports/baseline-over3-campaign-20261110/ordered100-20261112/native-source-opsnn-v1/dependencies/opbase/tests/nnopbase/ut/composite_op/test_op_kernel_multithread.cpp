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
#include <random>
#include <thread>
#include <pthread.h>

#include "acl/acl.h"
#include "acl/acl_base.h"
#include "acl/acl_rt.h"

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

using namespace op::internal;
using namespace op::internal::test;
using char_t = char;

typedef std::function<void(void)> Functional;

extern "C" void InitPTACacheThreadLocal();
extern "C" aclOpExecutor* PTAGetExecCache(uint64_t hash, uint64_t* workspaceSize);
extern "C" aclOpExecutor* PTAFindExecCache(uint8_t* buf, size_t len, uint64_t* workspaceSize);

class OpKernelMultiThreadUT : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        // aclInit(nullptr)
        setenv("ASCEND_OPP_PATH", OP_API_COMMON_UT_SRC_DIR, 1); // does overwrite
        GetThreadLocalContext().cacheHasFull_ = true;
        op::GenOpTypeId("QuantBatchMatmulV3");
    }

    static void TearDownTestCase() {}
};

static std::unique_ptr<OpKernelBin> CreateFakeOpKernelBin(bool genPlaceholder, bool hasDevPtrArg)
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

    std::unique_ptr<OpKernelBin> kernelBin = std::make_unique<OpKernelBin>(
        opType, jsonPath, jsonPath, binPath, key, hashKey, BinType::DYNAMIC_BIN, genPlaceholder, hasDevPtrArg);
    return std::move(kernelBin);
}

static bool CheckRtKernelLaunchCfgSchemMode(aclrtLaunchKernelCfg* cfg, uint8_t schemMode)
{
    for (size_t idx = 0U; idx < cfg->numAttrs; idx++) {
        if (cfg->attrs[idx].id == ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE) {
            EXPECT_EQ(cfg->attrs[idx].value.schemMode, schemMode);
            return true;
        }
    }
    return false;
}

static bool CheckRtKernelLaunchCfgLocalMemSize(aclrtLaunchKernelCfg* cfg, uint32_t dynUBufSize)
{
    for (size_t idx = 0U; idx < cfg->numAttrs; idx++) {
        if (cfg->attrs[idx].id == LAUNCH_KERNEL_ATTR_DYN_UBUF_SIZE) {
            EXPECT_EQ(cfg->attrs[idx].value.localMemorySize, dynUBufSize);
            return true;
        }
    }
    return false;
}

class MultiDoLaunchNormalTest2DumpStub : public Adx::DumpStub {
public:
    void* AdumpGetDFXInfoAddrForDynamic(uint32_t space, uint64_t& atomicIndex)
    {
        dfxInfoSpace_ = space;
        atomicIndex = 666;
        return (void*)dfxInfoMem_;
    }

    void CheckDFXInfoDump(const std::vector<uint64_t>& tensorSizeInfo, const std::vector<uint64_t>& shapeInfo)
    {
        std::cout << std::endl;
        EXPECT_EQ(dfxInfoSpace_, 46);
        EXPECT_EQ(tensorSizeInfo.size() + shapeInfo.size(), 46);
        for (size_t i = 0; i < tensorSizeInfo.size(); i++) {
            EXPECT_EQ(tensorSizeInfo[i], dfxInfoMem_[i]);
        }
        size_t shapeInfoOffset = tensorSizeInfo.size();
        for (size_t i = 0; i < shapeInfo.size(); i++) {
            EXPECT_EQ(shapeInfo[i], dfxInfoMem_[shapeInfoOffset + i]);
        }
        for (size_t i = 0; i < (tensorSizeInfo.size() + shapeInfo.size()); i++) {
            dfxInfoMem_[i] = 0;
        }
    }

    uint32_t dfxInfoSpace_{0};
    uint64_t dfxInfoMem_[1000] = {0};
};

class MultiDoLaunchNormalTest2AclrtStub : public AclrtStub {
public:
    aclError aclrtBinaryGetFunctionByEntry(aclrtBinHandle binHandle, uint64_t funcEntry, aclrtFuncHandle* funcHandle)
    {
        OP_LOGI("MultiDoLaunchNormalTest2AclrtStub aclrtBinaryGetFunctionByEntry start");
        EXPECT_EQ(binHandle, nullptr);
        EXPECT_EQ(funcEntry, 10020);
        *funcHandle = (void*)0x12341234;
        return ACL_SUCCESS;
    }

    aclError aclrtLaunchKernelWithHostArgs(aclrtFuncHandle funcHandle, uint32_t blockDim, aclrtStream stream,
                                           aclrtLaunchKernelCfg* cfg, void* hostArgs, size_t argsSize,
                                           aclrtPlaceHolderInfo* placeHolderArray, size_t placeHolderNum)
    {
        OP_LOGI("MultiDoLaunchNormalTest2AclrtStub aclrtLaunchKernelWithHostArgs start");
        // check rts params
        EXPECT_EQ(funcHandle, (void*)0x12341234);
        EXPECT_EQ(blockDim, 16);
        EXPECT_EQ(stream, nullptr);
        EXPECT_EQ(argsSize, 400);

        EXPECT_NE(cfg, nullptr);
        EXPECT_TRUE(CheckRtKernelLaunchCfgSchemMode(cfg, 1));
        EXPECT_TRUE(CheckRtKernelLaunchCfgLocalMemSize(cfg, 888));

        // check ptr
        void** ptrArgs = reinterpret_cast<void**>(hostArgs);
        EXPECT_EQ(ptrArgs[0], input1);
        EXPECT_EQ(ptrArgs[1], input2_1);
        EXPECT_EQ(ptrArgs[2], input2_2);
        EXPECT_EQ(ptrArgs[3], nullptr);
        EXPECT_EQ(ptrArgs[4], nullptr);

        EXPECT_EQ(ptrArgs[6], output1);
        EXPECT_EQ(ptrArgs[7], output2_1);
        EXPECT_EQ(ptrArgs[8], output2_2);

        EXPECT_EQ(ptrArgs[9], outShape);

        EXPECT_EQ(ptrArgs[10], workspace1);
        EXPECT_EQ(ptrArgs[11], workspace2);

        EXPECT_EQ(ptrArgs[13], overflow);

        // check host value
        EXPECT_NE(placeHolderArray, nullptr);
        EXPECT_EQ(placeHolderNum, 2);

        EXPECT_EQ(placeHolderArray[0].addrOffset, 40);
        EXPECT_EQ(placeHolderArray[0].dataOffset, 38 * 8);
        int64_t* hostVal = reinterpret_cast<int64_t*>(reinterpret_cast<char*>(hostArgs) + 38 * 8);
        for (int64_t i = 0; i < 10; i++) {
            EXPECT_EQ(hostVal[i], 1);
        }

        // check tiling data
        EXPECT_EQ(placeHolderArray[1].addrOffset, 96);
        EXPECT_EQ(placeHolderArray[1].dataOffset, 112);
        int64_t* tilingVal = reinterpret_cast<int64_t*>(reinterpret_cast<char*>(hostArgs) + 112);
        for (int64_t i = 0; i < 21; i++) {
            EXPECT_EQ(tilingVal[i], i);
        }

        return ACL_SUCCESS;
    }

    void* input1;
    void* input2_1;
    void* input2_2;
    void* output1;
    void* output2_1;
    void* output2_2;
    void* outShape;
    void* workspace1;
    void* workspace2;
    void* overflow;
};

class MultiDoLaunchNormalTest2CacheAclrtStub : public AclrtStub {
public:
    aclError aclrtLaunchKernelWithHostArgs(aclrtFuncHandle funcHandle, uint32_t blockDim, aclrtStream stream,
                                           aclrtLaunchKernelCfg* cfg, void* hostArgs, size_t argsSize,
                                           aclrtPlaceHolderInfo* placeHolderArray, size_t placeHolderNum)
    {
        OP_LOGI("MultiDoLaunchNormalTest2CacheAclrtStub aclrtLaunchKernelWithHostArgs start");
        // check rts params
        EXPECT_EQ(funcHandle, (void*)0x12341234);
        EXPECT_EQ(blockDim, 16);
        EXPECT_EQ(stream, nullptr);
        EXPECT_EQ(argsSize, 400);

        EXPECT_NE(cfg, nullptr);
        EXPECT_TRUE(CheckRtKernelLaunchCfgSchemMode(cfg, 1));
        EXPECT_TRUE(CheckRtKernelLaunchCfgLocalMemSize(cfg, 888));

        // check ptr
        void** ptrArgs = reinterpret_cast<void**>(hostArgs);
        EXPECT_EQ(ptrArgs[0], input1);
        EXPECT_EQ(ptrArgs[1], input2_1);
        EXPECT_EQ(ptrArgs[2], input2_2);
        EXPECT_EQ(ptrArgs[3], nullptr);
        EXPECT_EQ(ptrArgs[4], nullptr);

        EXPECT_EQ(ptrArgs[6], output1);
        EXPECT_EQ(ptrArgs[7], output2_1);
        EXPECT_EQ(ptrArgs[8], output2_2);

        EXPECT_EQ(ptrArgs[9], outShape);

        EXPECT_EQ(ptrArgs[10], workspace1);
        EXPECT_EQ(ptrArgs[11], workspace2);

        EXPECT_EQ(ptrArgs[13], overflow);

        // check host value
        EXPECT_NE(placeHolderArray, nullptr);
        EXPECT_EQ(placeHolderNum, 2);

        EXPECT_EQ(placeHolderArray[0].addrOffset, 40);
        EXPECT_EQ(placeHolderArray[0].dataOffset, 112);
        int64_t* hostVal = reinterpret_cast<int64_t*>(reinterpret_cast<char*>(hostArgs) + 112);
        for (int64_t i = 0; i < 10; i++) {
            EXPECT_EQ(hostVal[i], 1);
        }

        // check tiling data
        EXPECT_EQ(placeHolderArray[1].addrOffset, 96);
        EXPECT_EQ(placeHolderArray[1].dataOffset, 208);
        int64_t* tilingVal = reinterpret_cast<int64_t*>(reinterpret_cast<char*>(hostArgs) + 208);
        for (int64_t i = 0; i < 21; i++) {
            EXPECT_EQ(tilingVal[i], i);
        }

        return ACL_SUCCESS;
    }

    void* input1;
    void* input2_1;
    void* input2_2;
    void* output1;
    void* output2_1;
    void* output2_2;
    void* outShape;
    void* workspace1;
    void* workspace2;
    void* overflow;
};

// MultiDoLaunchNormalTest2: genPlaceholder_=true, hasDevPtrArg_=false;
static void MultiDoLaunchNormalTest2()
{
    OP_LOGI("Start to run UT MultiDoLaunchNormalTest2 -->");
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
    snprintf_s(
        jsonPath, sizeof(jsonPath), sizeof(jsonPath),
        "%s/built-in/ /ai_core/tbe/kernel/ascend910/axpy/Axpy_233851a3505389e43928a8bba133a74d_high_performance.json",
        p);
    snprintf_s(binPath, sizeof(binPath), sizeof(binPath),
               "%s/built-in/op_impl/ai_core/tbe/kernel/ascend910/axpy/"
               "Axpy_233851a3505389e43928a8bba133a74d_high_performance.o",
               p);
    uint32_t opType = op::OpTypeDict::ToOpType("QuantBatchMatmulV3");
    OpKernelBin kernelBin(opType, jsonPath, jsonPath, binPath, key, hashKey, BinType::DYNAMIC_BIN, false, false);
    kernelBin.interCoreSync_ = false;
    kernelBin.genPlaceholder_ = true;
    kernelBin.hasDevPtrArg_ = false;
    kernelBin.isFatbin_ = true;
    kernelBin.currDevId_ = 0;
    auto f = [](void*& hdl) -> aclnnStatus {
        hdl = nullptr;
        return ACLNN_SUCCESS;
    };
    kernelBin.binHandle_[0].InitVar(f);

    // create input output workspace outputshape
    op::Shape selfShape{33, 15, 64};
    op::Shape outShape{33, 15, 64};
    op::Shape idxShape{33, 15, 64};
    op::Shape wsShape{32};

    // input
    int* ptr1 = new int;
    aclTensor tensor1(selfShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND, ptr1);
    tensor1.SetFromWorkspace(false);
    int* ptr2 = new int;
    auto tensor2 = std::make_unique<aclTensor>(selfShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND, ptr2);
    int* ptr3 = new int;
    auto tensor3 = std::make_unique<aclTensor>(selfShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND, ptr3);
    aclTensor* tensor4 = nullptr;
    std::vector<int64_t> hostValue(10, 1);
    aclIntArray intArray(hostValue.data(), hostValue.size());
    aclTensor hostTensor(&intArray, op::DataType::DT_INT64);

    aclTensor* input1 = &tensor1;
    aclTensor* tensor2Ptr = tensor2.release();
    tensor2Ptr->SetFromWorkspace(false);
    aclTensor* tensor3Ptr = tensor3.release();
    tensor3Ptr->SetFromWorkspace(false);

    const aclTensor* inputTensors[3] = {tensor2Ptr, tensor3Ptr, tensor4};
    aclTensorList* input2 = aclCreateTensorList(inputTensors, 3);
    aclTensor* input3 = tensor4;
    aclTensorList* input4 = nullptr;
    aclTensor* input5 = &hostTensor;

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

    auto input = OP_INPUT(input1, input2, input3, input4, input5);
    auto output = OP_OUTPUT(output1, output2, output3, output4);
    auto outshape = OP_OUTSHAPE(outshapeTensor, 0);

    auto ctx = op::MakeOpArgContext(input, output, outshape);
    ctx->AppendOpWorkspaceArg(workspace1);

    // init cache
    InitExecutorCacheThreadLocal();
    auto in = std::make_tuple(input1, input2, input3, input4, input5);
    auto out = std::make_tuple(output1, output2, output3, output4, outshapeTensor);
    CalculateHashKey(in);
    CalculateHashKey(out);

    // init tiling res
    TilingCtxOutput tilingOutput{};

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

    uint64_t tilingKey = 10020;
    int64_t numBlocks = 16;
    uint8_t scheduleMode = 1;
    uint32_t dynUBufSize = 888;

    tilingOutput.tilingKey_ = &tilingKey;
    tilingOutput.numBlocks_ = &numBlocks;
    tilingOutput.tilingData_ = tilingData;
    tilingOutput.scheduleMode_ = &scheduleMode;
    tilingOutput.dynUBufSize_ = &dynUBufSize;
    tilingOutput.rtsArgBuffer_ = &buffer;
    buffer.RegisterOutputTilingDataPtr(&tilingOutput.tilingData_);

    // run launch
    MultiDoLaunchNormalTest2AclrtStub aclrtStub;
    aclrtStub.input1 = ptr1;
    aclrtStub.input2_1 = ptr2;
    aclrtStub.input2_2 = ptr3;
    aclrtStub.output1 = ptr1;
    aclrtStub.output2_1 = ptr2;
    aclrtStub.output2_2 = ptr3;
    aclrtStub.outShape = ptr1;
    aclrtStub.workspace1 = workspacePtr;
    aclrtStub.workspace2 = reinterpret_cast<char*>(workspacePtr) + 512;
    aclrtStub.overflow = (void*)0x005;

    AclrtStub::GetInstance()->Install(&aclrtStub);
    MultiDoLaunchNormalTest2DumpStub dumpStub;
    Adx::DumpStub::GetInstance()->Install(&dumpStub);

    std::vector<int32_t> tensorOffset;
    auto opExecCache = new OpExecCache();
    opExecCache->hashKey_ = 1;
    opExecCache->SetCacheBuf(GetCacheBuf());
    GetOpCacheContext().SetOpCache(opExecCache);
    auto rc = kernelBin.DoLaunch(&tilingOutput, stream, false, ctx, tensorOffset);

    for (int32_t i = 0; i < 12; i++) {
        EXPECT_EQ(tensorOffset[i], i);
    }

    const std::vector<uint64_t> tensorSizeInfo = {63360, 63360, 63360, 0,     0,     96,
                                                  63360, 63360, 63360, 63360, 63360, 63360};
    const std::vector<uint64_t> shapeInfo = {3,  33, 15, 64, 3,  33, 15, 64, 3,  33, 15, 64, 1,  10, 3,  33, 15,
                                             64, 3,  33, 15, 64, 3,  33, 15, 64, 3,  33, 15, 64, 3,  33, 15, 64};
    dumpStub.CheckDFXInfoDump(tensorSizeInfo, shapeInfo);

    AclrtStub::GetInstance()->UnInstall();

    MultiDoLaunchNormalTest2CacheAclrtStub cacheAclrtStub;
    cacheAclrtStub.input1 = ptr1;
    cacheAclrtStub.input2_1 = ptr2;
    cacheAclrtStub.input2_2 = ptr3;
    cacheAclrtStub.output1 = ptr1;
    cacheAclrtStub.output2_1 = ptr2;
    cacheAclrtStub.output2_2 = ptr3;
    cacheAclrtStub.outShape = ptr1;
    cacheAclrtStub.workspace1 = workspacePtr;
    cacheAclrtStub.workspace2 = reinterpret_cast<char*>(workspacePtr) + 512;
    cacheAclrtStub.overflow = (void*)0x005;
    AclrtStub::GetInstance()->Install(&cacheAclrtStub);

    OpExecCacheWrap* cacheWrap = CreateCacheWrap(opExecCache);
    cacheWrap->Run(workspacePtr, stream);

    AclrtStub::GetInstance()->UnInstall();
    Adx::DumpStub::GetInstance()->UnInstall();

    // tear down
    delete ptr1;
    delete ptr2;
    delete ptr3;
    delete workspacePtr;
    delete opExecCache;
    delete cacheWrap;
    aclDestroyTensorList(input2);
    aclDestroyTensorList(workspace1);
}

class MultiDoLaunchNormalTest3DumpStub : public Adx::DumpStub {
public:
    void* AdumpGetDFXInfoAddrForDynamic(uint32_t space, uint64_t& atomicIndex)
    {
        dfxInfoSpace_ = space;
        atomicIndex = 666;
        return (void*)dfxInfoMem_;
    }

    void CheckDFXInfoDump(const std::vector<uint64_t>& tensorSizeInfo, const std::vector<uint64_t>& shapeInfo)
    {
        std::cout << std::endl;
        EXPECT_EQ(dfxInfoSpace_, 26);
        EXPECT_EQ(tensorSizeInfo.size() + shapeInfo.size(), 26);
        for (size_t i = 0; i < tensorSizeInfo.size(); i++) {
            EXPECT_EQ(tensorSizeInfo[i], dfxInfoMem_[i]);
        }
        size_t shapeInfoOffset = tensorSizeInfo.size();
        for (size_t i = 0; i < shapeInfo.size(); i++) {
            EXPECT_EQ(shapeInfo[i], dfxInfoMem_[shapeInfoOffset + i]);
        }
        for (size_t i = 0; i < (tensorSizeInfo.size() + shapeInfo.size()); i++) {
            dfxInfoMem_[i] = 0;
        }
    }

    uint32_t dfxInfoSpace_{0};
    uint64_t dfxInfoMem_[1000] = {0};
};

class MultiDoLaunchNormalTest3AclrtStub : public AclrtStub {
public:
    aclError aclrtBinaryGetFunctionByEntry(aclrtBinHandle binHandle, uint64_t funcEntry, aclrtFuncHandle* funcHandle)
    {
        OP_LOGI("MultiDoLaunchNormalTest3AclrtStub aclrtBinaryGetFunctionByEntry start");
        EXPECT_EQ(binHandle, nullptr);
        EXPECT_EQ(funcEntry, 10020);
        *funcHandle = (void*)0x12341234;
        return ACL_SUCCESS;
    }

    aclError aclrtLaunchKernelWithHostArgs(aclrtFuncHandle funcHandle, uint32_t blockDim, aclrtStream stream,
                                           aclrtLaunchKernelCfg* cfg, void* hostArgs, size_t argsSize,
                                           aclrtPlaceHolderInfo* placeHolderArray, size_t placeHolderNum)
    {
        OP_LOGI("MultiDoLaunchNormalTest3AclrtStub aclrtLaunchKernelWithHostArgs start");
        // check rts params
        EXPECT_EQ(funcHandle, (void*)0x12341234);
        EXPECT_EQ(blockDim, 16);
        EXPECT_EQ(stream, nullptr);
        EXPECT_EQ(argsSize, 624);

        EXPECT_NE(cfg, nullptr);
        EXPECT_TRUE(CheckRtKernelLaunchCfgSchemMode(cfg, 1));
        EXPECT_TRUE(CheckRtKernelLaunchCfgLocalMemSize(cfg, 888));

        // check ptr
        void** ptrArgs = reinterpret_cast<void**>(hostArgs);
        EXPECT_EQ(ptrArgs[0], input1);

        EXPECT_EQ(ptrArgs[3], output1);

        EXPECT_EQ(ptrArgs[5], outShape);

        EXPECT_EQ(ptrArgs[6], workspace1);
        EXPECT_EQ(ptrArgs[7], workspace2);

        EXPECT_EQ(ptrArgs[9], overflow);

        // check host value
        EXPECT_NE(placeHolderArray, nullptr);
        EXPECT_EQ(placeHolderNum, 4);

        EXPECT_EQ(placeHolderArray[0].addrOffset, 8);
        EXPECT_EQ(placeHolderArray[0].dataOffset, 272);
        ptrArgs = reinterpret_cast<void**>(hostArgs + 272);
        int64_t shapeOffset = reinterpret_cast<int64_t>(ptrArgs[0]) / sizeof(void*);
        EXPECT_EQ(ptrArgs[shapeOffset++], input2_1);
        EXPECT_EQ(ptrArgs[shapeOffset++], input2_2);
        EXPECT_EQ(ptrArgs[shapeOffset], nullptr);

        EXPECT_EQ(placeHolderArray[1].addrOffset, 16);
        EXPECT_EQ(placeHolderArray[1].dataOffset, 400);
        int64_t* hostVal = reinterpret_cast<int64_t*>(reinterpret_cast<char*>(hostArgs) + 400);
        for (int64_t i = 0; i < 10; i++) {
            EXPECT_EQ(hostVal[i], 1);
        }

        EXPECT_EQ(placeHolderArray[2].addrOffset, 32);
        EXPECT_EQ(placeHolderArray[2].dataOffset, 496);
        ptrArgs = reinterpret_cast<void**>(hostArgs + 496);
        shapeOffset = reinterpret_cast<int64_t>(ptrArgs[0]) / sizeof(void*);
        EXPECT_EQ(ptrArgs[shapeOffset++], output2_1);
        EXPECT_EQ(ptrArgs[shapeOffset++], output2_2);
        EXPECT_EQ(ptrArgs[shapeOffset], nullptr);

        // check tiling data
        EXPECT_EQ(placeHolderArray[3].addrOffset, 64);
        EXPECT_EQ(placeHolderArray[3].dataOffset, 80);
        int64_t* tilingVal = reinterpret_cast<int64_t*>(reinterpret_cast<char*>(hostArgs) + 80);
        for (int64_t i = 0; i < 21; i++) {
            EXPECT_EQ(tilingVal[i], i);
        }

        return ACL_SUCCESS;
    }

    void* input1;
    void* input2_1;
    void* input2_2;
    void* output1;
    void* output2_1;
    void* output2_2;
    void* outShape;
    void* workspace1;
    void* workspace2;
    void* overflow;
};

class MultiDoLaunchNormalTest3CacheAclrtStub : public AclrtStub {
public:
    aclError aclrtLaunchKernelWithHostArgs(aclrtFuncHandle funcHandle, uint32_t blockDim, aclrtStream stream,
                                           aclrtLaunchKernelCfg* cfg, void* hostArgs, size_t argsSize,
                                           aclrtPlaceHolderInfo* placeHolderArray, size_t placeHolderNum)
    {
        OP_LOGI("MultiDoLaunchNormalTest3CacheRuntimeStub aclrtLaunchKernelWithHostArgs start");
        // check rts params
        EXPECT_EQ(funcHandle, (void*)0x12341234);
        EXPECT_EQ(blockDim, 16);
        EXPECT_EQ(stream, nullptr);
        EXPECT_EQ(argsSize, 624);

        EXPECT_NE(cfg, nullptr);
        EXPECT_TRUE(CheckRtKernelLaunchCfgSchemMode(cfg, 1));
        EXPECT_TRUE(CheckRtKernelLaunchCfgLocalMemSize(cfg, 888));

        // check ptr
        void** ptrArgs = reinterpret_cast<void**>(hostArgs);
        EXPECT_EQ(ptrArgs[0], input1);

        EXPECT_EQ(ptrArgs[3], output1);

        EXPECT_EQ(ptrArgs[5], outShape);

        EXPECT_EQ(ptrArgs[6], workspace1);
        EXPECT_EQ(ptrArgs[7], workspace2);

        EXPECT_EQ(ptrArgs[9], overflow);

        // check host value
        EXPECT_NE(placeHolderArray, nullptr);
        EXPECT_EQ(placeHolderNum, 4);

        EXPECT_EQ(placeHolderArray[0].addrOffset, 8);
        EXPECT_EQ(placeHolderArray[0].dataOffset, 80);
        ptrArgs = reinterpret_cast<void**>(hostArgs + 80);
        int64_t shapeOffset = reinterpret_cast<int64_t>(ptrArgs[0]) / sizeof(void*);
        EXPECT_EQ(ptrArgs[shapeOffset++], input2_1);
        EXPECT_EQ(ptrArgs[shapeOffset++], input2_2);
        EXPECT_EQ(ptrArgs[shapeOffset], nullptr);

        EXPECT_EQ(placeHolderArray[1].addrOffset, 16);
        EXPECT_EQ(placeHolderArray[1].dataOffset, 208);
        int64_t* hostVal = reinterpret_cast<int64_t*>(reinterpret_cast<char*>(hostArgs) + 208);
        for (int64_t i = 0; i < 10; i++) {
            EXPECT_EQ(hostVal[i], 1);
        }

        EXPECT_EQ(placeHolderArray[2].addrOffset, 32);
        EXPECT_EQ(placeHolderArray[2].dataOffset, 304);
        ptrArgs = reinterpret_cast<void**>((hostArgs) + 304);
        shapeOffset = reinterpret_cast<int64_t>(ptrArgs[0]) / sizeof(void*);
        EXPECT_EQ(ptrArgs[shapeOffset++], output2_1);
        EXPECT_EQ(ptrArgs[shapeOffset++], output2_2);
        EXPECT_EQ(ptrArgs[shapeOffset], nullptr);

        // check tiling data
        EXPECT_EQ(placeHolderArray[3].addrOffset, 64);
        EXPECT_EQ(placeHolderArray[3].dataOffset, 432);
        int64_t* tilingVal = reinterpret_cast<int64_t*>(reinterpret_cast<char*>(hostArgs) + 432);
        for (int64_t i = 0; i < 21; i++) {
            EXPECT_EQ(tilingVal[i], i);
        }

        return ACL_SUCCESS;
    }

    void* input1;
    void* input2_1;
    void* input2_2;
    void* output1;
    void* output2_1;
    void* output2_2;
    void* outShape;
    void* workspace1;
    void* workspace2;
    void* overflow;
};

// MultiDoLaunchNormalTest3: genPlaceholder_=false, hasDevPtrArg_=true;
static void MultiDoLaunchNormalTest3()
{
    OP_LOGI("Start to run UT MultiDoLaunchNormalTest3 -->");
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
    uint32_t opType = op::OpTypeDict::ToOpType("QuantBatchMatmulV3");
    OpKernelBin kernelBin(opType, jsonPath, jsonPath, binPath, key, hashKey, BinType::DYNAMIC_BIN, false, false);
    kernelBin.interCoreSync_ = false;
    kernelBin.genPlaceholder_ = false;
    kernelBin.hasDevPtrArg_ = true;
    kernelBin.isFatbin_ = true;
    kernelBin.currDevId_ = 0;
    auto f = [](void*& hdl) -> aclnnStatus {
        hdl = nullptr;
        return ACLNN_SUCCESS;
    };
    kernelBin.binHandle_[0].InitVar(f);

    // create input output workspace outputshape
    op::Shape selfShape{33, 15, 64};
    op::Shape outShape{33, 15, 64};
    op::Shape idxShape{33, 15, 64};
    op::Shape wsShape{32};

    // input
    int* ptr1 = new int;
    aclTensor tensor1(selfShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND, ptr1);
    tensor1.SetFromWorkspace(false);
    int* ptr2 = new int;
    auto tensor2 = std::make_unique<aclTensor>(selfShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND, ptr2);
    int* ptr3 = new int;
    auto tensor3 = std::make_unique<aclTensor>(selfShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND, ptr3);
    aclTensor* tensor4 = nullptr;
    std::vector<int64_t> hostValue(10, 1);
    aclIntArray intArray(hostValue.data(), hostValue.size());
    aclTensor hostTensor(&intArray, op::DataType::DT_INT64);

    aclTensor* input1 = &tensor1;
    aclTensor* tensor2Ptr = tensor2.release();
    tensor2Ptr->SetFromWorkspace(false);
    aclTensor* tensor3Ptr = tensor3.release();
    tensor3Ptr->SetFromWorkspace(false);

    const aclTensor* inputTensors[3] = {tensor2Ptr, tensor3Ptr, tensor4};
    aclTensorList* input2 = aclCreateTensorList(inputTensors, 3);
    aclTensor* input3 = tensor4;
    aclTensorList* input4 = nullptr;
    aclTensor* input5 = &hostTensor;

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

    auto input = OP_INPUT(input1, input2, input3, input4, input5);
    auto output = OP_OUTPUT(output1, output2, output3, output4);
    auto outshape = OP_OUTSHAPE(outshapeTensor, 0);

    auto ctx = op::MakeOpArgContext(input, output, outshape);
    ctx->AppendOpWorkspaceArg(workspace1);

    // init cache
    InitExecutorCacheThreadLocal();
    auto in = std::make_tuple(input1, input2, input3, input4, input5);
    auto out = std::make_tuple(output1, output2, output3, output4, outshapeTensor);
    CalculateHashKey(in);
    CalculateHashKey(out);

    // init tiling res
    TilingCtxOutput tilingOutput{};

    op::internal::ExpandableRtsArgBuffer buffer;
    // launchArgCap=32 < 64 触发 launch arg 扩容
    // tilingHostDataCap=200: tilingData(168) 不扩容，AppendDevicePtrArg 阶段(168+104=272>200) 触发扩容
    buffer.Init(32, 200);
    int64_t* tilingDataData = static_cast<int64_t*>(buffer.GetTilingDataAddr());
    for (size_t i = 0; i < 21; i++) {
        tilingDataData[i] = i;
    }
    op::internal::TilingData* tilingData = buffer.GetTilingDataPtr();
    tilingData->data_ = buffer.GetTilingDataAddr();
    tilingData->data_size_ = sizeof(size_t) * 21;
    tilingData->capacity_ = TEST_TILING_HOST_DATA_INIT_CAP;

    uint64_t tilingKey = 10020;
    int64_t numBlocks = 16;
    uint8_t scheduleMode = 1;
    uint32_t dynUBufSize = 888;

    tilingOutput.tilingKey_ = &tilingKey;
    tilingOutput.numBlocks_ = &numBlocks;
    tilingOutput.tilingData_ = tilingData;
    tilingOutput.scheduleMode_ = &scheduleMode;
    tilingOutput.dynUBufSize_ = &dynUBufSize;
    tilingOutput.rtsArgBuffer_ = &buffer;
    buffer.RegisterOutputTilingDataPtr(&tilingOutput.tilingData_);

    // run launch
    MultiDoLaunchNormalTest3AclrtStub aclrtStub;
    aclrtStub.input1 = ptr1;
    aclrtStub.input2_1 = ptr2;
    aclrtStub.input2_2 = ptr3;
    aclrtStub.output1 = ptr1;
    aclrtStub.output2_1 = ptr2;
    aclrtStub.output2_2 = ptr3;
    aclrtStub.outShape = ptr1;
    aclrtStub.workspace1 = workspacePtr;
    aclrtStub.workspace2 = reinterpret_cast<char*>(workspacePtr) + 512;
    aclrtStub.overflow = (void*)0x005;

    AclrtStub::GetInstance()->Install(&aclrtStub);
    MultiDoLaunchNormalTest3DumpStub dumpStub;
    Adx::DumpStub::GetInstance()->Install(&dumpStub);

    std::vector<int32_t> tensorOffset;
    auto opExecCache = new OpExecCache();
    opExecCache->hashKey_ = 1;
    opExecCache->SetCacheBuf(GetCacheBuf());
    GetOpCacheContext().SetOpCache(opExecCache);
    auto rc = kernelBin.DoLaunch(&tilingOutput, stream, false, ctx, tensorOffset);

    EXPECT_EQ(tensorOffset[0], 0);
    EXPECT_EQ(tensorOffset[1], 44);
    EXPECT_EQ(tensorOffset[2], 45);
    EXPECT_EQ(tensorOffset[3], 46);
    EXPECT_EQ(tensorOffset[4], 2);
    EXPECT_EQ(tensorOffset[5], 3);
    EXPECT_EQ(tensorOffset[6], 72);
    EXPECT_EQ(tensorOffset[7], 73);
    EXPECT_EQ(tensorOffset[8], 74);
    EXPECT_EQ(tensorOffset[9], 5);
    EXPECT_EQ(tensorOffset[10], 6);
    EXPECT_EQ(tensorOffset[11], 7);

    const std::vector<uint64_t> tensorSizeInfo = {63360, 80, 96, 63360, 80, 63360, 63360, 63360};
    const std::vector<uint64_t> shapeInfo = {3, 33, 15, 64, 1, 10, 3, 33, 15, 64, 3, 33, 15, 64, 3, 33, 15, 64};
    dumpStub.CheckDFXInfoDump(tensorSizeInfo, shapeInfo);

    AclrtStub::GetInstance()->UnInstall();

    MultiDoLaunchNormalTest3CacheAclrtStub cacheAclrtStub;
    cacheAclrtStub.input1 = ptr1;
    cacheAclrtStub.input2_1 = ptr2;
    cacheAclrtStub.input2_2 = ptr3;
    cacheAclrtStub.output1 = ptr1;
    cacheAclrtStub.output2_1 = ptr2;
    cacheAclrtStub.output2_2 = ptr3;
    cacheAclrtStub.outShape = ptr1;
    cacheAclrtStub.workspace1 = workspacePtr;
    cacheAclrtStub.workspace2 = reinterpret_cast<char*>(workspacePtr) + 512;
    cacheAclrtStub.overflow = (void*)0x005;
    AclrtStub::GetInstance()->Install(&cacheAclrtStub);

    OpExecCacheWrap* cacheWrap = CreateCacheWrap(opExecCache);
    cacheWrap->Run(workspacePtr, stream);

    AclrtStub::GetInstance()->UnInstall();
    Adx::DumpStub::GetInstance()->UnInstall();

    // tear down
    delete ptr1;
    delete ptr2;
    delete ptr3;
    delete workspacePtr;
    delete opExecCache;
    delete cacheWrap;
    aclDestroyTensorList(input2);
    aclDestroyTensorList(workspace1);
}

class MultiDoLaunchNormalTest4DumpStub : public Adx::DumpStub {
public:
    void* AdumpGetDFXInfoAddrForDynamic(uint32_t space, uint64_t& atomicIndex)
    {
        dfxInfoSpace_ = space;
        atomicIndex = 666;
        return (void*)dfxInfoMem_;
    }

    void CheckDFXInfoDump(const std::vector<uint64_t>& tensorSizeInfo, const std::vector<uint64_t>& shapeInfo)
    {
        std::cout << std::endl;
        EXPECT_EQ(dfxInfoSpace_, 27);
        EXPECT_EQ(tensorSizeInfo.size() + shapeInfo.size(), 27);
        for (size_t i = 0; i < tensorSizeInfo.size(); i++) {
            EXPECT_EQ(tensorSizeInfo[i], dfxInfoMem_[i]);
        }
        size_t shapeInfoOffset = tensorSizeInfo.size();
        for (size_t i = 0; i < shapeInfo.size(); i++) {
            EXPECT_EQ(shapeInfo[i], dfxInfoMem_[shapeInfoOffset + i]);
        }
        for (size_t i = 0; i < (tensorSizeInfo.size() + shapeInfo.size()); i++) {
            dfxInfoMem_[i] = 0;
        }
    }

    uint32_t dfxInfoSpace_{0};
    uint64_t dfxInfoMem_[1000] = {0};
};

class MultiDoLaunchNormalTest4AclrtStub : public AclrtStub {
public:
    aclError aclrtBinaryGetFunctionByEntry(aclrtBinHandle binHandle, uint64_t funcEntry, aclrtFuncHandle* funcHandle)
    {
        OP_LOGI("MultiDoLaunchNormalTest4AclrtStub aclrtBinaryGetFunctionByEntry start");
        EXPECT_EQ(binHandle, nullptr);
        EXPECT_EQ(funcEntry, 10020);
        *funcHandle = (void*)0x12341234;
        return ACL_SUCCESS;
    }

    aclError aclrtLaunchKernelWithHostArgs(aclrtFuncHandle funcHandle, uint32_t blockDim, aclrtStream stream,
                                           aclrtLaunchKernelCfg* cfg, void* hostArgs, size_t argsSize,
                                           aclrtPlaceHolderInfo* placeHolderArray, size_t placeHolderNum)
    {
        OP_LOGI("MultiDoLaunchNormalTest4AclrtStub aclrtLaunchKernelWithHostArgs start");
        // check rts params
        EXPECT_EQ(funcHandle, (void*)0x12341234);
        EXPECT_EQ(blockDim, 16);
        EXPECT_EQ(stream, nullptr);
        EXPECT_EQ(argsSize, 632);

        EXPECT_NE(cfg, nullptr);
        EXPECT_TRUE(CheckRtKernelLaunchCfgSchemMode(cfg, 1));
        EXPECT_TRUE(CheckRtKernelLaunchCfgLocalMemSize(cfg, 888));

        // check ptr
        void** ptrArgs = reinterpret_cast<void**>(hostArgs);
        EXPECT_EQ(ptrArgs[0], input1);
        EXPECT_EQ(ptrArgs[2], nullptr);

        EXPECT_EQ(ptrArgs[4], output1);

        EXPECT_EQ(ptrArgs[6], outShape);

        EXPECT_EQ(ptrArgs[7], workspace1);
        EXPECT_EQ(ptrArgs[8], workspace2);

        EXPECT_EQ(ptrArgs[10], overflow);

        // check host value
        EXPECT_NE(placeHolderArray, nullptr);
        EXPECT_EQ(placeHolderNum, 4);

        EXPECT_EQ(placeHolderArray[0].addrOffset, 8);
        EXPECT_EQ(placeHolderArray[0].dataOffset, 280);
        ptrArgs = reinterpret_cast<void**>(hostArgs + 280);
        int64_t shapeOffset = reinterpret_cast<int64_t>(ptrArgs[0]) / sizeof(void*);
        EXPECT_EQ(ptrArgs[shapeOffset++], input2_1);
        EXPECT_EQ(ptrArgs[shapeOffset++], input2_2);
        EXPECT_EQ(ptrArgs[shapeOffset], nullptr);

        EXPECT_EQ(placeHolderArray[1].addrOffset, 24);
        EXPECT_EQ(placeHolderArray[1].dataOffset, 408);
        int64_t* hostVal = reinterpret_cast<int64_t*>(reinterpret_cast<char*>(hostArgs) + 408);
        for (int64_t i = 0; i < 10; i++) {
            EXPECT_EQ(hostVal[i], 1);
        }

        EXPECT_EQ(placeHolderArray[2].addrOffset, 40);
        EXPECT_EQ(placeHolderArray[2].dataOffset, 504);
        ptrArgs = reinterpret_cast<void**>(hostArgs + 504);
        shapeOffset = reinterpret_cast<int64_t>(ptrArgs[0]) / sizeof(void*);
        EXPECT_EQ(ptrArgs[shapeOffset++], output2_1);
        EXPECT_EQ(ptrArgs[shapeOffset++], output2_2);
        EXPECT_EQ(ptrArgs[shapeOffset], nullptr);

        // check tiling data
        EXPECT_EQ(placeHolderArray[3].addrOffset, 72);
        EXPECT_EQ(placeHolderArray[3].dataOffset, 88);
        int64_t* tilingVal = reinterpret_cast<int64_t*>(reinterpret_cast<char*>(hostArgs) + 88);
        for (int64_t i = 0; i < 21; i++) {
            EXPECT_EQ(tilingVal[i], i);
        }

        return ACL_SUCCESS;
    }

    void* input1;
    void* input2_1;
    void* input2_2;
    void* output1;
    void* output2_1;
    void* output2_2;
    void* outShape;
    void* workspace1;
    void* workspace2;
    void* overflow;
};

class MultiDoLaunchNormalTest4CacheAclrtStub : public AclrtStub {
public:
    aclError aclrtLaunchKernelWithHostArgs(aclrtFuncHandle funcHandle, uint32_t blockDim, aclrtStream stream,
                                           aclrtLaunchKernelCfg* cfg, void* hostArgs, size_t argsSize,
                                           aclrtPlaceHolderInfo* placeHolderArray, size_t placeHolderNum)
    {
        OP_LOGI("MultiDoLaunchAlignTestCacheAclrtStub aclrtLaunchKernelWithHostArgs start");
        // check rts params
        EXPECT_EQ(funcHandle, (void*)0x12341234);
        EXPECT_EQ(blockDim, 16);
        EXPECT_EQ(stream, nullptr);
        EXPECT_EQ(argsSize, 632);

        EXPECT_NE(cfg, nullptr);
        EXPECT_TRUE(CheckRtKernelLaunchCfgSchemMode(cfg, 1));
        EXPECT_TRUE(CheckRtKernelLaunchCfgLocalMemSize(cfg, 888));

        // check ptr
        void** ptrArgs = reinterpret_cast<void**>(hostArgs);
        EXPECT_EQ(ptrArgs[0], input1);
        EXPECT_EQ(ptrArgs[2], nullptr);

        EXPECT_EQ(ptrArgs[4], output1);

        EXPECT_EQ(ptrArgs[6], outShape);

        EXPECT_EQ(ptrArgs[7], workspace1);
        EXPECT_EQ(ptrArgs[8], workspace2);

        EXPECT_EQ(ptrArgs[10], overflow);

        // check host value
        EXPECT_NE(placeHolderArray, nullptr);
        EXPECT_EQ(placeHolderNum, 4);

        EXPECT_EQ(placeHolderArray[0].addrOffset, 8);
        EXPECT_EQ(placeHolderArray[0].dataOffset, 88);
        ptrArgs = reinterpret_cast<void**>(hostArgs + 88);
        int64_t shapeOffset = reinterpret_cast<int64_t>(ptrArgs[0]) / sizeof(void*);
        EXPECT_EQ(ptrArgs[shapeOffset++], input2_1);
        EXPECT_EQ(ptrArgs[shapeOffset++], input2_2);
        EXPECT_EQ(ptrArgs[shapeOffset], nullptr);

        EXPECT_EQ(placeHolderArray[1].addrOffset, 24);
        EXPECT_EQ(placeHolderArray[1].dataOffset, 216);
        int64_t* hostVal = reinterpret_cast<int64_t*>(reinterpret_cast<char*>(hostArgs) + 216);
        for (int64_t i = 0; i < 10; i++) {
            EXPECT_EQ(hostVal[i], 1);
        }

        EXPECT_EQ(placeHolderArray[2].addrOffset, 40);
        EXPECT_EQ(placeHolderArray[2].dataOffset, 312);
        ptrArgs = reinterpret_cast<void**>(hostArgs + 312);
        shapeOffset = reinterpret_cast<int64_t>(ptrArgs[0]) / sizeof(void*);
        EXPECT_EQ(ptrArgs[shapeOffset++], output2_1);
        EXPECT_EQ(ptrArgs[shapeOffset++], output2_2);
        EXPECT_EQ(ptrArgs[shapeOffset], nullptr);

        // check tiling data
        EXPECT_EQ(placeHolderArray[3].addrOffset, 72);
        EXPECT_EQ(placeHolderArray[3].dataOffset, 440);
        int64_t* tilingVal = reinterpret_cast<int64_t*>(reinterpret_cast<char*>(hostArgs) + 440);
        for (int64_t i = 0; i < 21; i++) {
            EXPECT_EQ(tilingVal[i], i);
        }

        return ACL_SUCCESS;
    }

    void* input1;
    void* input2_1;
    void* input2_2;
    void* output1;
    void* output2_1;
    void* output2_2;
    void* outShape;
    void* workspace1;
    void* workspace2;
    void* overflow;
};

// MultiDoLaunchNormalTest4: genPlaceholder_=true, hasDevPtrArg_=true;
static void MultiDoLaunchNormalTest4()
{
    OP_LOGI("Start to run UT MultiDoLaunchNormalTest4 -->");
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
    uint32_t opType = op::OpTypeDict::ToOpType("QuantBatchMatmulV3");
    OpKernelBin kernelBin(opType, jsonPath, jsonPath, binPath, key, hashKey, BinType::DYNAMIC_BIN, false, false);
    kernelBin.interCoreSync_ = false;
    kernelBin.genPlaceholder_ = true;
    kernelBin.hasDevPtrArg_ = true;
    kernelBin.isFatbin_ = true;
    kernelBin.currDevId_ = 0;
    auto f = [](void*& hdl) -> aclnnStatus {
        hdl = nullptr;
        return ACLNN_SUCCESS;
    };
    kernelBin.binHandle_[0].InitVar(f);

    // create input output workspace outputshape
    op::Shape selfShape{33, 15, 64};
    op::Shape outShape{33, 15, 64};
    op::Shape idxShape{33, 15, 64};
    op::Shape wsShape{32};

    // input
    int* ptr1 = new int;
    aclTensor tensor1(selfShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND, ptr1);
    tensor1.SetFromWorkspace(false);
    int* ptr2 = new int;
    auto tensor2 = std::make_unique<aclTensor>(selfShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND, ptr2);
    int* ptr3 = new int;
    auto tensor3 = std::make_unique<aclTensor>(selfShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND, ptr3);
    aclTensor* tensor4 = nullptr;
    std::vector<int64_t> hostValue(10, 1);
    aclIntArray intArray(hostValue.data(), hostValue.size());
    aclTensor hostTensor(&intArray, op::DataType::DT_INT64);

    aclTensor* input1 = &tensor1;
    aclTensor* tensor2Ptr = tensor2.release();
    tensor2Ptr->SetFromWorkspace(false);
    aclTensor* tensor3Ptr = tensor3.release();
    tensor3Ptr->SetFromWorkspace(false);

    const aclTensor* inputTensors[3] = {tensor2Ptr, tensor3Ptr, tensor4};
    aclTensorList* input2 = aclCreateTensorList(inputTensors, 3);
    aclTensor* input3 = tensor4;
    aclTensorList* input4 = nullptr;
    aclTensor* input5 = &hostTensor;

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

    auto input = OP_INPUT(input1, input2, input3, input4, input5);
    auto output = OP_OUTPUT(output1, output2, output3, output4);
    auto outshape = OP_OUTSHAPE(outshapeTensor, 0);

    auto ctx = op::MakeOpArgContext(input, output, outshape);
    ctx->AppendOpWorkspaceArg(workspace1);

    // init cache
    InitExecutorCacheThreadLocal();
    auto in = std::make_tuple(input1, input2, input3, input4, input5);
    auto out = std::make_tuple(output1, output2, output3, output4, outshapeTensor);
    CalculateHashKey(in);
    CalculateHashKey(out);

    // init tiling res
    TilingCtxOutput tilingOutput{};

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

    uint64_t tilingKey = 10020;
    int64_t numBlocks = 16;
    uint8_t scheduleMode = 1;
    uint32_t dynUBufSize = 888;

    tilingOutput.tilingKey_ = &tilingKey;
    tilingOutput.numBlocks_ = &numBlocks;
    tilingOutput.tilingData_ = tilingData;
    tilingOutput.scheduleMode_ = &scheduleMode;
    tilingOutput.dynUBufSize_ = &dynUBufSize;
    tilingOutput.rtsArgBuffer_ = &buffer;
    buffer.RegisterOutputTilingDataPtr(&tilingOutput.tilingData_);

    // run launch
    MultiDoLaunchNormalTest4AclrtStub aclrtStub;
    aclrtStub.input1 = ptr1;
    aclrtStub.input2_1 = ptr2;
    aclrtStub.input2_2 = ptr3;
    aclrtStub.output1 = ptr1;
    aclrtStub.output2_1 = ptr2;
    aclrtStub.output2_2 = ptr3;
    aclrtStub.outShape = ptr1;
    aclrtStub.workspace1 = workspacePtr;
    aclrtStub.workspace2 = reinterpret_cast<char*>(workspacePtr) + 512;
    aclrtStub.overflow = (void*)0x005;

    AclrtStub::GetInstance()->Install(&aclrtStub);
    MultiDoLaunchNormalTest4DumpStub dumpStub;
    Adx::DumpStub::GetInstance()->Install(&dumpStub);

    std::vector<int32_t> tensorOffset;
    auto opExecCache = new OpExecCache();
    opExecCache->hashKey_ = 1;
    opExecCache->SetCacheBuf(GetCacheBuf());
    GetOpCacheContext().SetOpCache(opExecCache);
    auto rc = kernelBin.DoLaunch(&tilingOutput, stream, false, ctx, tensorOffset);

    EXPECT_EQ(tensorOffset[0], 0);
    EXPECT_EQ(tensorOffset[1], 45);
    EXPECT_EQ(tensorOffset[2], 46);
    EXPECT_EQ(tensorOffset[3], 47);
    EXPECT_EQ(tensorOffset[4], 2);
    EXPECT_EQ(tensorOffset[5], 3);
    EXPECT_EQ(tensorOffset[6], 4);
    EXPECT_EQ(tensorOffset[7], 73);
    EXPECT_EQ(tensorOffset[8], 74);
    EXPECT_EQ(tensorOffset[9], 75);
    EXPECT_EQ(tensorOffset[10], 6);
    EXPECT_EQ(tensorOffset[11], 7);
    EXPECT_EQ(tensorOffset[12], 8);

    const std::vector<uint64_t> tensorSizeInfo = {63360, 80, 0, 96, 63360, 80, 63360, 63360, 63360};
    const std::vector<uint64_t> shapeInfo = {3, 33, 15, 64, 1, 10, 3, 33, 15, 64, 3, 33, 15, 64, 3, 33, 15, 64};
    dumpStub.CheckDFXInfoDump(tensorSizeInfo, shapeInfo);
    AclrtStub::GetInstance()->UnInstall();

    MultiDoLaunchNormalTest4CacheAclrtStub cacheAclrtStub;
    cacheAclrtStub.input1 = ptr1;
    cacheAclrtStub.input2_1 = ptr2;
    cacheAclrtStub.input2_2 = ptr3;
    cacheAclrtStub.output1 = ptr1;
    cacheAclrtStub.output2_1 = ptr2;
    cacheAclrtStub.output2_2 = ptr3;
    cacheAclrtStub.outShape = ptr1;
    cacheAclrtStub.workspace1 = workspacePtr;
    cacheAclrtStub.workspace2 = reinterpret_cast<char*>(workspacePtr) + 512;
    cacheAclrtStub.overflow = (void*)0x005;
    AclrtStub::GetInstance()->Install(&cacheAclrtStub);

    OpExecCacheWrap* cacheWrap = CreateCacheWrap(opExecCache);
    cacheWrap->Run(workspacePtr, stream);

    AclrtStub::GetInstance()->UnInstall();
    Adx::DumpStub::GetInstance()->UnInstall();

    // tear down
    delete ptr1;
    delete ptr2;
    delete ptr3;
    delete workspacePtr;
    delete opExecCache;
    delete cacheWrap;
    aclDestroyTensorList(input2);
    aclDestroyTensorList(workspace1);
}

class MultiDoLaunchAlignTestDumpStub : public Adx::DumpStub {
public:
    void* AdumpGetDFXInfoAddrForDynamic(uint32_t space, uint64_t& atomicIndex)
    {
        dfxInfoSpace_ = space;
        atomicIndex = 666;
        return (void*)dfxInfoMem_;
    }

    void CheckDFXInfoDump(const std::vector<uint64_t>& tensorSizeInfo, const std::vector<uint64_t>& shapeInfo)
    {
        std::cout << std::endl;
        EXPECT_EQ(dfxInfoSpace_, 47);
        EXPECT_EQ(tensorSizeInfo.size() + shapeInfo.size(), 47);
        for (size_t i = 0; i < tensorSizeInfo.size(); i++) {
            EXPECT_EQ(tensorSizeInfo[i], dfxInfoMem_[i]);
        }
        size_t shapeInfoOffset = tensorSizeInfo.size();
        for (size_t i = 0; i < shapeInfo.size(); i++) {
            EXPECT_EQ(shapeInfo[i], dfxInfoMem_[shapeInfoOffset + i]);
        }
        for (size_t i = 0; i < (tensorSizeInfo.size() + shapeInfo.size()); i++) {
            dfxInfoMem_[i] = 0;
        }
    }

    uint32_t dfxInfoSpace_{0};
    uint64_t dfxInfoMem_[1000] = {0};
};

class MultiDoLaunchAlignTestAclrtStub : public AclrtStub {
public:
    aclError aclrtBinaryGetFunctionByEntry(aclrtBinHandle binHandle, uint64_t funcEntry, aclrtFuncHandle* funcHandle)
    {
        OP_LOGI("MultiDoLaunchAlignTestAclrtStub aclrtBinaryGetFunctionByEntry start");
        EXPECT_EQ(binHandle, nullptr);
        EXPECT_EQ(funcEntry, 10020);
        *funcHandle = (void*)0x12341234;
        return ACL_SUCCESS;
    }

    aclError aclrtLaunchKernelWithHostArgs(aclrtFuncHandle funcHandle, uint32_t blockDim, aclrtStream stream,
                                           aclrtLaunchKernelCfg* cfg, void* hostArgs, size_t argsSize,
                                           aclrtPlaceHolderInfo* placeHolderArray, size_t placeHolderNum)
    {
        OP_LOGI("MultiDoLaunchAlignTestAclrtStub aclrtLaunchKernelWithHostArgs start");
        // check rts params
        EXPECT_EQ(funcHandle, (void*)0x12341234);
        EXPECT_EQ(blockDim, 16);
        EXPECT_EQ(stream, nullptr);
        EXPECT_EQ(argsSize, 424);

        EXPECT_NE(cfg, nullptr);
        EXPECT_TRUE(CheckRtKernelLaunchCfgSchemMode(cfg, 1));
        EXPECT_TRUE(CheckRtKernelLaunchCfgLocalMemSize(cfg, 888));

        // check ptr
        void** ptrArgs = reinterpret_cast<void**>(hostArgs);
        EXPECT_EQ(ptrArgs[0], input1);
        EXPECT_EQ(ptrArgs[1], input2_1);
        EXPECT_EQ(ptrArgs[2], input2_2);

        EXPECT_EQ(ptrArgs[5], output1);
        EXPECT_EQ(ptrArgs[6], output2_1);
        EXPECT_EQ(ptrArgs[7], output2_2);

        EXPECT_EQ(ptrArgs[8], outShape);

        EXPECT_EQ(ptrArgs[9], workspace1);
        EXPECT_EQ(ptrArgs[10], workspace2);

        EXPECT_EQ(ptrArgs[12], overflow);

        // check host value
        EXPECT_NE(placeHolderArray, nullptr);
        EXPECT_EQ(placeHolderNum, 3);

        EXPECT_EQ(placeHolderArray[0].addrOffset, 24);
        EXPECT_EQ(placeHolderArray[0].dataOffset, 296);
        int64_t* hostVal = reinterpret_cast<int64_t*>(reinterpret_cast<char*>(hostArgs) + 296);
        for (int64_t i = 0; i < 10; i++) {
            EXPECT_EQ(hostVal[i], 1);
        }
        for (int64_t i = 10; i < 12; i++) {
            EXPECT_EQ(hostVal[i], 0);
        }

        EXPECT_EQ(placeHolderArray[1].addrOffset, 32);
        EXPECT_EQ(placeHolderArray[1].dataOffset, 392);
        float* hostVal2 = reinterpret_cast<float*>(reinterpret_cast<char*>(hostArgs) + 392);
        EXPECT_FLOAT_EQ(*hostVal2, 3.2);

        // check tiling data
        EXPECT_EQ(placeHolderArray[2].addrOffset, 88);
        EXPECT_EQ(placeHolderArray[2].dataOffset, 104);
        int64_t* tilingVal = reinterpret_cast<int64_t*>(reinterpret_cast<char*>(hostArgs) + 104);
        for (int64_t i = 0; i < 21; i++) {
            EXPECT_EQ(tilingVal[i], i);
        }
        EXPECT_EQ(tilingVal[21], 666);
        EXPECT_EQ(tilingVal[22], 0);
        EXPECT_EQ(tilingVal[23], 0);

        return ACL_SUCCESS;
    }

    void* input1;
    void* input2_1;
    void* input2_2;
    void* input2_3;
    void* input3;
    void* input4;
    void* output1;
    void* output2_1;
    void* output2_2;
    void* output2_3;
    void* output3;
    void* output4;
    void* outShape;
    void* workspace1;
    void* workspace2;
    void* overflow;
};

class MultiDoLaunchAlignTestCacheAclrtStub : public AclrtStub {
public:
    aclError aclrtLaunchKernelWithHostArgs(aclrtFuncHandle funcHandle, uint32_t blockDim, aclrtStream stream,
                                           aclrtLaunchKernelCfg* cfg, void* hostArgs, size_t argsSize,
                                           aclrtPlaceHolderInfo* placeHolderArray, size_t placeHolderNum)
    {
        OP_LOGI("MultiDoLaunchAlignTestCacheAclrtStub aclrtLaunchKernelWithHostArgs start");
        // check rts params
        EXPECT_EQ(funcHandle, (void*)0x12341234);
        EXPECT_EQ(blockDim, 16);
        EXPECT_EQ(stream, nullptr);
        EXPECT_EQ(argsSize, 424);

        EXPECT_NE(cfg, nullptr);
        EXPECT_TRUE(CheckRtKernelLaunchCfgSchemMode(cfg, 1));
        EXPECT_TRUE(CheckRtKernelLaunchCfgLocalMemSize(cfg, 888));

        // check ptr
        void** ptrArgs = reinterpret_cast<void**>(hostArgs);
        EXPECT_EQ(ptrArgs[0], input1);
        EXPECT_EQ(ptrArgs[1], input2_1);
        EXPECT_EQ(ptrArgs[2], input2_2);

        EXPECT_EQ(ptrArgs[5], output1);
        EXPECT_EQ(ptrArgs[6], output2_1);
        EXPECT_EQ(ptrArgs[7], output2_2);

        EXPECT_EQ(ptrArgs[8], outShape);

        EXPECT_EQ(ptrArgs[9], workspace1);
        EXPECT_EQ(ptrArgs[10], workspace2);

        EXPECT_EQ(ptrArgs[12], overflow);

        // check host value
        EXPECT_NE(placeHolderArray, nullptr);
        EXPECT_EQ(placeHolderNum, 3);

        EXPECT_EQ(placeHolderArray[0].addrOffset, 24);
        EXPECT_EQ(placeHolderArray[0].dataOffset, 104);
        int64_t* hostVal = reinterpret_cast<int64_t*>(reinterpret_cast<char*>(hostArgs) + 104);
        for (int64_t i = 0; i < 10; i++) {
            EXPECT_EQ(hostVal[i], 1);
        }
        for (int64_t i = 10; i < 12; i++) {
            EXPECT_EQ(hostVal[i], 0);
        }

        EXPECT_EQ(placeHolderArray[1].addrOffset, 32);
        EXPECT_EQ(placeHolderArray[1].dataOffset, 200);
        float* hostVal2 = reinterpret_cast<float*>(reinterpret_cast<char*>(hostArgs) + 200);
        EXPECT_FLOAT_EQ(*hostVal2, 3.2);

        // check tiling data
        EXPECT_EQ(placeHolderArray[2].addrOffset, 88);
        EXPECT_EQ(placeHolderArray[2].dataOffset, 232);
        int64_t* tilingVal = reinterpret_cast<int64_t*>(reinterpret_cast<char*>(hostArgs) + 232);
        for (int64_t i = 0; i < 21; i++) {
            EXPECT_EQ(tilingVal[i], i);
        }
        EXPECT_EQ(tilingVal[21], 666);
        EXPECT_EQ(tilingVal[22], 0);
        EXPECT_EQ(tilingVal[23], 0);

        return ACL_SUCCESS;
    }

    void* input1;
    void* input2_1;
    void* input2_2;
    void* input2_3;
    void* input3;
    void* input4;
    void* output1;
    void* output2_1;
    void* output2_2;
    void* output2_3;
    void* output3;
    void* output4;
    void* outShape;
    void* workspace1;
    void* workspace2;
    void* overflow;
};

// MultiDoLaunchAlignTest: genPlaceholder_=false, hasDevPtrArg_=false;
static void MultiDoLaunchAlignTest()
{
    OP_LOGI("Start to run UT MultiDoLaunchAlignTest -->");
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
    uint32_t opType = op::OpTypeDict::ToOpType("QuantBatchMatmulV3");
    OpKernelBin kernelBin(opType, jsonPath, jsonPath, binPath, key, hashKey, BinType::DYNAMIC_BIN, false, false);
    kernelBin.interCoreSync_ = false;
    kernelBin.genPlaceholder_ = false;
    kernelBin.hasDevPtrArg_ = false;
    kernelBin.isFatbin_ = true;
    kernelBin.currDevId_ = 0;
    auto f = [](void*& hdl) -> aclnnStatus {
        hdl = nullptr;
        return ACLNN_SUCCESS;
    };
    kernelBin.binHandle_[0].InitVar(f);

    // create input output workspace outputshape
    op::Shape selfShape{33, 15, 64};
    op::Shape outShape{33, 15, 64};
    op::Shape idxShape{33, 15, 64};
    op::Shape wsShape{32};

    // input
    int* ptr1 = new int;
    aclTensor tensor1(selfShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND, ptr1);
    tensor1.SetFromWorkspace(false);
    int* ptr2 = new int;
    auto tensor2 = std::make_unique<aclTensor>(selfShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND, ptr2);
    int* ptr3 = new int;
    auto tensor3 = std::make_unique<aclTensor>(selfShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND, ptr3);
    aclTensor* tensor4 = nullptr;
    std::vector<int64_t> hostValue(10, 1);
    aclIntArray intArray(hostValue.data(), hostValue.size());
    aclTensor hostTensor(&intArray, op::DataType::DT_INT64);
    float fpValue = 3.2;
    aclScalar scalar(&fpValue, op::DataType::DT_FLOAT);

    aclTensor* input1 = &tensor1;
    aclTensor* tensor2Ptr = tensor2.release();
    tensor2Ptr->SetFromWorkspace(false);
    aclTensor* tensor3Ptr = tensor3.release();
    tensor3Ptr->SetFromWorkspace(false);

    const aclTensor* inputTensors[3] = {tensor2Ptr, tensor3Ptr, tensor4};
    aclTensorList* input2 = aclCreateTensorList(inputTensors, 3);
    aclTensor* input3 = tensor4;
    aclTensorList* input4 = nullptr;
    aclTensor* input5 = &hostTensor;
    aclTensor scalar2Tensor(&scalar, op::DataType::DT_FLOAT);
    aclTensor* input6 = &scalar2Tensor;

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

    // init cache
    InitExecutorCacheThreadLocal();
    auto in = std::make_tuple(input1, input2, input3, input4, input5, input6);
    auto out = std::make_tuple(output1, output2, output3, output4, outshapeTensor);
    CalculateHashKey(in);
    CalculateHashKey(out);

    // init tiling res
    TilingCtxOutput tilingOutput{};

    op::internal::ExpandableRtsArgBuffer buffer;
    // launchArgCap=64 < 120 触发 launch arg 扩容
    // tilingHostDataCap=200: tilingData(168) 不扩容，AppendHostArg 阶段(168+80=248>200) 触发扩容
    buffer.Init(64, 200);
    int64_t* tilingDataData = static_cast<int64_t*>(buffer.GetTilingDataAddr());
    for (size_t i = 0; i < 21; i++) {
        tilingDataData[i] = i;
    }
    op::internal::TilingData* tilingData = buffer.GetTilingDataPtr();
    tilingData->data_ = buffer.GetTilingDataAddr();
    tilingData->data_size_ = sizeof(size_t) * 21;
    tilingData->capacity_ = TEST_TILING_HOST_DATA_INIT_CAP;

    uint64_t tilingKey = 10020;
    int64_t numBlocks = 16;
    uint8_t scheduleMode = 1;
    uint32_t dynUBufSize = 888;

    tilingOutput.tilingKey_ = &tilingKey;
    tilingOutput.numBlocks_ = &numBlocks;
    tilingOutput.tilingData_ = tilingData;
    tilingOutput.scheduleMode_ = &scheduleMode;
    tilingOutput.dynUBufSize_ = &dynUBufSize;
    tilingOutput.rtsArgBuffer_ = &buffer;
    buffer.RegisterOutputTilingDataPtr(&tilingOutput.tilingData_);

    // run launch
    MultiDoLaunchAlignTestAclrtStub aclrtStub;
    aclrtStub.input1 = ptr1;
    aclrtStub.input2_1 = ptr2;
    aclrtStub.input2_2 = ptr3;
    aclrtStub.input2_3 = nullptr;
    aclrtStub.input3 = nullptr;
    aclrtStub.input4 = nullptr;
    aclrtStub.output1 = ptr1;
    aclrtStub.output2_1 = ptr2;
    aclrtStub.output2_2 = ptr3;
    aclrtStub.output2_3 = nullptr;
    aclrtStub.output3 = nullptr;
    aclrtStub.output4 = nullptr;
    aclrtStub.outShape = ptr1;
    aclrtStub.workspace1 = workspacePtr;
    aclrtStub.workspace2 = reinterpret_cast<char*>(workspacePtr) + 512;
    aclrtStub.overflow = (void*)0x005;

    AclrtStub::GetInstance()->Install(&aclrtStub);

    MultiDoLaunchAlignTestDumpStub dumpStub;
    Adx::DumpStub::GetInstance()->Install(&dumpStub);

    std::vector<int32_t> tensorOffset;
    auto opExecCache = new OpExecCache();
    opExecCache->hashKey_ = 1;
    opExecCache->SetCacheBuf(GetCacheBuf());
    GetOpCacheContext().SetOpCache(opExecCache);
    auto rc = kernelBin.DoLaunch(&tilingOutput, stream, false, ctx, tensorOffset);

    for (int32_t i = 0; i < 10; i++) {
        EXPECT_EQ(tensorOffset[i], i);
    }

    const std::vector<uint64_t> tensorSizeInfo = {63360, 63360, 63360, 96,    32,   63360,
                                                  63360, 63360, 63360, 63360, 63360};
    const std::vector<uint64_t> shapeInfo = {3,  33, 15, 64, 3,  33, 15, 64, 3,  33, 15, 64, 1,  10, 1, 1,  3,  33,
                                             15, 64, 3,  33, 15, 64, 3,  33, 15, 64, 3,  33, 15, 64, 3, 33, 15, 64};
    dumpStub.CheckDFXInfoDump(tensorSizeInfo, shapeInfo);

    AclrtStub::GetInstance()->UnInstall();

    MultiDoLaunchAlignTestCacheAclrtStub cacheAclrtStub;
    cacheAclrtStub.input1 = ptr1;
    cacheAclrtStub.input2_1 = ptr2;
    cacheAclrtStub.input2_2 = ptr3;
    cacheAclrtStub.input2_3 = nullptr;
    cacheAclrtStub.input3 = nullptr;
    cacheAclrtStub.input4 = nullptr;
    cacheAclrtStub.output1 = ptr1;
    cacheAclrtStub.output2_1 = ptr2;
    cacheAclrtStub.output2_2 = ptr3;
    cacheAclrtStub.output2_3 = nullptr;
    cacheAclrtStub.output3 = nullptr;
    cacheAclrtStub.output4 = nullptr;
    cacheAclrtStub.outShape = ptr1;
    cacheAclrtStub.workspace1 = workspacePtr;
    cacheAclrtStub.workspace2 = reinterpret_cast<char*>(workspacePtr) + 512;
    cacheAclrtStub.overflow = (void*)0x005;
    AclrtStub::GetInstance()->Install(&cacheAclrtStub);

    OpExecCacheWrap* cacheWrap = CreateCacheWrap(opExecCache);
    cacheWrap->Run(workspacePtr, stream);

    AclrtStub::GetInstance()->UnInstall();
    Adx::DumpStub::GetInstance()->UnInstall();

    // tear down
    delete ptr1;
    delete ptr2;
    delete ptr3;
    delete workspacePtr;
    delete opExecCache;
    delete cacheWrap;
    aclDestroyTensorList(input2);
    aclDestroyTensorList(workspace1);
}

class InvalidFunctionHandleAclrtStub : public AclrtStub {
public:
    aclError aclrtLaunchKernelWithHostArgs(aclrtFuncHandle funcHandle, uint32_t blockDim, aclrtStream stream,
                                           aclrtLaunchKernelCfg* cfg, void* hostArgs, size_t argsSize,
                                           aclrtPlaceHolderInfo* placeHolderArray, size_t placeHolderNum)
    {
        if (returnInvalidHandleFlag == true) {
            OP_LOGI("return ACL_ERROR_RT_INVALID_HANDLE");
            returnInvalidHandleFlag = false;
            return ACL_ERROR_RT_INVALID_HANDLE;
        }
        return ACL_SUCCESS;
    }

    bool returnInvalidHandleFlag{false};
};

// fat bin: true
static void TestInvalidFunctionHandle1()
{
    OP_LOGI("Start to run UT TestInvalidFunctionHandle 1 -->");
    // init OpArgContext
    op::Shape selfShape{10};
    op::Shape outShape{10};

    aclOpExecutor exe;
    int64_t inputData[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto self = exe.AllocIntArray(inputData, 10);
    auto inputTensor = exe.ConvertToTensor(self, op::DataType::DT_INT64);
    auto outputTensor = exe.AllocTensor(outShape, op::DataType::DT_INT32);

    auto input_arg = OP_INPUT(inputTensor);
    auto output_arg = OP_OUTPUT(outputTensor);
    auto ctx = op::MakeOpArgContext(input_arg, output_arg);

    // init op kernel bin
    bool genPlaceholder = false, hasDevPtrArg = false;
    auto kernelBin = CreateFakeOpKernelBin(genPlaceholder, hasDevPtrArg);
    kernelBin->interCoreSync_ = false;
    kernelBin->isFatbin_ = true;
    kernelBin->currDevId_ = 0;
    auto f = [](void*& hdl) -> aclnnStatus {
        hdl = (void*)0x11223344;
        return ACLNN_SUCCESS;
    };
    kernelBin->binHandle_[0].InitVar(f);

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

    uint64_t tilingKey = 10020;
    int64_t numBlocks = 16;
    uint8_t scheduleMode = 1;
    uint32_t dynUBufSize = 888;

    TilingCtxOutput tilingOutput;
    tilingOutput.tilingKey_ = &tilingKey;
    tilingOutput.numBlocks_ = &numBlocks;
    tilingOutput.tilingData_ = tilingData;
    tilingOutput.scheduleMode_ = &scheduleMode;
    tilingOutput.dynUBufSize_ = &dynUBufSize;
    tilingOutput.rtsArgBuffer_ = &buffer;
    buffer.RegisterOutputTilingDataPtr(&tilingOutput.tilingData_);

    InvalidFunctionHandleAclrtStub aclrtStub;
    aclrtStub.returnInvalidHandleFlag = true;
    AclrtStub::GetInstance()->Install(&aclrtStub);
    std::vector<int32_t> tensorOffset;
    auto opExecCache = new OpExecCache();
    opExecCache->hashKey_ = 1;
    opExecCache->SetCacheBuf(GetCacheBuf());
    GetOpCacheContext().SetOpCache(opExecCache);
    aclrtStream stream = 0;
    auto rc = kernelBin->DoLaunch(&tilingOutput, stream, false, ctx, tensorOffset);
    EXPECT_EQ(rc, ACLNN_SUCCESS);

    opExecCache->Finalize();
    opExecCache->SetUse();

    void* globalCacheBuf = GetCacheBuf();
    int64_t* bufPtr = static_cast<int64_t*>(globalCacheBuf);
    for (size_t i = 0; i < 200; i++) {
        bufPtr[i] = i;
    }

    aclrtStub.returnInvalidHandleFlag = true;
    OpExecCacheWrap* cacheWrap = CreateCacheWrap(opExecCache);
    rc = cacheWrap->Run(nullptr, stream);
    EXPECT_EQ(rc, ACLNN_SUCCESS);
    AclrtStub::GetInstance()->UnInstall();

    delete opExecCache;
    delete cacheWrap;
}

// fat bin: false
static void TestInvalidFunctionHandle2()
{
    OP_LOGI("Start to run UT TestInvalidFunctionHandle 2 -->");
    // init OpArgContext
    op::Shape workspaceShape{10};

    int workspace[10] = {0};
    int* workspacePtr = workspace;
    aclTensor wsTensor(workspaceShape, op::DataType::DT_INT32, op::Format::FORMAT_ND, workspacePtr);
    wsTensor.SetFromWorkspace(true);
    wsTensor.SetWorkspaceOffset(0);
    aclTensor* wsTensorPtr = &wsTensor;

    auto workspace_arg = OP_WORKSPACE(wsTensorPtr);
    auto ctx = op::MakeOpArgContext(workspace_arg);

    // creat OpKernelBin
    const char* p = std::getenv("ASCEND_OPP_PATH");
    EXPECT_NE(p, nullptr);
    KeyAndDetail key;
    key.key = "hahaha";
    size_t hashKey = 123;
    char jsonPath[1024];
    char binPath[1024];
    snprintf_s(jsonPath, sizeof(jsonPath), sizeof(jsonPath),
               "%s/built-in/op_impl/ai_core/tbe/kernel/ascend910/mem_set/"
               "MemSet_1a6864193b99ef93ef38616f04a712ab_high_performance.json",
               p);
    snprintf_s(binPath, sizeof(binPath), sizeof(binPath),
               "%s/built-in/op_impl/ai_core/tbe/kernel/ascend910/mem_set/"
               "MemSet_1a6864193b99ef93ef38616f04a712ab_high_performance.o.o",
               p);
    uint32_t opType = op::OpTypeDict::ToOpType("Memset");
    OpKernelBin kernelBin(opType, jsonPath, jsonPath, binPath, key, hashKey, BinType::DYNAMIC_BIN, false, false);

    kernelBin.interCoreSync_ = false;
    kernelBin.isFatbin_ = false;
    kernelBin.currDevId_ = 0;
    auto f = [](void*& hdl) -> aclnnStatus {
        hdl = (void*)0x11223344;
        return ACLNN_SUCCESS;
    };
    kernelBin.binHandle_[0].InitVar(f);

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

    uint64_t tilingKey = 10020;
    int64_t numBlocks = 16;
    uint8_t scheduleMode = 1;
    uint32_t dynUBufSize = 888;

    TilingCtxOutput tilingOutput;
    tilingOutput.tilingKey_ = &tilingKey;
    tilingOutput.numBlocks_ = &numBlocks;
    tilingOutput.tilingData_ = tilingData;
    tilingOutput.scheduleMode_ = &scheduleMode;
    tilingOutput.dynUBufSize_ = &dynUBufSize;
    tilingOutput.rtsArgBuffer_ = &buffer;
    buffer.RegisterOutputTilingDataPtr(&tilingOutput.tilingData_);

    InvalidFunctionHandleAclrtStub aclrtStub;
    aclrtStub.returnInvalidHandleFlag = true;
    AclrtStub::GetInstance()->Install(&aclrtStub);
    std::vector<int32_t> tensorOffset;
    auto opExecCache = new OpExecCache();
    opExecCache->hashKey_ = 1;
    opExecCache->SetCacheBuf(GetCacheBuf());
    GetOpCacheContext().SetOpCache(opExecCache);
    aclrtStream stream = 0;
    aclnnStatus rc = kernelBin.JsonLoad();
    EXPECT_EQ(rc, ACLNN_SUCCESS);
    rc = kernelBin.DoLaunch(&tilingOutput, stream, false, ctx, tensorOffset);
    EXPECT_EQ(rc, ACLNN_SUCCESS);

    opExecCache->Finalize();
    opExecCache->SetUse();

    void* globalCacheBuf = GetCacheBuf();
    int64_t* bufPtr = static_cast<int64_t*>(globalCacheBuf);
    for (size_t i = 0; i < 200; i++) {
        bufPtr[i] = i;
    }

    aclrtStub.returnInvalidHandleFlag = true;
    OpExecCacheWrap* cacheWrap = CreateCacheWrap(opExecCache);
    rc = cacheWrap->Run(nullptr, stream);
    EXPECT_EQ(rc, ACLNN_SUCCESS);
    AclrtStub::GetInstance()->UnInstall();

    delete opExecCache;
    delete cacheWrap;
}

TEST_F(OpKernelMultiThreadUT, SingleDoLaunchTest)
{
    vector<Functional> funVec = {MultiDoLaunchAlignTest,   MultiDoLaunchNormalTest2,   MultiDoLaunchNormalTest3,
                                 MultiDoLaunchNormalTest4, TestInvalidFunctionHandle1, TestInvalidFunctionHandle2};
    for (int i = 0; i < funVec.size(); i++) {
        funVec[i]();
    }
}

TEST_F(OpKernelMultiThreadUT, MultiDoLaunchTest)
{
    vector<Functional> funVec = {MultiDoLaunchAlignTest,   MultiDoLaunchNormalTest2,   MultiDoLaunchNormalTest3,
                                 MultiDoLaunchNormalTest4, TestInvalidFunctionHandle1, TestInvalidFunctionHandle2};

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, funVec.size() - 1);
    const uint64_t threadCount = funVec.size() * 180;
    vector<std::thread> threadVec;
    for (int i = 0; i < threadCount; i++) {
        threadVec.emplace_back(std::thread(funVec[distrib(gen)]));
    }
    for (int i = 0; i < threadCount; i++) {
        threadVec[i].join();
    }
}

OP_TYPE_REGISTER(TestStatic);

extern "C" aclnnStatus aclnnReselectStaticKernel();
extern "C" aclnnStatus aclnnReselectStaticKernelWithPath(const char* staticKernelPath);

static void TestReselectStaticKernelFunc(pthread_barrier_t* barrier, pthread_barrier_t* barrier2)
{
    uint32_t testStaticAddOpType = op::OpTypeDict::ToOpType("TestStaticAdd");
    uint32_t conv2DOpType = op::OpTypeDict::ToOpType("Conv2D");
    uint32_t testStaticOpType = op::OpTypeDict::ToOpType("TestStatic");
    op::internal::KernelMgr::getInstance().AclOpKernelInit(testStaticAddOpType);
    op::internal::KernelMgr::getInstance().AclOpKernelInit(conv2DOpType);

    auto& testStaticAddKernel = op::internal::KernelMgr::getInstance().kernel_[testStaticAddOpType];
    size_t numOfTSADynKernels = testStaticAddKernel.bins_.size();
    EXPECT_EQ(numOfTSADynKernels > 0, true);
    EXPECT_EQ(testStaticAddKernel.staticBins_.size(), 1);

    auto& conv2DKernel = op::internal::KernelMgr::getInstance().kernel_[conv2DOpType];
    size_t numOfConv2DDynKernels = conv2DKernel.bins_.size();
    EXPECT_EQ(numOfConv2DDynKernels > 0, true);
    EXPECT_EQ(conv2DKernel.staticBins_.size(), 2); // There is debug static kernel here

    auto& testStaticKernel = op::internal::KernelMgr::getInstance().kernel_[testStaticOpType];
    EXPECT_EQ(testStaticKernel.bins_.size(), 0);
    EXPECT_EQ(testStaticKernel.staticBins_.size(), 0);

    const char* api1 = "AclnnTestStaticAdd";
    op::internal::GetThreadLocalContext().usePTAHash_ = true;
    op::internal::GetThreadLocalContext().cacheApi_ = api1;

    op::internal::GetThreadLocalContext().hashKey_ = 111;
    op::internal::GetThreadLocalContext().cacheHashKey_ = nullptr;
    op::internal::GetThreadLocalContext().cacheHashKeyLen_ = 0;
    OpExecCache* cache1 = new OpExecCache();
    cache1->SetUse();
    EXPECT_EQ(AddOpExecCache(cache1), true);
    EXPECT_EQ(GetOpExecCache(111), cache1);

    op::internal::GetThreadLocalContext().hashKey_ = 222;
    op::internal::GetThreadLocalContext().cacheHashKey_ = nullptr;
    op::internal::GetThreadLocalContext().cacheHashKeyLen_ = 0;
    OpExecCache* cache2 = new OpExecCache();
    cache2->SetUse();
    EXPECT_EQ(AddOpExecCache(cache2), true);
    EXPECT_EQ(GetOpExecCache(222), cache2);
    uint64_t workspaceSize = 0;
    aclOpExecutor* exec = PTAGetExecCache(222, &workspaceSize);
    uint64_t* magicNum = reinterpret_cast<uint64_t*>(exec);
    EXPECT_EQ(*magicNum, op::internal::K_CACHE_WRAP_MAGIC_NUMBER);
    op::internal::OpExecCacheWrap* cacheWrap = reinterpret_cast<op::internal::OpExecCacheWrap*>(exec);
    EXPECT_EQ(cacheWrap->opExecCache_, cache2);
    delete cacheWrap;

    op::internal::GetThreadLocalContext().hashKey_ = 0;
    op::internal::GetThreadLocalContext().cacheHashKey_ = (uint8_t*)"hello111";
    op::internal::GetThreadLocalContext().cacheHashKeyLen_ = 8;
    OpExecCache* cache21 = new OpExecCache();
    cache21->SetUse();
    EXPECT_EQ(AddOpExecCache(cache21), true);
    OpCacheKey key21;
    key21.buf = op::internal::GetThreadLocalContext().cacheHashKey_;
    key21.len = op::internal::GetThreadLocalContext().cacheHashKeyLen_;
    EXPECT_EQ(GetOpExecCache(key21), cache21);

    op::internal::GetThreadLocalContext().hashKey_ = 0;
    op::internal::GetThreadLocalContext().cacheHashKey_ = (uint8_t*)"hello222";
    op::internal::GetThreadLocalContext().cacheHashKeyLen_ = 8;
    OpExecCache* cache22 = new OpExecCache();
    cache22->SetUse();
    EXPECT_EQ(AddOpExecCache(cache22), true);
    OpCacheKey key22;
    key22.buf = op::internal::GetThreadLocalContext().cacheHashKey_;
    key22.len = op::internal::GetThreadLocalContext().cacheHashKeyLen_;
    EXPECT_EQ(GetOpExecCache(key22), cache22);
    uint64_t workspaceSize2 = 0;
    aclOpExecutor* exec2 = PTAFindExecCache((uint8_t*)"hello222", 8, &workspaceSize2);
    uint64_t* magicNum2 = reinterpret_cast<uint64_t*>(exec2);
    EXPECT_EQ(*magicNum2, op::internal::K_CACHE_WRAP_MAGIC_NUMBER);
    op::internal::OpExecCacheWrap* cacheWrap2 = reinterpret_cast<op::internal::OpExecCacheWrap*>(exec2);
    EXPECT_EQ(cacheWrap2->opExecCache_, cache22);
    delete cacheWrap2;

    // 在这里等待所有线程到达屏障
    pthread_barrier_wait(barrier);
    aclnnReselectStaticKernel();

    EXPECT_EQ(GetOpExecCache(111), nullptr);
    EXPECT_EQ(GetOpExecCache(222), nullptr);
    EXPECT_EQ(GetOpExecCache(key21), nullptr);
    EXPECT_EQ(GetOpExecCache(key22), nullptr);

    EXPECT_EQ(testStaticAddKernel.bins_.size(), numOfTSADynKernels);
    EXPECT_EQ(testStaticAddKernel.staticBins_.size(), 1); // reselect 不清空 staticBins_, AppendStaticBin 按 binPath
                                                          // 去重, 重新 init 时相同 binPath 不重复加载, size 不变
    EXPECT_EQ(conv2DKernel.bins_.size(), numOfConv2DDynKernels);
    EXPECT_EQ(conv2DKernel.staticBins_.size(),
              2); // reselect 不清空 staticBins_, enableDebug 仍开, 含 1 static + 1 debug

    pthread_barrier_wait(barrier2);
    OP_LOGI("pthread_barrier_wait2");
    op::internal::KernelMgr::getInstance().AclOpKernelInit(testStaticAddOpType);
    op::internal::KernelMgr::getInstance().AclOpKernelInit(conv2DOpType);
    op::internal::KernelMgr::getInstance().AclOpKernelInit(testStaticOpType);

    EXPECT_EQ(testStaticAddKernel.bins_.size(), numOfTSADynKernels);
    EXPECT_EQ(testStaticAddKernel.staticBins_.size(), 1);
    EXPECT_EQ(conv2DKernel.bins_.size(), numOfConv2DDynKernels);
    EXPECT_EQ(conv2DKernel.staticBins_.size(),
              2); // reselect 不清空 staticBins_, enableDebug 仍开, AppendStaticBin 按 binPath 去重, size 不变
    EXPECT_EQ(testStaticKernel.bins_.size() > 0, true);
    EXPECT_EQ(testStaticKernel.staticBins_.size() > 0, true);

    // 再次add缓存，测试缓存的添加与opcacheManager的析构能力是否正常
    op::internal::GetThreadLocalContext().hashKey_ = 333;
    op::internal::GetThreadLocalContext().cacheHashKey_ = nullptr;
    op::internal::GetThreadLocalContext().cacheHashKeyLen_ = 0;
    OpExecCache* cache3 = new OpExecCache();
    cache3->SetUse();
    EXPECT_EQ(AddOpExecCache(cache3), true);
    EXPECT_EQ(GetOpExecCache(333), cache3);

    op::internal::GetThreadLocalContext().hashKey_ = 0;
    op::internal::GetThreadLocalContext().cacheHashKey_ = (uint8_t*)"hello333";
    op::internal::GetThreadLocalContext().cacheHashKeyLen_ = 8;
    OpExecCache* cache23 = new OpExecCache();
    cache23->SetUse();
    EXPECT_EQ(AddOpExecCache(cache23), true);
    OpCacheKey key23;
    key23.buf = op::internal::GetThreadLocalContext().cacheHashKey_;
    key23.len = op::internal::GetThreadLocalContext().cacheHashKeyLen_;
    EXPECT_EQ(GetOpExecCache(key23), cache23);
}

// 镜像 TestReselectStaticKernelFunc，唯一区别：调用 aclnnReselectStaticKernelWithPath(getenv("ASCEND_OPP_PATH"))
// 替代无参版本 aclnnReselectStaticKernel()，验证 reload 流程在传入合法 OPP 路径时与无参版本行为对称
static void TestReselectStaticKernelWithPathFunc(pthread_barrier_t* barrier, pthread_barrier_t* barrier2)
{
    uint32_t testStaticAddOpType = op::OpTypeDict::ToOpType("TestStaticAdd");
    uint32_t conv2DOpType = op::OpTypeDict::ToOpType("Conv2D");
    uint32_t testStaticOpType = op::OpTypeDict::ToOpType("TestStatic");
    op::internal::KernelMgr::getInstance().AclOpKernelInit(testStaticAddOpType);
    op::internal::KernelMgr::getInstance().AclOpKernelInit(conv2DOpType);

    auto& testStaticAddKernel = op::internal::KernelMgr::getInstance().kernel_[testStaticAddOpType];
    size_t numOfTSADynKernels = testStaticAddKernel.bins_.size();
    EXPECT_EQ(numOfTSADynKernels > 0, true);
    EXPECT_EQ(testStaticAddKernel.staticBins_.size(), 1);

    auto& conv2DKernel = op::internal::KernelMgr::getInstance().kernel_[conv2DOpType];
    size_t numOfConv2DDynKernels = conv2DKernel.bins_.size();
    EXPECT_EQ(numOfConv2DDynKernels > 0, true);
    EXPECT_EQ(conv2DKernel.staticBins_.size(), 2); // There is debug static kernel here

    auto& testStaticKernel = op::internal::KernelMgr::getInstance().kernel_[testStaticOpType];
    // 不验证 testStatic 初始状态 —— TestReselectStaticKernel 已对此覆盖；
    // 两个测试共享 KernelMgr 单例，此处 testStatic 可能已被加载（bins_/staticBins_ 非 0）；
    // 本测试核心目标是验证 reselect + withPath 行为对称

    const char* api1 = "AclnnTestStaticAdd";
    op::internal::GetThreadLocalContext().usePTAHash_ = true;
    op::internal::GetThreadLocalContext().cacheApi_ = api1;

    op::internal::GetThreadLocalContext().hashKey_ = 111;
    op::internal::GetThreadLocalContext().cacheHashKey_ = nullptr;
    op::internal::GetThreadLocalContext().cacheHashKeyLen_ = 0;
    OpExecCache* cache1 = new OpExecCache();
    cache1->SetUse();
    EXPECT_EQ(AddOpExecCache(cache1), true);
    EXPECT_EQ(GetOpExecCache(111), cache1);

    op::internal::GetThreadLocalContext().hashKey_ = 222;
    op::internal::GetThreadLocalContext().cacheHashKey_ = nullptr;
    op::internal::GetThreadLocalContext().cacheHashKeyLen_ = 0;
    OpExecCache* cache2 = new OpExecCache();
    cache2->SetUse();
    EXPECT_EQ(AddOpExecCache(cache2), true);
    EXPECT_EQ(GetOpExecCache(222), cache2);
    uint64_t workspaceSize = 0;
    aclOpExecutor* exec = PTAGetExecCache(222, &workspaceSize);
    uint64_t* magicNum = reinterpret_cast<uint64_t*>(exec);
    EXPECT_EQ(*magicNum, op::internal::K_CACHE_WRAP_MAGIC_NUMBER);
    op::internal::OpExecCacheWrap* cacheWrap = reinterpret_cast<op::internal::OpExecCacheWrap*>(exec);
    EXPECT_EQ(cacheWrap->opExecCache_, cache2);
    delete cacheWrap;

    op::internal::GetThreadLocalContext().hashKey_ = 0;
    op::internal::GetThreadLocalContext().cacheHashKey_ = (uint8_t*)"hello111";
    op::internal::GetThreadLocalContext().cacheHashKeyLen_ = 8;
    OpExecCache* cache21 = new OpExecCache();
    cache21->SetUse();
    EXPECT_EQ(AddOpExecCache(cache21), true);
    OpCacheKey key21;
    key21.buf = op::internal::GetThreadLocalContext().cacheHashKey_;
    key21.len = op::internal::GetThreadLocalContext().cacheHashKeyLen_;
    EXPECT_EQ(GetOpExecCache(key21), cache21);

    op::internal::GetThreadLocalContext().hashKey_ = 0;
    op::internal::GetThreadLocalContext().cacheHashKey_ = (uint8_t*)"hello222";
    op::internal::GetThreadLocalContext().cacheHashKeyLen_ = 8;
    OpExecCache* cache22 = new OpExecCache();
    cache22->SetUse();
    EXPECT_EQ(AddOpExecCache(cache22), true);
    OpCacheKey key22;
    key22.buf = op::internal::GetThreadLocalContext().cacheHashKey_;
    key22.len = op::internal::GetThreadLocalContext().cacheHashKeyLen_;
    EXPECT_EQ(GetOpExecCache(key22), cache22);
    uint64_t workspaceSize2 = 0;
    aclOpExecutor* exec2 = PTAFindExecCache((uint8_t*)"hello222", 8, &workspaceSize2);
    uint64_t* magicNum2 = reinterpret_cast<uint64_t*>(exec2);
    EXPECT_EQ(*magicNum2, op::internal::K_CACHE_WRAP_MAGIC_NUMBER);
    op::internal::OpExecCacheWrap* cacheWrap2 = reinterpret_cast<op::internal::OpExecCacheWrap*>(exec2);
    EXPECT_EQ(cacheWrap2->opExecCache_, cache22);
    delete cacheWrap2;

    // 在这里等待所有线程到达屏障
    pthread_barrier_wait(barrier);
    // 与 TestReselectStaticKernelFunc 的唯一区别：调用 withPath 版本，传入环境变量 ASCEND_OPP_PATH
    // 多线程函数中禁用 ASSERT_*，用 if-return 模式处理 nullptr
    const char* oppPath = std::getenv("ASCEND_OPP_PATH");
    if (oppPath == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER, "ASCEND_OPP_PATH not set");
        return;
    }
    aclnnReselectStaticKernelWithPath(oppPath);

    EXPECT_EQ(GetOpExecCache(111), nullptr);
    EXPECT_EQ(GetOpExecCache(222), nullptr);
    EXPECT_EQ(GetOpExecCache(key21), nullptr);
    EXPECT_EQ(GetOpExecCache(key22), nullptr);

    EXPECT_EQ(testStaticAddKernel.bins_.size(), numOfTSADynKernels);
    EXPECT_EQ(testStaticAddKernel.staticBins_.size(), 1); // reselect 不清空 staticBins_, AppendStaticBin 按 binPath
                                                          // 去重, 重新 init 时相同 binPath 不重复加载, size 不变
    EXPECT_EQ(conv2DKernel.bins_.size(), numOfConv2DDynKernels);
    EXPECT_EQ(conv2DKernel.staticBins_.size(),
              2); // reselect 不清空 staticBins_, enableDebug 仍开, 含 1 static + 1 debug

    pthread_barrier_wait(barrier2);
    OP_LOGI("pthread_barrier_wait2");
    op::internal::KernelMgr::getInstance().AclOpKernelInit(testStaticAddOpType);
    op::internal::KernelMgr::getInstance().AclOpKernelInit(conv2DOpType);
    op::internal::KernelMgr::getInstance().AclOpKernelInit(testStaticOpType);

    EXPECT_EQ(testStaticAddKernel.bins_.size(), numOfTSADynKernels);
    EXPECT_EQ(testStaticAddKernel.staticBins_.size(), 1);
    EXPECT_EQ(conv2DKernel.bins_.size(), numOfConv2DDynKernels);
    EXPECT_EQ(conv2DKernel.staticBins_.size(),
              2); // reselect 不清空 staticBins_, enableDebug 仍开, AppendStaticBin 按 binPath 去重, size 不变
    EXPECT_EQ(testStaticKernel.bins_.size() > 0, true);
    EXPECT_EQ(testStaticKernel.staticBins_.size() > 0, true);

    // 再次add缓存，测试缓存的添加与opcacheManager的析构能力是否正常
    op::internal::GetThreadLocalContext().hashKey_ = 333;
    op::internal::GetThreadLocalContext().cacheHashKey_ = nullptr;
    op::internal::GetThreadLocalContext().cacheHashKeyLen_ = 0;
    OpExecCache* cache3 = new OpExecCache();
    cache3->SetUse();
    EXPECT_EQ(AddOpExecCache(cache3), true);
    EXPECT_EQ(GetOpExecCache(333), cache3);

    op::internal::GetThreadLocalContext().hashKey_ = 0;
    op::internal::GetThreadLocalContext().cacheHashKey_ = (uint8_t*)"hello333";
    op::internal::GetThreadLocalContext().cacheHashKeyLen_ = 8;
    OpExecCache* cache23 = new OpExecCache();
    cache23->SetUse();
    EXPECT_EQ(AddOpExecCache(cache23), true);
    OpCacheKey key23;
    key23.buf = op::internal::GetThreadLocalContext().cacheHashKey_;
    key23.len = op::internal::GetThreadLocalContext().cacheHashKeyLen_;
    EXPECT_EQ(GetOpExecCache(key23), cache23);
}

TEST_F(OpKernelMultiThreadUT, TestReselectStaticKernel)
{
    op::GenOpTypeId("TestStatic");
    op::GenOpTypeId("TestStaticAdd");
    op::GenOpTypeId("Conv2D");
    // 启用 debug kernel，使 Conv2D 能同时加载 static 与 debug static 二进制（满足 line 1988 的初始期望）
    op::internal::systemConfig.SetEnableDebugKernelFlag(true);
    const char_t* const cacheLimit = std::getenv("ACLNN_CACHE_LIMIT");
    setenv("ACLNN_CACHE_LIMIT", "10000", 1);

    // 单线程预初始化被测 op，避免多线程首次 init 时 mock 环境下 dynamic kernel 加载竞态/失败
    // 注意：故意不预 init testStaticOpType —— 原测试设计要求 testStatic 在初始加载阶段不加载
    // （line 1992 期望 staticBins_ == 0），仅在 reselect 后的 line 2068 首次 init 才加载
    uint32_t testStaticAddOpType = op::OpTypeDict::ToOpType("TestStaticAdd");
    uint32_t conv2DOpType = op::OpTypeDict::ToOpType("Conv2D");
    op::internal::KernelMgr::getInstance().AclOpKernelInit(testStaticAddOpType);
    op::internal::KernelMgr::getInstance().AclOpKernelInit(conv2DOpType);

    constexpr uint32_t LAUNCH_THREADS = 20;
    pthread_barrier_t barrier;
    auto ret = pthread_barrier_init(&barrier, nullptr, LAUNCH_THREADS); // 初始化屏障，指定需要等待的线程数
    if (ret != 0) {
        OP_LOGW("pthread_barrier_init failed (Error: %s)", strerror(ret));
    }

    pthread_barrier_t barrier2;
    ret = pthread_barrier_init(&barrier2, nullptr, LAUNCH_THREADS);

    vector<std::thread> threadVec;
    for (uint32_t i = 0; i < LAUNCH_THREADS; i++) {
        threadVec.emplace_back(std::thread(TestReselectStaticKernelFunc, &barrier, &barrier2));
    }
    for (auto& t : threadVec) {
        t.join();
    }
    ret = pthread_barrier_destroy(&barrier);
    ret = pthread_barrier_destroy(&barrier2);
    setenv("ACLNN_CACHE_LIMIT", cacheLimit, 1);
    // 还原 debug kernel flag，避免污染同 fixture 其他测试
    op::internal::systemConfig.SetEnableDebugKernelFlag(false);
}

// 镜像 TestReselectStaticKernel，唯一区别：线程函数从 TestReselectStaticKernelFunc 改为
// TestReselectStaticKernelWithPathFunc（调用 withPath 版本 reload）。fixture 适配（注册 op type、
// 启用 debug、预 init testStaticAdd 和 Conv2D、LAUNCH_THREADS=20、结束前 restore enableDebug）完全对称
TEST_F(OpKernelMultiThreadUT, TestReselectStaticKernelWithPath)
{
    op::GenOpTypeId("TestStatic");
    op::GenOpTypeId("TestStaticAdd");
    op::GenOpTypeId("Conv2D");
    // 启用 debug kernel，使 Conv2D 能同时加载 static 与 debug static 二进制（与 TestReselectStaticKernel 对称）
    op::internal::systemConfig.SetEnableDebugKernelFlag(true);
    const char_t* const cacheLimit = std::getenv("ACLNN_CACHE_LIMIT");
    setenv("ACLNN_CACHE_LIMIT", "10000", 1);

    // 单线程预初始化被测 op，避免多线程首次 init 时 mock 环境下 dynamic kernel 加载竞态/失败
    // 注意：故意不预 init testStaticOpType —— 原测试设计要求 testStatic 在初始加载阶段不加载
    uint32_t testStaticAddOpType = op::OpTypeDict::ToOpType("TestStaticAdd");
    uint32_t conv2DOpType = op::OpTypeDict::ToOpType("Conv2D");
    op::internal::KernelMgr::getInstance().AclOpKernelInit(testStaticAddOpType);
    op::internal::KernelMgr::getInstance().AclOpKernelInit(conv2DOpType);

    constexpr uint32_t LAUNCH_THREADS = 20;
    pthread_barrier_t barrier;
    auto ret = pthread_barrier_init(&barrier, nullptr, LAUNCH_THREADS);
    if (ret != 0) {
        OP_LOGW("pthread_barrier_init failed (Error: %s)", strerror(ret));
    }

    pthread_barrier_t barrier2;
    ret = pthread_barrier_init(&barrier2, nullptr, LAUNCH_THREADS);

    vector<std::thread> threadVec;
    for (uint32_t i = 0; i < LAUNCH_THREADS; i++) {
        threadVec.emplace_back(std::thread(TestReselectStaticKernelWithPathFunc, &barrier, &barrier2));
    }
    for (auto& t : threadVec) {
        t.join();
    }
    ret = pthread_barrier_destroy(&barrier);
    ret = pthread_barrier_destroy(&barrier2);
    setenv("ACLNN_CACHE_LIMIT", cacheLimit, 1);
    // 还原 debug kernel flag，避免污染同 fixture 其他测试
    op::internal::systemConfig.SetEnableDebugKernelFlag(false);
}
