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
#include "opdev/op_arg_def.h"
#include "opdev/op_errno.h"
#include "op_ctx_def.h"
#include "register/op_impl_registry.h"
#include "thread_local_context.h"
#include "depends/op/aclnn_mul_stub.h"
#include "depends/dump/dump_stub.h"
#include "depends/runtime/runtime_stub.h"
#include "depends/acl/aclrt_stub.h"

OP_TYPE_REGISTER(Axpy);
OP_TYPE_REGISTER(MemSet);
OP_TYPE_REGISTER(AddN);

extern inline uint32_t SortOpTypeId();

class KernelLaunchUT : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        // aclInit(nullptr);
        setenv("ASCEND_OPP_PATH", OP_API_COMMON_UT_SRC_DIR, 1); // does overwrite
        op::internal::GetThreadLocalContext().cacheHasFull_ = true;
    }
    static void TearDownTestCase() {}
};

TEST_F(KernelLaunchUT, KernelLaunchUTCase1)
{
    op::Shape selfShape{33, 15, 1, 48};
    op::Shape otherShape{33, 15, 14, 48};
    op::Shape outShape{33, 15, 14, 48};

    auto self = std::make_unique<aclTensor>(selfShape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND, nullptr);
    auto other = std::make_unique<aclTensor>(otherShape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND, nullptr);
    float alpha = 13.37;
    auto out = std::make_unique<aclTensor>(outShape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND, nullptr);

    AxpyOpTypeId();
    uint32_t opType = op::OpTypeDict::ToOpType("Axpy");
    auto input = OP_INPUT(self.get(), other.get());
    auto output = OP_OUTPUT(out.get());
    auto attr = OP_ATTR(alpha);
    auto ws = OP_WORKSPACE(out.get());
    auto ctx = op::MakeOpArgContext(input, output, attr, ws);
    int dummyStream = 0;
    void* stream = &dummyStream;
    auto rc = op::internal::gKernelMgr.Run(opType, stream, ctx);
    op::DestroyOpArgContext(ctx);
    EXPECT_EQ(rc, ACL_SUCCESS);
}

TEST_F(KernelLaunchUT, KernelLaunchUTCase2)
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

    SortOpTypeId();
    uint32_t opType = op::OpTypeDict::ToOpType("Sort");
    auto input = OP_INPUT(self.get());
    auto output = OP_OUTPUT(out.get(), idx.get(), static_cast<aclTensor*>(nullptr),
                            static_cast<aclTensorList*>(nullptr));
    auto attr = OP_ATTR(dim, descending);

    auto ws1 = std::make_unique<aclTensor>(wsShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND, nullptr);
    auto ws2 = std::make_unique<aclTensor>(wsShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND, nullptr);
    auto ws3 = std::make_unique<aclTensor>(wsShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND, nullptr);

    const aclTensor* wsArr[] = {ws1.get(), ws2.get(), ws3.get()};
    aclTensorList* wsList = aclCreateTensorList(wsArr, 3);

    auto wsArg = OP_WORKSPACE(wsList);
    auto ctx = op::MakeOpArgContext(input, output, attr, wsArg);
    int dummyStream = 0;
    void* stream = &dummyStream;
    auto rc = op::internal::gKernelMgr.Run(opType, stream, ctx);

    EXPECT_EQ(rc, ACL_SUCCESS);
    delete wsList;
}

TEST_F(KernelLaunchUT, KernelLaunchUTCase3)
{
    op::Shape selfShape{33, 15, 64};
    op::Shape outShape{33, 15, 64};
    op::Shape idxShape{33, 15, 64};

    int alpha = 1337;

    auto self = std::make_unique<aclTensor>(selfShape, op::DataType::DT_INT32, op::Format::FORMAT_ND, nullptr);
    auto other = std::make_unique<aclTensor>(outShape, op::DataType::DT_INT32, op::Format::FORMAT_ND, nullptr);
    auto out = std::make_unique<aclTensor>(idxShape, op::DataType::DT_INT32, op::Format::FORMAT_ND, nullptr);

    const aclTensor* inputTensor[2] = {self.get(), other.get()};
    aclTensorList* inputList = aclCreateTensorList(inputTensor, 2);

    uint32_t opType = op::OpTypeDict::ToOpType("AddN");
    auto input = OP_INPUT(inputList);
    auto output = OP_OUTPUT(out.get());
    auto attr = OP_ATTR(alpha);
    auto ws = OP_WORKSPACE(out.get());
    auto ctx = op::MakeOpArgContext(input, output, attr, ws);
    int dummyStream = 0;
    void* stream = &dummyStream;
    auto rc = op::internal::gKernelMgr.Run(opType, stream, ctx);
    delete inputList;
    op::DestroyOpArgContext(ctx);
    EXPECT_EQ(rc, ACL_SUCCESS);
}

TEST_F(KernelLaunchUT, KernelLaunchUTCase4)
{
    op::Shape selfShape{33, 15, 64};
    op::Shape outShape{33, 15, 64};
    op::Shape idxShape{33, 15, 64};

    auto self = std::make_unique<aclTensor>(selfShape, op::DataType::DT_INT32, op::Format::FORMAT_ND, nullptr);
    auto other = std::make_unique<aclTensor>(outShape, op::DataType::DT_INT32, op::Format::FORMAT_ND, nullptr);
    auto out = std::make_unique<aclTensor>(idxShape, op::DataType::DT_INT32, op::Format::FORMAT_ND, nullptr);

    const aclTensor* inputTensor[2] = {self.get(), other.get()};
    aclTensorList* inputList = aclCreateTensorList(inputTensor, 2);

    auto input = OP_INPUT(inputList);
    auto output = OP_OUTPUT(out.get());
    auto ctx = op::MakeOpArgContext(input, output);

    size_t tn_list = op::internal::GetAclTensorCount(*ctx->GetOpArg(op::OpArgDef::OP_INPUT_ARG));
    size_t tn = op::internal::GetAclTensorCount(*ctx->GetOpArg(op::OpArgDef::OP_OUTPUT_ARG));
    delete inputList;
    op::DestroyOpArgContext(ctx);
    EXPECT_EQ(tn, 1u);
    EXPECT_EQ(tn_list, 2u);
}

TEST_F(KernelLaunchUT, MemSetTiling1)
{
    op::internal::MemSetKernelContextHolder ctx;

    op::internal::MemSetTensorInfo tensor{0,       ge::DT_FLOAT, 0.0f, 0, 100, 256, op::OpArgType::OPARG_ACLTENSOR,
                                          nullptr, nullptr};
    std::vector<op::internal::MemSetTensorInfo> tensorInfo{tensor};
    ctx.UpdateComputeNodeInfo(tensorInfo);

    EXPECT_EQ(ctx.inputNum_, 2ul);
}

TEST_F(KernelLaunchUT, KernelLaunchUT_Outshape)
{
    op::Shape selfShape{33, 15, 64};
    op::Shape outShape{33, 15, 64};
    op::Shape idxShape{33, 15, 64};

    op::Shape outShapeShape{4, 9};

    op::Shape wsShape{64};

    auto self = std::make_unique<aclTensor>(selfShape, op::DataType::DT_INT32, op::Format::FORMAT_ND, nullptr);
    auto other = std::make_unique<aclTensor>(outShape, op::DataType::DT_INT32, op::Format::FORMAT_ND, nullptr);
    auto out = std::make_unique<aclTensor>(idxShape, op::DataType::DT_INT32, op::Format::FORMAT_ND, nullptr);
    auto ws = std::make_unique<aclTensor>(outShapeShape, op::DataType::DT_INT32, op::Format::FORMAT_ND, nullptr);
    auto outshape = std::make_unique<aclTensor>(outShapeShape, op::DataType::DT_INT32, op::Format::FORMAT_ND, nullptr);

    auto input_arg = OP_INPUT(self.get(), other.get());
    auto output_arg = OP_OUTPUT(out.get());
    auto ws_arg = OP_WORKSPACE(ws.get());
    auto outshape_arg = OP_OUTSHAPE(outshape.get(), 0);
    auto attr_arg = OP_ATTR(123);
    auto ctx = op::MakeOpArgContext(input_arg, output_arg, ws_arg, outshape_arg, attr_arg);

    AxpyOpTypeId();
    uint32_t opType = op::OpTypeDict::ToOpType("Axpy");
    int dummyStream = 0;
    void* stream = &dummyStream;
    auto rc = op::internal::gKernelMgr.Run(opType, stream, ctx);
    op::DestroyOpArgContext(ctx);
    EXPECT_EQ(rc, ACL_SUCCESS);
}

TEST_F(KernelLaunchUT, GetAclTensorCountTerst1)
{
    aclTensor* nullTensor = nullptr;
    aclTensorList* nullTensorList = nullptr;
    auto input_arg = OP_INPUT(nullTensor, nullTensorList);
    auto ctx = op::MakeOpArgContext(input_arg);

    size_t num = op::internal::GetAclTensorCount(*ctx->GetOpArg(op::OpArgDef::OP_INPUT_ARG));
    op::DestroyOpArgContext(ctx);
    EXPECT_EQ(num, 0u);
}

TEST_F(KernelLaunchUT, SetMemSetFlagFromJsonTest)
{
    const char* p = std::getenv("ASCEND_OPP_PATH");
    EXPECT_NE(p, nullptr);

    op::internal::KeyAndDetail key;
    key.key = "hahaha";
    size_t hashKey = 123;

    for (int i = 1; i <= 4; i++) {
        char jsonPath[1024];
        char binPath[1024];
        snprintf_s(jsonPath, sizeof(jsonPath), sizeof(jsonPath),
                   "%s/built-in/op_impl/ai_core/tbe/kernel/ascend910/dummy/dummy_%d.json", p, i);
        snprintf_s(binPath, sizeof(binPath), sizeof(binPath),
                   "%s/built-in/op_impl/ai_core/tbe/kernel/ascend910/dummy/dummy_%d.o", p, i);
        op::internal::OpKernelBin kernel(9999, jsonPath, jsonPath, binPath, key, hashKey,
                                         op::internal::BinType::DYNAMIC_BIN, false, false);
        aclnnStatus rc = kernel.JsonLoad();
        EXPECT_EQ(rc, ACLNN_SUCCESS);
        kernel.GetTaskInfo(0);
    }
    op::internal::ReportNodeContextIdInfo(10);
}

TEST_F(KernelLaunchUT, RtsArgTest)
{
    op::Shape selfShape{100};
    op::Shape outShape{100};

    aclOpExecutor exe;

    int64_t inputData[100];
    auto self = exe.AllocIntArray(inputData, 100);
    auto out = exe.AllocTensor(outShape, op::DataType::DT_INT32);
    auto inputTensor = exe.ConvertToTensor(self, op::DataType::DT_INT32);

    auto input_arg = OP_INPUT(inputTensor);
    auto output_arg = OP_OUTPUT(out);
    auto ctx = op::MakeOpArgContext(input_arg, output_arg);

    op::internal::ExpandableRtsArgBuffer buffer;
    buffer.Init(1000, 2000);
    op::internal::TilingData* tilingData = buffer.GetTilingDataPtr();
    tilingData->data_size_ = 60;

    op::internal::LaunchArgInfo argInfo(false, false, ctx);
    op::internal::RtsArg arg(true, argInfo, &buffer);
    arg.FillArgs();

    op::internal::KernelLaunchConfig launchCfg;
    launchCfg.funcHandle = (void*)0x12345678;
    launchCfg.numBlocks = 32;
    launchCfg.schemMode = 1;
    launchCfg.dynUBufSize = 0;
    launchCfg.blockDimOffset = 0;
    launchCfg.engineType = op::internal::LaunchKernelEngineType::NO_VECTOR_CORE;
    aclnnStatus rc = arg.LaunchKernel(nullptr, launchCfg);

    EXPECT_EQ(rc, ACLNN_SUCCESS);
    op::DestroyOpArgContext(ctx);
}

TEST_F(KernelLaunchUT, TestWithHandleBlockDimOffset1)
{
    op::Shape selfShape{100};
    op::Shape outShape{100};

    aclOpExecutor exe;

    int64_t inputData[100];
    auto self = exe.AllocIntArray(inputData, 100);
    auto out = exe.AllocTensor(outShape, op::DataType::DT_INT32);
    auto inputTensor = exe.ConvertToTensor(self, op::DataType::DT_INT32);

    auto input_arg = OP_INPUT(inputTensor);
    auto output_arg = OP_OUTPUT(out);
    auto ctx = op::MakeOpArgContext(input_arg, output_arg);

    op::internal::ExpandableRtsArgBuffer buffer;
    buffer.Init(1000, 2000);
    op::internal::TilingData* tilingData = buffer.GetTilingDataPtr();
    tilingData->data_size_ = 60;

    op::internal::LaunchArgInfo argInfo(false, false, ctx);
    op::internal::RtsArg arg(true, argInfo, &buffer);
    arg.FillArgs();

    op::internal::KernelLaunchConfig launchCfg;
    launchCfg.funcHandle = (void*)0x12345678;
    launchCfg.numBlocks = 32;
    launchCfg.schemMode = 1;
    launchCfg.dynUBufSize = 0;
    launchCfg.blockDimOffset = 10;
    launchCfg.engineType = op::internal::LaunchKernelEngineType::NO_VECTOR_CORE;
    aclnnStatus rc = arg.LaunchKernel(nullptr, launchCfg);

    EXPECT_EQ(rc, ACLNN_SUCCESS);
    op::DestroyOpArgContext(ctx);
}

TEST_F(KernelLaunchUT, TestWithFlagBlockDimOffset1)
{
    op::Shape selfShape{100};
    op::Shape outShape{100};

    aclOpExecutor exe;

    int64_t inputData[100];
    auto self = exe.AllocIntArray(inputData, 100);
    auto out = exe.AllocTensor(outShape, op::DataType::DT_INT32);
    auto inputTensor = exe.ConvertToTensor(self, op::DataType::DT_INT32);

    auto input_arg = OP_INPUT(inputTensor);
    auto output_arg = OP_OUTPUT(out);
    auto ctx = op::MakeOpArgContext(input_arg, output_arg);

    op::internal::ExpandableRtsArgBuffer buffer;
    buffer.Init(1000, 2000);
    op::internal::TilingData* tilingData = buffer.GetTilingDataPtr();
    tilingData->data_size_ = 60;

    op::internal::LaunchArgInfo argInfo(false, false, ctx);
    op::internal::RtsArg arg(true, argInfo, &buffer);
    arg.FillArgs();

    op::internal::KernelLaunchConfig launchCfg;
    launchCfg.funcHandle = (void*)0x12345678;
    launchCfg.numBlocks = 32;
    launchCfg.schemMode = 1;
    launchCfg.dynUBufSize = 0;
    launchCfg.blockDimOffset = 10;
    launchCfg.engineType = op::internal::LaunchKernelEngineType::NO_VECTOR_CORE;
    aclnnStatus rc = arg.LaunchKernel(nullptr, launchCfg);

    EXPECT_EQ(rc, ACLNN_SUCCESS);
    op::DestroyOpArgContext(ctx);
}

class DoLaunchNormalTestAclrtStub : public AclrtStub {
public:
    aclError aclrtLaunchKernelWithHostArgs(aclrtFuncHandle funcHandle, uint32_t blockDim, aclrtStream stream,
                                           aclrtLaunchKernelCfg* cfg, void* hostArgs, size_t argsSize,
                                           aclrtPlaceHolderInfo* placeHolderArray, size_t placeHolderNum)
    {
        OP_LOGI("DoLaunchNormalTestAclrtStub rtsLaunchKernelWithHostArgs start");
        // check rts params
        EXPECT_EQ(funcHandle, (void*)0x12341234);
        EXPECT_EQ(blockDim, 17);
        EXPECT_EQ(stream, nullptr);
        EXPECT_EQ(argsSize, 72);

        // check ptr
        void** ptrArgs = reinterpret_cast<void**>(hostArgs);
        EXPECT_EQ(ptrArgs[0], input1);
        EXPECT_EQ(ptrArgs[1], input2);
        EXPECT_EQ(ptrArgs[2], output);
        EXPECT_EQ(ptrArgs[3], nullptr);
        EXPECT_EQ(ptrArgs[4], overflow);

        EXPECT_NE(placeHolderArray, nullptr);
        EXPECT_EQ(placeHolderNum, 1);

        // check tiling data
        EXPECT_EQ(placeHolderArray[0].addrOffset, 24);
        EXPECT_EQ(placeHolderArray[0].dataOffset, 40);
        int32_t* tilingVal = reinterpret_cast<int32_t*>(reinterpret_cast<char*>(hostArgs) + 40);
        EXPECT_EQ(tilingVal[0], 8910);
        EXPECT_EQ(tilingVal[1], 5525); // dfx info dump atomic index
        for (int i = 2; i < 8; i++) {
            EXPECT_EQ(tilingVal[i], 0);
        }

        return RT_ERROR_NONE;
    }

    void* input1;
    void* input2;
    void* output;
    void* overflow;

    void* argPtr;
    uint32_t argsSize;
};

TEST_F(KernelLaunchUT, TestWithHostapi)
{
    int fakeData = 1;
    int fakeData2 = 2;
    int fakeData3 = 3;
    std::vector<int64_t> shape = {1, 1, 1, 1, 1};
    aclTensor* input1 = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT16, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), &fakeData);

    aclTensor* input2 = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT16, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), &fakeData2);

    aclTensor* output = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT16, nullptr, 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), &fakeData3);
    aclOpExecutor* executor = nullptr;
    size_t workspaceLen = 0U;
    auto ret = aclnnMulStubGetWorkspaceSize(input1, input2, output, &workspaceLen, &executor);
    EXPECT_EQ(ret, ACLNN_SUCCESS);

    DoLaunchNormalTestAclrtStub aclrtStub;
    aclrtStub.input1 = &fakeData;
    aclrtStub.input2 = &fakeData2;
    aclrtStub.output = &fakeData3;
    aclrtStub.overflow = (void*)0x005;
    AclrtStub::GetInstance()->Install(&aclrtStub);

    aclrtStream stream = nullptr;
    ret = aclnnMulStub(nullptr, 0, executor, stream);
    EXPECT_EQ(ret, ACLNN_SUCCESS);

    AclrtStub::GetInstance()->UnInstall();
}
