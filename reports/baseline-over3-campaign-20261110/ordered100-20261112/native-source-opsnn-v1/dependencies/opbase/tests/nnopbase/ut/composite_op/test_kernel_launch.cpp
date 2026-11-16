/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <iostream>
#include <memory>
#include <stdlib.h>

#include "gtest/gtest.h"
#include "gtest/gtest_prod.h"
#include "acl/acl.h"
#include "register/op_impl_registry.h"

#include "aclnn/acl_meta.h"
#include "aclnn/aclnn_base.h"

#include "kernel_mgr.h"
#include "memset_ctx_holder.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/op_arg_def.h"
#include "opdev/op_errno.h"
#include "op_ctx_def.h"
#include "thread_local_context.h"
#include "op_kernel.h"
#include "memset_op.h"

#include "depends/platform/platform_stub.h"
#include "kernel_launcher.h"
#include "depends/acl/aclrt_stub.h"
#include "test_comp_op_common.h"

using namespace op;
using namespace op::internal;
using namespace op::internal::test;
extern inline uint32_t SortOpTypeId();
OP_TYPE_REGISTER(Axpy);
OP_TYPE_REGISTER(MemSet);
OP_TYPE_REGISTER(MemSetV2);
OP_TYPE_REGISTER(AddN);

class KernelLaunchUT : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        // aclInit(nullptr);
        setenv("ASCEND_OPP_PATH", OP_API_COMMON_UT_SRC_DIR, 1); // does overwrite
        op::internal::GetThreadLocalContext().cacheHasFull_ = true;
        AxpyOpTypeId();
        SortOpTypeId();
        AddNOpTypeId();
    }
    static void TearDownTestCase() {}

    op::internal::OpKernelBin* CreateFakeOpKernelBin(bool hasDevPtrArg)
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
        op::internal::OpKernelBin* fakeBin = new op::internal::OpKernelBin(
            opType, jsonPath, jsonPath, binPath, key, hashKey, BinType::DYNAMIC_BIN, false, hasDevPtrArg);
        return fakeBin;
    }
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

    uint32_t opType = op::OpTypeDict::ToOpType("Axpy");
    auto input = OP_INPUT(self.get(), other.get());
    auto output = OP_OUTPUT(out.get());
    auto attr = OP_ATTR(alpha);
    auto ws = OP_WORKSPACE(out.get());
    int dummyStream = 0;
    void* stream = &dummyStream;
    // create arg
    auto ctx = op::MakeOpArgContext(input, output, attr, ws);
    auto rc = op::internal::gKernelMgr.Run(opType, stream, ctx);

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
    int dummyStream = 0;
    void* stream = &dummyStream;
    // create arg
    auto ctx = op::MakeOpArgContext(input, output, attr, wsArg);
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
    int dummyStream = 0;
    void* stream = &dummyStream;
    auto ctx = op::MakeOpArgContext(input, output, attr, ws);
    auto rc = op::internal::gKernelMgr.Run(opType, stream, ctx);
    delete inputList;
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

    size_t tn_list = op::internal::GetAclTensorCount(*(ctx->GetOpArg(op::OpArgDef::OP_INPUT_ARG)));
    size_t tn = op::internal::GetAclTensorCount(*(ctx->GetOpArg(op::OpArgDef::OP_OUTPUT_ARG)));
    delete inputList;
    EXPECT_EQ(tn, 1u);
    EXPECT_EQ(tn_list, 2u);
}

TEST_F(KernelLaunchUT, MemSetTiling1)
{
    op::internal::MemSetKernelContextHolder ctx;

    op::internal::MemSetTensorInfo tensor{0, ge::DT_FLOAT, 0.0f, 0, 100, 256, op::OpArgType::OPARG_FLOAT};
    std::vector<op::internal::MemSetTensorInfo> tensorInfo{tensor};
    ctx.UpdateComputeNodeInfo(tensorInfo);

    EXPECT_EQ(ctx.inputNum_, 2ul);
}

extern "C" int InitHugeMemThreadLocal(void* arg, bool sync);
extern "C" void UnInitHugeMemThreadLocal(void* arg, bool sync);
extern "C" void ReleaseHugeMem(void* arg, bool sync);

static void MemSetV2OutputTensorNoDevPtr(op::internal::OpKernelBin* kernelBin)
{
    OP_LOGI("not support dev ptr");
    InitHugeMemThreadLocal(nullptr, false);
    PlatformInfoStub::GetInstance()->SetSoCVersion("Ascend950", "Ascend950DT_9591");
    // create input output workspace outputshape
    op::Shape selfShape{33, 15, 64};
    op::Shape wsShape{32};
    // create tensors
    int addr[6] = {0};
    aclTensor tensor1(selfShape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND, &addr[0]);
    tensor1.SetFromWorkspace(false);
    aclTensor* tensorPtr1 = &tensor1;
    aclTensor tensor2(selfShape, op::DataType::DT_INT32, op::Format::FORMAT_ND, &addr[1]);
    tensor2.SetFromWorkspace(false);
    aclTensor* tensorPtr2 = &tensor2;
    aclTensor tensor3(selfShape, op::DataType::DT_INT32, op::Format::FORMAT_ND, &addr[2]);
    tensor3.SetFromWorkspace(false);
    aclTensor* tensorPtr3 = &tensor3;
    aclTensor tensor4(wsShape, op::DataType::DT_UINT32, op::Format::FORMAT_ND, &addr[3]);
    tensor4.SetFromWorkspace(false);
    aclTensor* tensorPtr4 = &tensor4;
    aclTensor tensor5(wsShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND, &addr[4]);
    tensor5.SetFromWorkspace(false);
    aclTensor* tensorPtr5 = &tensor5;
    aclTensor tensor6(wsShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND, &addr[5]);
    tensor6.SetFromWorkspace(false);
    aclTensor* tensorPtr6 = &tensor6;
    // int32 tensor list
    const aclTensor* tensorArr1[3] = {tensorPtr2, tensorPtr3, nullptr};
    aclTensorList* tensorList1 = aclCreateTensorList(tensorArr1, 3);
    const aclTensor* tensorsArr2[2] = {tensorPtr5, tensorPtr6};
    aclTensorList* workspaceTensorList = aclCreateTensorList(tensorsArr2, 2);

    auto input = OP_INPUT(tensorPtr1, tensorList1);
    auto output = OP_OUTPUT(tensorPtr1, tensorList1);
    auto workspace = OP_WORKSPACE(tensorPtr4, workspaceTensorList);
    auto outshape = OP_OUTSHAPE(tensorPtr4, 0);
    // create arg
    auto ctx = op::MakeOpArgContext(input, output, workspace, outshape);

    kernelBin->memSetValue_ = {
        {3, op::DataType::DT_FLOAT, 1.1f, 0, 100, 256, op::OpArgType::OPARG_ACLTENSOR, nullptr, nullptr, nullptr},
        {4, op::DataType::DT_INT32, 0.0f, 1, 100, 256, op::OpArgType::OPARG_ACLTENSOR, nullptr, nullptr, nullptr},
        {5, op::DataType::DT_INT32, 0.0f, 2, 100, 256, op::OpArgType::OPARG_ACLTENSOR, nullptr, nullptr, nullptr},
        {6, op::DataType::DT_UINT32, 0.0f, 3, 100, 256, op::OpArgType::OPARG_ACLTENSOR, nullptr, nullptr, nullptr},
        {7, op::DataType::DT_FLOAT16, 2.2f, 0, 100, 256, op::OpArgType::OPARG_ACLTENSOR, nullptr, nullptr, nullptr},
        {8, op::DataType::DT_FLOAT16, 3.3f, 0, 100, 256, op::OpArgType::OPARG_ACLTENSOR, nullptr, nullptr, nullptr},
        {9, op::DataType::DT_UINT32, 0.0f, 4, 100, 256, op::OpArgType::OPARG_ACLTENSOR, nullptr, nullptr, nullptr}};

    auto unique_executor = CREATE_EXECUTOR();
    aclOpExecutor* executor = unique_executor.get();
    op::internal::GetThreadLocalContext().executor_ = executor;
    aclrtStream stream = 0;
    auto ret = kernelBin->MemsetOutputTensor(stream, ctx);
    EXPECT_EQ(ret, ACLNN_SUCCESS);

    EXPECT_EQ(kernelBin->memSetValueCtx_.size(), 7);
    EXPECT_EQ((kernelBin->memSetValueCtx_)[0].argIdx_, 3);
    EXPECT_EQ((kernelBin->memSetValueCtx_)[0].tensor_, tensorPtr1);
    EXPECT_EQ((kernelBin->memSetValueCtx_)[0].tensorList_, nullptr);
    EXPECT_EQ((kernelBin->memSetValueCtx_)[1].argIdx_, 4);
    EXPECT_EQ((kernelBin->memSetValueCtx_)[1].tensor_, tensorPtr2);
    EXPECT_EQ((kernelBin->memSetValueCtx_)[1].tensorList_, nullptr);
    EXPECT_EQ((kernelBin->memSetValueCtx_)[2].argIdx_, 5);
    EXPECT_EQ((kernelBin->memSetValueCtx_)[2].tensor_, tensorPtr3);
    EXPECT_EQ((kernelBin->memSetValueCtx_)[2].tensorList_, nullptr);
    EXPECT_EQ((kernelBin->memSetValueCtx_)[3].argIdx_, 6);
    EXPECT_EQ((kernelBin->memSetValueCtx_)[3].tensor_, tensorPtr4);
    EXPECT_EQ((kernelBin->memSetValueCtx_)[3].tensorList_, nullptr);
    EXPECT_EQ((kernelBin->memSetValueCtx_)[4].argIdx_, 7);
    EXPECT_EQ((kernelBin->memSetValueCtx_)[4].tensor_, tensorPtr5);
    EXPECT_EQ((kernelBin->memSetValueCtx_)[4].tensorList_, nullptr);
    EXPECT_EQ((kernelBin->memSetValueCtx_)[5].argIdx_, 8);
    EXPECT_EQ((kernelBin->memSetValueCtx_)[5].tensor_, tensorPtr6);
    EXPECT_EQ((kernelBin->memSetValueCtx_)[5].tensorList_, nullptr);
    EXPECT_EQ((kernelBin->memSetValueCtx_)[6].argIdx_, 9);
    EXPECT_EQ((kernelBin->memSetValueCtx_)[6].tensor_, tensorPtr4);
    EXPECT_EQ((kernelBin->memSetValueCtx_)[6].tensorList_, nullptr);

    MemsetV2ArgContext memsetV2ArgCtx;
    ret = memsetV2ArgCtx.Init(kernelBin->memSetValueCtx_);
    EXPECT_EQ(ret, ACLNN_SUCCESS);
    op::OpArgContext* memsetV2OpArgCtx = memsetV2ArgCtx.GetMemsetV2OpArgContext();
    EXPECT_EQ(memsetV2OpArgCtx->ContainsOpArgType(op::OP_INPUT_ARG), true);
    EXPECT_EQ(memsetV2OpArgCtx->ContainsOpArgType(op::OP_OUTPUT_ARG), true);
    EXPECT_EQ(memsetV2OpArgCtx->ContainsOpArgType(op::OP_WORKSPACE_ARG), true);
    EXPECT_EQ(memsetV2ArgCtx.memsetTensors_->Size(), 7);
    EXPECT_EQ((*memsetV2ArgCtx.memsetTensors_)[0], tensorPtr1);
    EXPECT_EQ((*memsetV2ArgCtx.memsetTensors_)[1], tensorPtr2);
    EXPECT_EQ((*memsetV2ArgCtx.memsetTensors_)[2], tensorPtr3);
    EXPECT_EQ((*memsetV2ArgCtx.memsetTensors_)[3], tensorPtr4);
    EXPECT_EQ((*memsetV2ArgCtx.memsetTensors_)[4], tensorPtr5);
    EXPECT_EQ((*memsetV2ArgCtx.memsetTensors_)[5], tensorPtr6);
    EXPECT_EQ((*memsetV2ArgCtx.memsetTensors_)[6], tensorPtr4);
    EXPECT_EQ(memsetV2ArgCtx.intAttrArray_->Size(), 4);
    EXPECT_EQ((*memsetV2ArgCtx.intAttrArray_)[0], 1);
    EXPECT_EQ((*memsetV2ArgCtx.intAttrArray_)[1], 2);
    EXPECT_EQ((*memsetV2ArgCtx.intAttrArray_)[2], 3);
    EXPECT_EQ((*memsetV2ArgCtx.intAttrArray_)[3], 4);
    EXPECT_EQ(memsetV2ArgCtx.floatAttrArray_->Size(), 3);
    EXPECT_FLOAT_EQ((*memsetV2ArgCtx.floatAttrArray_)[0], 1.1);
    EXPECT_FLOAT_EQ((*memsetV2ArgCtx.floatAttrArray_)[1], 2.2);
    EXPECT_FLOAT_EQ((*memsetV2ArgCtx.floatAttrArray_)[2], 3.3);

    unique_executor.ReleaseTo(&executor);
    delete executor;
    delete tensorList1;
    delete workspaceTensorList;
    DestroyOpArgContext(ctx);
    ReleaseHugeMem(nullptr, false);
    UnInitHugeMemThreadLocal(nullptr, false);
}

static void MemSetV2OutputTensorWithDevPtr(op::internal::OpKernelBin* kernelBin)
{
    OP_LOGI("support dev ptr");
    PlatformInfoStub::GetInstance()->SetSoCVersion("Ascend950", "Ascend950DT_9591");
    // create input output workspace outputshape
    op::Shape selfShape{33, 15, 64};
    op::Shape wsShape{32};
    // create tensors
    int addr[6] = {0};
    aclTensor tensor1(selfShape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND, &addr[0]);
    tensor1.SetFromWorkspace(false);
    aclTensor* tensorPtr1 = &tensor1;
    aclTensor tensor2(selfShape, op::DataType::DT_INT32, op::Format::FORMAT_ND, &addr[1]);
    tensor2.SetFromWorkspace(false);
    aclTensor* tensorPtr2 = &tensor2;
    aclTensor tensor3(selfShape, op::DataType::DT_INT32, op::Format::FORMAT_ND, &addr[2]);
    tensor3.SetFromWorkspace(false);
    aclTensor* tensorPtr3 = &tensor3;
    aclTensor tensor4(selfShape, op::DataType::DT_UINT32, op::Format::FORMAT_ND, &addr[3]);
    tensor4.SetFromWorkspace(false);
    aclTensor* tensorPtr4 = &tensor4;
    aclTensor tensor5(wsShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND, &addr[4]);
    tensor5.SetFromWorkspace(false);
    aclTensor* tensorPtr5 = &tensor5;
    aclTensor tensor6(wsShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND, &addr[5]);
    tensor6.SetFromWorkspace(false);
    aclTensor* tensorPtr6 = &tensor6;
    // int32 tensor list
    const aclTensor* tensorArr1[3] = {tensorPtr2, tensorPtr3, nullptr};
    aclTensorList* tensorList1 = aclCreateTensorList(tensorArr1, 3);

    auto input = OP_INPUT(tensorPtr1, tensorList1);
    auto output = OP_OUTPUT(tensorPtr1, tensorList1, nullptr);
    auto outshape = OP_OUTSHAPE(tensorPtr4, 0);
    // create arg
    auto ctx = op::MakeOpArgContext(input, output, outshape);
    // append workspace float16 tensor list
    const aclTensor* tensorsArr2[2] = {tensorPtr5, tensorPtr6};
    aclTensorList* workspaceTensorList = aclCreateTensorList(tensorsArr2, 2);
    EXPECT_EQ(ctx->ContainsOpArgType(OP_WORKSPACE_ARG), false);
    ctx->AppendOpWorkspaceArg(workspaceTensorList);
    EXPECT_EQ(ctx->ContainsOpArgType(OP_WORKSPACE_ARG), true);

    kernelBin->memSetValue_ = {
        {2, op::DataType::DT_FLOAT, 6.6f, 0, 100, 256, op::OpArgType::OPARG_ACLTENSOR, nullptr, nullptr, nullptr},
        {3, op::DataType::DT_INT32, 0.0f, 6, 100, 256, op::OpArgType::OPARG_ACLTENSOR_LIST, nullptr, nullptr, nullptr},
        {4, op::DataType::DT_FLOAT16, 8.8f, 0, 100, 256, op::OpArgType::OPARG_ACLTENSOR_LIST, nullptr, nullptr,
         nullptr},
        {5, op::DataType::DT_UINT32, 0.0f, 8, 100, 256, op::OpArgType::OPARG_ACLTENSOR, nullptr, nullptr, nullptr}};

    auto unique_executor = CREATE_EXECUTOR();
    aclOpExecutor* executor = unique_executor.get();
    op::internal::GetThreadLocalContext().executor_ = executor;
    aclrtStream stream = 0;
    auto ret = kernelBin->MemsetOutputTensor(stream, ctx);
    EXPECT_EQ(ret, ACLNN_SUCCESS);

    EXPECT_EQ(kernelBin->memSetValueCtx_.size(), 4);
    EXPECT_EQ((kernelBin->memSetValueCtx_)[0].argIdx_, 2);
    EXPECT_EQ((kernelBin->memSetValueCtx_)[0].tensor_, tensorPtr1);
    EXPECT_EQ((kernelBin->memSetValueCtx_)[0].tensorList_, nullptr);
    EXPECT_EQ((kernelBin->memSetValueCtx_)[1].argIdx_, 3);
    EXPECT_EQ((kernelBin->memSetValueCtx_)[1].tensor_, nullptr);
    EXPECT_EQ((kernelBin->memSetValueCtx_)[1].tensorList_, tensorList1);
    EXPECT_EQ((kernelBin->memSetValueCtx_)[2].argIdx_, 4);
    EXPECT_EQ((kernelBin->memSetValueCtx_)[2].tensor_, nullptr);
    EXPECT_EQ((kernelBin->memSetValueCtx_)[2].tensorList_, workspaceTensorList);
    EXPECT_EQ((kernelBin->memSetValueCtx_)[3].argIdx_, 5);
    EXPECT_EQ((kernelBin->memSetValueCtx_)[3].tensor_, tensorPtr4);
    EXPECT_EQ((kernelBin->memSetValueCtx_)[3].tensorList_, nullptr);

    MemsetV2ArgContext memsetV2ArgCtx;
    ret = memsetV2ArgCtx.Init(kernelBin->memSetValueCtx_);
    EXPECT_EQ(ret, ACLNN_SUCCESS);
    op::OpArgContext* memsetV2OpArgCtx = memsetV2ArgCtx.GetMemsetV2OpArgContext();
    EXPECT_EQ(memsetV2OpArgCtx->ContainsOpArgType(op::OP_INPUT_ARG), true);
    EXPECT_EQ(memsetV2OpArgCtx->ContainsOpArgType(op::OP_OUTPUT_ARG), true);
    EXPECT_EQ(memsetV2OpArgCtx->ContainsOpArgType(op::OP_WORKSPACE_ARG), true);

    EXPECT_EQ(memsetV2ArgCtx.memsetTensors_->Size(), 6);
    EXPECT_EQ((*memsetV2ArgCtx.memsetTensors_)[0], tensorPtr1);
    EXPECT_EQ((*memsetV2ArgCtx.memsetTensors_)[1], tensorPtr2);
    EXPECT_EQ((*memsetV2ArgCtx.memsetTensors_)[2], tensorPtr3);
    EXPECT_EQ((*memsetV2ArgCtx.memsetTensors_)[3], tensorPtr5);
    EXPECT_EQ((*memsetV2ArgCtx.memsetTensors_)[4], tensorPtr6);
    EXPECT_EQ((*memsetV2ArgCtx.memsetTensors_)[5], tensorPtr4);
    EXPECT_EQ(memsetV2ArgCtx.intAttrArray_->Size(), 3);
    EXPECT_EQ((*memsetV2ArgCtx.intAttrArray_)[0], 6);
    EXPECT_EQ((*memsetV2ArgCtx.intAttrArray_)[1], 6);
    EXPECT_EQ((*memsetV2ArgCtx.intAttrArray_)[2], 8);
    EXPECT_EQ(memsetV2ArgCtx.floatAttrArray_->Size(), 3);
    EXPECT_FLOAT_EQ((*memsetV2ArgCtx.floatAttrArray_)[0], 6.6);
    EXPECT_FLOAT_EQ((*memsetV2ArgCtx.floatAttrArray_)[1], 8.8);
    EXPECT_FLOAT_EQ((*memsetV2ArgCtx.floatAttrArray_)[2], 8.8);

    unique_executor.ReleaseTo(&executor);
    delete executor;
    delete tensorList1;
    delete workspaceTensorList;
    DestroyOpArgContext(ctx);
}

static void MemSetV2OutputTensorWithDevPtr2(op::internal::OpKernelBin* kernelBin)
{
    OP_LOGI("support dev ptr 2");
    PlatformInfoStub::GetInstance()->SetSoCVersion("Ascend950", "Ascend950DT_9591");
    // create input output workspace outputshape
    op::Shape selfShape{33, 15, 64};
    op::Shape wsShape{31, 32};
    // create tensors
    int addr[6] = {0};
    aclTensor tensor1(selfShape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND, &addr[0]);
    tensor1.SetFromWorkspace(false);
    aclTensor* tensorPtr1 = &tensor1;
    aclTensor tensor2(wsShape, op::DataType::DT_INT32, op::Format::FORMAT_ND, &addr[1]);
    tensor2.SetFromWorkspace(false);
    aclTensor* tensorPtr2 = &tensor2;
    aclTensor tensor3(wsShape, op::DataType::DT_INT32, op::Format::FORMAT_ND, &addr[2]);
    tensor3.SetFromWorkspace(false);
    aclTensor* tensorPtr3 = &tensor3;

    // int32 tensor list
    const aclTensor* tensorArr1[3] = {tensorPtr2, tensorPtr3, nullptr};
    aclTensorList* tensorList1 = aclCreateTensorList(tensorArr1, 3);

    kernelBin->memSetValueCtx_ = {
        {2, op::DataType::DT_FLOAT, 6.6f, 0, 100, 256, op::OpArgType::OPARG_ACLTENSOR, nullptr, tensorPtr1, nullptr},
        {3, op::DataType::DT_FLOAT, 8.8f, 6, 100, 256, op::OpArgType::OPARG_ACLTENSOR, nullptr, nullptr, nullptr},
        {4, op::DataType::DT_INT32, 0.0f, 6, 100, 256, op::OpArgType::OPARG_ACLTENSOR_LIST, nullptr, nullptr,
         tensorList1},
        {5, op::DataType::DT_INT32, 0.0f, 8, 100, 256, op::OpArgType::OPARG_ACLTENSOR_LIST, nullptr, nullptr, nullptr}};

    auto unique_executor = CREATE_EXECUTOR();
    aclOpExecutor* executor = unique_executor.get();
    op::internal::GetThreadLocalContext().executor_ = executor;

    MemsetV2ArgContext memsetV2ArgCtx;
    auto ret = memsetV2ArgCtx.Init(kernelBin->memSetValueCtx_);
    EXPECT_EQ(ret, ACLNN_SUCCESS);
    op::OpArgContext* memsetV2OpArgCtx = memsetV2ArgCtx.GetMemsetV2OpArgContext();

    EXPECT_EQ(memsetV2ArgCtx.memsetTensors_->Size(), 3);
    EXPECT_EQ((*memsetV2ArgCtx.memsetTensors_)[0], tensorPtr1);
    EXPECT_EQ((*memsetV2ArgCtx.memsetTensors_)[1], tensorPtr2);
    EXPECT_EQ((*memsetV2ArgCtx.memsetTensors_)[2], tensorPtr3);
    EXPECT_EQ(memsetV2ArgCtx.intAttrArray_->Size(), 2);
    EXPECT_EQ((*memsetV2ArgCtx.intAttrArray_)[0], 6);
    EXPECT_EQ((*memsetV2ArgCtx.intAttrArray_)[1], 6);
    EXPECT_EQ(memsetV2ArgCtx.floatAttrArray_->Size(), 1);
    EXPECT_FLOAT_EQ((*memsetV2ArgCtx.floatAttrArray_)[0], 6.6);

    unique_executor.ReleaseTo(&executor);
    delete executor;
    delete tensorList1;
}

TEST_F(KernelLaunchUT, MemSetV2LaunchTest)
{
    auto* kernelBinWithDevPtr = CreateFakeOpKernelBin(true);
    aclnnStatus ret = kernelBinWithDevPtr->JsonLoad();

    auto* kernelBinNoDevPtr = CreateFakeOpKernelBin(false);
    ret = kernelBinNoDevPtr->JsonLoad();

    const uint64_t threadCount = 0;
    vector<std::thread> threadVec;
    for (uint64_t i = 0; i < threadCount; i++) {
        threadVec.emplace_back(std::thread(MemSetV2OutputTensorWithDevPtr, kernelBinWithDevPtr));
        threadVec.emplace_back(std::thread(MemSetV2OutputTensorWithDevPtr2, kernelBinWithDevPtr));
        threadVec.emplace_back(std::thread(MemSetV2OutputTensorNoDevPtr, kernelBinNoDevPtr));
    }
    for (uint64_t i = 0; i < threadCount * 3; i++) {
        threadVec[i].join();
    }

    for (auto& [key, value] : kernelBinWithDevPtr->tilingParseCtxHolder_) {
        if (value.get()) {
            value.get()->ReleaseTilingParse();
        }
    }

    for (auto& [key, value] : kernelBinNoDevPtr->tilingParseCtxHolder_) {
        if (value.get()) {
            value.get()->ReleaseTilingParse();
        }
    }
    delete kernelBinWithDevPtr;
    delete kernelBinNoDevPtr;
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
    auto outshape_arg = OP_OUTSHAPE({outshape.get(), 0});
    auto attr_arg = OP_ATTR(123);

    uint32_t opType = op::OpTypeDict::ToOpType("Axpy");
    int dummyStream = 0;
    void* stream = &dummyStream;
    auto ctx = op::MakeOpArgContext(input_arg, output_arg, attr_arg, ws_arg, outshape_arg);
    auto rc = op::internal::gKernelMgr.Run(opType, stream, ctx);

    EXPECT_EQ(rc, ACL_SUCCESS);
}

TEST_F(KernelLaunchUT, GetAclTensorCountTerst1)
{
    aclTensor* nullTensor = nullptr;
    aclTensorList* nullTensorList = nullptr;
    auto input_arg = OP_INPUT(nullTensor, nullTensorList);
    // create arg
    auto ctx = op::MakeOpArgContext(input_arg);
    size_t num = GetAclTensorCount(*(ctx->GetOpArg(op::OpArgDef::OP_INPUT_ARG)));
    EXPECT_EQ(num, 0u);
}

TEST_F(KernelLaunchUT, SetMemSetFlagFromJsonTest)
{
    const char* p = std::getenv("ASCEND_OPP_PATH");
    EXPECT_NE(p, nullptr);

    op::internal::KeyAndDetail key;
    key.key = "hahaha";
    size_t hashKey = 123;

    constexpr size_t binNum = 4;
    op::internal::OpKernelBin* bins[binNum];
    for (size_t i = 1; i <= binNum; i++) {
        char jsonPath[1024];
        char binPath[1024];
        snprintf_s(jsonPath, sizeof(jsonPath), sizeof(jsonPath),
                   "%s/built-in/op_impl/ai_core/tbe/kernel/ascend910/dummy/dummy_%zu.json", p, i);
        snprintf_s(binPath, sizeof(binPath), sizeof(binPath),
                   "%s/built-in/op_impl/ai_core/tbe/kernel/ascend910/dummy/dummy_%zu.o", p, i);
        op::internal::OpKernelBin* bin = new op::internal::OpKernelBin(
            9999, jsonPath, jsonPath, binPath, key, hashKey, op::internal::BinType::DYNAMIC_BIN, false, false);
        aclnnStatus rc = bin->JsonLoad();
        EXPECT_EQ(rc, ACLNN_SUCCESS);
        bins[i - 1] = bin;
    }
    EXPECT_EQ(bins[0]->memSetValue_.size(), 1);
    EXPECT_EQ(bins[0]->memSetValue_[0].argIdx_, 3);
    EXPECT_EQ(bins[0]->memSetValue_[0].dtype_, op::DataType::DT_FLOAT16);
    EXPECT_EQ(bins[0]->memSetValue_[0].valueFloat_, 65504.0f);

    EXPECT_EQ(bins[1]->memSetValue_.size(), 2);
    EXPECT_EQ(bins[1]->memSetValue_[0].argIdx_, 3);
    EXPECT_EQ(bins[1]->memSetValue_[0].dtype_, op::DataType::DT_FLOAT);
    EXPECT_EQ(bins[1]->memSetValue_[0].valueFloat_, 65504.0f);
    EXPECT_EQ(bins[1]->memSetValue_[1].argIdx_, 4);
    EXPECT_EQ(bins[1]->memSetValue_[1].dtype_, op::DataType::DT_INT32);
    EXPECT_EQ(bins[1]->memSetValue_[1].valueInt_, 655);

    EXPECT_EQ(bins[2]->memSetValue_.size(), 2);
    EXPECT_EQ(bins[2]->memSetValue_[0].argIdx_, 3);
    EXPECT_EQ(bins[2]->memSetValue_[0].dtype_, op::DataType::DT_INT32);
    EXPECT_EQ(bins[2]->memSetValue_[0].valueInt_, 655);
    EXPECT_EQ(bins[2]->memSetValue_[1].argIdx_, 4);
    EXPECT_EQ(bins[2]->memSetValue_[1].dtype_, op::DataType::DT_FLOAT);
    EXPECT_EQ(bins[2]->memSetValue_[1].valueInt_, 0);

    EXPECT_EQ(bins[3]->memSetValue_.size(), 2);
    EXPECT_EQ(bins[3]->memSetValue_[0].argIdx_, 3);
    EXPECT_EQ(bins[3]->memSetValue_[0].dtype_, op::DataType::DT_UINT32);
    EXPECT_EQ(bins[3]->memSetValue_[0].valueInt_, 655);
    EXPECT_EQ(bins[3]->memSetValue_[1].argIdx_, 4);
    EXPECT_EQ(bins[3]->memSetValue_[1].dtype_, op::DataType::DT_FLOAT);
    EXPECT_EQ(bins[3]->memSetValue_[1].valueFloat_, 65504.0f);

    for (size_t i = 0; i < binNum; i++) {
        delete bins[i];
    }
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

    op::internal::ExpandableRtsArgBuffer buffer;
    buffer.Init(TEST_LAUNCH_ARG_INIT_CAP, TEST_TILING_HOST_DATA_INIT_CAP);
    op::internal::TilingData* tilingData = buffer.GetTilingDataPtr();
    tilingData->data_size_ = 60;

    auto ctx = op::MakeOpArgContext(input_arg, output_arg);

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

    launchCfg.numBlocks = 32;
    launchCfg.schemMode = 1;
    launchCfg.dynUBufSize = 0;
    launchCfg.blockDimOffset = 0;
    launchCfg.engineType = op::internal::LaunchKernelEngineType::VECTOR_CORE_ENGINE_AIC;
    rc = arg.LaunchKernel(nullptr, launchCfg);
    EXPECT_EQ(rc, ACLNN_SUCCESS);

    launchCfg.numBlocks = 7;
    launchCfg.schemMode = 1;
    launchCfg.dynUBufSize = 0;
    launchCfg.blockDimOffset = 8;
    launchCfg.engineType = op::internal::LaunchKernelEngineType::VECTOR_CORE_ENGINE_AIV;
    rc = arg.LaunchKernel(nullptr, launchCfg);
    EXPECT_EQ(rc, ACLNN_SUCCESS);
}

TEST_F(KernelLaunchUT, TestWithHandleTensorPtrList)
{
    op::Shape selfShape{100};
    op::Shape outShape{100};

    aclOpExecutor exe;

    int64_t inputData[100];
    auto self = exe.AllocIntArray(inputData, 100);
    auto out = exe.AllocTensor(outShape, op::DataType::DT_INT32);
    auto inputTensor = exe.ConvertToTensor(self, op::DataType::DT_INT32);
    auto inputTensor2 = exe.ConvertToTensor(self, op::DataType::DT_INT32);
    const aclTensor* inputArr[] = {inputTensor, inputTensor2};
    aclTensorList* inputTensors = aclCreateTensorList(inputArr, 2);

    auto input_arg = OP_INPUT(inputTensors);
    auto output_arg = OP_OUTPUT(out);
    auto ctx = op::MakeOpArgContext(input_arg, output_arg);

    op::internal::ExpandableRtsArgBuffer buffer;
    buffer.Init(TEST_LAUNCH_ARG_INIT_CAP, TEST_TILING_HOST_DATA_INIT_CAP);
    op::internal::TilingData* tilingData = buffer.GetTilingDataPtr();
    tilingData->data_size_ = 60;

    op::internal::LaunchArgInfo argInfo(false, true, ctx);
    op::internal::RtsArg arg(true, argInfo, &buffer);
    arg.FillArgs();

    op::internal::KernelLaunchConfig launchCfg;
    launchCfg.funcHandle = (void*)0x12345678;
    launchCfg.numBlocks = 32;
    launchCfg.schemMode = 1;
    launchCfg.dynUBufSize = 0;
    launchCfg.blockDimOffset = 0;
    launchCfg.engineType = op::internal::LaunchKernelEngineType::VECTOR_CORE_ENGINE_AIC;
    aclnnStatus rc = arg.LaunchKernel(nullptr, launchCfg);

    delete inputTensors;
    op::DestroyOpArgContext(ctx);
}

TEST_F(KernelLaunchUT, Launch1982Test)
{
    setenv("ENABLE_1982", "1", 1);
    PlatformInfoStub::GetInstance()->SetSoCVersion("Ascend910_93", "Ascend910_9391");
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
    buffer.Init(TEST_LAUNCH_ARG_INIT_CAP, TEST_TILING_HOST_DATA_INIT_CAP);
    op::internal::TilingData* tilingData = buffer.GetTilingDataPtr();
    tilingData->data_size_ = 60;

    op::internal::LaunchArgInfo argInfo(false, false, ctx);
    op::internal::RtsArg arg(true, argInfo, &buffer);
    arg.FillArgs();

    op::internal::KernelLaunchConfig launchCfg;
    launchCfg.funcHandle = (void*)0x12345678;
    launchCfg.numBlocks = 32;
    launchCfg.schemMode = 1;
    launchCfg.dynUBufSize = 1;
    launchCfg.blockDimOffset = 0;
    launchCfg.engineType = op::internal::LaunchKernelEngineType::NO_VECTOR_CORE;
    aclnnStatus rc = arg.LaunchKernel(nullptr, launchCfg);

    op::DestroyOpArgContext(ctx);
    PlatformInfoStub::GetInstance()->Reset();
    unsetenv("ENABLE_1982");
}

// ---- Capture Scenario Stub ----

class CaptureActiveStub : public AclrtStub {
public:
    aclError aclmdlRICaptureGetInfo(aclrtStream stream, aclmdlRICaptureStatus* status, aclmdlRI* captureMdl) override
    {
        *status = ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE;
        return ACL_SUCCESS;
    }
};

class CaptureGetInfoFailStub : public AclrtStub {
public:
    aclError aclmdlRICaptureGetInfo(aclrtStream stream, aclmdlRICaptureStatus* status, aclmdlRI* captureMdl) override
    {
        return ACL_ERROR_FAILURE;
    }
};

static void TestLaunchWithCaptureStub(AclrtStub* captureStub)
{
    if (captureStub != nullptr) {
        AclrtStub::GetInstance()->Install(captureStub);
    }

    op::Shape selfShape{33, 15, 1, 48};
    op::Shape otherShape{33, 15, 14, 48};
    op::Shape outShape{33, 15, 14, 48};

    auto self = std::make_unique<aclTensor>(selfShape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND, nullptr);
    auto other = std::make_unique<aclTensor>(otherShape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND, nullptr);
    float alpha = 13.37;
    auto out = std::make_unique<aclTensor>(outShape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND, nullptr);

    uint32_t opType = op::OpTypeDict::ToOpType("Axpy");
    auto uniqueExecutor = CREATE_EXECUTOR();
    aclOpExecutor* executor = uniqueExecutor.get();
    thread_local uint64_t kernelLaunchIdDefinedInL0Dfx = op::internal::GenKernelLauncherId("Axpy");
    op::internal::ProfilingInfoId profilingInfoId(0, kernelLaunchIdDefinedInL0Dfx, 0);

    auto ctx = op::MakeOpArgContext(OP_INPUT(self.get(), other.get()), OP_OUTPUT(out.get()), OP_ATTR(alpha),
                                    OP_WORKSPACE(out.get()));
    auto launcher = new op::AiCoreKernelLauncher{opType, op::AI_CORE, profilingInfoId, executor, ctx};

    auto rc = launcher->Launch();
    EXPECT_EQ(rc, ACL_SUCCESS);

    delete launcher;
    if (captureStub != nullptr) {
        AclrtStub::GetInstance()->UnInstall();
    }
}

TEST_F(KernelLaunchUT, KernelLaunch_CaptureActive_SkipOverflow)
{
    CaptureActiveStub captureStub;
    TestLaunchWithCaptureStub(&captureStub);
}

TEST_F(KernelLaunchUT, KernelLaunch_CaptureNone_NormalOverflow) { TestLaunchWithCaptureStub(nullptr); }

TEST_F(KernelLaunchUT, KernelLaunch_CaptureGetInfoFail_NormalOverflow)
{
    CaptureGetInfoFailStub failStub;
    TestLaunchWithCaptureStub(&failStub);
}
