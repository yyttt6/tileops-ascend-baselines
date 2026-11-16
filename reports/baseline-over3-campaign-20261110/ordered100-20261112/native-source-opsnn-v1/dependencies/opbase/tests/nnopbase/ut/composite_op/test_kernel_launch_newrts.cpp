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

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "kernel_mgr.h"
#include "memset_ctx_holder.h"
#include "rts_arg.h"
#include "op_ctx_def.h"
#include "op_dfx_internal.h"
#include "outshape.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_arg_def.h"
#include "opdev/op_errno.h"
#include "register/op_impl_registry.h"
#include "kernel_launcher.h"
#include "thread_local_context.h"
#include "tiling_parse_ctx_holder.h"
#include "test_comp_op_common.h"

using Json = nlohmann::json;
using namespace op::internal::test;

extern inline uint32_t SortOpTypeId();
extern inline uint32_t AxpyOpTypeId();
extern inline uint32_t AddNOpTypeId();

class KernelLaunchNewRtsUT : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        // aclInit(nullptr);
        setenv("ASCEND_OPP_PATH", OP_API_COMMON_UT_SRC_DIR, 1); // does overwrite
        op::internal::RtsApiFlag::GetRtsApiFlag().UseNewApi(true);
        op::internal::GetThreadLocalContext().cacheHasFull_ = true;
        AxpyOpTypeId();
        SortOpTypeId();
        AddNOpTypeId();
    }
    static void TearDownTestCase() {}
};

TEST_F(KernelLaunchNewRtsUT, KernelLaunchUTCase0)
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
}

TEST_F(KernelLaunchNewRtsUT, KernelLaunchUTCase1)
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
    auto rc = op::internal::gKernelMgr.Run(opType, stream, ctx);
    op::DestroyOpArgContext(ctx);

    EXPECT_EQ(rc, ACL_SUCCESS);
}

TEST_F(KernelLaunchNewRtsUT, KernelLaunchUTCase2)
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
    auto output = OP_OUTPUT(out.get(), idx.get());
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
    op::DestroyOpArgContext(ctx);
    delete wsList;
}

TEST_F(KernelLaunchNewRtsUT, KernelLaunchUTCase3)
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
    op::DestroyOpArgContext(ctx);
    delete inputList;
    EXPECT_EQ(rc, ACL_SUCCESS);
}

TEST_F(KernelLaunchNewRtsUT, KernelLaunchUTCase4)
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
    op::DestroyOpArgContext(ctx);
    delete inputList;
    EXPECT_EQ(tn, 1u);
    EXPECT_EQ(tn_list, 2u);
}

TEST_F(KernelLaunchNewRtsUT, MemSetTiling1)
{
    op::internal::MemSetKernelContextHolder ctx;

    op::internal::MemSetTensorInfo tensor{0,       ge::DT_FLOAT, 0.0f, 0, 100, 256, op::OpArgType::OPARG_ACLTENSOR,
                                          nullptr, nullptr};
    std::vector<op::internal::MemSetTensorInfo> tensorInfo{tensor};
    ctx.UpdateComputeNodeInfo(tensorInfo);

    EXPECT_EQ(ctx.inputNum_, 2ul);
}

TEST_F(KernelLaunchNewRtsUT, KernelLaunchUT_Outshape)
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

    uint32_t opType = op::OpTypeDict::ToOpType("Axpy");
    int dummyStream = 0;
    void* stream = &dummyStream;
    auto rc = op::internal::gKernelMgr.Run(opType, stream, ctx);
    op::DestroyOpArgContext(ctx);
    EXPECT_EQ(rc, ACL_SUCCESS);
}

TEST_F(KernelLaunchNewRtsUT, abnormalCase1)
{
    op::Shape selfShape{33, 15, 1, 48};
    op::Shape otherShape{33, 15, 14, 48};
    op::Shape outShape{33, 15, 14, 48};

    auto self = std::make_unique<aclTensor>(selfShape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND, nullptr);
    auto other = std::make_unique<aclTensor>(otherShape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND, nullptr);
    float alpha = 13.37;
    auto out = std::make_unique<aclTensor>(outShape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND, nullptr);

    auto input = OP_INPUT(self.get(), other.get());
    auto output = OP_OUTPUT(out.get());
    auto attr = OP_ATTR(alpha);
    auto ws = OP_WORKSPACE(out.get());
    auto ctx = op::MakeOpArgContext(input, output, attr, ws);
    int dummyStream = 0;
    void* stream = &dummyStream;

    uint32_t opType = static_cast<uint32_t>(op::internal::MAX_OP_TYPE_COUNT) + 1;
    auto rc2 = op::internal::gKernelMgr.Run(opType, stream, ctx);
    EXPECT_NE(rc2, ACL_SUCCESS);
    const size_t* workspaceSize = nullptr;
    size_t workspaceNum = 0;
    auto rc3 = op::internal::gKernelMgr.GetWorkspace(
        opType, workspaceSize, workspaceNum, *ctx->GetOpArg(op::OpArgDef::OP_OPTION_ARG),
        *ctx->GetOpArg(op::OpArgDef::OP_OUTPUT_ARG), *ctx->GetOpArg(op::OpArgDef::OP_ATTR_ARG));
    EXPECT_NE(rc3, ACL_SUCCESS);

    uint32_t opType1 = op::OpTypeDict::ToOpType("Axpy1");
    auto rc = op::internal::gKernelMgr.Run(opType1, stream, ctx);
    EXPECT_NE(rc, ACL_SUCCESS);
    auto rc1 = op::internal::gKernelMgr.GetWorkspace(
        opType1, workspaceSize, workspaceNum, *ctx->GetOpArg(op::OpArgDef::OP_OPTION_ARG),
        *ctx->GetOpArg(op::OpArgDef::OP_OUTPUT_ARG), *ctx->GetOpArg(op::OpArgDef::OP_ATTR_ARG));
    EXPECT_NE(rc1, ACL_SUCCESS);
    op::DestroyOpArgContext(ctx);
}

TEST_F(KernelLaunchNewRtsUT, KernelLaunchUT_Outshape2)
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
    int64_t value = 1;
    aclIntArray strides(&value, 1);
    float fvalue = 1.0;
    aclFloatArray fstrides(&fvalue, 1);
    double dvalue = 1.0;
    std::string str = "123";
    auto scalar = std::make_unique<aclScalar>(&value, op::DataType::DT_INT64);
    const aclScalar* s = scalar.get();

    auto input_arg = OP_INPUT(self.get(), other.get());
    auto output_arg = OP_OUTPUT(out.get());
    auto ws_arg = OP_WORKSPACE(ws.get());
    auto outshape_arg = OP_OUTSHAPE(outshape.get(), 0);
    auto attr_arg = OP_ATTR(123, &strides, &fstrides, str, s, fvalue, dvalue);
    auto ctx = op::MakeOpArgContext(input_arg, output_arg, ws_arg, outshape_arg, attr_arg);

    uint32_t opType = op::OpTypeDict::ToOpType("Axpy");
    int dummyStream = 0;
    void* stream = &dummyStream;
    std::vector<op::internal::AttrInfo> attrInfos;
    std::string attrStr;
    op::internal::ReportAttrInfo(*ctx->GetOpArg(op::OpArgDef::OP_ATTR_ARG), attrStr, attrInfos);

    auto outshape2 = std::make_unique<aclTensor>(outShapeShape, op::DataType::DT_INT64, op::Format::FORMAT_ND, nullptr);
    auto outshape_arg2 = OP_OUTSHAPE(outshape2.get(), 0);
    auto ctx2 = op::MakeOpArgContext(input_arg, output_arg, ws_arg, outshape_arg2, attr_arg);
    op::internal::RefreshOutputShape(0, *ctx2->GetOpArg(op::OpArgDef::OP_OUTSHAPE_ARG),
                                     *ctx2->GetOpArg(op::OpArgDef::OP_OUTPUT_ARG));
    auto outshape3 = std::make_unique<aclTensor>(outShapeShape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND, nullptr);
    auto outshape_arg3 = OP_OUTSHAPE(outshape3.get(), 0);
    auto ctx3 = op::MakeOpArgContext(input_arg, output_arg, ws_arg, outshape_arg3, attr_arg);
    op::internal::RefreshOutputShape(0, *ctx3->GetOpArg(op::OpArgDef::OP_OUTSHAPE_ARG),
                                     *ctx3->GetOpArg(op::OpArgDef::OP_OUTPUT_ARG));
    auto outshape4 = std::make_unique<aclTensor>(outShapeShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND,
                                                 nullptr);
    auto outshape_arg4 = OP_OUTSHAPE(outshape4.get(), 0);
    auto ctx4 = op::MakeOpArgContext(input_arg, output_arg, ws_arg, outshape_arg4, attr_arg);
    op::internal::RefreshOutputShape(0, *ctx4->GetOpArg(op::OpArgDef::OP_OUTSHAPE_ARG),
                                     *ctx4->GetOpArg(op::OpArgDef::OP_OUTPUT_ARG));

    auto bad_input_arg = OP_INPUT(self.get(), other.get(), value);
    auto ctx5 = op::MakeOpArgContext(bad_input_arg, output_arg, ws_arg, outshape_arg4, attr_arg);
    auto rc = op::internal::gKernelMgr.Run(opType, stream, ctx5);
    op::DestroyOpArgContext(ctx);
    op::DestroyOpArgContext(ctx2);
    op::DestroyOpArgContext(ctx3);
    op::DestroyOpArgContext(ctx4);
    op::DestroyOpArgContext(ctx5);
}

TEST_F(KernelLaunchNewRtsUT, GetAclTensorCountTerst1)
{
    aclTensor* nullTensor = nullptr;
    aclTensorList* nullTensorList = nullptr;
    auto input_arg = OP_INPUT(nullTensor, nullTensorList);
    auto ctx = op::MakeOpArgContext(input_arg);

    size_t num = op::internal::GetAclTensorCount(*ctx->GetOpArg(op::OpArgDef::OP_INPUT_ARG));
    op::DestroyOpArgContext(ctx);
    EXPECT_EQ(num, 0u);
}

TEST_F(KernelLaunchNewRtsUT, SetMemSetFlagFromJsonTest)
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
    }
}

TEST_F(KernelLaunchNewRtsUT, hasHostDataTest)
{
    op::Shape selfShape{100};
    op::Shape outShape{100};

    aclOpExecutor exe;

    int64_t inputData[100];
    auto self = exe.AllocIntArray(inputData, 100);
    auto out = exe.AllocTensor(outShape, op::DataType::DT_INT32);
    auto inputTensor = exe.ConvertToTensor(self, op::DataType::DT_INT32);

    auto tensor0 = exe.AllocHostTensor({100}, op::DataType::DT_FLOAT, op::Format::FORMAT_ND);

    auto input_arg = OP_INPUT(inputTensor, tensor0);
    auto output_arg = OP_OUTPUT(out);
    auto ctx = op::MakeOpArgContext(input_arg, output_arg);

    op::internal::ExpandableRtsArgBuffer buffer;
    buffer.Init(TEST_LAUNCH_ARG_INIT_CAP, TEST_TILING_HOST_DATA_INIT_CAP);
    op::internal::TilingData* tilingData = buffer.GetTilingDataPtr();
    tilingData->data_size_ = 100;
    op::internal::LaunchArgInfo argInfo(false, false, ctx);
    op::internal::RtsArg arg(true, argInfo, &buffer);
    arg.DumpToCache();
}

TEST_F(KernelLaunchNewRtsUT, CalcMixCoreNumTest)
{
    Json mix1 = {{"taskRation", "0:1"}};
    Json mix2 = {{"taskRation", "1:0"}};
    Json mix3 = {{"taskRation", "0:0"}};
    Json mix4 = {{"taskRation", "1:1"}};
    Json mix5 = {{"taskRation", "0:2"}};
    Json mix6 = {{"taskRation", "1:2"}};
    Json mix7 = {{"taskRation", "1:3"}};
    Json mix8 = {{"taskRation", "100:1"}};

    EXPECT_EQ(op::internal::CalcMixCoreNum(24, 48, mix1), 48u);
    EXPECT_EQ(op::internal::CalcMixCoreNum(24, 48, mix2), 24u);
    EXPECT_EQ(op::internal::CalcMixCoreNum(24, 48, mix3), 48u);
    EXPECT_EQ(op::internal::CalcMixCoreNum(24, 48, mix4), 24u);
    EXPECT_EQ(op::internal::CalcMixCoreNum(24, 48, mix5), 24u);
    EXPECT_EQ(op::internal::CalcMixCoreNum(24, 48, mix6), 24u);
    EXPECT_EQ(op::internal::CalcMixCoreNum(24, 48, mix7), 16u);
    EXPECT_EQ(op::internal::CalcMixCoreNum(24, 48, mix8), 1u);
}

TEST_F(KernelLaunchNewRtsUT, SetCoreNumTest)
{
    Json mix1 = {{"coreType", "AiCore"}};
    Json mix2 = {{"coreType", "MIX"}};
    Json mix3 = {{"coreType", "VectorCore"}};
    Json mix4 = {{"coreType", "MIX_AIV"}};
    fe::PlatFormInfos plat;
    uint32_t coreNum = 0;

    op::internal::SetCoreNum(mix1, &plat, coreNum);
    op::internal::SetCoreNum(mix2, &plat, coreNum);
    op::internal::SetCoreNum(mix3, &plat, coreNum);
    op::internal::SetCoreNum(mix4, &plat, coreNum);
    EXPECT_EQ(1, 1);
}
