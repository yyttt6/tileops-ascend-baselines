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
#include <memory>

#include "acl/acl.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "adump_pub.h"
#include "depends/dump/dump_stub.h"
#include "launch_arg_info.h"
#include "bridge_pool.h"
#include "op_dfx_internal.h"
#include "op_ctx_def.h"
#include "rts_arg.h"
#include "bridge_dfx.h"
#include "acl/acl_rt.h"
#include "test_comp_op_common.h"

using namespace op;
using namespace op::internal::test;

OP_TYPE_REGISTER(NonFiniteCheck);

namespace op {
namespace internal {
void OpCacheTid();
bool IsExceptionDumpEnable();
} // namespace internal
} // namespace op

char arr[100] = {0};
bool getSizeInfoAddr = false;
uint32_t spaceNum = 0;

class UtArgsExceptionDump : public Adx::DumpStub {
public:
    void* AdumpGetSizeInfoAddr(uint32_t space, uint32_t& atomicIndex)
    {
        spaceNum = space;
        atomicIndex = 1;
        getSizeInfoAddr = true;
        return (void*)arr;
    }
};

class DumpUt : public testing::Test {
protected:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}
};

TEST_F(DumpUt, dump_l2)
{
    op::internal::GetLogApiInfo();
    op::internal::GenSummaryItemId("aaaaa", "bbbbb");
    op::GenOpTypeId("aaaaa");
    op::internal::GenKernelLauncherId("aaaaa");
    op::internal::OpGetTid();
    op::internal::OpCacheTid();
    op::internal::OpGetLogSequence();

    vector<int64_t> shapeA = {2, 1, 32, 16};
    aclDataType dtype1 = aclDataType::ACL_FLOAT16;
    auto storageShapeA = shapeA;
    void* deviceDataA = nullptr;
    vector<int64_t> stridesA = {2, 1, 32, 16};

    const aclTensor* tensor = aclCreateTensor(shapeA.data(), shapeA.size(), dtype1, stridesA.data(), 0,
                                              aclFormat::ACL_FORMAT_ND, storageShapeA.data(), storageShapeA.size(),
                                              deviceDataA);
    ;
    std::vector<const aclTensor*> in;
    op::internal::OpLogInfo info;
    in.push_back(tensor);
    const aclrtStream stream = 0;
    DumpL2(in, info, OpInputType, stream);
    aclDestroyTensor(tensor);
}

TEST_F(DumpUt, exception_dump) { op::internal::IsExceptionDumpEnable(); }

TEST_F(DumpUt, exception_dump_assert)
{
    UtArgsExceptionDump dumpStub;
    Adx::DumpStub::GetInstance()->Install(&dumpStub);

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
    tilingData->data_size_ = 60;

    op::internal::LaunchArgInfo argInfo(false, false, ctx);
    op::internal::RtsArg arg(true, argInfo, &buffer);
    arg.FillArgs(true);
    EXPECT_EQ(spaceNum, 4);
    EXPECT_EQ(getSizeInfoAddr, true);
    Adx::DumpStub::GetInstance()->UnInstall();
    op::DestroyOpArgContext(ctx);
}

TEST_F(DumpUt, TraitsAclTensorAndIdxTest)
{
    op::Shape outShape{100};
    int64_t inputData[8];
    aclIntArray self(inputData, 8);
    static aclTensor out(outShape, op::DataType::DT_INT32, ge::FORMAT_ND, nullptr);
    static aclTensor inputTensor(&self, op::DataType::DT_INT32);

    aclTensor* inputTensorNull = nullptr;

    op::Shape selfShape{33, 15, 64};
    aclTensor input1(&self, op::DataType::DT_INT32);
    aclTensor input2(&self, op::DataType::DT_INT32);
    aclTensor* input3 = nullptr;
    aclTensor* input4 = nullptr;
    const aclTensor* inputTensorList[4] = {&input1, input3, &input2, input4};
    aclTensorList* inputList = aclCreateTensorList(inputTensorList, 4);
    aclTensorList* inputTensorListNull = nullptr;

    aclTensorList* workSapceList = aclCreateTensorList(inputTensorList, 4);

    auto input_arg = OP_INPUT(&inputTensor, inputTensorNull, inputList, inputTensorListNull);
    auto output_arg = OP_OUTPUT(&out);
    auto workspace_arg = OP_WORKSPACE(workSapceList);
    auto ctx = op::MakeOpArgContext(input_arg, output_arg, workspace_arg);

    std::vector<const aclTensor*> aclInTensors;
    std::vector<const aclTensor*> aclOutTensors;
    std::vector<const aclTensor*> aclWorkSpaceTensors;
    std::vector<uint32_t> inTensorsIdxList;
    std::vector<uint32_t> outTensorsIdxList;
    std::vector<uint32_t> workSpaceTensorsIdxList;

    int32_t currentIdx = -1;
    // not support for placeholders, not support core sync
    op::internal::TraitsAclTensorAndIdx(aclInTensors, inTensorsIdxList, *ctx->GetOpArg(op::OP_INPUT_ARG), false,
                                        currentIdx);
    EXPECT_EQ(inTensorsIdxList.size(), 3);
    EXPECT_EQ(inTensorsIdxList[0], 0);
    EXPECT_EQ(inTensorsIdxList[1], 1);
    EXPECT_EQ(inTensorsIdxList[2], 2);

    op::internal::TraitsAclTensorAndIdx(aclOutTensors, outTensorsIdxList, *ctx->GetOpArg(op::OP_OUTPUT_ARG), false,
                                        currentIdx);
    EXPECT_EQ(outTensorsIdxList.size(), 1);
    EXPECT_EQ(outTensorsIdxList[0], 3);

    op::internal::TraitsAclTensorAndIdx(aclWorkSpaceTensors, workSpaceTensorsIdxList,
                                        *ctx->GetOpArg(op::OP_WORKSPACE_ARG), false, currentIdx);
    EXPECT_EQ(workSpaceTensorsIdxList.size(), 2);
    EXPECT_EQ(workSpaceTensorsIdxList[0], 4);
    EXPECT_EQ(workSpaceTensorsIdxList[1], 5);

    // support for placeholders, not support core sync
    currentIdx = -1;
    inTensorsIdxList.clear();
    outTensorsIdxList.clear();
    workSpaceTensorsIdxList.clear();
    aclInTensors.clear();
    aclOutTensors.clear();
    aclWorkSpaceTensors.clear();
    op::internal::TraitsAclTensorAndIdx(aclInTensors, inTensorsIdxList, *ctx->GetOpArg(op::OP_INPUT_ARG), true,
                                        currentIdx);
    EXPECT_EQ(inTensorsIdxList.size(), 3);
    EXPECT_EQ(inTensorsIdxList[0], 0);
    EXPECT_EQ(inTensorsIdxList[1], 2);
    EXPECT_EQ(inTensorsIdxList[2], 4);

    op::internal::TraitsAclTensorAndIdx(aclOutTensors, outTensorsIdxList, *ctx->GetOpArg(op::OP_OUTPUT_ARG), true,
                                        currentIdx);
    EXPECT_EQ(outTensorsIdxList.size(), 1);
    EXPECT_EQ(outTensorsIdxList[0], 6);

    op::internal::TraitsAclTensorAndIdx(aclWorkSpaceTensors, workSpaceTensorsIdxList,
                                        *ctx->GetOpArg(op::OP_WORKSPACE_ARG), true, currentIdx);
    EXPECT_EQ(workSpaceTensorsIdxList.size(), 2);
    EXPECT_EQ(workSpaceTensorsIdxList[0], 7);
    EXPECT_EQ(workSpaceTensorsIdxList[1], 9);

    // support for placeholders, support core sync
    currentIdx = 0;
    inTensorsIdxList.clear();
    outTensorsIdxList.clear();
    workSpaceTensorsIdxList.clear();
    aclInTensors.clear();
    aclOutTensors.clear();
    aclWorkSpaceTensors.clear();
    op::internal::TraitsAclTensorAndIdx(aclInTensors, inTensorsIdxList, *ctx->GetOpArg(op::OP_INPUT_ARG), true,
                                        currentIdx);
    EXPECT_EQ(inTensorsIdxList.size(), 3);
    EXPECT_EQ(inTensorsIdxList[0], 1);
    EXPECT_EQ(inTensorsIdxList[1], 3);
    EXPECT_EQ(inTensorsIdxList[2], 5);

    op::internal::TraitsAclTensorAndIdx(aclOutTensors, outTensorsIdxList, *ctx->GetOpArg(op::OP_OUTPUT_ARG), true,
                                        currentIdx);
    EXPECT_EQ(outTensorsIdxList.size(), 1);
    EXPECT_EQ(outTensorsIdxList[0], 7);

    op::internal::TraitsAclTensorAndIdx(aclWorkSpaceTensors, workSpaceTensorsIdxList,
                                        *ctx->GetOpArg(op::OP_WORKSPACE_ARG), true, currentIdx);
    EXPECT_EQ(workSpaceTensorsIdxList.size(), 2);
    EXPECT_EQ(workSpaceTensorsIdxList[0], 8);
    EXPECT_EQ(workSpaceTensorsIdxList[1], 10);

    op::internal::DeAllocate(ctx);
    delete inputList;
    delete workSapceList;
    delete inputTensorListNull;
}

TEST_F(DumpUt, overflow_dump_check) { op::internal::IsOverflowDumpEnable(); }

TEST_F(DumpUt, overflow_dump_saturation)
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
    auto ctx = op::MakeOpArgContext(input, output, attr, wsArg);
    aclOpExecutor executor;
    int dummyStream = 0;
    void* stream = &dummyStream;
    op::internal::OpLogInfo logInfo;
    aclnnStatus ret = OverflowDumpProcess(ctx, &executor, stream, logInfo);
    EXPECT_EQ(ret, ACLNN_SUCCESS);
    delete wsList;
    DestroyOpArgContext(ctx);
}

TEST_F(DumpUt, overflow_dump_infnan)
{
    aclrtFloatOverflowMode mode;
    aclrtGetDeviceSatMode(&mode);
    aclrtSetDeviceSatMode(ACL_RT_OVERFLOW_MODE_INFNAN);
    op::Shape selfShape{33, 15, 64};
    op::Shape outShape{33, 15, 64};
    op::Shape idxShape{33, 15, 64};
    op::Shape wsShape{32};

    int64_t dim = 0;
    bool descending = true;

    auto self = std::make_unique<aclTensor>(selfShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND, nullptr);
    auto out = std::make_unique<aclTensor>(outShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND, nullptr);
    auto idx = std::make_unique<aclTensor>(idxShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND, nullptr);

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
    aclOpExecutor executor;
    int dummyStream = 0;
    void* stream = &dummyStream;
    op::internal::OpLogInfo logInfo;
    aclnnStatus ret = OverflowDumpProcess(ctx, &executor, stream, logInfo);
    EXPECT_EQ(ret, ACLNN_SUCCESS);
    delete wsList;
    DestroyOpArgContext(ctx);
    aclrtSetDeviceSatMode(mode);
}