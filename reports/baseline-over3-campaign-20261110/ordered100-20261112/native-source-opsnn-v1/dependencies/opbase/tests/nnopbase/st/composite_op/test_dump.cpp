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
#include "dump/adump_pub.h"
#include "op_dfx_internal.h"
#include "op_ctx_def.h"
#include "launch_arg_info.h"
#include "rts_arg.h"
#include "depends/dump/dump_stub.h"

using namespace op;

namespace op {
namespace internal {
void OpCacheTid();
bool IsExceptionDumpEnable();
} // namespace internal
} // namespace op

char arr[100] = {0};
bool getSizeInfoAddr = false;
uint32_t spaceNum = 0;

class StArgsExceptionDump : public Adx::DumpStub {
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
    StArgsExceptionDump dumpStub;
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
    buffer.Init(1000, 2000);
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