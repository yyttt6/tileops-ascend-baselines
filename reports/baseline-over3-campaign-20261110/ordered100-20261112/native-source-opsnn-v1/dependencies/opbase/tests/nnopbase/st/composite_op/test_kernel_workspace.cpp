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
#include "thread_local_context.h"
#include "kernel_workspace.h"

OP_TYPE_REGISTER(Sub)
OP_TYPE_REGISTER(Transpose);
OP_TYPE_REGISTER(Sort);

class KernelWorkspaceUt : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        setenv("ASCEND_OPP_PATH", OP_API_COMMON_UT_SRC_DIR, 1); // does overwrite
        op::internal::GetThreadLocalContext().cacheHasFull_ = true;
    }

    static void TearDownTestCase() {}
};

// TEST_F(KernelWorkspaceUt, KernelWorkspaceTestCase0)
// {
//     op::Shape selfShape{33, 15, 1, 48};
//     op::Shape otherShape{33, 15, 14, 48};

//     auto self = std::make_unique<aclTensor>(selfShape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND, nullptr);
//     auto other = std::make_unique<aclTensor>(otherShape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND, nullptr);
//     auto out =
//         std::make_unique<aclTensor>(op::Shape{33, 15, 1, 48}, op::DataType::DT_FLOAT, op::Format::FORMAT_ND,
//         nullptr);

//     auto uniqueExecutor = CREATE_EXECUTOR();
//     aclOpExecutor *executor = uniqueExecutor.get();
//     aclTensorList *workspace = nullptr;
//     auto ret = GET_WORKSPACE(Sub, &workspace,
//                              OP_INPUT(self.get(), other.get()),
//                              OP_OUTPUT(out.get()));
//     EXPECT_EQ(ret, ACL_SUCCESS);
//     EXPECT_EQ(workspace, nullptr);
// }

// TEST_F(KernelWorkspaceUt, KernelWorkspaceTestCase1)
// {
//     op::Shape xShape{4, 5};
//     auto x = std::make_unique<aclTensor>(xShape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND, nullptr);
//     std::array<int64_t, 2> perm{1, 0};
//     aclIntArray permArray{perm.data(), perm.size()};
//     auto permTensor = std::make_unique<aclTensor>(&permArray, op::DataType::DT_INT64);
//     auto out = std::make_unique<aclTensor>(op::Shape{5, 4}, op::DataType::DT_FLOAT, op::Format::FORMAT_ND, nullptr);
//     auto uniqueExecutor = CREATE_EXECUTOR();
//     aclOpExecutor *executor = uniqueExecutor.get();
//     aclTensorList *workspace = nullptr;
//     auto ret = GET_WORKSPACE(Transpose, &workspace,
//                              OP_INPUT(x.get(), permTensor.get()),
//                              OP_OUTPUT(out.get()));
//     EXPECT_EQ(ret, ACL_SUCCESS);
//     ASSERT_NE(workspace, nullptr);
//     ASSERT_EQ(workspace->Size(), 1);
//     EXPECT_EQ((*workspace)[0]->GetOriginalShape(), op::Shape{1024});
// }

TEST_F(KernelWorkspaceUt, KernelWorkspaceTestCase2)
{
    op::Shape selfShape{1, 1, 1, 1, 2, 4};
    auto self = std::make_unique<aclTensor>(selfShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND, nullptr);

    auto values = std::make_unique<aclTensor>(selfShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND, nullptr);
    auto indices = std::make_unique<aclTensor>(selfShape, op::DataType::DT_INT32, op::Format::FORMAT_ND, nullptr);

    auto uniqueExecutor = CREATE_EXECUTOR();
    aclOpExecutor* executor = uniqueExecutor.get();
    aclTensorList* workspace = nullptr;
    auto ctx = op::MakeOpArgContext(OP_INPUT(self.get()), OP_OUTPUT(values.get(), indices.get()), OP_ATTR(-1, false));
    SortOpTypeId();
    auto ret = op::internal::GetWorkspace(
        op::OpTypeDict::ToOpType("Sort"), &workspace, executor, *ctx->GetOpArg(op::OpArgDef::OP_INPUT_ARG),
        *ctx->GetOpArg(op::OpArgDef::OP_OUTPUT_ARG), *ctx->GetOpArg(op::OpArgDef::OP_ATTR_ARG));
    EXPECT_EQ(ret, ACL_SUCCESS);
    ASSERT_NE(workspace, nullptr);
    ASSERT_EQ(workspace->Size(), 3u);
    EXPECT_EQ((*workspace)[0]->GetOriginalShape(), op::Shape{32});
    EXPECT_EQ((*workspace)[1]->GetOriginalShape(), op::Shape{32});
    EXPECT_EQ((*workspace)[2]->GetOriginalShape(), op::Shape{32});
    op::DestroyOpArgContext(ctx);
}
