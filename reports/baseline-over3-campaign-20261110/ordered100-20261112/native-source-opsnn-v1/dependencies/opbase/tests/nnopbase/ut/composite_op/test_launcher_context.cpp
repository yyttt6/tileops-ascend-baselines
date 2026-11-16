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
#include "kernel_context_holder.h"
#include "op_kernel.h"
#include "launcher_ctx.h"
#include "op_ctx_def.h"
#include "tilingctx_builder.h"
#include "op_run_context.h"
#include "opdev/op_def.h"
#include "opdev/make_op_executor.h"
#include "thread_local_context.h"
#include "kernel_workspace.h"

#include "opdev/op_errno.h"

extern inline uint32_t SortOpTypeId();

class LauncherCtxUT : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        // aclInit(nullptr)
        setenv("ASCEND_OPP_PATH", OP_API_COMMON_UT_SRC_DIR, 1); // does overwrite
        op::internal::GetThreadLocalContext().cacheHasFull_ = true;
    }
    static void TearDownTestCase() {}
};

TEST_F(LauncherCtxUT, LauncherCtxUTCase1)
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
    auto ctx = op::MakeOpArgContext(input, output, attr);

    auto ws1 = std::make_unique<aclTensor>(wsShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND, nullptr);
    auto ws2 = std::make_unique<aclTensor>(wsShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND, nullptr);
    auto ws3 = std::make_unique<aclTensor>(wsShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND, nullptr);

    op::internal::GetLauncherCtx().Reset();

    auto uniqueExecutor = CREATE_EXECUTOR();
    aclOpExecutor* executor = uniqueExecutor.get();
    aclTensorList* workspace = nullptr;
    op::internal::GetWorkspace(opType, &workspace, executor, *ctx->GetOpArg(op::OpArgDef::OP_INPUT_ARG),
                               *ctx->GetOpArg(op::OpArgDef::OP_OUTPUT_ARG), *ctx->GetOpArg(op::OpArgDef::OP_ATTR_ARG));
    op::DestroyOpArgContext(ctx);

    const op::internal::TilingResCache* tilingCache = nullptr;
    tilingCache = op::internal::GetLauncherCtx().GetTilingResCache();
    EXPECT_NE(tilingCache, nullptr);

    op::internal::GetLauncherCtx().Reset();
    tilingCache = op::internal::GetLauncherCtx().GetTilingResCache();
    EXPECT_EQ(tilingCache, nullptr);

    op::internal::GetLauncherCtx().SetImplMode(op::OpImplMode::IMPL_MODE_DEFAULT);
    auto mode = op::internal::GetLauncherCtx().GetImplMode();
    EXPECT_EQ(mode, op::OpImplMode::IMPL_MODE_DEFAULT);
}
