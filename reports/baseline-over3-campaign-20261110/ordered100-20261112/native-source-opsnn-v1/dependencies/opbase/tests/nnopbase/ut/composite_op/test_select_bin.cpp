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
#include "opdev/op_cache.h"
#include "opdev/op_errno.h"
#include "register/op_impl_registry.h"
#include "rts_arg.h"

using namespace op::internal;

extern op::internal::OpExecCacheWrap* GetOpExecCacheFromExecutor(aclOpExecutor* executor);
extern "C" aclOpExecutor* PTAFindExecCache(uint8_t* buf, size_t len, uint64_t* workspaceSize);
extern "C" aclOpExecutor* PTAGetExecCache(uint64_t hash, uint64_t* workspaceSize);
extern "C" void InitPTACacheThreadLocal();
extern "C" void ResetCacheThreadLocal();
extern "C" bool CanUsePTACache(const char* api);

class SelectBinUt : public testing::Test {
protected:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}
};

TEST_F(SelectBinUt, SelectBinTest)
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

    auto input = OP_INPUT(inputList);
    auto output = OP_OUTPUT(out.get());
    auto attr = OP_ATTR(alpha);
    auto ctx = op::MakeOpArgContext(input, output, attr);

    uint32_t opType = op::OpTypeDict::ToOpType("AddN");
    auto opKernel = op::internal::gKernelMgr.GetKernel(opType);
    EXPECT_NE(opKernel, nullptr);
    auto bin = opKernel->SelectBin(*ctx->GetOpArg(op::OpArgDef::OP_INPUT_ARG),
                                   *ctx->GetOpArg(op::OpArgDef::OP_OUTPUT_ARG),
                                   *ctx->GetOpArg(op::OpArgDef::OP_ATTR_ARG));

    EXPECT_NE(bin, nullptr);
    delete inputList;
    op::DestroyOpArgContext(ctx);
}
