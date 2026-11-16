/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_mul_stub.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

OP_TYPE_REGISTER(Mul);

static const aclTensor* MulAiCore(const aclTensor* self, const aclTensor* other, aclTensor* addOut,
                                  aclOpExecutor* executor)
{
    L0_DFX(MulAiCore, self, other, addOut);
    // 使用框架宏ADD_TO_LAUNCHER_LIST_AICORE，将AiCore Add算子加入任务队列
    // Add是算子的OpType，self、other是算子的输入，addOut是算子的输出
    ADD_TO_LAUNCHER_LIST_AICORE(Mul, OP_INPUT(self, other), OP_OUTPUT(addOut));
    return addOut;
}

aclnnStatus aclnnMulStubGetWorkspaceSize(const aclTensor* intput1, const aclTensor* intput2, aclTensor* out,
                                         uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnMulStub, DFX_IN(intput1, intput2), DFX_OUT(out));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    const aclTensor* addOpOut = MulAiCore(intput1, intput2, out, uniqueExecutor.get());
    CHECK_RET(addOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor); // 需要把 uniqueExecutor持有executor转移给executor
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMulStub(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMulStub);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif