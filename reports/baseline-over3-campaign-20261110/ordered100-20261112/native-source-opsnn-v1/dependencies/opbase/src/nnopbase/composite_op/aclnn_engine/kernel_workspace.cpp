/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_workspace.h"
#include "opdev/op_executor.h"

namespace op {
namespace internal {
inline aclnnStatus UpdateWorkspace(aclOpExecutor* executor, const size_t* workspaceSize, size_t workspaceNum,
                                   aclTensorList** workspace)
{
    FVector<aclTensor*> workspaces;
    for (size_t i = 0; i != workspaceNum; ++i) {
        op::Shape workspaceShape({static_cast<int64_t>(*(workspaceSize + i))});
        workspaces.push_back(executor->AllocTensor(workspaceShape, op::DataType::DT_UINT8));
    }

    *workspace = executor->AllocTensorList(workspaces.data(), workspaces.size());
    return ACLNN_SUCCESS;
}

aclnnStatus GetWorkspace(uint32_t optype, aclTensorList** workspace, aclOpExecutor* executor, OpArgList& inputs,
                         OpArgList& outputs, OpArgList& attrs)
{
    const size_t* workspaceSize = nullptr;
    size_t workspaceNum = 0;
    auto ret = gKernelMgr.GetWorkspace(optype, workspaceSize, workspaceNum, inputs, outputs, attrs);
    if (ret != ACLNN_SUCCESS) {
        return ACLNN_ERR_INNER;
    }

    if (workspaceSize != nullptr && workspaceNum > 0) {
        ret = UpdateWorkspace(executor, workspaceSize, workspaceNum, workspace);
    }
    return ret;
}
} // namespace internal
} // namespace op