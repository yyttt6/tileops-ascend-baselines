/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dsa_task.h"

namespace op {
namespace internal {
constexpr uint64_t DSA_WORKSPACE_SIZE = 512;
aclnnStatus DSATask::ParamMemCpy(const aclrtStream stream) const
{
    constexpr size_t destMax = 512;
    const auto rc = aclrtMemcpyAsync(workspaceBaseAddr_, destMax, workspaceHolder_, sizeof(workspaceHolder_),
                                     ACL_MEMCPY_HOST_TO_BUF_TO_DEVICE, stream);
    if (rc != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "aclrtMemcpyAsync workspace input failed: %d", rc);
        return ACLNN_ERR_RUNTIME_ERROR;
    }
    return ACLNN_SUCCESS;
}

void GetDSATaskWorkspace(aclOpExecutor* executor, aclTensorList** workspace)
{
    FVector<aclTensor*> workspaces;
    workspaces.push_back(executor->AllocTensor({DSA_WORKSPACE_SIZE}, op::DataType::DT_UINT8));

    *workspace = executor->AllocTensorList(workspaces.data(), workspaces.size());
}
} // namespace internal
} // namespace op
