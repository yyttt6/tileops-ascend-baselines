/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_BNINFERENCE_D_KERNEL_STUB_H_
#define ACLNN_BNINFERENCE_D_KERNEL_STUB_H_
#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnBninferenceDKernelGetWorkspaceSize(const aclTensor* x1, const aclTensor* x2, const aclTensor* x3,
                                                    const aclTensor* out, uint64_t* workspaceSize,
                                                    aclOpExecutor** executor);

aclnnStatus aclnnBninferenceDKernel(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                    const aclrtStream stream);

aclnnStatus aclnnBninferenceDKernelNoOutputGetWorkspaceSize(const aclTensor* x1, const aclTensor* x2,
                                                            const aclTensor* x3, const aclTensor* x4,
                                                            uint64_t* workspaceSize, aclOpExecutor** executor);

aclnnStatus aclnnBninferenceDKernelNoOutput(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                            const aclrtStream stream);

aclnnStatus aclnnBninferenceDKernelRefGetWorkspaceSize(const aclTensor* x1Ref, const aclTensor* x2, const aclTensor* x3,
                                                       uint64_t* workspaceSize, aclOpExecutor** executor);

aclnnStatus aclnnBninferenceDKernelRef(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                       const aclrtStream stream);

#ifdef __cplusplus
}
#endif
#endif