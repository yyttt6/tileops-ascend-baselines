/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ACLNN_MUL_STUB_H_
#define ACLNN_MUL_STUB_H_
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnMulStubGetWorkspaceSize(const aclTensor* intput1, const aclTensor* intput2, aclTensor* out,
                                         uint64_t* workspaceSize, aclOpExecutor** executor);

aclnnStatus aclnnMulStub(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream);

#ifdef __cplusplus
}
#endif
#endif