/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_OP_API_COMMON_INC_DLOPEN_API_H_
#define OP_API_OP_API_COMMON_INC_DLOPEN_API_H_

#include <cstdint>
#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

aclOpExecutor* PTAGetExecCache(uint64_t hash, uint64_t* workspaceSize);
aclOpExecutor* PTAFindExecCache(uint8_t* buf, size_t len, uint64_t* workspaceSize);

aclnnStatus aclnnReselectStaticKernel();
aclnnStatus aclnnReselectStaticKernelWithPath(const char* staticKernelPath);

aclnnStatus InitSystemConfig(const char* configPath); // Init

#ifdef __cplusplus
}
#endif
#endif // OP_API_OP_API_COMMON_INC_DLOPEN_API_H_