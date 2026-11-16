/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef THREAD_VAR_CONTAINER_H_
#define THREAD_VAR_CONTAINER_H_
#include "aclnn/acl_meta.h"
#include "hccl/base.h"
#include <string>
namespace nnopbase {
namespace utils {
class ThreadVarContainer {
public:
    static aclnnStatus SetCurMc2RankIdInThread(HcclComm commHandle);
    static uint32_t GetCurMc2RankIdInThread();
    static void SetStaticKernelBasePathInThread(const std::string& staticKernelBasePathValue);
    static std::string GetStaticKernelBasePathInThread();
};
} // namespace utils
} // namespace nnopbase

#endif
