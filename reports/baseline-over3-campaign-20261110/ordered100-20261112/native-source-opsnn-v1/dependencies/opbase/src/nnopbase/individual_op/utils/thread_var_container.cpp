/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "thread_var_container.h"
#include <mutex>
#include <thread>
#include "op_def.h"
#include "indv_lib_wrapper.h"

static thread_local uint32_t mc2RankId = 0U;
static thread_local std::string staticKernelBasePath;
static std::mutex staticKernelBasePathMutex;
namespace nnopbase {
namespace utils {
aclnnStatus ThreadVarContainer::SetCurMc2RankIdInThread(HcclComm commHandle)
{
    uint32_t rankId;
    CHECK_COND(nnopbase::IndvHcclWrapper::GetInstance().HcclGetRankId(commHandle, &rankId) == OK, ACLNN_ERR_INNER,
               "Failed to invode the HcclGetRankId function of the hccl module. HcclCom = %p", commHandle);
    mc2RankId = rankId;
    return OK;
}

uint32_t ThreadVarContainer::GetCurMc2RankIdInThread()
{
    OP_LOGD("Current mc2RankId is %u.", mc2RankId);
    return mc2RankId;
}

void ThreadVarContainer::SetStaticKernelBasePathInThread(const std::string& staticKernelBasePathValue)
{
    const std::lock_guard<std::mutex> lock(staticKernelBasePathMutex);
    OP_LOGD("Set static kernel base path to %s.", staticKernelBasePathValue.c_str());
    staticKernelBasePath = staticKernelBasePathValue;
}

std::string ThreadVarContainer::GetStaticKernelBasePathInThread()
{
    const std::lock_guard<std::mutex> lock(staticKernelBasePathMutex);
    OP_LOGD("Get static kernel base path %s.", staticKernelBasePath.c_str());
    return staticKernelBasePath;
}
} // namespace utils
} // namespace nnopbase
