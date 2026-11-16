/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "opdev/op_log.h"
#include "file_utils.h"
#include "kernel_utils.h"
#include "nnopbase_error_msg.h"

namespace op {
namespace internal {

using std::string;

gert::OppImplVersionTag GetOppImplVersion()
{
    const char* homePath = nullptr;
    MM_SYS_GET_ENV(MM_ENV_ASCEND_HOME_PATH, homePath);
    if (homePath == nullptr) {
        OP_LOGW("ASCEND_HOME_PATH is null.");
        return gert::OppImplVersionTag::kOpp;
    }
    string p = homePath;
    const std::string kernelPath = p + "/opp_latest";
    std::string oppKernelPath = RealPath(kernelPath);
    OP_CHECK(oppKernelPath.empty(), OP_LOGI("opp kernel path %s", oppKernelPath.c_str()),
             return gert::OppImplVersionTag::kOppKernel);
    return gert::OppImplVersionTag::kOpp;
}

aclnnStatus GetOppKernelPath(std::string& oppKernelPath)
{
    const char* homePath = nullptr;
    MM_SYS_GET_ENV(MM_ENV_ASCEND_HOME_PATH, homePath);
    if (homePath != nullptr) {
        string p = homePath;
        const std::string kernelPath = p + "/opp_latest";
        oppKernelPath = RealPath(kernelPath);
        OP_CHECK(oppKernelPath.empty(), OP_LOGI("opp kernel path %s", oppKernelPath.c_str()), return ACLNN_SUCCESS);
    } else {
        OP_LOGW("ASCEND_HOME_PATH is null.");
    }
    const char* oppPath = nullptr;
    MM_SYS_GET_ENV(MM_ENV_ASCEND_OPP_PATH, oppPath);
    if (oppPath == nullptr) {
        OP_LOGE_FOR_CONFIG_ERROR_INVALID_ENVIRONMENT_VARIABLE("Get opp kernel path", "ASCEND_OPP_PATH");
        return ACLNN_ERR_INNER_OPP_PATH_NOT_FOUND;
    }
    string p = oppPath;
    oppKernelPath = RealPath(p);
    if (!oppKernelPath.empty()) {
        OP_LOGI("opp kernel path %s", oppKernelPath.c_str());
        return ACLNN_SUCCESS;
    } else {
        OP_LOGE_FOR_CONFIG_ERROR_INVALID_ENVIRONMENT_VARIABLE("Get opp kernel path", "ASCEND_OPP_PATH");
        return ACLNN_ERR_INNER_OPP_PATH_NOT_FOUND;
    }
}

} // namespace internal
} // namespace op