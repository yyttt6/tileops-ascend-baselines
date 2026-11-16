/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <nlohmann/json.hpp>
#include <fstream>

#include "mmpa/mmpa_api.h"

#include "opdev/op_log.h"
#include "opp_resource_loader.h"
#include "op_dfx_internal.h"
#include "kernel_mgr.h"
#include "opdev/aicpu/aicpu_task.h"
#include "file_utils.h"
#include "dlopen_api.h"
#include "aclnn/aclnn_base.h"

using namespace std;
using Json = nlohmann::json;

#ifdef __cplusplus
extern "C" {
#endif
aclnnStatus InitSystemConfig(const char* configPath)
{
    // 获取环境变量中的配置
    const char* enableDebugKernelEnv = nullptr;
    MM_SYS_GET_ENV(MM_ENV_NPU_COLLECT_PATH, enableDebugKernelEnv);
    if (enableDebugKernelEnv != nullptr && strlen(enableDebugKernelEnv) != 0) {
        return op::internal::systemConfig.SetEnableDebugKernelFlag(true);
    }
    OP_LOGI("The environment variable NPU_COLLECT_PATH is not set");

    if (configPath == nullptr || strlen(configPath) == 0) {
        OP_LOGI("The config path is null or empty");
        return op::internal::systemConfig.SetEnableDebugKernelFlag(false);
    }

    string configPathStr(configPath);
    string realConfigPathStr = op::RealPath(configPathStr);
    OP_CHECK(!realConfigPathStr.empty(), OP_LOGI("real path of the input config path [%s] is not exist", configPath),
             return op::internal::systemConfig.SetEnableDebugKernelFlag(false));

    // 解析json文件中的配置
    try {
        std::ifstream f(configPath);
        Json sysConfig = Json::parse(f);
        Json opDebugConfig;
        if (sysConfig.find("op_debug_config") == sysConfig.end()) {
            OP_LOGI("op_debug_config not in json file: %s", configPath);
            return op::internal::systemConfig.SetEnableDebugKernelFlag(false);
        }
        opDebugConfig = sysConfig.at("op_debug_config");
        if (opDebugConfig.find("enable_debug_kernel") == opDebugConfig.end()) {
            OP_LOGI("enable_debug_kernel not in json file: %s", configPath);
            return op::internal::systemConfig.SetEnableDebugKernelFlag(false);
        }
        std::string enableDebugKernel = opDebugConfig.at("enable_debug_kernel");
        OP_LOGI("enable_debug_kernel value is: %s", enableDebugKernel.c_str());
        if (enableDebugKernel.compare("on") == 0) {
            return op::internal::systemConfig.SetEnableDebugKernelFlag(true);
        }
    } catch (...) {
        OP_LOGW("json parse failed. json file: %s", configPath);
    }
    return op::internal::systemConfig.SetEnableDebugKernelFlag(false);
}

aclnnStatus aclnnInit([[maybe_unused]] const char* configPath)
{
    OP_LOGI("Entering func: aclnnInit");
    op::opploader::LoadAllOppPackage();
    auto ret = InitSystemConfig(configPath);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    OP_LOGI("Leaving func: aclnnInit");
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFinalize()
{
    op::internal::aclnnAicpuFinalize();
    op::internal::gKernelMgr.ReleaseTilingParse();
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
