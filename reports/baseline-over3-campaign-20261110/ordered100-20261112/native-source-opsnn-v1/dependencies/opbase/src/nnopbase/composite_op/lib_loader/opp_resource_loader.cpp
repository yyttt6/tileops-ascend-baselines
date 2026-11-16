/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "opp_resource_loader.h"

#include <mutex>
#include <dlfcn.h>

#include "base/registry/opp_package_utils.h"
#include "graph/ge_error_codes.h"
#include "mmpa/mmpa_api.h"

#include "op_proto_loader.h"
#include "op_tiling_loader.h"
#include "opdev/op_log.h"
#include "file_utils.h"

using namespace std;
namespace op {
namespace opploader {

class ResourceHandlersManager {
public:
    static ResourceHandlersManager& GetInstance()
    {
        static ResourceHandlersManager instance;
        return instance;
    }

    std::vector<void*> resourceHandlers_;
};

static bool GetBuiltinOppPath(string& oppPath)
{
    string oppPathEnv;
    const char* currHomePath = nullptr;
    MM_SYS_GET_ENV(MM_ENV_ASCEND_HOME_PATH, currHomePath);
    if (currHomePath) {
        const std::string kernelPath = std::string(currHomePath) + "/opp_latest";
        oppPathEnv = RealPath(kernelPath);
        OP_CHECK(oppPathEnv.empty(), OP_LOGI("opp kernel path %s", oppPathEnv.c_str()), return true);
    }

    const char* currOppPath = nullptr;
    MM_SYS_GET_ENV(MM_ENV_ASCEND_OPP_PATH, currOppPath);
    OP_CHECK(currOppPath != nullptr, OP_LOGW("ASCEND_OPP_PATH not config."), return false);
    oppPathEnv = currOppPath;

    OP_CHECK(!oppPathEnv.empty(), OP_LOGW("ASCEND_OPP_PATH is empty"), return false);

    oppPath = oppPathEnv;
    string filePath = RealPath(oppPath);
    OP_CHECK(!filePath.empty(), OP_LOGW("ASCEND_OPP_PATH is invalid, path is: %s", oppPathEnv.c_str()), return false);
    OP_LOGI("ASCEND_OPP_PATH is: %s", oppPathEnv.c_str());
    if (oppPath.back() != '/') {
        oppPath += '/';
    }
    return true;
}

aclnnStatus LoadOppResource()
{
    OP_LOGI("Entering func: LoadOppResource.");
    string oppPath;
    auto ret = GetBuiltinOppPath(oppPath);
    OP_CHECK(ret, OP_LOGI("Leaving func: LoadOppResource, ASCEND_OPP_PATH not config."),
             return ACLNN_ERR_PARAM_INVALID);
    auto loadRet = LoadOpProto(oppPath);
    OP_CHECK(loadRet == ACLNN_SUCCESS, OP_LOGI("Leaving func: LoadOppResource with status: %d", loadRet),
             return loadRet);
    loadRet = LoadOpTiling(oppPath, ResourceHandlersManager::GetInstance().resourceHandlers_);
    OP_CHECK(loadRet == ACLNN_SUCCESS, OP_LOGI("Leaving func: LoadOppResource with status: %d", loadRet),
             return loadRet);
    return ACLNN_SUCCESS;
}

void ReleaseOppResource()
{
    for (void* handler : ResourceHandlersManager::GetInstance().resourceHandlers_) {
        OP_CHECK(dlclose(handler) == 0, OP_LOGW("dlclose opp resource failed"), ;);
    }
}

void LoadAllOppPackage()
{
    static std::once_flag loadFlag;
    std::call_once(loadFlag, [&]() {
        OP_LOGI("Start to load all OPP package.");
        OP_CHECK_NO_RETURN(gert::OppPackageUtils::LoadAllOppPackage() == ge::GRAPH_SUCCESS,
                           OP_LOGW("LoadAllOppPackage failed."));
        OP_LOGI("Load OPP package completed.");
    });
}
} // namespace opploader
} // namespace op
