/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "op_tiling_loader.h"
#include <string>
#include <dlfcn.h>
#include <vector>
#include "file_utils.h"
#include "opdev/op_log.h"

using namespace std;
namespace op {
namespace opploader {

constexpr char const* OP_TILING_SO_SUFFIX = ".so";
constexpr char const* OP_TILING_PATH_FROM_NEW_OPP_PKG = "built-in/op_impl/ai_core/tbe/op_tiling/";
constexpr char const* OP_TILING_PATH_FROM_OLD_OPP_PKG = "op_impl/built-in/ai_core/tbe/op_tiling/";

inline void GetBuiltinOpTilingLibPath(const string& oppPath, string& opTilingPath)
{
    // new version opp path is ASCEND_OPP_PATH/built-in
    if (IsDir(oppPath + "built-in")) {
        opTilingPath = oppPath + OP_TILING_PATH_FROM_NEW_OPP_PKG;
    } else {
        opTilingPath = oppPath + OP_TILING_PATH_FROM_OLD_OPP_PKG;
    }
}

static aclnnStatus LoadBuiltinOpTiling(const std::string& oppPath, std::vector<void*>& handlers)
{
    string opTilingPath;
    GetBuiltinOpTilingLibPath(oppPath, opTilingPath);
    OP_LOGI("Load op tiling in path: %s", opTilingPath.c_str());
    const string osType = "linux";
#ifdef __x86_64__
    const string cpuType = "x86_64";
#elif defined(__aarch64__)
    const string cpuType = "aarch64";
#else
    const string cpuType = "unknown";
#endif
    string opTilingOsPath = opTilingPath + "lib/" + osType + "/" + cpuType + "/";
    vector<string> opTilingSoList;
    GetFilesWithSuffix(opTilingOsPath, OP_TILING_SO_SUFFIX, opTilingSoList);
    for (const auto& soRealPath : opTilingSoList) {
        void* handle = dlopen(soRealPath.c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (handle != nullptr) {
            handlers.push_back(handle);
            OP_LOGI("Load op tiling so successfully, so path: %s", soRealPath.c_str());
        } else {
            OP_LOGW("Load op tiling so failed, so path: %s", soRealPath.c_str());
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus LoadOpTiling(const std::string& oppPath, std::vector<void*>& handlers)
{
    return LoadBuiltinOpTiling(oppPath, handlers);
}

} // namespace opploader
} // namespace op
