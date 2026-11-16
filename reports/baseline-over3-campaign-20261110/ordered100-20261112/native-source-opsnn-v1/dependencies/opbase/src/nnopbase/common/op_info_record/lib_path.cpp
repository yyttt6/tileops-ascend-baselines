/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "lib_path.h"
#include "mmpa/mmpa_api.h"
#include "opdev/op_log.h"
#include "opdev/op_errno.h"
#include "utils.h"

namespace aclnnOpInfoRecord {
constexpr char LIBSUFFIX[] = ".so";

LibPath& LibPath::Instance()
{
    static LibPath libPath;
    return libPath;
}

Path LibPath::GetInstallParentPath() const
{
    // get install path from self path : <install path>/x86_64-linux/lib64/ --> <install path>/x86_64-linux
    Path currentPath(LibPath::GetSelfLibraryDir());
    auto installPath = currentPath.ParentPath();
    return installPath;
}

/**
 * @brief       Get directory path where self dynamic library in
 * @return      Path
 */
Path LibPath::GetSelfLibraryDir() const
{
    mmDlInfo info;
    LibPath& (*instancePtr)() = &LibPath::Instance;
    const auto ret = mmDladdr(reinterpret_cast<void*>(instancePtr), &info);
    if (ret != EN_OK) {
        OP_LOGE(ACLNN_ERR_INNER, "Cannot find symbol GetSelfLibraryDir");
        return Path();
    }
    OP_LOGD("Find symbol GetSelfLibraryDir in %s", info.dli_fname);
    Path path(info.dli_fname); // entire so name
    const auto name = path.GetFileName();
    if (name.length() <= strlen(LIBSUFFIX) ||
        name.compare(name.length() - strlen(LIBSUFFIX), strlen(LIBSUFFIX), LIBSUFFIX) != 0) {
        path = GetSelfPath();
    }

    if (!path.RealPath()) {
        OP_LOGW("Can not get realpath self library path %s.", path.GetCString());
        return Path();
    }

    OP_LOGI("Get self library path: %s", path.GetCString());

    return path.ParentPath();
}

/**
 * @brief       Get directory path where self binary
 * @return      Path
 */
Path LibPath::GetSelfPath() const
{
    Path selfPath;
    char curPath[MMPA_MAX_PATH + 1] = {0};
    const auto len = readlink("/proc/self/exe", curPath, MMPA_MAX_PATH);
    if (len <= 0 || len > MMPA_MAX_PATH) {
        OP_LOGE(ACLNN_ERR_INNER, "Get self path failed.");
        return selfPath;
    }
    curPath[len] = '\0';

    selfPath = curPath;
    OP_LOGI("Get self path: %s", selfPath.GetCString());
    return selfPath;
}

} // namespace aclnnOpInfoRecord
