/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "indv_path.h"
#include "indv_linux.h"

namespace nnopbase {
void IndvPath::GetChildDirs(std::string& baseDir, std::vector<std::string>& childDirs, uint32_t depth,
                            uint32_t maxDepth)
{
    if (baseDir.empty() || depth > maxDepth) {
        return;
    }
    // 扫描子文件
    IndvDirent** nameList = nullptr;
    const int32_t fileCount = IndvScandir(baseDir.c_str(), &nameList, nullptr, nullptr);
    if (fileCount == INDV_EN_ERROR || fileCount == INDV_EN_INVALID_PARAM) {
        return;
    }

    if (nameList == nullptr) {
        return;
    }
    // 遍历查找
    for (int32_t i = 0; i < fileCount; ++i) {
        std::string childName = nameList[i]->d_name;
        std::string childPath = baseDir + "/" + childName;
        // 过滤上级目录和本级目录符号
        if (childName.compare(".") == 0 || childName.compare("..") == 0) {
            continue;
        }
        childDirs.push_back(childPath);
        // 递归查找
        GetChildDirs(childPath, childDirs, depth + 1, maxDepth);
    }

    IndvScandirFree(nameList, fileCount);
}
} // namespace nnopbase