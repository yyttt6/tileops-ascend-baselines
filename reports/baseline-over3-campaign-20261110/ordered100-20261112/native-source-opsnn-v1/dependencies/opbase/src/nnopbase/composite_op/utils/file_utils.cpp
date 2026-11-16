/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "file_utils.h"
#include <fstream>
#include <iostream>
#include <cstring>
#include <dirent.h>
#include "mmpa/mmpa_api.h"
#include "opdev/op_dfx.h"

using namespace std;
namespace op {

bool IsDir(const std::string& path)
{
    DIR* p = opendir(path.c_str());
    if (p != nullptr) {
        (void)closedir(p);
        return true;
    }
    return false;
}

std::unique_ptr<char[]> GetBinFromFile(const std::string& path, uint32_t& dataLen)
{
    ifstream ifs(path, ios::in | ios::binary);
    if (!ifs.is_open()) {
        return nullptr;
    }

    ifs.seekg(0, ios_base::end);
    streampos fileSize = ifs.tellg();
    ifs.seekg(0, ios_base::beg);

    auto p = unique_ptr<char[]>(new (nothrow) char[fileSize]);
    if (p == nullptr) {
        ifs.close();
        return nullptr;
    }

    if (!ifs.read(internal::PtrCastTo<char>(p.get()), fileSize)) {
        p = nullptr;
    } else {
        dataLen = static_cast<uint32_t>(fileSize);
    }

    ifs.close();
    return p;
}

void GetFilesWithSuffix(const std::string& path, const std::string& suffix, std::vector<std::string>& files)
{
    if (path.empty()) {
        return;
    }

    string realPath = RealPath(path);
    if (realPath.empty()) {
        return;
    }

    if (!IsDir(realPath)) {
        return;
    }

    struct dirent** entries = nullptr;
    const auto fileNum = scandir(realPath.c_str(), &entries, nullptr, nullptr);
    if (entries == nullptr) {
        return;
    }

    if (fileNum < 0) {
        free(entries);
        return;
    }

    for (int i = 0; i < fileNum; ++i) {
        const dirent* const dirEnt = entries[i];
        const string name = string(dirEnt->d_name);
        if ((strcmp(name.c_str(), ".") == 0) || (strcmp(name.c_str(), "..") == 0)) {
            continue;
        }

        if (dirEnt->d_type == DT_DIR) {
            continue;
        }

        if (name.size() < suffix.size() || name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0) {
            continue;
        }

        const string fullName = realPath + "/" + name;
        files.push_back(fullName);
    }

    for (int i = 0; i < fileNum; i++) {
        free(entries[i]);
    }
    free(entries);
}

std::string RealPath(const std::string& path)
{
    string res;
    char realPath[MMPA_MAX_PATH] = {0};
    int ret = mmRealPath(path.c_str(), realPath, MMPA_MAX_PATH);
    if (ret != EN_OK) {
        res.assign("");
    } else {
        res.assign(realPath);
    }
    return res;
}

} // namespace op
