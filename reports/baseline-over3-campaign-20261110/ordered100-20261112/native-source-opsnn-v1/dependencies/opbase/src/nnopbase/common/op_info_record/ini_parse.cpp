/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ini_parse.h"
#include <fstream>
#include "opdev/op_log.h"
#include "opdev/op_errno.h"
#include "path.h"
#include "utils.h"
#include "lib_path.h"

namespace aclnnOpInfoRecord {
const std::string DEFAULT_CONF_PATH = "conf/dump_tool_config.ini";
constexpr size_t MAX_LINE_SIZE = 10000UL;

bool IniParse::CheckInputFile(const std::string& file) const
{
    if (file.empty()) {
        OP_LOGE(ACLNN_ERR_INNER, "Invalid file empty");
        return false;
    }
    aclnnOpInfoRecord::Path filePath(file);
    if (!filePath.RealPath()) {
        OP_LOGE(ACLNN_ERR_INNER, "Get real path of %s failed", file.c_str());
        return false;
    }
    if (!filePath.Exist()) {
        OP_LOGE(ACLNN_ERR_INNER, "Cannot access file %s", file.c_str());
        return false;
    }
    return true;
}

void IniParse::ParseLine(const std::string& line, std::string& mapKey, std::map<std::string, std::string>& contentMap)
{
    if (line.find('[') == 0) {
        if (!mapKey.empty() && !contentMap.empty()) {
            contentInfoMap_.emplace(make_pair(mapKey, contentMap));
            contentMap.clear();
        }
        const size_t pos = line.rfind(']');
        if (pos == std::string::npos) {
            return;
        }
        mapKey = line.substr(1, pos - 1);
        mapKey = aclnnOpInfoRecord::Utils::Trim(mapKey);
        return;
    }

    const size_t posOfEqual = line.find('=');
    if (posOfEqual == std::string::npos) {
        return;
    }

    std::string key = line.substr(0, posOfEqual);
    key = aclnnOpInfoRecord::Utils::Trim(key);
    std::string value = line.substr(posOfEqual + 1, line.length() - posOfEqual - 1);
    value = aclnnOpInfoRecord::Utils::Trim(value);
    if (value != "tune" && value != "static_kernel" && value != "all") {
        OP_LOGE(ACLNN_ERR_INNER, "Config value only support tune/static_kernel/all.");
    }
    if (!key.empty()) {
        contentMap.emplace(make_pair(key, value));
    }
}

bool IniParse::LoadIniFile(const std::string& iniFileRealPath)
{
    if (!CheckInputFile(iniFileRealPath)) {
        return false;
    }
    std::ifstream ifs(iniFileRealPath);
    if (!ifs.is_open()) {
        return false;
    }
    std::map<std::string, std::string> contentMap;
    std::string line;
    std::string mapKey;
    for (size_t i = 0; i < MAX_LINE_SIZE; ++i) {
        if (!std::getline(ifs, line)) {
            break;
        }
        line = aclnnOpInfoRecord::Utils::Trim(line);
        if (!line.empty() && line.front() == '#') {
            continue;
        }
        ParseLine(line, mapKey, contentMap);
    }
    if (!contentMap.empty() && !mapKey.empty()) {
        contentInfoMap_.emplace(make_pair(mapKey, contentMap));
        contentMap.clear();
    }

    ifs.close();
    return true;
}

int32_t IniParse::GetIniParams(std::map<std::string, std::string>& params)
{
    aclnnOpInfoRecord::Path installBasePath = aclnnOpInfoRecord::LibPath::Instance().GetInstallParentPath();
    if (installBasePath.Empty()) {
        OP_LOGE(ACLNN_ERR_INNER, "Get install path empty.");
        return -1;
    }
    auto iniPath = installBasePath.Append(DEFAULT_CONF_PATH).GetString();
    iniPath.erase(iniPath.size() - 1);
    if (!LoadIniFile(iniPath)) {
        OP_LOGE(ACLNN_ERR_INNER, "Invalid config file.");
        return -1;
    }
    auto it = contentInfoMap_.find("COMMON");
    if (it != contentInfoMap_.end()) {
        params = it->second;
    }
    return 0;
}
} // namespace aclnnOpInfoRecord