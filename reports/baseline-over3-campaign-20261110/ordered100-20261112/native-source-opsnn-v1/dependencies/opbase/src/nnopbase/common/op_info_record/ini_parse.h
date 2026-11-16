/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef TUNING_INI_PARSE_H
#define TUNING_INI_PARSE_H
#include <string>
#include <map>

namespace aclnnOpInfoRecord {
class IniParse {
public:
    IniParse() {}
    ~IniParse() {}
    int32_t GetIniParams(std::map<std::string, std::string>& params);

private:
    bool CheckInputFile(const std::string& file) const;
    bool LoadIniFile(const std::string& iniFileRealPath);
    void ParseLine(const std::string& line, std::string& mapKey, std::map<std::string, std::string>& contentMap);
    std::map<std::string, std::map<std::string, std::string>> contentInfoMap_;
};
} // namespace aclnnOpInfoRecord
#endif