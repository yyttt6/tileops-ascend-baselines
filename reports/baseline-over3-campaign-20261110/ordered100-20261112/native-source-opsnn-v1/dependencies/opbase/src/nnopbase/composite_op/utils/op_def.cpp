/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdlib>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include <mutex>
#include <dirent.h>
#include <strings.h>
#include "opdev/op_def.h"
#include "opdev/op_log.h"

namespace op {

using std::map;
using std::pair;
using std::string;
using std::vector;

static std::atomic<uint32_t> opTypeNum = 0;
static std::mutex opTypeMutex __attribute__((init_priority(200)));
std::unordered_map<std::string, uint32_t> globalOpTypeName2Id__ __attribute__((init_priority(201)));
std::vector<ge::AscendString> globalOpTypeName__ __attribute__((init_priority(201)));
std::unordered_map<std::string, uint32_t>& OpTypeDict::opTypeName2Id_ = globalOpTypeName2Id__;
std::vector<ge::AscendString>& OpTypeDict::opTypeName_ = globalOpTypeName__;

std::vector<std::vector<std::string>> BinConfigJsonDict::opConfigJsonPath_;
std::unordered_map<std::string, uint32_t> BinConfigJsonDict::opConfigJsonPath2Id_;
uint32_t BinConfigJsonDict::transDataId_ = INVALID_OP_TYPE_ID;
constexpr char const* TRANS_DATA = "TransData";

aclnnStatus ReadDirBySuffix(const string& dir, const string& suffix, vector<string>& paths)
{
    // GX
    DIR* dirp = opendir(dir.c_str());
    if (dirp == nullptr) {
        OP_LOGW("Dir %s is invalid.", dir.c_str());
        return ACLNN_ERR_INNER;
    }
    struct dirent* dp;
    while ((dp = readdir(dirp)) != nullptr) {
        const string& fn = string(dp->d_name);
        size_t fnlen = fn.size();
        if (fnlen >= suffix.size() && fn.substr(fnlen - suffix.size()) == suffix) {
            paths.emplace_back(fn);
        }
    }
    closedir(dirp);
    return ACLNN_SUCCESS;
}

static std::string GetConfigJsonNameOld(const ge::AscendString& s)
{
    std::string result;
    size_t letterCountFromLastUnderline = 0;
    auto str = s.GetString();
    for (size_t i = 0; i < s.GetLength(); ++i) {
        char char_curr = str[i];

        if (isdigit(char_curr)) {
            result.push_back(char_curr);
        } else if (isupper(char_curr)) {
            if (letterCountFromLastUnderline >= 1 && i != 0 && !isdigit(str[i - 1])) {
                result.push_back('_');
                letterCountFromLastUnderline = 0;
            }
            result.push_back(static_cast<char>(tolower(char_curr)));
        } else {
            result.push_back(char_curr);
        }
        letterCountFromLastUnderline++;
    }
    result += JSON_SUFFIX;
    return result;
}

static std::string GetConfigJsonName(const ge::AscendString& s)
{
    std::string result;
    size_t letterCountFromLastUnderline = 0;
    auto str = s.GetString();
    for (size_t i = 0; i < s.GetLength(); ++i) {
        char char_curr = str[i];

        if (isdigit(char_curr)) {
            if (i > 0) {
                result.push_back('_');
                letterCountFromLastUnderline = 0;
            }
            result.push_back(char_curr);
        } else if (isupper(char_curr)) {
            if (letterCountFromLastUnderline >= 1 && i != 0) {
                result.push_back('_');
                letterCountFromLastUnderline = 0;
            }
            result.push_back(static_cast<char>(tolower(char_curr)));
        } else {
            result.push_back(char_curr);
        }
        letterCountFromLastUnderline++;
    }
    result += JSON_SUFFIX;
    return result;
}

aclnnStatus OpTypeDict::Add(uint32_t& id, const char* opName)
{
    const std::lock_guard<std::mutex> lock(opTypeMutex);
    if (opTypeName2Id_.find(opName) != opTypeName2Id_.end()) {
        id = opTypeName2Id_[opName];
        OP_LOGI("Op %s is already added, id is %u", opName, id);
        return ACLNN_SUCCESS;
    }
    if (opTypeName_.empty()) {
        opTypeName_.emplace_back(ge::AscendString("BEGIN__"));
        std::vector<std::string> temp;
        temp.emplace_back();
        BinConfigJsonDict::opConfigJsonPath_.emplace_back(temp);
        opTypeNum++;
    }

    // check duplicate
    id = opTypeNum;
    opTypeName_.emplace_back(ge::AscendString(opName));
    string opConfigFileStr(GetConfigJsonName(opTypeName_.back()));
    OP_LOGI("Add op %s with id %u, config json name [%s].\n", opName, id, opConfigFileStr.c_str());
    opTypeName2Id_[opName] = id;

    // compatible with the operator naming rules with old version
    string opConfigFileStrOld(GetConfigJsonNameOld(opTypeName_.back()));

    // updata BinConfigJsonDict
    std::vector<std::string> configFiles;
    configFiles.emplace_back(opConfigFileStr);
    if (opConfigFileStrOld != opConfigFileStr) {
        configFiles.emplace_back(opConfigFileStrOld);
    }
    BinConfigJsonDict::opConfigJsonPath_.emplace_back(configFiles);
    BinConfigJsonDict::opConfigJsonPath2Id_[opConfigFileStr] = id;
    BinConfigJsonDict::opConfigJsonPath2Id_[opConfigFileStrOld] = id;

    /* Special operation for the following case:
     * Two */
    if (BinConfigJsonDict::transDataId_ == INVALID_OP_TYPE_ID && opTypeName_.back() == ge::AscendString(TRANS_DATA)) {
        OP_LOGI("TransData id is %u. config json name [%s].\n", id, opConfigFileStr.c_str());
        BinConfigJsonDict::transDataId_ = id;
    }
    opTypeNum++;
    return ACLNN_SUCCESS;
}

uint32_t OpTypeDict::ToOpType(const std::string& opName)
{
    const std::lock_guard<std::mutex> lock(opTypeMutex);
    auto iter = opTypeName2Id_.find(opName);
    if (iter == opTypeName2Id_.end()) {
        return 0;
    }
    return iter->second;
}

const ge::AscendString OpTypeDict::ToString(uint32_t opType)
{
    const std::lock_guard<std::mutex> lock(opTypeMutex);
    if (opType >= opTypeNum) {
        return opTypeName_[0];
    }
    return opTypeName_[opType];
}

size_t OpTypeDict::GetAllOpTypeSize() { return static_cast<size_t>(opTypeNum); }

uint32_t BinConfigJsonDict::ToOpTypeByConfigJson(const std::string& op_config_json)
{
    const std::lock_guard<std::mutex> lock(opTypeMutex);
    auto iter = opConfigJsonPath2Id_.find(op_config_json);
    if (iter == opConfigJsonPath2Id_.end()) {
        return 0;
    }
    return iter->second;
}

void BinConfigJsonDict::UpdateConfigJsonPath(uint32_t opType, const std::string& opFile)
{
    const std::lock_guard<std::mutex> lock(opTypeMutex);
    if (opType < opConfigJsonPath_.size()) {
        for (std::string& fileName : opConfigJsonPath_[opType]) {
            if (fileName == opFile) {
                return;
            }
        }
        opConfigJsonPath_[opType].emplace_back(opFile);
        opConfigJsonPath2Id_[opFile] = opType;
        OP_LOGD("opFile for op %u %s is %s.", opType, OpTypeDict::opTypeName_[opType].GetString(), opFile.c_str());
    }
}

std::vector<std::string> GetOpConfigJsonFileName(uint32_t opType)
{
    const std::lock_guard<std::mutex> lock(opTypeMutex);
    return BinConfigJsonDict::opConfigJsonPath_[opType];
}

aclnnStatus ReadFile2String(const char* filename, string& content)
{
    std::FILE* fp = std::fopen(filename, "rb");
    if (fp == nullptr) {
        return ACLNN_ERR_INNER;
    }
    std::fseek(fp, 0, SEEK_END);
    content.resize(std::ftell(fp));
    std::rewind(fp);
    size_t nr = std::fread(&content[0], 1, content.size(), fp);
    if (nr != content.size()) {
        std::fclose(fp);
        return ACLNN_ERR_INNER;
    }
    std::fclose(fp);
    return ACLNN_SUCCESS;
}

static const std::map<std::string, OpImplMode> STRING_TO_OP_IMPL_MODE_MAP = {
    {"high_performance", OpImplMode::IMPL_MODE_HIGH_PERFORMANCE},
    {"high_precision", OpImplMode::IMPL_MODE_HIGH_PRECISION},
    {"super_performance", OpImplMode::IMPL_MODE_SUPER_PERFORMANCE},
    {"support_out_of_bound_index", OpImplMode::IMPL_MODE_SUPPORT_OUT_OF_BOUND_INDEX},
    {"enable_float32_execution", OpImplMode::IMPL_MODE_ENABLE_FLOAT32_EXECUTION},
    {"enable_hi_float32_execution", OpImplMode::IMPL_MODE_ENABLE_HI_FLOAT32_EXECUTION},
    {"keep_fp16", OpImplMode::IMPL_MODE_KEEP_FP16},
    {"default", OpImplMode::IMPL_MODE_DEFAULT}};

static const std::map<OpImplMode, ge::AscendString> OP_IMPL_MODE_TO_STRING_MAP = {
    {OpImplMode::IMPL_MODE_HIGH_PERFORMANCE, ge::AscendString("high_performance")},
    {OpImplMode::IMPL_MODE_HIGH_PRECISION, ge::AscendString("high_precision")},
    {OpImplMode::IMPL_MODE_SUPER_PERFORMANCE, ge::AscendString("super_performance")},
    {OpImplMode::IMPL_MODE_SUPPORT_OUT_OF_BOUND_INDEX, ge::AscendString("support_out_of_bound_index")},
    {OpImplMode::IMPL_MODE_ENABLE_FLOAT32_EXECUTION, ge::AscendString("enable_float32_execution")},
    {OpImplMode::IMPL_MODE_ENABLE_HI_FLOAT32_EXECUTION, ge::AscendString("enable_hi_float32_execution")},
    {OpImplMode::IMPL_MODE_KEEP_FP16, ge::AscendString("keep_fp16")},
    {OpImplMode::IMPL_MODE_DEFAULT, ge::AscendString("default")},
    {OpImplMode::IMPL_MODE_RESERVED, ge::AscendString("unknown")}};

OpImplMode ToOpImplMode(const std::string& implModeStr)
{
    OpImplMode implMode = OpImplMode::IMPL_MODE_RESERVED;
    auto found = STRING_TO_OP_IMPL_MODE_MAP.find(implModeStr);
    if (found != STRING_TO_OP_IMPL_MODE_MAP.end()) {
        implMode = found->second;
    } else {
        OP_LOGW("unknown OpImplMode:%s.", implModeStr.c_str());
    }
    return implMode;
}

ge::AscendString ToString(OpImplMode implMode)
{
    auto found = OP_IMPL_MODE_TO_STRING_MAP.find(implMode);
    if (found != OP_IMPL_MODE_TO_STRING_MAP.end()) {
        return found->second;
    }
    return OP_IMPL_MODE_TO_STRING_MAP.at(OpImplMode::IMPL_MODE_RESERVED);
}

const ge::AscendString& ImplModeToString(OpImplMode implMode)
{
    auto found = OP_IMPL_MODE_TO_STRING_MAP.find(implMode);
    if (found != OP_IMPL_MODE_TO_STRING_MAP.end()) {
        return found->second;
    }
    return OP_IMPL_MODE_TO_STRING_MAP.at(OpImplMode::IMPL_MODE_RESERVED);
}

int64_t ToIndex(OpImplMode implMode) { return ffs(static_cast<int64_t>(implMode)) - 1; }

wchar_t ToIndexChar(OpImplMode implMode)
{
    return static_cast<wchar_t>('0') + static_cast<wchar_t>(ffs(static_cast<int64_t>(implMode)) - 1);
}
} // namespace op
