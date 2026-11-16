/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ops_json_parse.h"
#include <fstream>
#include <sstream>
#include "ops_struct.h"
#include "opdev/op_log.h"
#include "opdev/op_errno.h"

using namespace std;
using namespace nlohmann;

namespace op {
namespace internal {
OpsJsonFile& OpsJsonFile::Instance()
{
    static OpsJsonFile instance;
    return instance;
}

static aclnnStatus ReadJsonFile(const string& filePath, json& jsonRead)
{
    OP_LOGI("Read custom operator repository %s file", filePath.c_str());
    ifstream ifs(filePath);
    if (!ifs.is_open()) {
        OP_LOGW("Open %s file failed.", filePath.c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }

    try {
        ifs >> jsonRead;
        ifs.close();
    } catch (const json::exception& e) {
        if (ifs.is_open()) {
            ifs.close();
        }
        OP_LOGW("[Check][JsonFile]invalid json string, exception:%s.", e.what());
        return ACLNN_ERR_PARAM_INVALID;
    }

    OP_LOGI("Read %s file success.", filePath.c_str());
    return ACLNN_SUCCESS;
}

aclnnStatus OpsJsonFile::ParseUnderPath(const string& filePath, json& jsonRead) const
{
    aclnnStatus ret = ReadJsonFile(filePath, jsonRead);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGW("Read kernel json file failed, file path is %s .", filePath.c_str());
        return ret;
    }
    return ConvertJsonFormat(jsonRead) ? ACLNN_SUCCESS : ACLNN_ERR_PARAM_INVALID;
}

bool OpsJsonFile::ConvertJsonFormat(json& jsonRead) const
{
    json opInfos = json::array();
    json jsonNull;
    for (auto it = jsonRead.cbegin(); it != jsonRead.cend(); ++it) {
        json newJson = it.value();
        string opType = it.key();
        newJson[kConfigOpType] = opType;

        bool userDefined = true;
        auto buff = newJson[kConfigOpInfo][kConfigUserDefined];
        if (!CheckAndGetUserDefine(buff, opType, "userDefined", userDefined)) {
            OP_LOGW("No userDefined information found in the json file.");
            return false;
        }
        newJson[kConfigOpInfo][kConfigUserDefined] = userDefined;

        buff = newJson[kConfigOpInfo][kConfigFunctionName];
        if (buff == jsonNull) {
            OP_LOGW("No functionName information found in the json file.");
            return false;
        }

        buff = newJson[kConfigOpInfo][kConfigKernelSo];
        if (buff == jsonNull) {
            OP_LOGW("No kernelSo information found in the json file.");
            return false;
        }

        buff = newJson[kConfigOpInfo][kConfigKernelLib];
        if (buff == jsonNull) {
            OP_LOGW("No opKernelLib information found in the json file.");
            return false;
        }
        opInfos.push_back(newJson);
    }

    jsonRead = {};
    if (!opInfos.empty()) {
        std::string kernelLib = opInfos[0][kConfigOpInfo][kConfigKernelSo];
        OP_LOGI("Convert json success, kernel config is %s.", kernelLib.c_str());
    }
    jsonRead[kConfigOpInfos] = opInfos;
    return true;
}

aclnnStatus OpsJsonFile::StringToBool(const std::string& str, bool& result) const
{
    result = false;
    std::string buff = str;
    try {
        (void)transform(buff.begin(), buff.end(), buff.begin(), ::tolower);
        if ((buff == "false") || (buff == "true")) {
            (istringstream(buff) >> boolalpha) >> result;
            return ACLNN_SUCCESS;
        }
    } catch (std::exception& e) {
        OP_LOGW("[Check][JsonFile]invalid json string, exception:%s.", e.what());
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_ERR_PARAM_INVALID;
}

bool OpsJsonFile::CheckAndGetUserDefine(const json& buff, const string& opType, const string& fieldStr,
                                        bool& value) const
{
    if (!buff.empty()) {
        if (StringToBool(buff.get<string>(), value) != ACLNN_SUCCESS) {
            OP_LOGW("Invalid ops info %s[%s], should be False or True, op[%s].", fieldStr.c_str(),
                    buff.get<string>().c_str(), opType.c_str());
            return false;
        }
    }
    return true;
}

template <typename T>
inline void Assignment(T& varible, const string& key, const json& jsonRead)
{
    auto iter = jsonRead.find(key);
    if (iter != jsonRead.end()) {
        varible = iter.value().get<T>();
    }
}

void from_json(const json& jsonRead, OpInfoDescs& infos) { Assignment(infos.opInfos, kConfigOpInfos, jsonRead); }

void from_json(const json& jsonRead, OpFullInfo& opInfo)
{
    Assignment(opInfo.opKernelLib, kConfigKernelLib, jsonRead);

    opInfo.userDefined = true;
    Assignment(opInfo.userDefined, kConfigUserDefined, jsonRead);

    opInfo.kernelSo = "";
    Assignment(opInfo.kernelSo, kConfigKernelSo, jsonRead);

    opInfo.functionName = "";
    Assignment(opInfo.functionName, kConfigFunctionName, jsonRead);
}

void from_json(const json& jsonRead, OpInfoDesc& desc)
{
    Assignment(desc.opType, kConfigOpType, jsonRead);
    Assignment(desc.opInfo, kConfigOpInfo, jsonRead);
}
} // namespace internal
} // namespace op
