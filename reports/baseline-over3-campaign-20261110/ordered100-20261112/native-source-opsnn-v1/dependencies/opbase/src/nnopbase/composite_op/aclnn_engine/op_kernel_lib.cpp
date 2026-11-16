/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "op_kernel_lib.h"

#include <fstream>
#include <algorithm>

#include "utils/string_utils.h"
#include "file_utils.h"
#include "opdev/op_log.h"
#include "acl/acl_rt.h"
#include "nnopbase_error_msg.h"

namespace op {
namespace internal {
namespace {
constexpr char const* OP_FILE = "opFile";
constexpr char const* VALUE = "value";
constexpr char const* AICORE_IMPL_PATH_SUFFIX = "/built-in/op_impl/ai_core/tbe/";
constexpr char const* CUSTOM_IMPL_PATH_SUFFIX = "/op_impl/ai_core/tbe/";
constexpr char const* KERNEL_CONFIG_SUFFIX = "config/";
} // namespace

using namespace std;
using Json = nlohmann::json;
OpKernelLib::OpKernelLib() : initFlag_(false) {}

OpKernelLib::~OpKernelLib() {}

template <typename T>
aclnnStatus GetJsonValue(const nlohmann::json& json, const std::string& key1, const std::string& key2, T& value)
{
    auto iter1 = json.find(key1);
    if (iter1 == json.end()) {
        return ACLNN_ERR_INNER_JSON_VALUE_NOT_FOUND;
    }
    if (key2.empty()) {
        try {
            value = iter1->get<T>();
        } catch (nlohmann::json::exception& e) {
            OP_LOGE(ACLNN_ERR_INNER_OP_FILE_INVALID, "Exception:%s occurs when getting json value for key1 %s.",
                    e.what(), key1.c_str());
        }
    } else {
        auto iter2 = iter1->find(key2);
        if (iter2 == iter1->end()) {
            return ACLNN_ERR_INNER_JSON_VALUE_NOT_FOUND;
        }
        try {
            value = iter2->get<T>();
        } catch (nlohmann::json::exception& e) {
            OP_LOGE(ACLNN_ERR_INNER_OP_FILE_INVALID,
                    "Exception:%s occurs when getting json value for key1 %s and key2 %s.", e.what(), key1.c_str(),
                    key2.c_str());
        }
    }

    return ACLNN_SUCCESS;
}
// ============================= KernelLibInfo ==================================
KernelLibInfo::KernelLibInfo(uint32_t opTypeId, const std::string& opType) : opTypeId_(opTypeId), opType_(opType) {}

aclnnStatus KernelLibInfo::Initialize(nlohmann::json& singleKernelJson)
{
    (void)GetJsonValue(singleKernelJson, OP_FILE, VALUE, opFile_);
    if (!opFile_.empty()) {
        opFile_ += JSON_SUFFIX;
        BinConfigJsonDict::UpdateConfigJsonPath(opTypeId_, opFile_);
    }

    return ACLNN_SUCCESS;
}

const string& KernelLibInfo::GetOpFile() const { return opFile_; }

// ============================== OpKernelLib ====================================
static std::unordered_map<std::string, std::string> socOpMapV2 = {
    {"Ascend910_95", "ascend910_95/"}, {"Ascend950", "ascend950/"}, {"Ascend350", "ascend350/"}};

static std::string emptyString = "";

static const std::string& GetSocPathV2(const std::string& soc)
{
    for (auto& socPair : socOpMapV2) {
        const std::string& socPrefix = socPair.first;
        if (soc.compare(0, socPrefix.length(), socPrefix) == 0) {
            return socPair.second;
        }
    }
    return emptyString;
}

static map<string, string> socOpMap = {
    {"Ascend910A", "ascend910/"},        {"Ascend910B", "ascend910/"},        {"Ascend910ProA", "ascend910/"},
    {"Ascend910PremiumA", "ascend910/"}, {"Ascend910ProB", "ascend910/"},     {"Ascend910B1", "ascend910b/"},
    {"Ascend910B2", "ascend910b/"},      {"Ascend910B3", "ascend910b/"},      {"Ascend910B4", "ascend910b/"},
    {"Ascend910B4-1", "ascend910b/"},    {"Ascend910B2C", "ascend910b/"},     {"Ascend910_9391", "ascend910_93/"},
    {"Ascend910_9381", "ascend910_93/"}, {"Ascend910_9372", "ascend910_93/"}, {"Ascend910_9392", "ascend910_93/"},
    {"Ascend910_9382", "ascend910_93/"}, {"Ascend910_9362", "ascend910_93/"}, {"Ascend310P1", "ascend310p/"},
    {"Ascend310P3", "ascend310p/"},      {"Ascend310P5", "ascend310p/"},      {"Ascend310P7", "ascend310p/"},
    {"Ascend310B1", "ascend310b/"},      {"Ascend310B4", "ascend310b/"},      {"Ascend610Lite", "ascend610lite/"},
    {"KirinX90", "kirinx90/"},           {"Kirin9030", "kirin9030/"},         {"Ascend910_9363", "ascend910_93/"}};

const std::string& OpKernelLib::GetSocPath()
{
    if (!socPath_.empty()) {
        return socPath_;
    }

    auto soc = aclrtGetSocName();
    OP_CHECK(soc != nullptr, OP_LOGW("aclrtGetSocName return nullptr."), return socPath_);
    const string& devtype = string(soc);
    const auto& iter = socOpMap.find(devtype);
    if (iter != socOpMap.end()) {
        socPath_ = socOpMap[devtype];
    } else {
        socPath_ = GetSocPathV2(devtype);
    }
    OP_CHECK_NO_RETURN(!socPath_.empty(), OP_LOGW("Invalid device type:%s.", soc));
    OP_LOGI("aclrtGetSocName: %s, soc path: %s", soc, socPath_.c_str());
    return socPath_;
}

const std::string& OpKernelLib::GetAiCoreImplPath()
{
    if (!aiCoreImplPath_.empty()) {
        return aiCoreImplPath_;
    }

    std::string oppRealPath;
    auto ret = GetOppKernelPath(oppRealPath);
    OP_CHECK(ret == ACLNN_SUCCESS && !oppRealPath.empty(),
             OP_LOGW("opp kernel real path can not be found. ret %d", ret), return aiCoreImplPath_);

    aiCoreImplPath_.append(oppRealPath);
    aiCoreImplPath_.append(AICORE_IMPL_PATH_SUFFIX);
    return aiCoreImplPath_;
}

const std::vector<std::string>& OpKernelLib::GetCustomImplPath()
{
    if (!customImplPath_.empty()) {
        return customImplPath_;
    }

    const char* customOppPath = nullptr;
    MM_SYS_GET_ENV(MM_ENV_ASCEND_CUSTOM_OPP_PATH, customOppPath);
    if (customOppPath == nullptr) {
        OP_LOGI("ASCEND_CUSTOM_OPP_PATH is null.");
        return customImplPath_;
    }

    std::vector<std::string> strs = SplitWith(customOppPath, ':');
    for (const auto& str : strs) {
        const std::string implPath = str + CUSTOM_IMPL_PATH_SUFFIX;
        const std::string realImplPath = RealPath(implPath);
        OP_LOGI("custom impl path: %s, real path: %s", implPath.c_str(), realImplPath.c_str());
        if (realImplPath != "") {
            customImplPath_.emplace_back(realImplPath);
        }
    }

    return customImplPath_;
}

const std::vector<std::string>& OpKernelLib::GetConfigImplPath()
{
    if (!configImplPath_.empty()) {
        return configImplPath_;
    }

    const char* oppPath = nullptr;
    MM_SYS_GET_ENV(MM_ENV_ASCEND_OPP_PATH, oppPath);
    OP_CHECK(oppPath != nullptr, OP_LOGW("ASCEND_OPP_PATH is null."), return configImplPath_);

    const std::string oppPathStr = oppPath;
    const vector<std::string> vendorNames = GetVendorNames();
    for (const auto& vendorName : vendorNames) {
        const std::string implPath = oppPathStr + "/vendors/" + vendorName + CUSTOM_IMPL_PATH_SUFFIX;
        const std::string realImplPath = RealPath(implPath);
        OP_LOGD("config impl path: %s, real path: %s", implPath.c_str(), realImplPath.c_str());
        OP_CHECK_NO_RETURN(realImplPath == "", configImplPath_.emplace_back(realImplPath));
    }

    return configImplPath_;
}

aclnnStatus OpKernelLib::ParseKernelLibInfos(uint32_t opType)
{
    kernelLibInfos_[opType] = KernelLibInfo(opType, OpTypeDict::ToString(opType).GetString());
    uint32_t allOpTypeSize = OpTypeDict::GetAllOpTypeSize();

    for (auto& elem : allKernelsJson_.items()) {
        auto& opTypeStr = elem.key();
        uint32_t opTypeId = OpTypeDict::ToOpType(opTypeStr);
        if (opTypeId >= allOpTypeSize || opTypeId == 0) {
            continue;
        }

        if (opType == opTypeId) {
            CHECK_COND(elem.value().is_object(), ACLNN_ERR_INNER, "Op kernel library is not object.");
            kernelLibInfos_[opTypeId].Initialize(elem.value());
            break;
        }
    }

    return ACLNN_SUCCESS;
}

const std::vector<std::string> OpKernelLib::GetVendorNames() const
{
    std::vector<std::string> vendorNames;
    const char* oppPath = nullptr;
    MM_SYS_GET_ENV(MM_ENV_ASCEND_OPP_PATH, oppPath);
    OP_CHECK(oppPath != nullptr, OP_LOGW("ASCEND_OPP_PATH is null."), return vendorNames);

    const std::string oppPathStr = oppPath;
    const std::string configPath = oppPathStr + "/vendors/config.ini";
    const std::string realConfigPath = RealPath(configPath);
    OP_LOGI("config.ini file path [%s], real path: [%s].", configPath.c_str(), realConfigPath.c_str());
    if (realConfigPath == "") {
        OP_LOGI("config.ini file path [%s] is invalid or does not exist.", configPath.c_str());
        return vendorNames;
    }

    std::ifstream ifs(realConfigPath);
    std::string line;
    while (std::getline(ifs, line)) {
        TrimWith(line, '\r');
        TrimWith(line, '\n');
        if (line.empty() || (line.find('#') == 0)) {
            continue;
        }
        if (line.find("priority") != std::string::npos) {
            const size_t posOfEqual = line.find('=');
            if (posOfEqual != std::string::npos) {
                const std::string value = line.substr(posOfEqual + 1U);
                std::vector<std::string> strs = SplitWith(value, ',');
                vendorNames.insert(vendorNames.end(), strs.begin(), strs.end());
            }
            break;
        }
    }
    ifs.close();
    return vendorNames;
}

const std::vector<std::string> OpKernelLib::GetBuiltInFilePaths()
{
    std::vector<std::string> configFilePaths;
    std::vector<std::string> configFileNames;
    std::vector<std::string> orderedConfigFileNames;
    std::string configFileDir = GetAiCoreImplPath();
    configFileDir.append(KERNEL_CONFIG_SUFFIX);
    configFileDir.append(GetSocPath());
    OP_CHECK(ReadDirBySuffix(configFileDir, ".json", configFileNames) == ACLNN_SUCCESS,
             OP_LOGW("Failed to read dir: %s", configFileDir.c_str()), return configFilePaths);
    OP_CHECK(!configFileNames.empty(), OP_LOGW("configFileNames is emtpy in %s", configFileDir.c_str()),
             return configFilePaths);
    bool hasLegacyFile = false;
    std::string lebacyFileName;
    for (auto& fileName : configFileNames) {
        if (fileName.find("ops-info-legacy") != std::string::npos) {
            hasLegacyFile = true;
            lebacyFileName = fileName;
        } else {
            orderedConfigFileNames.emplace_back(fileName);
        }
    }
    if (hasLegacyFile) {
        orderedConfigFileNames.emplace_back(lebacyFileName);
    }

    for (const std::string& fileName : orderedConfigFileNames) {
        const std::string configFilePath = configFileDir + fileName;
        const std::string realConfigFilePath = RealPath(configFilePath);
        OP_CHECK_NO_RETURN(realConfigFilePath == "", configFilePaths.emplace_back(realConfigFilePath));
    }

    return configFilePaths;
}

const std::vector<std::string> OpKernelLib::GetOppVendorsFilePaths()
{
    std::vector<std::string> oppVendorsFilePaths;
    const char* oppPath = nullptr;
    MM_SYS_GET_ENV(MM_ENV_ASCEND_OPP_PATH, oppPath);
    OP_CHECK(oppPath != nullptr, OP_LOGW("ASCEND_OPP_PATH is null."), return oppVendorsFilePaths);

    const std::string oppPathStr = oppPath;
    const std::vector<std::string> vendorNames = GetVendorNames();
    for (const auto& vendorName : vendorNames) {
        const std::string customFileDir = oppPathStr + "/vendors/" + vendorName + CUSTOM_IMPL_PATH_SUFFIX +
                                          KERNEL_CONFIG_SUFFIX + GetSocPath();
        std::vector<std::string> customFileNames;
        std::string realCustomFileDir = RealPath(customFileDir);
        if (realCustomFileDir == "") {
            OP_LOGW("custom file dir [%s] is null, skip vendor [%s].", customFileDir.c_str(), vendorName.c_str());
            continue;
        }
        OP_CHECK_NO_RETURN(ReadDirBySuffix(realCustomFileDir, ".json", customFileNames) == ACLNN_SUCCESS,
                           OP_LOGW("Failed to read dir: %s", customFileDir.c_str()));
        for (const auto& fileName : customFileNames) {
            const std::string customFilePath = customFileDir + fileName;
            const std::string realCustomFilePath = RealPath(customFilePath);
            if (realCustomFilePath.empty()) {
                OP_LOGW("Skip path [%s], real path is empty.", customFilePath.c_str());
            } else {
                oppVendorsFilePaths.emplace_back(realCustomFilePath);
            }
        }
    }
    return oppVendorsFilePaths;
}

const std::vector<std::string> OpKernelLib::GetCustomOppFilePaths()
{
    std::vector<std::string> customOppFilePaths;
    const std::vector<std::string>& customImplPaths = GetCustomImplPath();
    for (const auto& element : customImplPaths) {
        // element 来自 RealPath(str + CUSTOM_IMPL_PATH_SUFFIX)，尾部斜杠行为不确定，
        // 显式补 '/' 兼容 mmRealPath 是否保留尾部斜杠两种实现
        const std::string customFileDir = element + "/" + KERNEL_CONFIG_SUFFIX + GetSocPath();
        std::string realCustomFileDir = RealPath(customFileDir);
        if (realCustomFileDir.empty()) {
            OP_LOGW("custom opp file dir [%s] is null, skip.", customFileDir.c_str());
            continue;
        }
        std::vector<std::string> customFileNames;
        OP_CHECK_NO_RETURN(ReadDirBySuffix(realCustomFileDir, ".json", customFileNames) == ACLNN_SUCCESS,
                           OP_LOGW("Failed to read dir: %s", customFileDir.c_str()));
        for (const auto& fileName : customFileNames) {
            const std::string customFilePath = customFileDir + fileName;
            const std::string realCustomFilePath = RealPath(customFilePath);
            if (realCustomFilePath.empty()) {
                OP_LOGW("Skip path [%s], real path is empty.", customFilePath.c_str());
            } else {
                customOppFilePaths.emplace_back(realCustomFilePath);
            }
        }
    }
    return customOppFilePaths;
}

aclnnStatus OpKernelLib::Initialize()
{
    OP_LOGI("Initialize OpKernelLib.");
    if (initFlag_) {
        OP_LOGD("OpKernelLib has been initialized.");
        return ACLNN_SUCCESS;
    }
    std::vector<std::string> customOppFilePaths = GetCustomOppFilePaths();
    std::vector<std::string> oppVendorsFilePaths = GetOppVendorsFilePaths();
    std::vector<std::string> opKernelLibFilePaths = GetBuiltInFilePaths();
    opKernelLibFilePaths.insert(opKernelLibFilePaths.begin(), oppVendorsFilePaths.begin(), oppVendorsFilePaths.end());
    opKernelLibFilePaths.insert(opKernelLibFilePaths.begin(), customOppFilePaths.begin(), customOppFilePaths.end());
    std::reverse(opKernelLibFilePaths.begin(), opKernelLibFilePaths.end());
    for (const auto& filePath : opKernelLibFilePaths) {
        OP_LOGI("OpKernelLib start parse json file: %s.", filePath.c_str());
        try {
            std::ifstream f(filePath);
            allKernelsJson_.merge_patch(Json::parse(f));
            OP_CHECK(allKernelsJson_.is_object(),
                     OP_LOGE_FOR_FILE_OPERATION_ERROR_PARSE_WITH_INVALID_CONTENT(
                         filePath.c_str(), "The operator JSON file is not in the standard key-value structure"),
                     return ACLNN_ERR_INNER_LOAD_JSON_FAILED);
        } catch (std::exception& e) {
            OP_LOGE_FOR_FILE_OPERATION_ERROR_PARSE(filePath.c_str(), e.what());
            return ACLNN_ERR_INNER_LOAD_JSON_FAILED;
        }
    }

    initFlag_ = true;
    OP_LOGI("Successfully initialized op kernel library.");
    return ACLNN_SUCCESS;
}

const KernelLibInfo& OpKernelLib::GetKernelLibInfo(uint32_t opTypeId) const { return kernelLibInfos_[opTypeId]; }
} // namespace internal
} // namespace op
