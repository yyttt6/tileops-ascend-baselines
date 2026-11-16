/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <fstream>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>
#include <dirent.h>
#include <nlohmann/json.hpp>

#include "register/op_binary_resource_manager.h"

#include "file_utils.h"
#include "op_kernel_lib.h"
#include "kernel_utils.h"
#include "op_run_context.h"
#include "../../common/inc/kernel_mgr.h"
#include "nnopbase_error_msg.h"

namespace op {
namespace internal {
using std::ifstream;
using std::map;
using std::pair;
using std::string;
using std::stringstream;
using std::vector;

using json = nlohmann::json;

constexpr const char* STATIC_CONFIG_JSON_PATH = "/static_kernel/ai_core/config/";
constexpr const char* STATIC_CONFIG_FILE_NAME = "binary_info_config.json";
constexpr const char* STATIC_BIN_AND_JSON_DIR_PATH = "/static_kernel/ai_core/";

constexpr const char* DEBUG_CONFIG_JSON_PATH = "/debug_kernel/config/";
constexpr const char* DEBUG_CONFIG_FILE_NAME = "binary_info_config.json";
constexpr const char* DEBUG_BIN_AND_JSON_DIR_PATH = "/debug_kernel/";

KernelMgr& gKernelMgr = KernelMgr::getInstance();

static void StrSplit(const std::string& src, std::vector<std::string>& dst, const string& sep)
{
    std::string strs = src + sep;
    size_t pos = strs.find(sep);
    std::string subStr;
    while (pos != string::npos) {
        subStr = strs.substr(0, pos);
        if (!subStr.empty()) {
            dst.push_back(subStr);
        }
        strs = strs.substr(pos + 1U);
        pos = strs.find(sep);
    }
}

OpKernel* KernelMgr::GetKernel(uint32_t opType)
{
    if (opType >= OpTypeDict::GetAllOpTypeSize() || opType == 0 || opType >= kernel_.size()) {
        OP_LOGE(ACLNN_ERR_INNER, "Op type [%u, %s] is invalid.", opType, op::OpTypeDict::ToString(opType).GetString());
        return nullptr;
    }
    OP_LOGD("Found kernel of op type %s.", op::OpTypeDict::ToString(opType).GetString());
    auto& kernel = kernel_[opType];
    kernel.SetOpType(opType);
    return &kernel;
}

aclnnStatus KernelMgr::LoadStaticBinJson()
{
    staticConfigJson_.clear();

    std::string staticKernelBasePath;
    if (!customStaticKernelBasePath_.empty()) {
        // 路径已在 aclnnReselectStaticKernelWithPath 入口做过 RealPath 校验
        OP_LOGI("Load static bin json from custom base path: %s", customStaticKernelBasePath_.c_str());
        staticKernelBasePath = customStaticKernelBasePath_;
        customStaticKernelBasePath_.clear();
    } else {
        auto ret = GetOppKernelPath(staticKernelBasePath);
        OP_CHECK(ret == ACLNN_SUCCESS && !staticKernelBasePath.empty(),
                 OP_LOGE_FOR_CONFIG_ERROR_INVALID_ENVIRONMENT_VARIABLE("Loading static bin json", "ASCEND_OPP_PATH"),
                 return ACLNN_ERR_INNER_OPP_PATH_NOT_FOUND);
    }
    std::string configJsonPath = staticKernelBasePath;
    auto& knlLib = OpKernelLib::GetInstance();
    configJsonPath.append(STATIC_CONFIG_JSON_PATH);
    configJsonPath.append(knlLib.GetSocPath());
    configJsonPath.append(STATIC_CONFIG_FILE_NAME);
    staticBinAndJsonDir_ = staticKernelBasePath;
    staticBinAndJsonDir_.append(STATIC_BIN_AND_JSON_DIR_PATH);
    std::string realConfigJsonPath = RealPath(configJsonPath);
    OP_CHECK(!realConfigJsonPath.empty(),
             OP_LOGW("static kernel config json is not exist, path: %s", configJsonPath.c_str()), return ACLNN_SUCCESS);
    ifstream f(realConfigJsonPath);
    try {
        staticConfigJson_ = nlohmann::json::parse(f);
    } catch (nlohmann::json::exception& e) {
        OP_LOGW("Cannot parse static json for config file [%s] because %s.", realConfigJsonPath.c_str(), e.what());
        return ACLNN_SUCCESS;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus KernelMgr::LoadDebugStaticBinJson()
{
    string oppRealPath;
    auto ret = GetOppKernelPath(oppRealPath);
    OP_CHECK(ret == ACLNN_SUCCESS && !oppRealPath.empty(),
             OP_LOGE_FOR_CONFIG_ERROR_INVALID_ENVIRONMENT_VARIABLE("Loading debug static bin json", "ASCEND_OPP_PATH"),
             return ACLNN_ERR_INNER_OPP_PATH_NOT_FOUND);
    string debugConfigJsonPath = oppRealPath;
    auto& knlLib = OpKernelLib::GetInstance();
    debugConfigJsonPath.append(DEBUG_CONFIG_JSON_PATH);
    debugConfigJsonPath.append(knlLib.GetSocPath());
    debugConfigJsonPath.append(DEBUG_CONFIG_FILE_NAME);
    debugStaticBinAndJsonDir_ = oppRealPath;
    debugStaticBinAndJsonDir_.append(DEBUG_BIN_AND_JSON_DIR_PATH);
    ifstream f(debugConfigJsonPath);
    try {
        debugStaticConfigJson_ = nlohmann::json::parse(f);
    } catch (nlohmann::json::exception& e) {
        OP_LOGW("Cannot parse static json for config file [%s] because %s.", debugConfigJsonPath.c_str(), e.what());
        return ACLNN_SUCCESS;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus KernelMgr::ParseStaticKernelConfig(uint32_t opType)
{
    CHECK_RET(opType < MAX_OP_TYPE_COUNT, ACLNN_ERR_PARAM_INVALID);
    auto& kernel = kernel_[opType];
    const string& opTypeStr = kernel.GetOpTypeStr();
    OP_LOGD("Parse kernel %s", opTypeStr.c_str());
    auto opIter = staticConfigJson_.find(opTypeStr);
    if (opIter == staticConfigJson_.end() || opIter->is_null()) {
        return ACLNN_SUCCESS;
    }

    kernel.AppendStaticBin(*opIter, staticBinAndJsonDir_);

    bool enableDebug = false;
    aclnnStatus ret = op::internal::systemConfig.GetEnableDebugKernelFlag(enableDebug);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGW("GetEnableDebugKernelFlag failed.");
        return ACLNN_SUCCESS;
    }
    if (enableDebug) {
        auto opDebugIter = debugStaticConfigJson_.find(opTypeStr);
        opDebugIter = debugStaticConfigJson_.find(opTypeStr);
        if (opDebugIter == debugStaticConfigJson_.end() || opDebugIter->is_null()) {
            return ACLNN_SUCCESS;
        }
        OP_LOGI("Start to parse debug static kernel for %s, bin and json dir is %s.", opTypeStr.c_str(),
                debugStaticBinAndJsonDir_.c_str());
        kernel.AppendStaticBin(*opDebugIter, debugStaticBinAndJsonDir_);
    }
    return ACLNN_SUCCESS;
}

aclnnStatus KernelMgr::ParseDynamicKernelInStaticLib(const string& configFileName)
{
    auto opType = op::BinConfigJsonDict::ToOpTypeByConfigJson(configFileName);
    if (opType >= op::OpTypeDict::GetAllOpTypeSize() || opType == 0) {
        OP_LOGD("Cannot find op type for config json %s.", configFileName.c_str());
        return ACLNN_ERR_INNER;
    }
    auto& kernel = kernel_[opType];
    kernel.SetOpType(opType);
    ge::AscendString opTypeAscendStr = op::OpTypeDict::ToString(opType);
    kernel.SetOpType(opTypeAscendStr.GetString());

    nlohmann::json configJson;
    auto ret = nnopbase::OpBinaryResourceManager::GetInstance().GetOpBinaryDesc(opTypeAscendStr.GetString(),
                                                                                configJson);
    OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGW("No builtin op desc info [%s]", opTypeAscendStr.GetString()),
             return ACLNN_ERR_INNER_LOAD_JSON_FAILED);
    ret = kernel.AppendDynBin("", "", false);
    OP_CHECK(ret == ACLNN_SUCCESS,
             OP_LOGW("Failed to append dynamic kernel from static lib, op name: %s.", opTypeAscendStr.GetString()),
             return ACLNN_ERR_INNER);
    OP_LOGI("Successfully parse dynamic kernel from static lib, op name: %s.", opTypeAscendStr.GetString());
    return ACLNN_SUCCESS;
}

aclnnStatus KernelMgr::ParseDynamicKernelConfig(const std::vector<std::string>& configFileNames)
{
    for (size_t i = 0; i < customConfigDirs_.size(); i++) {
        for (const auto& fileName : configFileNames) {
            const std::string jsonFile = customConfigDirs_[i] + fileName;
            const std::string realJsonFile = RealPath(jsonFile);
            if (realJsonFile == "") {
                OP_LOGI("json file path [%s] is invalid or does not exist, continue next priority", jsonFile.c_str());
                continue;
            }
            auto ret = ParseDynamicKernelConfig(customConfigDirs_[i], customBinAndJsonDirs_[i], fileName, "", false);
            if (ret == ACLNN_SUCCESS) {
                OP_LOGI("Parse dynamic kernel config success, file path: %s", realJsonFile.c_str());
                return ACLNN_SUCCESS;
            }
        }
    }

    for (size_t i = 0; i < configJsonOpsFolders_.size(); i++) {
        for (const auto& fileName : configFileNames) {
            const std::string jsonFile = builtInConfigDir_ + configJsonOpsFolders_[i] + "/" + fileName;
            const std::string realJsonFile = RealPath(jsonFile);
            if (realJsonFile == "") {
                OP_LOGI("json file path [%s] is invalid or does not exist, continue next priority", jsonFile.c_str());
                continue;
            }
            auto ret = ParseDynamicKernelConfig(builtInConfigDir_, builtInBinAndJsonDir_, fileName,
                                                configJsonOpsFolders_[i], false);
            if (ret == ACLNN_SUCCESS) {
                OP_LOGI("Parse dynamic kernel config of builtIn success, file path: %s", realJsonFile.c_str());
                return ACLNN_SUCCESS;
            }
        }
    }

    if (configFileNames.size() > 0) {
        OP_CHECK(ParseDynamicKernelInStaticLib(configFileNames[0]) != ACLNN_SUCCESS,
                 OP_LOGI("Parse dynamic kernel from static lib successfully"), return ACLNN_SUCCESS);
    }
    OP_LOGE_FOR_CONFIG_FILE_NOT_EXIST_IN_DYNAMIC_SHAPE();
    return ACLNN_ERR_INNER;
}

aclnnStatus KernelMgr::ParseDynamicKernelConfig(const string& configDir, const string& binJsonDir,
                                                const string& configFileName, const string& opsRepoName,
                                                bool debug = false)
{
    auto suffixPos = configFileName.find(JSON_SUFFIX);
    if (suffixPos == string::npos) {
        OP_LOGE(ACLNN_ERR_INNER, "Config json name %s is invalid.", configFileName.c_str());
        return ACLNN_ERR_INNER;
    }

    auto opType = op::BinConfigJsonDict::ToOpTypeByConfigJson(configFileName);
    if (opType >= op::OpTypeDict::GetAllOpTypeSize() || opType == 0) {
        OP_LOGD("Cannot find op type for config json %s.", configFileName.c_str());
        return ACLNN_ERR_INNER;
    }

    auto& kernel = kernel_[opType];
    kernel.SetOpType(opType);
    ge::AscendString opTypeAscendStr = op::OpTypeDict::ToString(opType);
    kernel.SetOpType(opTypeAscendStr.GetString());
    kernel.SetOpsRepoName(opsRepoName);
    OP_LOGD("Set op type for kernel %u %s%s/%s", opType, configDir.c_str(), opsRepoName.c_str(),
            configFileName.c_str());
    auto ret = kernel.AppendDynBin(configDir + opsRepoName + "/" + configFileName, binJsonDir, debug);
    OP_CHECK(ret == ACLNN_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_LOAD_JSON_FAILED, "Failed to parse kernel in %s.", configFileName.c_str()),
             return ACLNN_ERR_INNER_LOAD_JSON_FAILED);
    OP_LOGI("Successfully parse kernel in %s.", configFileName.c_str());
    return ACLNN_SUCCESS;
}

aclnnStatus KernelMgr::ParseDynamicKernelConfig(const std::vector<std::string>& configDir,
                                                const std::vector<std::string>& binJsonDir,
                                                const std::vector<std::string>& configFileNames)
{
    for (size_t i = 0; i < configDir.size(); i++) {
        for (size_t j = 0; j < configFileNames.size(); j++) {
            const std::string jsonFile = configDir[i] + configFileNames[j];
            const std::string realJsonFile = RealPath(jsonFile);
            if (realJsonFile == "") {
                OP_LOGI("json file path [%s] is invalid or does not exist, continue next priority", jsonFile.c_str());
                continue;
            }
            auto ret = ParseDynamicKernelConfig(configDir[i], binJsonDir[i], configFileNames[j], "", false);
            if (ret == ACLNN_SUCCESS) {
                OP_LOGI("Parse dynamic kernel config success, file path: %s", realJsonFile.c_str());
                return ACLNN_SUCCESS;
            }
        }
    }
    if (configFileNames.size() > 0) {
        OP_CHECK(ParseDynamicKernelInStaticLib(configFileNames[0]) != ACLNN_SUCCESS,
                 OP_LOGI("Parse dynamic kernel from static lib successfully"), return ACLNN_SUCCESS);
    }
    OP_LOGE_FOR_CONFIG_FILE_NOT_EXIST_IN_DYNAMIC_SHAPE();
    return ACLNN_ERR_INNER;
}

void KernelMgr::GetDirPath()
{
    auto& knlLib = OpKernelLib::GetInstance();
    static bool isDirsInited = false;
    if (!isDirsInited) {
        // append custom opp path
        std::vector<std::string> customImplPath = knlLib.GetCustomImplPath();
        if (!customImplPath.empty()) {
            for (const auto& element : customImplPath) {
                OP_LOGI("add custom opp path: %s", element.c_str());
                customConfigDirs_.emplace_back(element + DYN_KERNEL_CONFIG_PATH + knlLib.GetSocPath());
                customBinAndJsonDirs_.emplace_back(element + DYN_BIN_AND_JSON_PATH);
            }
        }

        // append opp vendors
        std::vector<std::string> configImplPath = knlLib.GetConfigImplPath();
        if (!configImplPath.empty()) {
            for (const auto& element : configImplPath) {
                OP_LOGI("add vector config.imi opp path: %s", element.c_str());
                customConfigDirs_.emplace_back(element + DYN_KERNEL_CONFIG_PATH + knlLib.GetSocPath());
                customBinAndJsonDirs_.emplace_back(element + DYN_BIN_AND_JSON_PATH);
            }
        }

        // append opp path
        builtInConfigDir_ = knlLib.GetAiCoreImplPath() + DYN_KERNEL_CONFIG_PATH + knlLib.GetSocPath();
        builtInBinAndJsonDir_ = knlLib.GetAiCoreImplPath() + DYN_BIN_AND_JSON_PATH;
        isDirsInited = true;
    }

    string oppRealPath;
    auto ret = GetOppKernelPath(oppRealPath);
    OP_CHECK(ret == ACLNN_SUCCESS && !oppRealPath.empty(),
             OP_LOGW("opp kernel real path can not be found. ret %d", ret), return);
    if (debugConfigDir_.empty()) {
        debugConfigDir_ = oppRealPath;
        debugConfigDir_ += DYN_DEBUG_KERNEL_CONFIG_PATH;
        debugConfigDir_ += knlLib.GetSocPath();
    }

    if (debugDynBinAndJsonDir_.empty()) {
        debugDynBinAndJsonDir_ = oppRealPath;
        debugDynBinAndJsonDir_ += DYN_DEBUG_BIN_AND_JSON_PATH;
    }
}

static aclnnStatus PrintConfigJsonOpsFolders(std::vector<std::string>& folders)
{
    std::string foldersName = "";
    for (auto& folder : folders) {
        foldersName += "[" + folder + "],";
    }
    OP_LOGI("Config json ops folders: %s", foldersName.c_str());
    return ACLNN_SUCCESS;
}

void KernelMgr::GetConfigJsonOpsFolders()
{
    static std::once_flag foldersInitFlag;
    std::call_once(foldersInitFlag, [&]() {
        configJsonOpsFolders_.emplace_back("");
        DIR* curDir = opendir(builtInConfigDir_.c_str());
        if (curDir == nullptr) {
            OP_LOGW("The built-in config dir [%s] is invalid.", builtInConfigDir_.c_str());
            return;
        }
        struct dirent* entry;
        bool hasLegacyFold = false;
        const std::string legacyFolderName = "ops_legacy";
        while ((entry = readdir(curDir)) != nullptr) {
            std::string curEntryName = std::string(entry->d_name);
            OP_LOGD("Entry d_name: %s, d_type: %d", entry->d_name, entry->d_type);
            if (curEntryName == "." || curEntryName == "..") {
                continue;
            }
            if (curEntryName == legacyFolderName) {
                hasLegacyFold = true;
            } else {
                configJsonOpsFolders_.emplace_back(curEntryName);
            }
        }
        if (hasLegacyFold) {
            configJsonOpsFolders_.emplace_back(legacyFolderName);
        }
        closedir(curDir);
    });
    OP_LOGI("The built-in config dir: %s, print folders result: %d", builtInConfigDir_.c_str(),
            PrintConfigJsonOpsFolders(configJsonOpsFolders_));
}

aclnnStatus KernelMgr::Prepare()
{
    aclnnStatus ret = ACLNN_ERR_INNER;

    ret = OpKernelLib::GetInstance().Initialize();
    CHECK_COND(ret == ACLNN_SUCCESS, ret, "Initialize OpKernelLib failed.");
    GetDirPath();
    GetConfigJsonOpsFolders();

    bool enableDebug = false;
    ret = op::internal::systemConfig.GetEnableDebugKernelFlag(enableDebug);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGW("GetEnableDebugKernelFlag failed.");
        return ACLNN_SUCCESS;
    }
    if (enableDebug) {
        ret = LoadDebugStaticBinJson();
        if (ret != ACLNN_SUCCESS) {
            OP_LOGW("LoadDebugStaticBinJson failed.");
        }
    }

    return ACLNN_SUCCESS;
}

aclnnStatus KernelMgr::ParseDynamicKernels(uint32_t opType)
{
    std::vector<std::string> configFileNames = GetOpConfigJsonFileName(opType);
    aclnnStatus ret = internal::gKernelMgr.ParseDynamicKernelConfig(configFileNames);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }

    bool enableDebug = false;
    ret = op::internal::systemConfig.GetEnableDebugKernelFlag(enableDebug);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGW("GetEnableDebugKernelFlag failed.");
        return ACLNN_SUCCESS;
    }
    if (enableDebug) {
        OP_LOGI("Start to parse debug kernel, debugStaticBinAndJsonDir is %s, debugDynBinAndJsonDir is %s.",
                debugStaticBinAndJsonDir_.c_str(), debugDynBinAndJsonDir_.c_str());

        for (size_t i = 0; i < configFileNames.size(); i++) {
            std::string jsonPath = debugConfigDir_ + configFileNames[i];
            ifstream f(jsonPath);
            if (!f.good()) {
                OP_LOGI("Can not find json file: %s, continue next priority", jsonPath.c_str());
                continue;
            }
            try {
                auto j = nlohmann::json::parse(f);
            } catch (nlohmann::json::exception& e) {
                OP_LOGW("Cannot parse debug json for config file [%s].", jsonPath.c_str());
                continue;
            }
            ret = internal::gKernelMgr.ParseDynamicKernelConfig(debugConfigDir_, debugDynBinAndJsonDir_,
                                                                configFileNames[i], "", true);
            if (ret == ACLNN_SUCCESS) {
                return ACLNN_SUCCESS;
            }
        }
        OP_LOGW("ParseDynamicKernelConfig for debug kernel failed.");
    }

    return ACLNN_SUCCESS;
}

aclnnStatus KernelMgr::ParseStaticKernels(uint32_t opType)
{
    return internal::gKernelMgr.ParseStaticKernelConfig(opType);
}

aclnnStatus KernelMgr::AclOpKernelInit(uint32_t opType)
{
    // init_ change name to initDynKernelFlags_. 2025.7.8
    if (opType >= MAX_OP_TYPE_COUNT) {
        OP_LOGD("Op type %u %s is larger than %zu has been initialized.", opType,
                OpTypeDict::ToString(opType).GetString(), MAX_OP_TYPE_COUNT);
        return ACLNN_SUCCESS;
    }

    aclnnStatus rc = ACLNN_SUCCESS;
    thread_local std::once_flag platFlag;
    std::call_once(platFlag, [&]() { rc = SocContext::SetupPlatformInfo(); });
    CHECK_COND(rc == ACLNN_SUCCESS, rc, "SetupPlatformInfo failed.");

    static std::once_flag flag;
    std::call_once(flag, [&]() { rc = Prepare(); });
    if (rc != ACLNN_SUCCESS) {
        OP_LOGE(rc, "Failed to Prepare.");
        return rc;
    }

    loadStaticBinJsonFlag_.CallOnce([&]() { rc = LoadStaticBinJson(); });
    CHECK_COND(rc == ACLNN_SUCCESS, rc, "LoadStaticBinJson failed.");

    static std::once_flag opFlag[MAX_OP_TYPE_COUNT];
    std::call_once(opFlag[opType], [&]() { rc = OpKernelLib::GetInstance().ParseKernelLibInfos(opType); });
    CHECK_COND(rc == ACLNN_SUCCESS, rc, "Failed to ParseKernelLibInfos.");

    std::call_once(initDynKernelFlags_[opType], [&]() { rc = internal::gKernelMgr.ParseDynamicKernels(opType); });
    CHECK_COND(rc == ACLNN_SUCCESS, rc, "Failed to ParseDynamicKernels.");

    initStaticKernelFlags_[opType].CallOnce([&]() { rc = internal::gKernelMgr.ParseStaticKernels(opType); });
    CHECK_COND(rc == ACLNN_SUCCESS, rc, "Failed to ParseStaticKernels.");

    return rc;
}

aclnnStatus OpKernelBin::JsonLoad()
{
    auto f = [this](nlohmann::json& jsonObj) -> aclnnStatus { return JsonLoadImpl(jsonObj); };
    return binJson_.InitVar(f);
}

aclnnStatus OpKernelBin::GetBinJson(nlohmann::json& jsonObj)
{
    (void)jsonObj;
    std::tuple<nlohmann::json, nnopbase::Binary> binInfo;
    auto ret = nnopbase::OpBinaryResourceManager::GetInstance().GetOpBinaryDescByPath(relativeJsonPath_.c_str(),
                                                                                      binInfo);
    if (ret == ACLNN_SUCCESS) {
        OP_LOGI("Get builtin op kernel bin json [%s]", jsonPath_.c_str());
        jsonObj = std::get<0>(binInfo);
        return ACLNN_SUCCESS;
    }

    OP_LOGW("No builtin op kernel bin json [%s]", relativeJsonPath_.c_str());

    const std::string realJsonPath = RealPath(jsonPath_);
    const std::string& openPath = realJsonPath.empty() ? jsonPath_ : realJsonPath;
    std::ifstream f(openPath);
    std::string errMsg = "[Errno " + std::to_string(errno) + "] " + std::generic_category().message(errno);
    OP_CHECK(f.is_open(), OP_LOGE_FOR_FILE_OPERATION_ERROR_OPEN(openPath.c_str(), errMsg.c_str()),
             return ACLNN_ERR_INNER);
    try {
        jsonObj = json::parse(f);
    } catch (nlohmann::json::exception& e) {
        OP_LOGE_FOR_FILE_OPERATION_ERROR_PARSE(jsonPath_.c_str(), e.what());
        return ACLNN_ERR_INNER;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus OpKernelBin::JsonLoadImpl(nlohmann::json& jsonObj)
{
    auto ret = GetBinJson(jsonObj);
    CHECK_COND(ret == ACLNN_SUCCESS, ACLNN_ERR_INNER, "Failed to get op kernel bin json [%s]", jsonPath_.c_str());

    if (jsonObj.contains("intercoreSync") && jsonObj["intercoreSync"] == 1) {
        interCoreSync_ = true;
    }

    if (jsonObj.contains("schedule_mode") && jsonObj["schedule_mode"] == 1) {
        scheduleMode_ = 1;
        OP_LOGD("OpKernelBin::JsonLoadImpl scheduleMode_: %u", scheduleMode_);
    }

    auto workspaceJson = jsonObj.find("workspace");
    if (workspaceJson != jsonObj.end()) {
        hasWorkspace_ = true;
        if (binType_ == BinType::STATIC_BIN) {
            ret = ParseStaticWorkSpace(*workspaceJson);
            if (ret != ACLNN_SUCCESS) {
                return ret;
            }
        }
    }
    if (binType_ == BinType::STATIC_BIN) {
        ret = ParseStaticBlockdim(jsonObj);
        if (ret != ACLNN_SUCCESS) {
            return ret;
        }
        ParseStaticImplMode(jsonObj);
        ParseStaticDevPtrMode(jsonObj);
        ParseStaticDynUBufSize(jsonObj);
    }

    ParseOpDebugConfig(jsonObj);
    ParseOriOpParaSize(jsonObj);

    // get coreType value
    if (jsonObj.contains("coreType")) {
        coreType_ = jsonObj["coreType"].get<std::string>();
    }

    if (jsonObj.contains("kernelList")) {
        isFatbin_ = true;
        for (const auto& elem : jsonObj["kernelList"]) {
            if (elem.contains("coreType") && elem["coreType"] == "MIX" && elem.contains("kernelName")) {
                const auto& kernelName = elem["kernelName"];
                std::vector<std::string> parts;
                StrSplit(kernelName, parts, "_");
                if (parts.size() > 1) {
                    mixKernel.emplace(atol(parts[parts.size() - 1].c_str()));
                }
                hasWorkspace_ = true; // Get workspace in phase 1 if Fatbin contains both MIX and non-MIX sub kernels
            }
        }
    } else {
        isFatbin_ = false;
    }

    SetMemSetFlagFromJson();

    ParseKernelDfxConfig(jsonObj);
    return ACLNN_SUCCESS;
}

} // namespace internal
} // namespace op
