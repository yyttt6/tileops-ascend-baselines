/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <set>
#include "aicpu_json_load_manager.h"
#include <fstream>
#include "ops_json_parse.h"
#include "mmpa/mmpa_api.h"
#include "opdev/op_log.h"
#include "opdev/aicpu/aicpu_utils.h"
#include "opdev/op_errno.h"
#include "file_utils.h"

namespace op {
namespace internal {
namespace {
const std::string kAicpuOpsFileEnvPath = "/built-in/op_impl/aicpu/aicpu_kernel/config/aicpu_kernel.json";
const std::string kTfOpsFileEnvPath = "/built-in/op_impl/aicpu/tf_kernel/config/tf_kernel.json";
const std::string kAicpuCustJsonFilePath = "/op_impl/cpu/config/cust_aicpu_kernel.json";
const std::string kAicpuCustOpsFilePath = "/op_impl/cpu/aicpu_kernel/impl/";
const std::string kAicpuBuiltinJsonDir = "/built-in/op_impl/aicpu/config";
const std::string kAicpuBuiltinOpsFilePath = "/built-in/op_impl/aicpu/kernel/";
constexpr size_t SOC_VERSION_LEN = 128U;
constexpr int kMaxFileSizeLimit = INT_MAX;
const std::string kSplitSeparator = ":";
const std::string kCustOpsBlacklistVendorName = "custom_aicpu_ops";
const std::string kCustOpsBlacklistSoName = "libcust_cpu_kernels.so";

const std::string& GetKernelSoDir(const JsonLoadManger::OpPackageType packageType)
{
    static const std::string kCustomKernelSoDir = kAicpuCustOpsFilePath;
    static const std::string kBuiltinKernelSoDir = kAicpuBuiltinOpsFilePath;
    return (packageType == JsonLoadManger::OpPackageType::BUILTIN) ? kBuiltinKernelSoDir : kCustomKernelSoDir;
}
} // namespace

bool JsonLoadManger::hasAicpuLoadBin_ = false;
bool JsonLoadManger::hasTfLoadBin_ = false;
aclrtBinHandle JsonLoadManger::aicpuBinHandle_ = nullptr;
aclrtBinHandle JsonLoadManger::tfBinHandle_ = nullptr;
std::mutex JsonLoadManger::aicpuBinLoadMutex_ = std::mutex();
std::mutex JsonLoadManger::tfBinLoadMutex_ = std::mutex();
bool JsonLoadManger::isSupportNewLaunch_ = true;
std::string JsonLoadManger::socVersion_ = "";
std::mutex JsonLoadManger::getSocVersionMutex_ = std::mutex();
std::mutex JsonLoadManger::custMutex_ = std::mutex();
bool JsonLoadManger::aicpuCustLoadFlag_ = false;
std::mutex JsonLoadManger::aicpuCustBinLoadMutex_ = std::mutex();
std::mutex JsonLoadManger::bufferCacheMutex_ = std::mutex();
std::vector<JsonLoadManger::OpJsonFileInfo> JsonLoadManger::custOpJsonInfo_ = {};
std::map<std::string, OpFullInfo> JsonLoadManger::customOpsInfos_ = {};
std::map<std::string, JsonLoadManger::OpRegisterInfo> JsonLoadManger::custRegisterInfos_ = {};
std::map<std::string, JsonLoadManger::CustomBinManager> JsonLoadManger::customBinhandleInfos_ = {};
std::map<std::string, std::shared_ptr<std::vector<char>>> JsonLoadManger::bufferCache_ = {};

JsonLoadManger::~JsonLoadManger()
{
    hasAicpuLoadBin_ = false;
    aicpuBinHandle_ = nullptr;
    hasTfLoadBin_ = false;
    tfBinHandle_ = nullptr;
    isSupportNewLaunch_ = true;
    socVersion_ = "";
    aicpuCustLoadFlag_ = false;
    customOpsInfos_.clear();
    custOpJsonInfo_.clear();
    custRegisterInfos_.clear();
    customBinhandleInfos_.clear();
    bufferCache_.clear();
    OP_LOGI("JsonLoadManager destroyed.");
}

aclnnStatus JsonLoadManger::LoadBinaryFromJson(const std::string& opsPath, aclrtBinHandle& binHandle, const bool isCust)
{
    std::string filePath = "";
    if (!isCust) {
        // 1. load binary
        const char* pathEnv = nullptr;
        MM_SYS_GET_ENV(MM_ENV_ASCEND_OPP_PATH, pathEnv);
        AICPU_ASSERT_NOTNULL_RETVAL(pathEnv);
        const std::string oppEnvPath = std::string(pathEnv);
        filePath = oppEnvPath + opsPath;
    } else {
        filePath = opsPath;
    }
    OP_LOGI("Ops json or so path [%s] loaded successfully.", filePath.c_str());
    auto loadBinOption = std::make_unique<aclrtBinaryLoadOption>();
    AICPU_ASSERT_NOTNULL_RETVAL(loadBinOption);
    loadBinOption->type = ACL_RT_BINARY_LOAD_OPT_CPU_KERNEL_MODE;
    loadBinOption->value.cpuKernelMode = isCust ? 2 : 0; // 0: only load json, 1: load json and so, 2: load data
    aclrtBinaryLoadOptions optionalCfg = {loadBinOption.get(), 1U};
    if (!isCust) {
        AICPU_ASSERT_RTOK_RETVAL(aclrtBinaryLoadFromFile(filePath.c_str(), &optionalCfg, &binHandle));
    } else {
        auto buffer = GetOrCreateBuffer(filePath);
        AICPU_ASSERT_NOTNULL_RETVAL(buffer);
        AICPU_ASSERT_RTOK_RETVAL(aclrtBinaryLoadFromData(buffer->data(), buffer->size(), &optionalCfg, &binHandle));
    }
    AICPU_ASSERT_NOTNULL_RETVAL(binHandle);
    return ACLNN_SUCCESS;
}

aclnnStatus JsonLoadManger::LoadAicpuBinaryFromJson()
{
    std::unique_lock<std::mutex> lk(aicpuBinLoadMutex_);
    if (hasAicpuLoadBin_) {
        OP_LOGI("Bin loaded from aicpu json successfully, no need to reload.");
        return ACLNN_SUCCESS;
    }

    AICPU_ASSERT_OK_RETVAL(LoadBinaryFromJson(kAicpuOpsFileEnvPath, aicpuBinHandle_));
    hasAicpuLoadBin_ = true;
    OP_LOGI("Aicpu bin loaded from json successfully.");
    return ACLNN_SUCCESS;
}

aclnnStatus JsonLoadManger::LoadTfBinaryFromJson()
{
    std::unique_lock<std::mutex> lk(tfBinLoadMutex_);
    if (hasTfLoadBin_) {
        OP_LOGI("Bin loaded from tf json successfully, no need to reload.");
        return ACLNN_SUCCESS;
    }

    AICPU_ASSERT_OK_RETVAL(LoadBinaryFromJson(kTfOpsFileEnvPath, tfBinHandle_));
    hasTfLoadBin_ = true;
    OP_LOGI("Tf bin loaded from json successfully.");
    return ACLNN_SUCCESS;
}

aclnnStatus JsonLoadManger::SetSupportedNewLaunchFlag()
{
    std::unique_lock<std::mutex> lk(getSocVersionMutex_);
    if (socVersion_ != "") {
        OP_LOGI("Get soc version %s successfully, no need to reload.", socVersion_.c_str());
        return ACLNN_SUCCESS;
    }

    const char* const socVersion = aclrtGetSocName();
    if (socVersion == nullptr) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "Get SoC version failed.");
        return ACLNN_ERR_RUNTIME_ERROR;
    }
    if (strncmp(socVersion, "Ascend910_96", (sizeof("Ascend910_96") - 1UL)) == 0) {
        isSupportNewLaunch_ = false;
    }
    socVersion_ = std::string(socVersion);
    OP_LOGI("Get soc version %s successfully.", socVersion_.c_str());
    return ACLNN_SUCCESS;
}

aclnnStatus JsonLoadManger::LoadAicpuCustBinaryFromJson(const std::string& opType, const std::string& kernelSoName,
                                                        std::string& kernelSoPath)
{
    std::unique_lock<std::mutex> lk(aicpuCustBinLoadMutex_);
    auto iter = custRegisterInfos_.find(opType);
    if (iter == custRegisterInfos_.end()) {
        OP_LOGE(ACLNN_ERR_INNER, "The operator %s not found in package registry.", opType.c_str());
        return ACLNN_ERR_INNER;
    }

    const auto& registerInfo = iter->second;
    kernelSoPath = registerInfo.opsRootPath + GetKernelSoDir(registerInfo.packageType) + kernelSoName;
    if (customBinhandleInfos_[kernelSoPath].hasLoad) {
        OP_LOGI("The package kernel so %s has loaded, no need to reload.", kernelSoPath.c_str());
        return ACLNN_SUCCESS;
    }
    AICPU_ASSERT_OK_RETVAL(LoadBinaryFromJson(kernelSoPath, customBinhandleInfos_[kernelSoPath].binHandle, true));
    customBinhandleInfos_[kernelSoPath].hasLoad = true;
    OP_LOGI("The package kernel so %s load successfully.", kernelSoPath.c_str());
    return ACLNN_SUCCESS;
}

bool JsonLoadManger::ReadCustJsonFile(const std::string& opsRegisterName, const std::string& customJsonPath,
                                      const OpPackageType packageType)
{
    std::ifstream ifs(customJsonPath);
    if (!ifs.is_open()) {
        OP_LOGW("Open operator impl %s failed, do next operator repository.", customJsonPath.c_str());
        return false;
    }
    nlohmann::json custOpInfoFile;
    if (OpsJsonFile::Instance().ParseUnderPath(customJsonPath, custOpInfoFile) != ACLNN_SUCCESS) {
        OP_LOGW("Parse operator json file[%s] failed.", customJsonPath.c_str());
        return false;
    }
    OP_LOGI("Operator repository name is %s, operator info file = %s", opsRegisterName.c_str(),
            custOpInfoFile.dump().c_str());
    custOpJsonInfo_.emplace_back(OpJsonFileInfo{opsRegisterName, custOpInfoFile, packageType});
    return true;
}

static void SplitLine(const std::string& str, const std::string& pattern, std::vector<std::string>& result)
{
    // Easy to intercept the last piece of data
    std::string strs = str + pattern;

    size_t pos = strs.find(pattern);
    size_t size = strs.size();

    while (pos != std::string::npos) {
        std::string x = strs.substr(0, pos);
        if (!x.empty()) {
            result.push_back(x);
        }
        strs = strs.substr(pos + pattern.length(), size);
        pos = strs.find(pattern);
    }
}

bool JsonLoadManger::ReadCustOpInfoFromJsonFile(const std::string& path)
{
    std::vector<std::string> customOppPath;
    SplitLine(path, kSplitSeparator, customOppPath);
    size_t customPathSize = customOppPath.size();
    OP_LOGI("Get custom opp path size = %zu.", customPathSize);
    if (customPathSize < 1) {
        return false;
    }

    std::string customJsonPath = "";
    for (size_t i = 0; i < customPathSize; i++) {
        customJsonPath = customOppPath[i] + kAicpuCustJsonFilePath;
        OP_LOGI("Custom operator repository json path is %s.", customJsonPath.c_str());
        if (!ReadCustJsonFile(customOppPath[i], customJsonPath, OpPackageType::CUSTOM)) {
            continue;
        }
    }
    OP_LOGI("Custom operator repository file size is %zu.", custOpJsonInfo_.size());
    return true;
}

bool JsonLoadManger::ReadBuiltinOpInfoFromJsonFile(const std::string& oppPath)
{
    const std::string configDir = oppPath + kAicpuBuiltinJsonDir;
    std::vector<std::string> jsonFilePaths;
    GetFilesWithSuffix(configDir, ".json", jsonFilePaths);
    if (jsonFilePaths.empty()) {
        OP_LOGI("Builtin operator config dir %s is empty.", configDir.c_str());
        return false;
    }
    std::sort(jsonFilePaths.begin(), jsonFilePaths.end());

    for (const auto& jsonPath : jsonFilePaths) {
        OP_LOGI("Builtin operator repository json path is %s.", jsonPath.c_str());
        if (!ReadCustJsonFile(oppPath, jsonPath, OpPackageType::BUILTIN)) {
            continue;
        }
    }
    OP_LOGI("Builtin operator repository file size is %zu.", custOpJsonInfo_.size());
    return true;
}

bool JsonLoadManger::GetCustomOppPathFromVendors(std::string& customOppPath)
{
    const char* oppPathEnv = nullptr;
    MM_SYS_GET_ENV(MM_ENV_ASCEND_OPP_PATH, oppPathEnv);
    if ((oppPathEnv == nullptr) || (std::string(oppPathEnv) == "")) {
        OP_LOGI("ASCEND_OPP_PATH is not set or empty, cannot derive custom opp path.");
        return false;
    }
    const std::string oppPath = std::string(oppPathEnv);
    const std::string configPath = oppPath + "/vendors/config.ini";
    const std::string realConfigPath = RealPath(configPath);
    if (!realConfigPath.empty()) {
        OP_LOGI("config.ini real path [%s].", realConfigPath.c_str());
        std::ifstream ifs(realConfigPath);
        if (!ifs.is_open()) {
            OP_LOGW("Failed to open config.ini at [%s].", realConfigPath.c_str());
        } else {
            std::string line = "";
            std::vector<std::string> vendorNames;
            while (std::getline(ifs, line)) {
                if (!line.empty() && line.back() == '\r') {
                    line.pop_back();
                }
                if (line.empty() || (line.find('#') == 0U)) {
                    continue;
                }
                if (line.find("priority") != std::string::npos) {
                    const size_t posOfEqual = line.find('=');
                    if (posOfEqual != std::string::npos) {
                        const std::string value = line.substr(posOfEqual + 1U);
                        SplitLine(value, ",", vendorNames);
                    }
                    break;
                }
            }
            for (const auto& vendor : vendorNames) {
                std::string trimmed = vendor;
                size_t start = trimmed.find_first_not_of(" \t");
                size_t end = trimmed.find_last_not_of(" \t");
                if (start == std::string::npos) {
                    continue;
                }
                trimmed = trimmed.substr(start, end - start + 1U);
                const std::string vendorPath = oppPath + "/vendors/" + trimmed;
                const std::string custJsonPath = vendorPath + kAicpuCustJsonFilePath;
                OP_LOGI("Checking vendor %s cust json path: %s", trimmed.c_str(), custJsonPath.c_str());
                if (RealPath(custJsonPath).empty()) {
                    OP_LOGI("Vendor %s has no cust_aicpu_kernel.json, skip.", trimmed.c_str());
                    continue;
                }
                if (!customOppPath.empty()) {
                    customOppPath += kSplitSeparator;
                }
                customOppPath += vendorPath;
            }
        }
    }
    if (!customOppPath.empty()) {
        OP_LOGI("Derived custom opp path from ASCEND_OPP_PATH: %s", customOppPath.c_str());
    }
    return !customOppPath.empty();
}

// Read custom operator json file and store it
aclnnStatus JsonLoadManger::CustJsonLoadAndParse()
{
    std::unique_lock<std::mutex> lk(custMutex_);
    if (aicpuCustLoadFlag_) {
        OP_LOGI("The operator repository has already been loaded.");
        return ACLNN_SUCCESS;
    }
    const char* customPathEnv = nullptr;
    MM_SYS_GET_ENV(MM_ENV_ASCEND_CUSTOM_OPP_PATH, customPathEnv);
    if ((customPathEnv != nullptr) && (std::string(customPathEnv) != "")) {
        std::string pathEnv = std::string(customPathEnv);
        OP_LOGI("Use ASCEND_CUSTOM_OPP_PATH for custom operator loading.");
        if (!ReadCustOpInfoFromJsonFile(pathEnv)) {
            OP_LOGW("Failed to read custom operator info from json file.");
        }
    } else {
        std::string pathEnv = "";
        if (GetCustomOppPathFromVendors(pathEnv)) {
            OP_LOGI("Fallback to vendors path derived from ASCEND_OPP_PATH.");
            if (!ReadCustOpInfoFromJsonFile(pathEnv)) {
                OP_LOGW("Failed to read custom operator info from json file.");
            }
        } else {
            OP_LOGI("ASCEND_CUSTOM_OPP_PATH not set and no valid custom opp path derived from ASCEND_OPP_PATH.");
        }
    }
    const char* oppPathEnv = nullptr;
    MM_SYS_GET_ENV(MM_ENV_ASCEND_OPP_PATH, oppPathEnv);
    if (oppPathEnv == nullptr) {
        OP_LOGI("Builtin operator environment variable ASCEND_OPP_PATH is not set.");
    } else {
        std::string oppPath = std::string(oppPathEnv);
        if (!ReadBuiltinOpInfoFromJsonFile(oppPath)) {
            OP_LOGW("Failed to read builtin operator info from json file.");
        }
    }
    (void)ParseCustOpInfo();
    aicpuCustLoadFlag_ = true;
    return ACLNN_SUCCESS;
}

aclnnStatus JsonLoadManger::ParseCustOpInfo()
{
    for (auto iter = custOpJsonInfo_.cbegin(); iter != custOpJsonInfo_.cend(); ++iter) {
        if (iter->opJson.find(kConfigOpInfos) == iter->opJson.end()) {
            OP_LOGW("The operator json file does not contain 'op_infos'.");
            continue;
        }
        try {
            OpInfoDescs infoDesc = iter->opJson;
            FillCustOpInfos(iter->opsRegisterName, infoDesc, iter->packageType);
        } catch (const nlohmann::json::exception& e) {
            OP_LOGW("Failed to parse operator json file %s : %s.", iter->opJson.dump().c_str(), e.what());
            continue;
        }
    }
    return ACLNN_SUCCESS;
}

void JsonLoadManger::FillCustOpInfos(std::string opsRegisterName, const OpInfoDescs& infoDesc,
                                     const OpPackageType packageType)
{
    if (!opsRegisterName.empty() && opsRegisterName.back() == '/') {
        opsRegisterName.pop_back();
        OP_LOGI("Ops register name: %s", opsRegisterName.c_str());
    }

    const size_t lastSlash = opsRegisterName.find_last_of('/');
    const std::string dirName = (lastSlash != std::string::npos) ? opsRegisterName.substr(lastSlash + 1) :
                                                                   opsRegisterName;

    for (const auto& opDesc : infoDesc.opInfos) {
        if (opDesc.opType.empty()) {
            continue;
        }

        if ((packageType == OpPackageType::CUSTOM) && (dirName == kCustOpsBlacklistVendorName) &&
            (opDesc.opInfo.kernelSo == kCustOpsBlacklistSoName)) {
            OP_LOGI("vendor name[%s] and so name[%s] are in blacklist, skip to insert customer ops info. "
                    "ops register name is %s, op type is %s.",
                    dirName.c_str(), opDesc.opInfo.kernelSo.c_str(), opsRegisterName.c_str(), opDesc.opType.c_str());
            continue;
        }

        if (packageType == OpPackageType::BUILTIN) {
            if (opDesc.opInfo.kernelSo.empty()) {
                OP_LOGW("Builtin operator kernel so name is empty, skip op type %s.", opDesc.opType.c_str());
                continue;
            }
            const std::string kernelSoPath = opsRegisterName + kAicpuBuiltinOpsFilePath + opDesc.opInfo.kernelSo;
            if (RealPath(kernelSoPath).empty()) {
                OP_LOGW("Builtin operator so path %s is invalid, skip op type %s.", kernelSoPath.c_str(),
                        opDesc.opType.c_str());
                continue;
            }
        }

        if (customOpsInfos_.find(opDesc.opType) != customOpsInfos_.end()) {
            OP_LOGW("[%s] of operator [%s] is duplicated; discarding in favor of existing entry.",
                    opDesc.opType.c_str(), opsRegisterName.c_str());
        } else {
            auto ret = customOpsInfos_.emplace(std::pair<std::string, OpFullInfo>(opDesc.opType, opDesc.opInfo));
            if (!ret.second) {
                OP_LOGW("Failed to insert operator [%s] and its information.", opDesc.opType.c_str());
            }
            custRegisterInfos_.emplace(
                std::pair<std::string, OpRegisterInfo>(opDesc.opType, OpRegisterInfo{opsRegisterName, packageType}));
            OP_LOGI(
                "Reading operator json file: operator type is %s, operator register name is %s, package type is %u.",
                opDesc.opType.c_str(), opsRegisterName.c_str(), static_cast<uint32_t>(packageType));
        }
    }
    OP_LOGI("The number of elements in the operator registry container is %zu.", custRegisterInfos_.size());
    return;
}

bool JsonLoadManger::FindAndGetInCustomRegistry(const std::string& opType, std::string& kernelSo,
                                                std::string& functionName)
{
    auto iter = customOpsInfos_.find(opType);
    if (iter == customOpsInfos_.end()) {
        OP_LOGI("The operator %s not found in operator registry.", opType.c_str());
        return false;
    }
    kernelSo = iter->second.kernelSo;
    functionName = iter->second.functionName;
    OP_LOGI("Found operator %s from the operator information library %s with function name %s.", opType.c_str(),
            kernelSo.c_str(), functionName.c_str());
    return true;
}

bool JsonLoadManger::ReadBytesFromBinaryFile(const std::string& fileName, std::vector<char>& buffer)
{
    if (fileName.empty()) {
        OP_LOGE(false, "The file %s name is empty.", fileName.c_str());
        return false;
    }

    std::string realPath = RealPath(fileName);
    if (realPath.empty()) {
        OP_LOGE(false, "Invalid path %s.", fileName.c_str());
        return false;
    }

    std::ifstream file(realPath.c_str(), std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        OP_LOGE(false, "Open file %s failed.", fileName.c_str());
        return false;
    }

    std::streamsize size = file.tellg();
    if (size <= 0) {
        file.close();
        OP_LOGE(false, "Empty file %s.", fileName.c_str());
        return false;
    }
    if (size > kMaxFileSizeLimit) {
        file.close();
        OP_LOGE(false, "File %s size %ld is out of limit %d.", fileName.c_str(), size, kMaxFileSizeLimit);
        return false;
    }

    file.seekg(0, std::ios::beg);

    buffer.resize(size);
    file.read(&buffer[0], size);
    file.close();
    OP_LOGI("Binary file size is %ld", size);
    return true;
}

std::shared_ptr<std::vector<char>> JsonLoadManger::GetOrCreateBuffer(const std::string& filePath)
{
    std::unique_lock<std::mutex> lk(bufferCacheMutex_);
    auto it = bufferCache_.find(filePath);
    if (it != bufferCache_.end()) {
        OP_LOGI("Using cached buffer for: %s", filePath.c_str());
        return it->second;
    }

    // Create a new buffer
    auto buffer = std::make_shared<std::vector<char>>();
    if (!ReadBytesFromBinaryFile(filePath, *buffer)) {
        OP_LOGW("Failed to read file: %s", filePath.c_str());
        return nullptr;
    }

    bufferCache_[filePath] = buffer;
    OP_LOGI("Cached buffer for: %s, size: %zu", filePath.c_str(), buffer->size());
    return buffer;
}
} // namespace internal
} // namespace op
