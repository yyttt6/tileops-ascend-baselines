/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "indv_collector.h"

#include <map>
#include <string>
#include <algorithm>
#include <fstream>
#include <mutex>
#include <dirent.h>
#include "securec.h"
#include "mmpa/mmpa_api.h"
#include "mmpa/sub_inc/mmpa_linux.h"
#include "utils/indv_base.h"
#include "utils/indv_debug_assert.h"
#include "utils/indv_path.h"
#include "utils/thread_var_container.h"
#include "indv_bininfo.h"
#include "utils/indv_soc.h"
#include "indv_executor.h"
#include "opdev/data_type_utils.h"
#include "op_dfx_util.h"
#include "register/op_binary_resource_manager.h"

using namespace std;
using namespace nnopbase;

namespace {
constexpr char const* OP_TILING_SO_SUFFIX = ".so";
constexpr char* ERR_REASON_FOR_OPP_PACKAGE = "1.The operator package is not installed. "
                                             "2.The operator package is damaged. "
                                             "3.The binary_info_config.json file is damaged or does not exist. "
                                             "4.The user does not have sufficient permissions and the "
                                             "binary_info_config.json file in the operator package fails to be read";
constexpr size_t MAX_BIN_KEY_MULTIPLIER = 4UL;
static const std::vector<std::string> OPS_PATH_VEC = {
    "ops_math",  "ops_nn", "ops_cv", "ops_transformer", "ops_oam",
    "ops_legacy" // 低优先级
};

void NnopbaseTrim(string& result, const NnopbaseChar delims)
{
    string::size_type index = result.find_last_not_of(delims);
    if (index != string::npos) {
        (void)result.erase(++index);
    }

    index = result.find_first_not_of(delims);
    if (index != string::npos) {
        (void)result.erase(0, index);
    }
    return;
}

void GetCustomVendorName(std::vector<std::string>& customPaths)
{
    OP_LOGI("Start to get custom path from ASCEND_CUSTOM_OPP_PATH.");
    const NnopbaseChar* custOppPathEnv = nullptr;
    MM_SYS_GET_ENV(MM_ENV_ASCEND_CUSTOM_OPP_PATH, custOppPathEnv);
    if (custOppPathEnv == nullptr) {
        OP_LOGI("Environment variable ASCEND_CUSTOM_OPP_PATH is not defined.");
        return;
    }
    const std::string customOppPath = custOppPathEnv;
    if (customOppPath.empty()) {
        OP_LOGW("Environment variable ASCEND_CUSTOM_OPP_PATH is defined, but it's empty.");
        return;
    }
    OP_LOGI("Environment variable[ASCEND_CUSTOM_OPP_PATH] is %s.", customOppPath.c_str());
    NnopbaseSplitStr(customOppPath, ":", customPaths);
}

void NnopbaseGetBasePath(NnopbaseBinCollector* const collector,
                         std::vector<std::pair<std::string, gert::OppImplVersionTag>>& basePath,
                         int32_t& builtInStartIndex)
{
    // 处理自研 高优先级在前 低优先级在后
    OP_LOGI("Start to get basePath.");
    NnopbaseGetCustomOppPath(basePath);
    NnopbaseGetOppPath(collector, basePath, builtInStartIndex);
    OP_LOGI("Get basePath finished, basePath size is %zu.", basePath.size());
}

std::string GetBuiltInBasePath(gert::OppImplVersionTag& oppImplVersion)
{
    const NnopbaseChar* homePath = nullptr;
    MM_SYS_GET_ENV(MM_ENV_ASCEND_HOME_PATH, homePath);
    if (homePath != nullptr) {
        const std::string homePathStr = std::string(homePath);
        const std::string kernelPath = homePathStr + "/opp_latest";
        std::vector<NnopbaseChar> path(NNOPBASE_FILE_PATH_MAX_LEN, '\0');
        if (mmRealPath(kernelPath.c_str(), &(path[0U]), NNOPBASE_FILE_PATH_MAX_LEN) == EN_OK) {
            oppImplVersion = gert::OppImplVersionTag::kOppKernel;
            return kernelPath;
        }
    }

    const NnopbaseChar* oppPathEnv = nullptr;
    MM_SYS_GET_ENV(MM_ENV_ASCEND_OPP_PATH, oppPathEnv);
    if (oppPathEnv != nullptr) {
        oppImplVersion = gert::OppImplVersionTag::kOpp;
        return std::string(oppPathEnv);
    }
    return "";
}

void GetFilesWithSuffix(const std::string& path, const std::string& suffix, std::vector<std::string>& files)
{
    struct dirent** entries = nullptr;
    const auto fileNum = scandir(path.c_str(), &entries, nullptr, nullptr);
    if (entries == nullptr) {
        return;
    }
    if (fileNum <= 0) {
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
        const string fullName = path + "/" + name;
        files.push_back(fullName);
    }
    for (int i = 0; i < fileNum; i++) {
        free(entries[i]);
    }
    free(entries);
}

std::string GetOpSoPackageName(const std::string& path)
{
    // 新的自定义算子包路径：<opp-path>/vendors/<name>/op_proto/
    const std::string vendors_str = "vendors/";
    auto pos = path.find(vendors_str);
    if (pos != std::string::npos) {
        // vendors/ 后面一级目录
        pos += vendors_str.size();
        auto end_pos = path.find('/', pos);
        if (end_pos == std::string::npos) {
            end_pos = path.size();
        }
        return path.substr(pos, end_pos - pos);
    }

    // 老的自定义算子包路径：<opp-path>/op_proto/custom/
    pos = path.find("custom/");
    if (pos != std::string::npos) {
        return "custom";
    }

    // 内置算子包名
    return "built-in";
}

bool MatchBinWithPlatformInfo(const NnopbaseBinInfo* const binInfo, const StaticKernelPlatformInfo* const platformInfo)
{
    // 不具备校验的条件或未开启校验，直接返回匹配成功
    if (platformInfo == nullptr || binInfo->extraKernelDesc == nullptr ||
        binInfo->extraKernelDesc->platformInfo == nullptr) {
        return true;
    }
    const auto& binPlatformInfo = binInfo->extraKernelDesc->platformInfo;
    const auto& binCoreNum = binInfo->extraKernelDesc->platformInfo->coreNum;
    const auto& coreNum = platformInfo->coreNum;
    OP_LOGD("Match static bin with runtime coreNum[%u, %u], while coreNum of candidate static kernel is [%u, %u].",
            coreNum.aicNum, coreNum.aivNum, binCoreNum.aicNum, binCoreNum.aivNum);
    // 控核场景必须保证aic/aiv核数与当前核数相等
    bool coreEqual = (binCoreNum.aivNum == coreNum.aivNum) && ((binCoreNum.aicNum == coreNum.aicNum));
    // 如果静态kernel描述符里包含了deterministicLevel，做强校验
    if (binPlatformInfo->deterministicLevel != -1) {
        OP_LOGD(
            "Match static bin with deterministic level %d, while deterministic level of candidate static kernel is %d.",
            binPlatformInfo->deterministicLevel, platformInfo->deterministicLevel);
        return (binPlatformInfo->deterministicLevel == platformInfo->deterministicLevel) && coreEqual;
    } else {
        return coreEqual;
    }
}
} // namespace

#ifdef __cplusplus
extern "C" {
#endif
static constexpr int32_t NNOPBASE_OP_TYPE_INDEX = 0;
static constexpr int32_t NNOPBASE_COMPILE_ARGS_INDEX = 1;

NnopbaseBinCollector* gBinCollector = nullptr;

// if usingStride = false,
// [OpType, determin, precision, {dtype,format,shape}....attrs] to generate staticKey
// if usingStride = true,
// [OpType, determin, precision, {dtype,format,shape,stride,offset}....attrs] to generate staticKey
NnopbaseUChar* NnopbaseCollectorGenStaticKey(NnopbaseUChar* verKey, const NnopbaseRegInfoKey* const regInfoKey,
                                             const NnopbaseStaticTensorNumInfo* const tensorNumInfo,
                                             const aclTensor* tensors[], const NnopbaseAttrAddr* attrs[],
                                             const int64_t implMode, const int64_t deterMin,
                                             const int64_t* const vDepend, const bool usingStride)
{
    verKey = NnopbaseAppendBinary(verKey, NNOPBASE_MAX_STATICKEY_LEN, &(regInfoKey->opType[0U]),
                                  regInfoKey->opType.size());
    uint64_t determinLevel = (deterMin == 0U) ? 0U : 1U;
    verKey = NnopbaseAppend8Byte(verKey, determinLevel);
    verKey = NnopbaseAppend8Byte(verKey, static_cast<uint64_t>(implMode));
    OP_LOGI("Get deterministic is %llu, high precision is %lld", determinLevel, implMode);

    std::vector<bool> valueDepend(tensorNumInfo->numTensors, false);
    for (int64_t i = 0; i < tensorNumInfo->numValueDepend; i++) {
        valueDepend[vDepend[i]] = true;
    }
    ge::DataType dtype = ge::DT_INT32;
    ge::Format format = ge::FORMAT_ND;
    const NnopbaseUChar* addr = nullptr;
    // input and output
    for (int64_t i = 0; i < tensorNumInfo->numTensors; i++) {
        if (tensors[i] == nullptr) {
            verKey = NnopbaseAppend8Byte(verKey, '_');
            continue;
        }

        dtype = tensors[i]->GetDataType();
        verKey = NnopbaseAppend8Byte(verKey, dtype);
        format = tensors[i]->GetStorageFormat();
        verKey = NnopbaseAppend8Byte(verKey, format);
        OP_LOGI("Get tensor[%ld] datatype is %d, format is %d.", i, dtype, format);
        const gert::Shape& shape = tensors[i]->GetStorageShape();
        const size_t dimNum = shape.GetDimNum();
        for (size_t j = 0U; j < dimNum; j++) {
            verKey = NnopbaseAppend8Byte(verKey, static_cast<uint64_t>(shape.GetDim(j)));
        }
        if (usingStride) {
            const auto stride = tensors[i]->GetTensor()->GetStride();
            const size_t strideDimNum = stride.GetDimNum();
            const int64_t offset = tensors[i]->GetTensor()->GetOffset();
            std::string strideStr = "[";
            for (size_t j = 0U; j < strideDimNum; j++) {
                verKey = NnopbaseAppend8Byte(verKey, static_cast<uint64_t>(stride.GetStride(j)));
                strideStr += std::to_string(stride.GetStride(j));
                if (j != strideDimNum - 1) {
                    strideStr += ", ";
                }
            }
            strideStr += "]";
            OP_LOGI("Tensor[%ld] strideDim is %zu, stride is %s, offset is %lld.", i, strideDimNum, strideStr.c_str(),
                    offset);
            verKey = NnopbaseAppend8Byte(verKey, offset);
        }
        if (valueDepend[i]) {
            const int64_t elementSize = tensors[i]->Size();
            addr = op::internal::PtrCastTo<NnopbaseUChar>(tensors[i]->GetData());
            const size_t typeSize = op::TypeSize(tensors[i]->GetDataType());
            for (int64_t k = 0; k < elementSize; k++) {
                verKey = NnopbaseExecutor8ByteCopy(typeSize, verKey, addr + typeSize * k);
            }
        }
    }
    // attrs
    if (tensorNumInfo->numAttrs > 0) {
        size_t length = 0U;
        for (int64_t j = 0; j < tensorNumInfo->numAttrs; j++) {
            if (attrs[j] == nullptr) {
                verKey = NnopbaseAppend8Byte(verKey, '_');
                continue;
            }
            if (attrs[j]->addr == nullptr || attrs[j]->size == 0U) {
                OP_LOGW("Attr[%zu] addr is nullptr or size is 0, skip concat simplifiedKey.", j);
                continue;
            };
            if (!attrs[j]->isVector) {
                addr = op::internal::PtrCastTo<const NnopbaseUChar>(attrs[j]->addr);
                length = attrs[j]->size;
                verKey = NnopbaseExecutor8ByteCopy(length, verKey, addr);
            } else {
                const size_t elementSize = attrs[j]->elementSize;
                if (attrs[j]->elementSize == 0U) {
                    OP_LOGW("Attr[%zu] elementSize is 0, skip concat simplifiedKey.", j);
                    continue;
                }
                const size_t elementSizeNumber = attrs[j]->size / elementSize;
                for (size_t i = 0U; i < elementSizeNumber; i++) {
                    addr = op::internal::PtrCastTo<const NnopbaseUChar>(attrs[j]->addr) + elementSize * i;
                    verKey = NnopbaseExecutor8ByteCopy(elementSize, verKey, addr);
                }
            }
        }
    }
    return verKey;
}

const NnopbaseChar* NnopbaseCollectorGetStaticKernelBin(const NnopbaseChar* const opType, const uint64_t key,
                                                        const NnopbaseUChar* verbose, const uint32_t verbLen,
                                                        const StaticKernelPlatformInfo* const platformInfo)
{
    NnopbaseRegInfo* regInfo = NnopbaseCollectorFindRegInfoInTbl(gBinCollector, opType, key);
    if (regInfo == nullptr) {
        return nullptr;
    }
    const size_t hashKey = NnopbaseHashBinary(verbose, static_cast<size_t>(verbLen)) % NNOPBASE_NORM_MAX_BIN_BUCKETS;
    const NnopbaseBinInfo* binInfo = NnopbaseCollectorFindBinInfo(regInfo, hashKey, verbose, verbLen, platformInfo);
    if (binInfo == nullptr) {
        return nullptr;
    }
    OP_LOGI("Found %s static kernel, verboseLen: %u, hashKey is: %zu.", opType, verbLen, hashKey);
    return binInfo->binPath.c_str();
}

aclnnStatus NnopbaseCollectorSetTiling(const NnopbaseJsonInfo& jsonInfo, TilingFun* const tiling,
                                       GenSimplifiedKeyFun* const genSimplifiedKey,
                                       gert::OppImplVersionTag oppImplVersion)
{
    auto& registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry(oppImplVersion);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(registry);
    const auto opImpl = registry->GetOpImpl(jsonInfo.opType.c_str());
    if (opImpl != nullptr) {
        // check null at executor, some op no rt2 tiling
        *tiling = reinterpret_cast<TilingFun>(opImpl->tiling);
        *genSimplifiedKey = reinterpret_cast<GenSimplifiedKeyFun>(opImpl->gen_simplifiedkey);
        OP_LOGI("Get opImpl of %s successfully.", jsonInfo.opType.c_str());
    } else {
        OP_LOGW("Failed to get opImpl of %s, opImpl is nullptr.", jsonInfo.opType.c_str());
        *tiling = nullptr;
        *genSimplifiedKey = nullptr;
    }
    return OK;
}

void NnopbaseSplitStr(const std::string& configPath, const string& pattern, std::vector<std::string>& subPaths)
{
    OP_LOGD("Start split str %s.", configPath.c_str());
    std::string strs = configPath + pattern;
    size_t pos = strs.find(pattern);
    std::string subStr;
    while (pos != string::npos) {
        subStr = strs.substr(0, pos);
        if (!subStr.empty()) {
            subPaths.push_back(subStr);
        }
        strs = strs.substr(pos + 1U);
        pos = strs.find(pattern);
    }
}

aclnnStatus NnopbaseGetCurEnvPackageOsAndCpuType(std::string& hostEnvOs, std::string& hostEnvCpu)
{
    const NnopbaseChar* ascendHomePath = nullptr;
    MM_SYS_GET_ENV(MM_ENV_ASCEND_HOME_PATH, ascendHomePath);
    if (ascendHomePath == nullptr) {
        OP_LOGE_FOR_CONFIG_ERROR_INVALID_ENVIRONMENT_VARIABLE("Finding platform configuration file scene.info",
                                                              "ASCEND_HOME_PATH");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    std::string modelPath = ascendHomePath;
    std::string sceneInfoPath = modelPath + "/share/info/opbase/" + SCENE;
    OP_LOGI("Try to extract os and cpu info from %s.", sceneInfoPath.c_str());
    std::ifstream ifs(sceneInfoPath);
    if (!ifs.good()) {
        ifs.close();
        OP_LOGW("Failed to get %s, trying another path.", sceneInfoPath.c_str());
        const NnopbaseChar* oppPathEnv = nullptr;
        MM_SYS_GET_ENV(MM_ENV_ASCEND_OPP_PATH, oppPathEnv);
        OP_CHECK(oppPathEnv != nullptr,
                 OP_LOGE_FOR_CONFIG_ERROR_INVALID_ENVIRONMENT_VARIABLE("Finding platform configuration file scene.info",
                                                                       "ASCEND_OPP_PATH"),
                 return ACLNN_ERR_PARAM_NULLPTR);
        NNOPBASE_ASSERT_NOTNULL_RETVAL(oppPathEnv);
        modelPath = oppPathEnv;
        sceneInfoPath = modelPath + "/" + SCENE;
        ifs.open(sceneInfoPath);
    }
    if (!ifs.good()) {
        std::string errMsg = "[Errno " + std::to_string(errno) + "] " + std::generic_category().message(errno);
        OP_LOGE_FOR_FILE_OPERATION_ERROR_OPEN(sceneInfoPath.c_str(), errMsg.c_str());
        return ACLNN_ERR_INNER;
    }
    std::string line;
    while (std::getline(ifs, line)) {
        NnopbaseTrim(line, '\r');
        NnopbaseTrim(line, '\n');
        std::vector<std::string> value;
        NnopbaseSplitStr(line, "=", value);
        if (value.size() != SCENE_VALUE_COUNT) {
            continue;
        }
        if (value[SCENE_KEY_INDEX].compare(SCENE_OS) == 0) {
            hostEnvOs = value[SCENE_VALUE_INDEX];
            OP_LOGI("Get os:%s.", hostEnvOs.c_str());
        }
        if (value[SCENE_KEY_INDEX].compare(SCENE_ARCH) == 0) {
            hostEnvCpu = value[SCENE_VALUE_INDEX];
            OP_LOGI("Get cpu:%s.", hostEnvCpu.c_str());
        }
    }
    ifs.close();
    return OK;
}

// subPath 存储 从高优先级到低优先级
bool NnopbaseReadConfigFile(const std::string& configPath, std::vector<std::string>& subPath)
{
    OP_LOGI("The real path of config.ini is %s.", configPath.c_str());
    std::ifstream ifs(configPath);
    if (!ifs.good()) {
        OP_LOGW("Cannot open file: %s.", configPath.c_str());
        return false;
    }
    std::string line;
    while (std::getline(ifs, line)) {
        NnopbaseTrim(line, '\r');
        NnopbaseTrim(line, '\n');
        if (line.empty() || (line.find('#') == 0)) {
            continue;
        }
        if (line.find("priority") != std::string::npos) {
            const size_t posOfEqual = line.find('=');
            if (posOfEqual == std::string::npos) {
                ifs.close();
                OP_LOGE(ACLNN_ERR_INNER, "The format of file %s is invalid.", configPath.c_str());
                return false;
            }
            const std::string value = line.substr(posOfEqual + 1U);
            NnopbaseSplitStr(value, ",", subPath);
            OP_LOGI("Get number %zu custom path.", subPath.size());
            break;
        }
    }
    ifs.close();
    OP_LOGD("End to load opp custom config.");
    return true;
}

void NnopbaseGetOppPath(NnopbaseBinCollector* const collector,
                        std::vector<std::pair<std::string, gert::OppImplVersionTag>>& basePath,
                        int32_t& builtInStartIndex)
{
    OP_LOGI("Start to get opp kernel base path, default custom opp kernel is in ASCEND_OPP_PATH.");
    const NnopbaseChar* oppPathEnv = nullptr;
    MM_SYS_GET_ENV(MM_ENV_ASCEND_OPP_PATH, oppPathEnv);
    std::string oppPath;
    if (oppPathEnv != nullptr) {
        oppPath = oppPathEnv;
        const std::string configPath = oppPath + "/vendors/config.ini";
        std::vector<std::string> subPaths;

        if (NnopbaseReadConfigFile(configPath, subPaths)) {
            for (auto subPath : subPaths) {
                const std::string path = oppPath + "/vendors/" + subPath;
                basePath.push_back(std::make_pair(path, gert::OppImplVersionTag::kOpp));
                OP_LOGI("Add opp kernel subPath %s.", path.c_str());
            }
        }
    }

    // 走旧的opp路径
    gert::OppImplVersionTag oppImplVersion = gert::OppImplVersionTag::kVersionEnd;
    const std::string oppKernelBase = GetBuiltInBasePath(oppImplVersion);
    collector->oppPath = oppKernelBase;
    // 追加built_in
    if (!oppKernelBase.empty()) {
        basePath.push_back(std::make_pair(oppKernelBase + "/built-in", oppImplVersion));
        builtInStartIndex = static_cast<int32_t>(basePath.size() - 1U);
        OP_LOGI("Add opp kernel built-in base path %s.", oppKernelBase.c_str());
    }
}

void NnopbaseGetCustomOppPath(std::vector<std::pair<std::string, gert::OppImplVersionTag>>& basePath)
{
    std::vector<std::string> customPaths;
    GetCustomVendorName(customPaths);
    for (const auto& customPath : customPaths) {
        if ((!customPath.empty()) && (mmIsDir((customPath).c_str()) == EN_OK)) {
            basePath.push_back(std::make_pair(customPath, gert::OppImplVersionTag::kOpp));
            OP_LOGI("Valid customPath '%s'.", customPath.c_str());
        } else {
            OP_LOGW("CustomPath '%s' is invalid, which is skipped.", customPath.c_str());
        }
    }
    OP_LOGI("Get CustomOppPath finished.");
}

aclnnStatus NnopbaseLoadTilingSo(std::vector<std::pair<std::string, gert::OppImplVersionTag>>& basePath)
{
    std::string path;
    std::string osType;
    std::string cpuType;
    bool openSoSuccess = false;
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseGetCurEnvPackageOsAndCpuType(osType, cpuType));
    std::vector<NnopbaseChar> soPath(NNOPBASE_FILE_PATH_MAX_LEN, '\0');
    std::vector<std::string> tilingSoPaths;
    for (size_t i = 0U; i < basePath.size(); i++) {
        tilingSoPaths.clear();
        if (i == (basePath.size() - 1U)) {
            // 自研路径V2
            std::string builtInTilingSoBasePath = basePath[i].first + "/op_impl/ai_core/tbe/op_host/lib/" + osType +
                                                  "/" + cpuType + "/";
            if (mmRealPath(builtInTilingSoBasePath.c_str(), &(soPath[0U]), NNOPBASE_FILE_PATH_MAX_LEN) != EN_OK) {
                // 自研路径V1
                tilingSoPaths.push_back(basePath[i].first + "/op_impl/ai_core/tbe/op_tiling/lib/" + osType + "/" +
                                        cpuType + "/" + "libopmaster_rt2.0.so");
            } else {
                GetFilesWithSuffix(&(soPath[0U]), OP_TILING_SO_SUFFIX, tilingSoPaths);
            }
        } else {
            // 自定义路径
            path = basePath[i].first + "/op_impl/ai_core/tbe/op_tiling/lib/" + osType + "/" + cpuType + "/" +
                   "libcust_opmaster_rt2.0.so";
            if (mmRealPath(path.c_str(), &(soPath[0U]), NNOPBASE_FILE_PATH_MAX_LEN) == EN_OK) {
                tilingSoPaths.push_back(std::string(&(soPath[0U])));
            } else {
                OP_LOGW("Failed to get op tiling so path for %s, errmsg:%s.", path.c_str(), NnopbaseGetmmErrorMsg());
            }
        }
        for (auto tilingSoPath : tilingSoPaths) {
            OP_LOGI("Tiling so path: %s", tilingSoPath.c_str());
            auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry(basePath[i].second);
            if (registry == nullptr) {
                registry = std::make_shared<gert::OpImplSpaceRegistryV2>();
                NNOPBASE_ASSERT_NOTNULL_RETVAL(registry);
                gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(registry, basePath[i].second);
            }
            gert::OppSoDesc oppSoDesc({ge::AscendString(tilingSoPath.c_str())},
                                      ge::AscendString(GetOpSoPackageName(tilingSoPath).c_str()));
            if (registry->AddSoToRegistry(oppSoDesc) == ge::GRAPH_SUCCESS) {
                openSoSuccess = true;
            } else {
                OP_LOGW("Failed to load op tiling so path for %s.", tilingSoPath.c_str());
            }
        }
    }
    if (openSoSuccess) {
        return OK;
    }
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Failed to get op tiling so path.");
    return ACLNN_ERR_PARAM_INVALID;
}

NnopbaseRegInfo* NnopbaseCollectorFindRegInfoInTbl(const NnopbaseBinCollector* const collector,
                                                   const NnopbaseChar* const opType, const uint64_t hashKey)
{
    if (hashKey >= NNOPBASE_NORM_MAX_BIN_BUCKETS) {
        OP_LOGE(ACLNN_ERR_INNER, "HashKey[%lu] is too large, please check.", hashKey);
        return nullptr;
    }
    if (collector == nullptr) {
        return nullptr;
    }
    NnopbaseRegInfo* regInfo = nullptr;
    int32_t ret = 0;
    const DList* const head = &collector->regInfoTbl.buckets[hashKey].head;
    for (DoubleListNode* node = head->node.next; node != &(head->node); node = node->next) {
        regInfo = (op::internal::PtrCastTo<NnopbaseRegInfo>(op::internal::PtrCastTo<NnopbaseChar>(node) -
                                                            offsetof(NnopbaseRegInfo, dllNode)));
        ret = strcmp(opType, regInfo->key.opType.c_str());
        if (ret < 0) {
            return nullptr;
        }
        if (ret == 0) {
            return regInfo;
        }
    }
    return nullptr;
}

aclnnStatus NnopbaseCollectorOpRegInfoInit(NnopbaseRegInfo* regInfo, const NnopbaseJsonInfo& jsonInfo,
                                           const uint64_t hashKey, gert::OppImplVersionTag oppImplVersion)
{
    NNOPBASE_ASSERT_OK_RETVAL(
        NnopbaseCollectorSetTiling(jsonInfo, &regInfo->tiling, &regInfo->genSimplifiedKey, oppImplVersion));
    regInfo->key.opType = jsonInfo.opType;
    regInfo->isActive = true;
    regInfo->key.hashKey = hashKey;
    DoubleListNodeInit(&regInfo->dllNode);
    NnopbaseCollectorInitBinTbl(&regInfo->binTbl);
    return OK;
}

void NnopbaseCollectorOpRegInfoDestroy(NnopbaseRegInfo** regInfo)
{
    delete (*regInfo);
    *regInfo = nullptr;
}

aclnnStatus NnopbaseCollectorAddRegInfoToTbl(NnopbaseBinCollector* const collector, const NnopbaseJsonInfo& jsonInfo,
                                             const uint64_t hashKey, NnopbaseRegInfo*& reg,
                                             gert::OppImplVersionTag oppImplVersion)
{
    NNOPBASE_ASSERT_TRUE_RETVAL(hashKey < NNOPBASE_NORM_MAX_BIN_BUCKETS); // check index
    OP_LOGD("Start to add %s regInfo to table, hashkey is %ld.", jsonInfo.opType.c_str(), hashKey);
    auto regInfo = std::make_unique<NnopbaseRegInfo>();
    NNOPBASE_ASSERT_NOTNULL_RETVAL(regInfo);
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseCollectorOpRegInfoInit(regInfo.get(), jsonInfo, hashKey, oppImplVersion));
    DList* const head = &collector->regInfoTbl.buckets[regInfo->key.hashKey].head;

    NnopbaseRegInfo* other = nullptr;
    for (DoubleListNode* node = head->node.next; node != &(head->node); node = node->next) {
        other = (op::internal::PtrCastTo<NnopbaseRegInfo>(op::internal::PtrCastTo<NnopbaseChar>(node) -
                                                          offsetof(NnopbaseRegInfo, dllNode)));
        const auto ret = strcmp(regInfo->key.opType.c_str(), other->key.opType.c_str());
        if (ret < 0) { // 链表中的node按照opType字典序排序 小于0时即找到比other小的位置 节点插入在other前面
            break;
        }
        other = nullptr;
    }
    if (other == nullptr) {
        DoubleListAppend(&regInfo->dllNode, head);
    } else {
        DoubleListInsertBefore(&regInfo->dllNode, &other->dllNode);
    }
    OP_LOGD("Finish add %s regInfo to table.", jsonInfo.opType.c_str());
    // regInfo has insert in list
    reg = regInfo.release();
    return OK;
}

aclnnStatus NnopbaseCollectorAddRepoInfo(NnopbaseBinCollector* const collector, const NnopbaseJsonInfo& jsonInfo,
                                         const string& key, gert::OppImplVersionTag oppImplVersion)
{
    const uint64_t hashKey = static_cast<uint64_t>(
        NnopbaseHashBinary(op::internal::PtrCastTo<const NnopbaseUChar>(jsonInfo.opType.c_str()),
                           jsonInfo.opType.size()) %
        NNOPBASE_NORM_MAX_BIN_BUCKETS);
    NnopbaseRegInfo* regInfo = NnopbaseCollectorFindRegInfoInTbl(collector, jsonInfo.opType.c_str(), hashKey);
    if (regInfo == nullptr) {
        NNOPBASE_ASSERT_OK_RETVAL(
            NnopbaseCollectorAddRegInfoToTbl(collector, jsonInfo, hashKey, regInfo, oppImplVersion));
        NNOPBASE_ASSERT_NOTNULL_RETVAL(regInfo);
    }
    NNOPBASE_ASSERT_TRUE_RETVAL(regInfo->isActive);

    regInfo->customizedSimplifiedKey = jsonInfo.customizedSimplifiedKey;

    const size_t binKeyLen = key.size() * MAX_BIN_KEY_MULTIPLIER;
    std::vector<NnopbaseUChar> binKey(binKeyLen, '\0');
    uint32_t keySize = 0U;
    OP_LOGD("Check attributes of operator %s, static shape is %s, customizedSimplifiedKey is %s, "
            "genSimplifiedKey is %s.",
            jsonInfo.opType.c_str(), jsonInfo.isStaticShape ? "true" : "false",
            jsonInfo.customizedSimplifiedKey ? "true" : "false",
            regInfo->genSimplifiedKey != nullptr ? "registered" : "not registered");
    if (jsonInfo.isStaticShape) {
        regInfo->hasStaticShapeBin = true;
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseCollectorConvertStaticVerbKey(key.c_str(), &(binKey[0U]), &keySize));
    } else if (jsonInfo.customizedSimplifiedKey) {
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseCollectorConvertCustomizedVerbKey(key.c_str(), &(binKey[0U]), &keySize));
    } else {
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseCollectorConvertDynamicVerbKey(key.c_str(), &(binKey[0U]), &keySize));
    } // isDynamic

    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseCollectorAddBinInfo(key, regInfo, jsonInfo, &(binKey[0U]), keySize));
    return OK;
}

inline static aclnnStatus NnopbaseCollectorAddRepoInfos(NnopbaseBinCollector* const collector,
                                                        const NnopbaseJsonInfo& jsonInfo,
                                                        gert::OppImplVersionTag oppImplVersion)
{
    for (const auto& key : jsonInfo.keys) {
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseCollectorAddRepoInfo(collector, jsonInfo, key, oppImplVersion));
    }
    return OK;
}

aclnnStatus NnopbaseCollectorGcRegInfo(void* data)
{
    NnopbaseRegInfo* regInfo = op::internal::PtrCastTo<NnopbaseRegInfo>(data);
    for (size_t i = 0U; i < NNOPBASE_NORM_MAX_BIN_BUCKETS; i++) {
        BinInfoBucket* bucket = &regInfo->binTbl.buckets[i];
        if (bucket->isVisit) {
            return ACLNN_SUCCESS;
        }
        DList* const head = &bucket->head;
        if (head->count == 0U) {
            continue;
        }
        for (DoubleListNode *node = (head)->node.next, *tmp = node->next; node != &(head->node);
             node = tmp, tmp = (node)->next) {
            NnopbaseBinInfo* binInfo = (op::internal::PtrCastTo<NnopbaseBinInfo>(
                op::internal::PtrCastTo<NnopbaseChar>(node) - offsetof(NnopbaseBinInfo, dllNode)));
            DoubleListRemove(node, head);
            if (binInfo != nullptr) {
                NnopbaseBinInfoDestroy(&binInfo);
            }
        }
    }
    NnopbaseCollectorOpRegInfoDestroy(&regInfo);
    return ACLNN_SUCCESS;
}

void SetExtraKernelInfoToBin(const NnopbaseJsonInfo& jsonInfo, std::unique_ptr<NnopbaseBinInfo>& binInfo)
{
    auto& runInfo = jsonInfo.extraKernelDesc.runInfo;
    auto& platformInfo = jsonInfo.extraKernelDesc.platformInfo;
    if (runInfo != nullptr) {
        if (binInfo->extraKernelDesc == nullptr) {
            binInfo->extraKernelDesc = std::make_shared<ExtraKernelDesc>();
            NNOPBASE_ASSERT_NOTNULL(binInfo->extraKernelDesc);
        }
        binInfo->extraKernelDesc->runInfo = std::make_unique<StaticKernelJsonRunInfo>(*runInfo);
        NNOPBASE_ASSERT_NOTNULL(binInfo->extraKernelDesc->runInfo);
        OP_LOGI(
            "Update runInfo of %s json to binInfo, aicpuNumBlocks %u, numBlocks %u,"
            " dynUBufSize %u, scheduleMode %u, tilingCond %u, tilingKey %llu, clearAtomic %d, workspaceSizesNum %zu.",
            jsonInfo.opType.c_str(), binInfo->extraKernelDesc->runInfo->aicpuNumBlocks,
            binInfo->extraKernelDesc->runInfo->numBlocks, binInfo->extraKernelDesc->runInfo->dynUBufSize,
            binInfo->extraKernelDesc->runInfo->scheduleMode, binInfo->extraKernelDesc->runInfo->tilingCond,
            binInfo->extraKernelDesc->runInfo->tilingKey, binInfo->extraKernelDesc->runInfo->clearAtomic,
            binInfo->extraKernelDesc->runInfo->workspaceSizesNum);
    }
    if (platformInfo != nullptr) {
        if (binInfo->extraKernelDesc == nullptr) {
            binInfo->extraKernelDesc = std::make_shared<ExtraKernelDesc>();
            NNOPBASE_ASSERT_NOTNULL(binInfo->extraKernelDesc);
        }
        binInfo->extraKernelDesc->platformInfo = std::make_unique<StaticKernelPlatformInfo>(*platformInfo);
        NNOPBASE_ASSERT_NOTNULL(binInfo->extraKernelDesc->platformInfo);
        OP_LOGI(
            "Update platformInfo of %s json to binInfo, aic blocks is %u, aiv blocks is %u, deterministic level is %d.",
            jsonInfo.opType.c_str(), binInfo->extraKernelDesc->platformInfo->coreNum.aicNum,
            binInfo->extraKernelDesc->platformInfo->coreNum.aivNum,
            binInfo->extraKernelDesc->platformInfo->deterministicLevel);
    }
}

aclnnStatus NnopbaseCollectorAddBinInfo(const string& key, NnopbaseRegInfo* const regInfo,
                                        const NnopbaseJsonInfo& jsonInfo, const NnopbaseUChar* const verbose,
                                        const uint32_t len)
{
    OP_LOGD("Start to add %s binInfo, key size is %u.", regInfo->key.opType.c_str(), len);

    // incremental update static bin
    if (jsonInfo.isStaticShape) {
        size_t hashKey = NnopbaseHashBinary(verbose, len) % NNOPBASE_NORM_MAX_BIN_BUCKETS;
        NnopbaseBinInfo* findBin = NnopbaseCollectorFindBinInfo(regInfo, hashKey, verbose, len,
                                                                jsonInfo.extraKernelDesc.platformInfo.get());
        if (findBin != nullptr) {
            OP_LOGI("%s binInfo already exists, no need to add.", regInfo->key.opType.c_str());
            return OK;
        }
    }

    auto binInfo = std::make_unique<NnopbaseBinInfo>();
    NNOPBASE_ASSERT_NOTNULL_RETVAL(binInfo);
    NnopbaseBinInfoInit(binInfo.get());
    binInfo->simplifiedKey = key;
    binInfo->coreType = jsonInfo.coreType;
    binInfo->isStaticShape = jsonInfo.isStaticShape;
    binInfo->numBlocks = jsonInfo.numBlocks;
    binInfo->kernelName = jsonInfo.kernelName;
    binInfo->multiKernelType = jsonInfo.multiKernelType;
    binInfo->loadBinInfoType = jsonInfo.loadBinInfoType;
    if (binInfo->loadBinInfoType == kStaticBinInfo) {
        binInfo->binLen = jsonInfo.binLen;
        binInfo->bin = jsonInfo.bin;
    }
    SetExtraKernelInfoToBin(jsonInfo, binInfo);

    auto vec = op::internal::PtrCastTo<gert::ContinuousVector>(binInfo->staticWorkspaceSizes);
    NNOPBASE_ASSERT_TRUE_RETVAL(memcpy_s(vec->MutableData(), sizeof(size_t) * vec->GetCapacity(),
                                         jsonInfo.workspaceSizes, sizeof(jsonInfo.workspaceSizes)) == EOK);
    NNOPBASE_ASSERT_TRUE_RETVAL(vec->SetSize(jsonInfo.workspaceSizeNum) == ge::GRAPH_SUCCESS);
    binInfo->binPath = std::string(jsonInfo.path);
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseBinInfoSetOpBinInfoKey(binInfo.get(), verbose, len));
    NnopbaseCollectorInsertBinInfo(regInfo, binInfo.release()); // insert in list
    OP_LOGD("Finish add %s binInfo.", regInfo->key.opType.c_str());
    return OK;
}

void UpdateRunInfoForStaticJson(ExtraKernelDesc& kernelDesc, nlohmann::json& staticKernelJsonConfig)
{
    try {
        kernelDesc.runInfo = std::make_unique<StaticKernelJsonRunInfo>();
        if (staticKernelJsonConfig.contains("runInfo")) {
            auto& runInfo = staticKernelJsonConfig["runInfo"];
            kernelDesc.runInfo->aicpuNumBlocks = runInfo["aicpu_block_dim"].get<uint32_t>();
            kernelDesc.runInfo->numBlocks = runInfo["block_dim"].get<uint32_t>();
            kernelDesc.runInfo->dynUBufSize = runInfo["local_memory_size"].get<uint32_t>();
            kernelDesc.runInfo->scheduleMode = runInfo["schedule_mode"].get<uint32_t>();
            kernelDesc.runInfo->tilingCond = runInfo["tiling_cond"].get<uint32_t>();
            kernelDesc.runInfo->tilingKey = runInfo["tiling_key"].get<uint64_t>();
            kernelDesc.runInfo->clearAtomic = runInfo["clear_atomic"].get<uint32_t>();
            std::string tilingData = runInfo["tiling_data"].get<std::string>();
            if (tilingData.size() % 2U == 1U) {
                OP_LOGW("To convert static tiling data, the size of tiling data must be even!");
            } else {
                for (size_t i = 0U; i < tilingData.size(); i += 2U) { // String内容按每2个字符读取
                    std::string byteStr = tilingData.substr(i, 2);    // 每2个四比特的字符拼一个字节数据
                    uint8_t byte = static_cast<uint8_t>(
                        std::stoul(byteStr, nullptr, 16)); // 按16进制字符格式转为一个无符号字节
                    kernelDesc.runInfo->tilingData.emplace_back(byte);
                }
            }
        } else {
            // 非mc2静态算子场景
            kernelDesc.runInfo->numBlocks = staticKernelJsonConfig["blockDim"].get<uint32_t>();
            kernelDesc.runInfo->dynUBufSize = staticKernelJsonConfig["localMemorySize"].get<uint32_t>();
            kernelDesc.runInfo->scheduleMode = staticKernelJsonConfig["schedule_mode"].get<uint32_t>();
        }
        kernelDesc.runInfo->workspaceSizesNum = staticKernelJsonConfig["workspace"]["num"].get<size_t>();
        kernelDesc.runInfo->workspaceSizes.resize(kernelDesc.runInfo->workspaceSizesNum);
        for (size_t i = 0U; i < kernelDesc.runInfo->workspaceSizesNum; ++i) {
            kernelDesc.runInfo->workspaceSizes[i] = staticKernelJsonConfig["workspace"]["size"][i].get<size_t>();
        }
        OP_LOGI("Parse runInfo successfully, aicpuNumBlocks is %u, numBlocks is %u, dynUBufSize is %u,"
                " scheduleMode is %u, tilingCond is %u, tilingKey is %llu, clearAtomic is %u, staticTilingData size is "
                "%u, workspaceNum is %zu",
                kernelDesc.runInfo->aicpuNumBlocks, kernelDesc.runInfo->numBlocks, kernelDesc.runInfo->dynUBufSize,
                kernelDesc.runInfo->scheduleMode, kernelDesc.runInfo->tilingCond, kernelDesc.runInfo->tilingKey,
                kernelDesc.runInfo->clearAtomic, kernelDesc.runInfo->tilingData.size(),
                kernelDesc.runInfo->workspaceSizesNum);
    } catch (const nlohmann::json::exception& e) {
        kernelDesc.runInfo = nullptr;
        OP_LOGW("Failed to read static kernel json file of runInfo, reason: %s", e.what());
    }
    return;
}

void UpdatePlatformInfoForStaticJson(ExtraKernelDesc& kernelDesc, nlohmann::json& staticKernelJsonConfig)
{
    try {
        if (staticKernelJsonConfig.contains("platformInfo")) {
            auto& platformInfo = staticKernelJsonConfig["platformInfo"];
            kernelDesc.platformInfo = std::make_unique<StaticKernelPlatformInfo>();
            if (kernelDesc.platformInfo == nullptr) {
                return;
            }
            auto& coreNum = kernelDesc.platformInfo->coreNum;
            coreNum.aicNum = platformInfo["cubeCoreCnt"].get<uint32_t>();
            coreNum.aivNum = platformInfo["vectorCoreCnt"].get<uint32_t>();
            if (platformInfo.contains("deterministicLevel")) {
                kernelDesc.platformInfo->deterministicLevel = platformInfo["deterministicLevel"].get<int8_t>();
            }
            OP_LOGI("Parse platformInfo successfully, aicNum is %u, aivNum is %u, determinisitic level is %d.",
                    kernelDesc.platformInfo->coreNum.aicNum, kernelDesc.platformInfo->coreNum.aivNum,
                    kernelDesc.platformInfo->deterministicLevel);
        }
    } catch (const nlohmann::json::exception& e) {
        kernelDesc.platformInfo = nullptr;
        OP_LOGW("Failed to read static kernel json file of platformInfo, reason: %s", e.what());
    }
    return;
}

aclnnStatus UpdateStaticJsonExtraInfo(NnopbaseJsonInfo& jsonInfo)
{
    std::string staticKernelBinPath = jsonInfo.path;
    const size_t pos = staticKernelBinPath.find(".o");
    if (pos == std::string::npos) {
        OP_LOGW("Static kernel json %s is not a correct format, failed to read extra infos.",
                staticKernelBinPath.c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    std::string staticKernelJsonPath = staticKernelBinPath.substr(0U, pos + 1U) + "json";
    nlohmann::json staticKernelJsonConfig;
    if (NnopbaseReadJsonConfig(staticKernelJsonPath, staticKernelJsonConfig) != OK) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    OP_LOGI("Read static kernel json info %s successfully.", staticKernelJsonPath.c_str());
    auto& kernelDesc = jsonInfo.extraKernelDesc;
    // 静态算子的json配置读取失败不影响算子的动态执行
    UpdateRunInfoForStaticJson(kernelDesc, staticKernelJsonConfig);
    UpdatePlatformInfoForStaticJson(kernelDesc, staticKernelJsonConfig);
    return OK;
}

void NnopbaseCollectorInsertBinInfo(NnopbaseRegInfo* const regInfo, NnopbaseBinInfo* binInfo)
{
    const size_t key = binInfo->binInfoKey.hashKey;
    const uint32_t keyLen = binInfo->binInfoKey.len;
    OP_LOGI("%s insert bin key is %zu, key len is %u", regInfo->key.opType.c_str(), key, keyLen);
    while (!__sync_bool_compare_and_swap(&regInfo->binTbl.buckets[key].isVisit, false, true)) {
        /* nothing to do */
    };
    DList* const head = &regInfo->binTbl.buckets[key].head;
    DoubleListNodeInit(&binInfo->dllNode);
    DoubleListAppend(&binInfo->dllNode, head);
    binInfo->regInfo = regInfo;
    regInfo->binTbl.buckets[key].isVisit = false;
}

// 一样的key，会优先选择先放入hash表的
NnopbaseBinInfo* NnopbaseCollectorFindBinInfo(NnopbaseRegInfo* const regInfo, const size_t hashKey,
                                              const NnopbaseUChar* const verbose, const uint32_t verbLen,
                                              const StaticKernelPlatformInfo* const platformInfo)
{
    OP_LOGI("Start find binInfo, opType is %s, hashKey is %zu, verbLen is %u.", regInfo->key.opType.c_str(), hashKey,
            verbLen);
    if (hashKey >= NNOPBASE_NORM_MAX_BIN_BUCKETS) {
        OP_LOGE(ACLNN_ERR_INNER, "HashKey[%zu] is too large, please check.", hashKey);
        return nullptr;
    }
    while (!__sync_bool_compare_and_swap(&regInfo->binTbl.buckets[hashKey].isVisit, false, true)) {
        /* nothing to do */
    };
    const DList* const head = &regInfo->binTbl.buckets[hashKey].head;
    for (DoubleListNode* node = head->node.next; node != &(head->node); node = node->next) {
        NnopbaseBinInfo* binInfo = (op::internal::PtrCastTo<NnopbaseBinInfo>(
            op::internal::PtrCastTo<NnopbaseChar>(node) - offsetof(NnopbaseBinInfo, dllNode)));
        if (verbLen != binInfo->binInfoKey.len) {
            OP_LOGI("Op %s bin key len is %u, key in table len is %u.", regInfo->key.opType.c_str(), verbLen,
                    binInfo->binInfoKey.len);
            continue;
        }
        const NnopbaseUChar* const verb = &(binInfo->binInfoKey.verbose[0U]);
        uint32_t i;
        for (i = 0U; i < verbLen; i++) {
            if (verb[i] != verbose[i]) {
                OP_LOGI("Op %s bin key is not equal, bin key[%u] is %u, table key[%u] is %u.",
                        regInfo->key.opType.c_str(), i, verbose[i], i, verb[i]);
                break;
            }
        }
        if (i == verbLen && MatchBinWithPlatformInfo(binInfo, platformInfo)) {
            regInfo->binTbl.buckets[hashKey].isVisit = false;
            OP_LOGI("Found %s binInfo, hashKey is %zu.", regInfo->key.opType.c_str(), hashKey);
            return binInfo;
        }
    }
    regInfo->binTbl.buckets[hashKey].isVisit = false;
    OP_LOGI("There is no available %s binInfo for hashKey %zu.", regInfo->key.opType.c_str(), hashKey);
    return nullptr;
}

aclnnStatus NnopbaseCollectorConvertCustomizedVerbKey(const NnopbaseChar* const strKey, NnopbaseUChar* const binKey,
                                                      uint32_t* const size)
{
    const size_t len = strlen(strKey);
    OP_LOGD("Start convert customized verbose key %s, size=%zu.", strKey, len);
    NnopbaseUChar* key = binKey;
    size_t i = 0U;
    int32_t index = NNOPBASE_OP_TYPE_INDEX;
    while (i < len) {
        switch (index) {
            case NNOPBASE_OP_TYPE_INDEX: {
                if (strKey[i] != '/') {
                    key = NnopbaseAppend1Byte(key, static_cast<NnopbaseUChar>(strKey[i]));
                } else {
                    index++;
                }
                break;
            }
            case NNOPBASE_COMPILE_ARGS_INDEX: {
                if (strKey[i] != '/') {
                    if ((strKey[i] == '0') || (strKey[i] == '1') || (strKey[i] == '2')) {
                        uint64_t offsetNumber = static_cast<uint64_t>(strKey[i]) - static_cast<uint64_t>('0');
                        key = NnopbaseAppend1Byte(key, static_cast<NnopbaseUChar>(offsetNumber));
                    }
                } else {
                    index++;
                }
                break;
            }
            default: {
                key = NnopbaseAppend1Byte(key, static_cast<NnopbaseUChar>(strKey[i]));
                break;
            }
        }
        i++;
    }

    *size = static_cast<uint32_t>(key - binKey);
    OP_LOGD("Finish convert customized verbose key, key size is [%u].", *size);
    return OK;
}

aclnnStatus NnopbaseCollectorConvertDynamicVerbKey(const NnopbaseChar* const strKey, NnopbaseUChar* const binKey,
                                                   uint32_t* const size)
{
    const size_t len = strlen(strKey);
    OP_LOGD("Start convert dynamic verbose key %s, size=%zu.", strKey, len);
    NnopbaseUChar* key = binKey;
    size_t i = 0U;
    int32_t index = NNOPBASE_OP_TYPE_INDEX;
    NnopbaseUChar type = 0U;
    bool isReplayStaticKey = false;
    while (i < len) {
        switch (index) {
            case NNOPBASE_OP_TYPE_INDEX: {
                if (strKey[i] != '/') {
                    key = NnopbaseAppend1Byte(key, static_cast<NnopbaseUChar>(strKey[i]));
                } else {
                    index++;
                }
                break;
            }
            case NNOPBASE_COMPILE_ARGS_INDEX: {
                if (strKey[i] != '/') {
                    if ((strKey[i] == '0') || (strKey[i] == '1') || (strKey[i] == '2')) {
                        key = NnopbaseAppend1Byte(key, static_cast<NnopbaseUChar>(static_cast<uint64_t>(strKey[i]) -
                                                                                  static_cast<uint64_t>('0')));
                    }
                } else {
                    index++;
                }
                break;
            }
            default: {
                if (!isReplayStaticKey) {
                    isReplayStaticKey = true;
                }
                if ((strKey[i] >= '0') && (strKey[i] <= '9')) {
                    type = type * 10U + (static_cast<uint64_t>(strKey[i]) - static_cast<uint64_t>('0')); // 10 转换规则
                } else {
                    key = NnopbaseAppend1Byte(key, type);
                    type = 0U;
                }
                break;
            }
        }
        i++;
    }

    if (isReplayStaticKey) {
        key = NnopbaseAppend1Byte(key, type);
    }
    *size = static_cast<uint32_t>(key - binKey);
    OP_LOGD("Finish convert dynamic verbose key, key size is [%u].", *size);
    return OK;
}

NnopbaseUChar* NnopbaseBeyond8ByteCopy(const int32_t start, const int32_t end, const NnopbaseChar* const strKey,
                                       NnopbaseUChar* verKey)
{
    uint64_t type = 0U;
    int32_t bitNum = 0;
    uint32_t j = 0U;
    for (int32_t i = end; i >= start; i--) {
        if ((strKey[i] >= '0') && (strKey[i] <= '9')) {
            type = type | ((static_cast<uint64_t>(static_cast<uint64_t>(strKey[i]) - static_cast<uint64_t>('0')))
                           << ((4U * j))); // 4 is 4bit
            bitNum++;
        } else if ((strKey[i] >= 'a') && (strKey[i] <= 'f')) {
            type = type | (static_cast<uint64_t>(static_cast<uint64_t>(strKey[i]) - static_cast<uint64_t>('a') + 10)
                           << ((4U * j))); // 4 is 4bit 10 is decimal
            bitNum++;
        }
        j++;
        if (bitNum == 16) { // // 16 is hexadecimal
            verKey = NnopbaseAppend8Byte(verKey, type);
            type = 0U;
            bitNum = 0;
            j = 0U;
        }
    }
    if (bitNum > 0) {
        verKey = NnopbaseAppend8Byte(verKey, type);
    }
    return verKey;
}

aclnnStatus NnopbaseCollectorConvertStaticVerbKey(const NnopbaseChar* const strKey, NnopbaseUChar* const binKey,
                                                  uint32_t* const size)
{
    const size_t len = strlen(strKey);
    OP_LOGD("Start convert static verbose key %s, size=%zu.", strKey, len);
    NnopbaseUChar* key = binKey;
    size_t i = 0U;
    uint64_t type = 0U;
    int32_t index = NNOPBASE_OP_TYPE_INDEX;
    int32_t bitNum = 0;
    bool isAppend = false;
    int32_t start = -1;
    while (i < len) {
        switch (index) {
            case NNOPBASE_OP_TYPE_INDEX: {
                if (strKey[i] != '/') {
                    key = NnopbaseAppend1Byte(key, static_cast<NnopbaseUChar>(strKey[i]));
                } else {
                    index++;
                }
                break;
            }
            case NNOPBASE_COMPILE_ARGS_INDEX: {
                if (strKey[i] != '/') {
                    if ((strKey[i] == '0') || (strKey[i] == '1') || (strKey[i] == '2')) {
                        key = NnopbaseAppend8Byte(key, (static_cast<uint64_t>(strKey[i]) - static_cast<uint64_t>('0')));
                    }
                } else {
                    index++;
                }
                break;
            }
            default: {
                if (strKey[i] == '_') {
                    key = NnopbaseAppend8Byte(key, static_cast<uint64_t>(strKey[i]));
                    break;
                }
                if ((strKey[i] >= '0') && (strKey[i] <= '9')) {
                    type = (type << 4U) | (static_cast<uint64_t>(strKey[i]) - static_cast<uint64_t>('0')); // 4 is 4bit
                    bitNum++;
                    isAppend = true;
                } else if ((strKey[i] >= 'a') && (strKey[i] <= 'f')) {
                    // 4 is 4bit, 10 is decimal conversion
                    type = (type << 4U) | ((static_cast<uint64_t>(strKey[i]) - static_cast<uint64_t>('a')) + 10U);
                    bitNum++;
                    isAppend = true;
                } else {
                    if (start != -1) {
                        key = NnopbaseBeyond8ByteCopy(start, static_cast<int32_t>(i - 1U), strKey, key);
                        start = -1;
                        type = 0U;
                        bitNum = 0;
                        isAppend = false;
                    }
                    if (isAppend) {
                        key = NnopbaseAppend8Byte(key, type);
                        type = 0U;
                        bitNum = 0;
                        isAppend = false;
                    }
                }
                if (bitNum == 16) {                           // 16 is hexadecimal
                    if (start == -1) {
                        start = static_cast<int32_t>(i) - 15; // 15 is bitNum -1
                    }
                    type = 0U;
                    bitNum = 0;
                    isAppend = false;
                }
                break;
            }
        }
        i += 1U;
    }
    if (start != -1) {
        key = NnopbaseBeyond8ByteCopy(start, static_cast<int32_t>(i - 1U), strKey, key); // 超出8字节
        isAppend = false;
    }
    if (isAppend) {
        key = NnopbaseAppend8Byte(key, type); // 字符串属性处于末尾
    }
    *size = static_cast<uint32_t>(key - binKey);
    OP_LOGD("Finish convert static verbose key, key size is [%u].", *size);
    return OK;
}

static aclnnStatus NnopbaseGetOpBinPath(const std::string& filePath, std::string& binPath)
{
    const size_t pos = filePath.find(".json");
    if (pos != std::string::npos) {
        binPath = filePath.substr(0U, pos + 1U) + "o";
    } else {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Failed to get binPath from filePath %s.", filePath.c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    return OK;
}

static aclnnStatus NnopbaseCollectorReadStaticBinJsonInfo(NnopbaseJsonInfo& jsonInfo)
{
    std::tuple<nlohmann::json, nnopbase::Binary> binInfo;
    NNOPBASE_ASSERT_OK_RETVAL(
        nnopbase::OpBinaryResourceManager::GetInstance().GetOpBinaryDescByKey(jsonInfo.keys[0].c_str(), binInfo));
    auto& binConfigInfo = std::get<0>(binInfo);
    const auto& binContent = std::get<1>(binInfo);
    jsonInfo.bin = op::internal::PtrCastTo<const NnopbaseUChar>((binContent.content));
    jsonInfo.binLen = binContent.len;
    try {
        const std::string coreType = binConfigInfo["coreType"].get<std::string>();
        const auto& iterKernelType = g_nnopbaseKernelTypeMap.find(coreType);
        NNOPBASE_ASSERT_TRUE_RETVAL(iterKernelType != g_nnopbaseKernelTypeMap.end());
        jsonInfo.coreType = iterKernelType->second;
        const std::string filePath = binConfigInfo["filePath"].get<std::string>();
        std::string binPath;
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseGetOpBinPath(filePath, binPath));
        const errno_t ret = strcpy_s(jsonInfo.path, NNOPBASE_FILE_PATH_MAX_LEN, binPath.c_str());
        CHECK_COND(ret == EOK, ACLNN_ERR_PARAM_INVALID,
                   "Failed to execute strcpy_s from binPath[%s] to jsonInfo.path[%s], result is %d.", binPath.c_str(),
                   jsonInfo.path, ret);
    } catch (const nlohmann::json::exception& e) {
        OP_LOGD("%s not get coreType or filePath, reason %s", jsonInfo.opType.c_str(), e.what());
        return ACLNN_ERR_PARAM_INVALID;
    }
    OP_LOGI("Get %s coreType is [%d].", jsonInfo.opType.c_str(), jsonInfo.coreType);
    return OK;
}

static aclnnStatus NnopbaseGetSimplifiedKey(nlohmann::json& binInfo, NnopbaseJsonInfo& jsonInfo)
{
    try {
        try {
            jsonInfo.keys = binInfo["simplifiedKey"].get<std::vector<std::string>>();
        } catch (const nlohmann::json::exception& e) {
            OP_LOGD("Can't read op %s jsonfile simplifiedKey, reason %s", jsonInfo.opType.c_str(), e.what());
            const std::string key = binInfo["simplifiedKey"].get<std::string>();
            jsonInfo.keys = std::vector<std::string>({key});
        }
    } catch (const nlohmann::json::exception& e) {
        OP_LOGW("Failed to read op %s jsonfile, reason: %s", jsonInfo.opType.c_str(), e.what());
        return ACLNN_ERR_PARAM_INVALID;
    }
    return OK;
}

static aclnnStatus NnopbaseUpdateCommonJsonInfo(nlohmann::json& binInfo, const std::string& kernelSubPath,
                                                NnopbaseJsonInfo& jsonInfo, std::string pkgName = "")
{
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseGetSimplifiedKey(binInfo, jsonInfo));

    int32_t coreType = kCoreTypeEnd;
    std::string relativeBinPath;
    try {
        coreType = binInfo["coreType"].get<int32_t>();
        relativeBinPath = binInfo["binPath"].get<std::string>();
    } catch (const nlohmann::json::exception& e) {
        OP_LOGW("Failed to read op %s coreType or binPath, reason: %s", jsonInfo.opType.c_str(), e.what());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!pkgName.empty()) {
        std::vector<std::string> splitedRelativeBinPaths;
        NnopbaseSplitStr(relativeBinPath, "/", splitedRelativeBinPaths);
        std::string relativeBinPathV2 = "";
        for (size_t i = 0U; i < splitedRelativeBinPaths.size(); ++i) {
            relativeBinPathV2 += splitedRelativeBinPaths[i];
            if (i == 0U) {
                relativeBinPathV2 += "/" + pkgName + "/";
            } else if (i != splitedRelativeBinPaths.size() - 1U) {
                relativeBinPathV2 += "/";
            }
        }
        relativeBinPath = relativeBinPathV2;
        OP_LOGI("Get op[%s] binPath [%s]", jsonInfo.opType.c_str(), relativeBinPath.c_str());
    }
    const std::string& kernelPath = kernelSubPath + relativeBinPath;
    jsonInfo.coreType = static_cast<CoreType>(coreType);
    // get kernelPath
    const errno_t ret = strcpy_s(jsonInfo.path, NNOPBASE_FILE_PATH_MAX_LEN, kernelPath.c_str());
    CHECK_COND(ret == EOK, ACLNN_ERR_PARAM_INVALID,
               "Failed to execute strcpy_s from kernel path[%s] to jsonInfo.path[%s], result is %d.",
               kernelPath.c_str(), jsonInfo.path, ret);
    OP_LOGI("Get op[%s] coreType is [%d], binPath is [%s]", jsonInfo.opType.c_str(), jsonInfo.coreType, jsonInfo.path);
    jsonInfo.multiKernelType = 0U;

    if ((jsonInfo.coreType == kMix || jsonInfo.coreType == kMixAiCore || jsonInfo.coreType == kMixAiv)) {
        try {
            jsonInfo.multiKernelType = binInfo["multiKernelType"].get<uint32_t>();
        } catch (const nlohmann::json::exception& e) {
            OP_LOGD("Op[%s] can not get multiKernelType, reason %s", jsonInfo.opType.c_str(), e.what());
        }
    }
    return OK;
}

aclnnStatus NnopbaseUpdateStaticJsonInfo(nlohmann::json& binInfo, NnopbaseJsonInfo& jsonInfo)
{
    jsonInfo.isStaticShape = true;
    try {
        auto binDesc = binInfo["binDesc"];
        jsonInfo.numBlocks = binDesc["blockDim"].get<uint32_t>();
        jsonInfo.kernelName = binDesc["kernelName"].get<std::string>();
        jsonInfo.workspaceSizeNum = binDesc["workspace"].size();
        if (binDesc["workspace"].size() <= NNOPBASE_NORM_MAX_WORKSPACE_NUMS) {
            for (size_t i = 0U; i < binDesc["workspace"].size(); i++) {
                jsonInfo.workspaceSizes[i] = binDesc["workspace"][i].get<size_t>();
            }
        } else {
            OP_LOGW("WorkspaceSizes %lu is too large.", binDesc["workspace"].size());
            return ACLNN_ERR_PARAM_INVALID;
        }
    } catch (const nlohmann::json::exception& e) {
        OP_LOGW("Failed to read op %s jsonfile binDesc, reason: %s", jsonInfo.opType.c_str(), e.what());
        return ACLNN_ERR_PARAM_INVALID;
    }
    return OK;
}

aclnnStatus NnopbaseCollectorReadDebugKernelOpInfoConfig(NnopbaseBinCollector* const collector,
                                                         nlohmann::json& binaryInfoConfig, const std::string& basePath,
                                                         gert::OppImplVersionTag oppImplVersion)
{
    const std::string& kernelPath = basePath + "/debug_kernel/";
    for (auto iter = binaryInfoConfig.begin(); iter != binaryInfoConfig.end(); ++iter) {
        NnopbaseJsonInfo jsonInfo;
        jsonInfo.opType = iter.key();
        for (auto binInfo : (iter.value())["staticList"]) {
            if (NnopbaseUpdateCommonJsonInfo(binInfo, kernelPath, jsonInfo) != OK) {
                OP_LOGW("Failed to read op %s jsonfile.", jsonInfo.opType.c_str());
                continue;
            }
            if (NnopbaseUpdateStaticJsonInfo(binInfo, jsonInfo) != OK) {
                OP_LOGW("Failed to read op %s jsonfile.", jsonInfo.opType.c_str());
                continue;
            }
            if (UpdateStaticJsonExtraInfo(jsonInfo) != OK) {
                OP_LOGW("Failed to update extra info of static op %s.", jsonInfo.opType.c_str());
            }
            NNOPBASE_ASSERT_OK_RETVAL(NnopbaseCollectorAddRepoInfos(collector, jsonInfo, oppImplVersion));
        }

        for (auto binInfo : (iter.value())["binaryList"]) {
            if (NnopbaseUpdateCommonJsonInfo(binInfo, kernelPath, jsonInfo) != OK) {
                OP_LOGW("Failed to read op %s jsonfile.", jsonInfo.opType.c_str());
                continue;
            }
            jsonInfo.isStaticShape = false;
            NNOPBASE_ASSERT_OK_RETVAL(NnopbaseCollectorAddRepoInfos(collector, jsonInfo, oppImplVersion));
        }
    }
    return OK;
}

aclnnStatus NnopbaseCollectorReadDynamicKernelOpInfoConfig(NnopbaseBinCollector* const collector,
                                                           const nlohmann::json& binaryInfoConfig,
                                                           const std::string& basePath,
                                                           gert::OppImplVersionTag oppImplVersion,
                                                           const std::string pkgName = "")
{
    const std::string& kernelPath = basePath + "/op_impl/ai_core/tbe/kernel/";
    for (auto iter = binaryInfoConfig.begin(); iter != binaryInfoConfig.end(); ++iter) {
        NnopbaseJsonInfo jsonInfo;
        jsonInfo.opType = iter.key();
        jsonInfo.customizedSimplifiedKey = (iter.value()[NNOPBASE_SIMPLIFIED_KEY_MODE_JSON_KEY] ==
                                            NNOPBASE_SIMPLIFIED_KEY_MODE_CUSTOMIZED);
        for (auto binInfo : (iter.value())["binaryList"]) {
            if (NnopbaseUpdateCommonJsonInfo(binInfo, kernelPath, jsonInfo, pkgName) != OK) {
                OP_LOGW("Failed to read op %s jsonfile.", jsonInfo.opType.c_str());
                continue;
            }
            jsonInfo.isStaticShape = false;
            NNOPBASE_ASSERT_OK_RETVAL(NnopbaseCollectorAddRepoInfos(collector, jsonInfo, oppImplVersion));
        }
    }
    OP_LOGI("Read Op Info config successfully.");
    return OK;
}

aclnnStatus NnopbaseUpdateStaticBinJsonInfos(NnopbaseBinCollector* const collector, const NnopbaseChar* const opType)
{
    // 将运行态添加的静态库算子信息注册到collector中
    const auto allOpBinaryDesc = nnopbase::OpBinaryResourceManager::GetInstance().GetAllOpBinaryDesc();
    auto iter = allOpBinaryDesc.find(ge::AscendString(opType));
    if (iter != allOpBinaryDesc.end()) {
        OP_LOGI("Update static library kernel info, opType: %s", opType);
        const nlohmann::json& opJsonDesc = iter->second;
        for (auto binInfo : opJsonDesc["binList"]) {
            NnopbaseJsonInfo jsonInfo;
            jsonInfo.opType = opType;
            jsonInfo.loadBinInfoType = kStaticBinInfo;
            if (NnopbaseGetSimplifiedKey(binInfo, jsonInfo) != OK ||
                NnopbaseCollectorReadStaticBinJsonInfo(jsonInfo) != OK) {
                OP_LOGW("Cannot Update kernel bin info for opType: %s", opType);
                continue;
            }
            NNOPBASE_ASSERT_OK_RETVAL(
                NnopbaseCollectorAddRepoInfos(collector, jsonInfo, gert::OppImplVersionTag::kOpp));
            OP_LOGI("Update static kernel info successfully, opType: %s.", opType);
        }
    }
    return OK;
}

aclnnStatus NnopbaseCollectorReadStaticKernelOpInfoConfig(NnopbaseBinCollector* const collector,
                                                          nlohmann::json& binaryInfoConfig, const std::string& basePath,
                                                          gert::OppImplVersionTag oppImplVersion)
{
    const std::string& kernelPath = basePath + "/static_kernel/ai_core/";
    bool readSuccessFlag = false;
    for (auto iter = binaryInfoConfig.begin(); iter != binaryInfoConfig.end(); ++iter) {
        NnopbaseJsonInfo jsonInfo;
        jsonInfo.opType = iter.key();
        for (auto binInfo : (iter.value())["staticList"]) {
            if (NnopbaseUpdateCommonJsonInfo(binInfo, kernelPath, jsonInfo) != OK) {
                OP_LOGW("Failed to read op %s jsonfile.", jsonInfo.opType.c_str());
                continue;
            }
            if (NnopbaseUpdateStaticJsonInfo(binInfo, jsonInfo) != OK) {
                OP_LOGW("Failed to read op %s jsonfile.", jsonInfo.opType.c_str());
                continue;
            }
            if (UpdateStaticJsonExtraInfo(jsonInfo) != OK) {
                // 不是每个静态json都包含runInfo字段，该字段是否存在不影响staticBin的正常场景执行
                OP_LOGW("Failed to update extra info of static op %s.", jsonInfo.opType.c_str());
            }
            readSuccessFlag = true;
            NNOPBASE_ASSERT_OK_RETVAL(NnopbaseCollectorAddRepoInfos(collector, jsonInfo, oppImplVersion));
        }
    }
    if (!readSuccessFlag) {
        OP_LOGW("Failed to read and parse static kernel config.");
    } else {
        OP_LOGI("Get static kernel path and read config successfully.");
    }
    return OK;
}

aclnnStatus NnopbaseCollectorGetDebugKernelPathAndReadConfig(NnopbaseBinCollector* const collector)
{
    gert::OppImplVersionTag oppImplVersion = gert::OppImplVersionTag::kVersionEnd;
    const std::string basePath = GetBuiltInBasePath(oppImplVersion);
    // Debug算子库为可选项，算子公共路径为空不中断流程
    if (basePath.empty()) {
        return OK;
    };
    const std::string socVersion = nnopbase::IndvSoc::GetInstance().GetCurSocVersion();
    const std::string& binaryInfoPath = basePath + "/debug_kernel/config/" + socVersion + "/binary_info_config.json";
    // 若不存在debug kernel的目录，会在NnopbaseReadJsonConfig里面的realpath判断返回error，不打error日志
    nlohmann::json binaryInfoConfig;
    if (NnopbaseReadJsonConfig(binaryInfoPath, binaryInfoConfig) == OK) {
        NNOPBASE_ASSERT_OK_RETVAL(
            NnopbaseCollectorReadDebugKernelOpInfoConfig(collector, binaryInfoConfig, basePath, oppImplVersion));
    }
    return OK;
}

aclnnStatus NnopbaseCollectorDeleteStaticBins(NnopbaseRegInfo* regInfo)
{
    if (regInfo == nullptr) {
        // skip delete if there is no regInfo in table.
        OP_LOGW("Cannot find static regInfo.");
        return OK;
    }
    auto opType = regInfo->key.opType;
    OP_LOGD("Start find static binInfo, opType is %s, hashKey is %zu.", regInfo->key.opType.c_str(),
            regInfo->key.hashKey);
    for (size_t i = 0U; i < NNOPBASE_NORM_MAX_BIN_BUCKETS; i++) {
        while (!__sync_bool_compare_and_swap(&regInfo->binTbl.buckets[i].isVisit, false, true)) {
            /* nothing to do */
        };
        BinInfoBucket* bucket = &regInfo->binTbl.buckets[i];
        DList* const head = &bucket->head;
        if (head->count == 0U) {
            regInfo->binTbl.buckets[i].isVisit = false;
            continue;
        }
        OP_LOGD("Start to delete OpType %s binTable [%zu], cur Number is %u.", opType.c_str(), i, head->count);
        for (DoubleListNode *node = (head)->node.next, *tmp = node->next; node != &(head->node);
             node = tmp, tmp = (node)->next) {
            NnopbaseBinInfo* binInfo = (op::internal::PtrCastTo<NnopbaseBinInfo>(
                op::internal::PtrCastTo<NnopbaseChar>(node) - offsetof(NnopbaseBinInfo, dllNode)));
            if (binInfo->isStaticShape) {
                DoubleListRemove(node, head);
                NnopbaseBinInfoDestroy(&binInfo);
            }
        }
        regInfo->binTbl.buckets[i].isVisit = false;
    }
    return OK;
}

aclnnStatus NnopbaseRefreshStaticKernelInfos(NnopbaseBinCollector* const collector)
{
    if (collector == nullptr) {
        OP_LOGD("collector is nullptr.");
        return OK;
    }
    // reload static kernel info
    return NnopbaseCollectorGetStaticKernelPathAndReadConfig(
        collector, nnopbase::utils::ThreadVarContainer::GetStaticKernelBasePathInThread());
}

aclnnStatus NnopbaseCollectorGetStaticKernelPathAndReadConfig(NnopbaseBinCollector* const collector,
                                                              const std::string& basePath)
{
    gert::OppImplVersionTag oppImplVersion = gert::OppImplVersionTag::kVersionEnd;
    std::string staticPackageBasePath = GetBuiltInBasePath(oppImplVersion);
    if (!basePath.empty()) {
        staticPackageBasePath = basePath;
    }
    // 静态算子库为可选项，算子公共路径为空不中断流程
    if (staticPackageBasePath.empty()) {
        return OK;
    };
    const std::string socVersion = nnopbase::IndvSoc::GetInstance().GetCurSocVersion();
    const std::string& binaryInfoPath = staticPackageBasePath + "/static_kernel/ai_core/config/" + socVersion +
                                        "/binary_info_config.json";
    OP_LOGI("Start read binary_info_config.json for static kernel. Path: %s", binaryInfoPath.c_str());
    // 若不存在静态kernel的目录，会在NnopbaseReadJsonConfig里面的realpath判断返回error，不打error日志
    nlohmann::json binaryInfoConfig;
    if (NnopbaseReadJsonConfig(binaryInfoPath, binaryInfoConfig) == OK) {
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseCollectorReadStaticKernelOpInfoConfig(collector, binaryInfoConfig,
                                                                                staticPackageBasePath, oppImplVersion));
    }
    return OK;
}

aclnnStatus NnopbaseCollectorGetDynamicKernelPathAndReadConfig(
    NnopbaseBinCollector* const collector, const std::vector<std::pair<std::string, gert::OppImplVersionTag>>& basePath,
    int32_t builtInStartIndex)
{
    OP_LOGI("Start get path and read binary_info_config json.");
    bool readConfigSucc = false;
    const std::string socVersion = nnopbase::IndvSoc::GetInstance().GetCurSocVersion();
    for (size_t i = 0U; i < basePath.size(); i++) {
        nlohmann::json binaryInfoConfig;
        const std::string binaryBasePath = basePath[i].first + "/op_impl/ai_core/tbe/kernel/config/" + socVersion;
        bool foundConfig = false;
        if (static_cast<int32_t>(i) == builtInStartIndex) {
            OP_LOGI("Start finding built-in operator binary_info_config.json.");
            for (const std::string& pkgName : OPS_PATH_VEC) {
                const std::string binaryInfoPath = binaryBasePath + "/" + pkgName + "/binary_info_config.json";
                if (NnopbaseReadJsonConfig(binaryInfoPath, binaryInfoConfig) == OK &&
                    NnopbaseCollectorReadDynamicKernelOpInfoConfig(collector, binaryInfoConfig, basePath[i].first,
                                                                   basePath[i].second, pkgName) == OK) {
                    readConfigSucc = true;
                    foundConfig = true;
                }
            }
        }
        // 算子子包路径查找失败后再找旧有算子整包配置路径
        if (!foundConfig) {
            const std::string binaryInfoPath = binaryBasePath + "/binary_info_config.json";
            if (NnopbaseReadJsonConfig(binaryInfoPath, binaryInfoConfig) == OK &&
                NnopbaseCollectorReadDynamicKernelOpInfoConfig(collector, binaryInfoConfig, basePath[i].first,
                                                               basePath[i].second) == OK) {
                readConfigSucc = true;
            }
        }
    }
    if (readConfigSucc) {
        OP_LOGI("Get path and read binary_info_config.json successfully.");
        return OK;
    }
    return ACLNN_ERR_PARAM_INVALID;
}

aclnnStatus NnopbaseCollectorGetStaticBinaryInfo(NnopbaseBinCollector* const collector)
{
    bool getOpInfoSucc = false;
    const auto allOpBinaryDesc = nnopbase::OpBinaryResourceManager::GetInstance().GetAllOpBinaryDesc();
    for (auto iter = allOpBinaryDesc.begin(); iter != allOpBinaryDesc.end(); ++iter) {
        const nlohmann::json& opJsonDesc = iter->second;
        // 获取单算子jsonInfo
        for (auto binInfo : opJsonDesc["binList"]) {
            NnopbaseJsonInfo jsonInfo;
            jsonInfo.opType = iter->first.GetString();
            jsonInfo.loadBinInfoType = kStaticBinInfo;
            if (NnopbaseGetSimplifiedKey(binInfo, jsonInfo) != OK ||
                NnopbaseCollectorReadStaticBinJsonInfo(jsonInfo) != OK) {
                continue;
            }
            NNOPBASE_ASSERT_OK_RETVAL(
                NnopbaseCollectorAddRepoInfos(collector, jsonInfo, gert::OppImplVersionTag::kOpp));
            getOpInfoSucc = true;
        }
    }
    if (!getOpInfoSucc) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    return OK;
}

aclnnStatus NnopbaseCollectorWork(NnopbaseBinCollector* const collector)
{
    OP_LOGI("[NnopbaseCollector] Collector work start.");
    std::vector<std::pair<std::string, gert::OppImplVersionTag>> basePath;
    int32_t builtInStartIndex = -1;
    NnopbaseGetBasePath(collector, basePath, builtInStartIndex);
    RecordNnopbaseInitTime(collector, NnopbaseCollectorTimeIdx::kGetBasePathEnd);

    if (basePath.size() > 0) {
        (void)(NnopbaseLoadTilingSo(basePath));
    }
    RecordNnopbaseInitTime(collector, NnopbaseCollectorTimeIdx::kLoadTilingSoEnd);

    if (NnopbaseIsExceptionDumpEnable() || g_nnopbaseSysCfgParams.enableDebugKernel) {
        (void)NnopbaseCollectorGetDebugKernelPathAndReadConfig(collector);
    }
    RecordNnopbaseInitTime(collector, NnopbaseCollectorTimeIdx::kLoadDebugKernelEnd);

    (void)NnopbaseCollectorGetStaticKernelPathAndReadConfig(
        collector, nnopbase::utils::ThreadVarContainer::GetStaticKernelBasePathInThread());
    const aclnnStatus retForStaticBinaryInfo = NnopbaseCollectorGetStaticBinaryInfo(collector);
    if (retForStaticBinaryInfo != OK && basePath.empty()) {
        OP_LOGE_FOR_CONFIG_ERROR_INVALID_ENVIRONMENT_VARIABLE(
            "Reading binary_info_config.json file in the operator packages",
            "ASCEND_OPP_PATH or ASCEND_CUSTOM_OPP_PATH");
        return ACLNN_ERR_INNER_OPP_PATH_NOT_FOUND;
    }
    RecordNnopbaseInitTime(collector, NnopbaseCollectorTimeIdx::kLoadStaticKernelEnd);

    const aclnnStatus retForDynamicKernelInfo = NnopbaseCollectorGetDynamicKernelPathAndReadConfig(collector, basePath,
                                                                                                   builtInStartIndex);
    RecordNnopbaseInitTime(collector, NnopbaseCollectorTimeIdx::kLoadDynamicKernelEnd);
    OP_CHECK((retForStaticBinaryInfo == OK) || (retForDynamicKernelInfo == OK),
             OP_LOGE_FOR_FILE_OPERATION_ERROR_PARSE("binary_info_config.json", ERR_REASON_FOR_OPP_PACKAGE),
             return ACLNN_ERR_INNER_LOAD_JSON_FAILED);
    OP_LOGI("[NnopbaseCollector] Collector work end.");
    return OK;
}

aclnnStatus NnopbaseSetCollectorSocVersion(NnopbaseBinCollector* collector)
{
    const std::string socVersion = nnopbase::IndvSoc::GetInstance().GetCurSocVersion();
    OP_LOGI("Get current soc version: %s", socVersion.c_str());
    CHECK_COND(nnopbase::IndvSoc::GetInstance().SupportCurrentSoc(), ACLNN_ERR_PARAM_INVALID,
               "Not supported socVersion %s.", socVersion.c_str());

    collector->useCoreTypeMagic = nnopbase::IndvSoc::GetInstance().UseCoreTypeMagic();
    if (nnopbase::IndvSoc::GetInstance().SupportMc2FusionLaunch()) {
        collector->useCoreTypeMagic = true;
        collector->isMc2FusionLaunch = true;
    }
    return OK;
}

void NnopbaseGetOppApiPath(std::vector<std::string>& basePath)
{
    const NnopbaseChar* oppPathEnv = nullptr;
    MM_SYS_GET_ENV(MM_ENV_ASCEND_OPP_PATH, oppPathEnv);
    std::string oppPath;
    if (oppPathEnv != nullptr) {
        oppPath = oppPathEnv;
        const std::string configPath = oppPath + "/vendors/config.ini";
        std::vector<std::string> subPaths;

        if (NnopbaseReadConfigFile(configPath, subPaths)) {
            for (auto subPath : subPaths) {
                const std::string path = oppPath + "/vendors/" + subPath + "/op_api/lib/libcust_opapi.so";
                basePath.push_back(path);
                OP_LOGI("Add customize opp path %s.", path.c_str());
            }
        }
    }

    gert::OppImplVersionTag oppImplVersion = gert::OppImplVersionTag::kVersionEnd;
    std::string oppKernelBase = GetBuiltInBasePath(oppImplVersion);
    const NnopbaseChar* homePath = nullptr;
    MM_SYS_GET_ENV(MM_ENV_ASCEND_HOME_PATH, homePath);
    if (homePath != nullptr) {
        const std::string kernelPath = std::string(homePath);
        std::vector<NnopbaseChar> path(NNOPBASE_FILE_PATH_MAX_LEN, '\0');
        if (mmRealPath(kernelPath.c_str(), &(path[0U]), NNOPBASE_FILE_PATH_MAX_LEN) == EN_OK) {
            static std::vector<std::string> libopapiPaths = {
                "libopapi_math.so",  "libopapi_nn.so", "libopapi_cv.so", "libopapi_transformer.so", "libopapi_oam.so",
                "libopapi_legacy.so" // 低优先级
            };
            for (const auto& libopapiPath : libopapiPaths) {
                basePath.push_back(kernelPath + "/lib64/" + libopapiPath);
            }
        }
    }
    if (!oppKernelBase.empty()) {
        std::string osType = "linux";
        std::string cpuType = "x86_64";
        (void)(NnopbaseGetCurEnvPackageOsAndCpuType(osType, cpuType));
        const std::string path = oppKernelBase + "/built-in/op_impl/ai_core/tbe/op_api/lib/" + osType + "/" + cpuType +
                                 "/libopapi.so";
        basePath.push_back(path);
        OP_LOGI("Add opp kernel built-in base path %s.", path.c_str());
    }
}

void NnopbaseGetCustomOpApiPath(std::vector<std::string>& basePath)
{
    std::vector<std::string> customPaths;
    GetCustomVendorName(customPaths);
    for (const auto& customPath : customPaths) {
        if ((!customPath.empty()) && (mmIsDir((customPath).c_str()) == EN_OK)) {
            basePath.push_back(customPath + "/op_api/lib/libcust_opapi.so");
            OP_LOGI("Valid custom path '%s'.", customPath.c_str());
        } else {
            OP_LOGW("customPath '%s' is invalid, which is skipped.", customPath.c_str());
        }
    }
    OP_LOGI("Get CustomOppPath finished.");
}

#ifdef __cplusplus
}
#endif
