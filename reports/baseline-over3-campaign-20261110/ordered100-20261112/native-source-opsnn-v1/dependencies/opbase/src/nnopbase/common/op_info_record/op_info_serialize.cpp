/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "op_info_serialize.h"
#include "dump/adump_pub.h"
#include "mmpa/mmpa_api.h"
#include <set>
#include <mutex>
#include <fstream>
#include "opdev/op_log.h"
#include "opdev/op_errno.h"
#include "platform/platform_info.h"
#include "tiling_context_to_json.h"
#include "ini_parse.h"

namespace aclnnOpInfoRecord {
enum class OpInfoType { OP_INFO = 0, OP_INFO_FOR_DEBUG, OP_INFO_FOR_OPCOMPILE };

namespace {
std::mutex g_opInfoStatisticsLck;
std::set<nlohmann::json> g_opInfoStatistics;
std::set<nlohmann::json> g_opInfoStatisticsDebug;
std::set<nlohmann::json> g_opInfoStatisticsOpcompile;
constexpr char FILE_TYPE_SPLIT_CHAR = '.';

std::set<std::string> g_opTypeWhiteList = {
    "FlashAttentionScore",
    "FlashAttentionScoreGrad",
    "MatMulV2",
    "MatMul",
    "BatchMatMulV2",
    "BatchMatMul",
    "Transpose",
    "TransData",
    "GeluGrad",
    "Gelu",
    "FastGelu",
    "FastGeluGrad",
    "SoftmaxV2",
    "SoftmxGrad",
    "LayerNorm",
    "LayerNormXBackpropV2",
    "LayerNormXBackpropV3",
    "LayerNormBetaGammaBackpropV2",
    "LayerNormV3",
    "Renorm",
};

std::set<std::string> g_opTypeBlackList = {
    "ReduceSum", "ReduceMin",       "ReduceMax", "ArgMinWithValue", "FusedInferAttentionScore",
    "GatherV2",  "ArgMaxWithValue", "Range"};

constexpr int JSON_INDENT = 2;

int32_t ReadJsonInfo(const std::string& binaryInfoPath, nlohmann::json& binaryInfoConfig)
{
    char opInfoPath[MMPA_MAX_PATH] = {0};
    const int32_t ret = mmRealPath(binaryInfoPath.c_str(), opInfoPath, MMPA_MAX_PATH);
    if (ret != EN_OK) {
        OP_LOGW("Get jsonfile path for %s failed.", binaryInfoPath.c_str());
        return ACLNN_ERR_INNER_LOAD_JSON_FAILED;
    }
    OP_LOGI("Reading ascendc operator jsonfile path: %s", opInfoPath);
    std::ifstream ifs(opInfoPath);
    if (!ifs.is_open()) {
        return ACLNN_ERR_INNER_LOAD_JSON_FAILED;
    }
    try {
        ifs >> binaryInfoConfig;
        ifs.close();
    } catch (const nlohmann::json::exception& e) {
        OP_LOGE(ACLNN_ERR_INNER_LOAD_JSON_FAILED, "Read jsonfile [%s] failed, reason %s.", opInfoPath, e.what());
        ifs.close();
        return ACLNN_ERR_INNER_LOAD_JSON_FAILED;
    }
    OP_LOGI("Read jsonfile [%s] successfully.", opInfoPath);
    return 0;
}

std::string GetFirstShapeStr(const nlohmann::json& j)
{
    if (j.is_null()) {
        return "";
    }
    nlohmann::json tmpJ = j;
    if (j.is_array()) {
        if (j.size() < 1) {
            return "";
        }
        if (j[0].is_null()) {
            return "";
        }
        tmpJ = j[0];
    }
    OP_LOGD("[%s] try to get first shape str!", j.dump().c_str());
    std::string str = tmpJ["dtype"].get<std::string>() + "_" + tmpJ["format"].get<std::string>() + "_";
    for (size_t idx = 0UL; idx < tmpJ["shape"].size(); idx++) {
        str += std::to_string(tmpJ["shape"][idx].get<int64_t>()) + "_";
    }
    return str;
}

int32_t DumpJsonToFile(const nlohmann::json& opJson, const OpInfoType type)
{
    static uint32_t cnt = 0;
    static uint32_t cntDebug = 0;
    static uint32_t cntOpcompile = 0;
    if (opJson.is_null()) {
        return 0;
    }
    // input 首节点为dynamic需要处理
    if (!opJson["inputs"].is_array()) {
        OP_LOGE(ACLNN_ERR_INNER, "DumpJsonToFile cannot record node[%s] because the input quantity is null!",
                opJson["op_type"].get<std::string>().c_str());
        return 0;
    }
    std::string fileName = opJson["op_type"].get<std::string>() + "_";
    if (opJson["inputs"].size() >= 1) {
        fileName += GetFirstShapeStr(opJson["inputs"][0]);
    }
    if (type == OpInfoType::OP_INFO) {
        fileName += std::to_string(cnt++) + ".json";
        fileName = std::to_string(getpid()) + "/" + fileName;
    } else if (type == OpInfoType::OP_INFO_FOR_DEBUG) {
        fileName += std::to_string(cntDebug++) + ".json";
        fileName = std::to_string(getpid()) + "_debug/" + fileName;
    } else {
        fileName += std::to_string(cntOpcompile++) + ".json";
        fileName = std::to_string(getpid()) + "_opcompile/" + fileName;
    }
    OP_LOGI("DumpJsonToFile start to record json[%s]!", fileName.c_str());
    std::string jStr = opJson.dump(JSON_INDENT);
    if (Adx::AdumpSaveToFile(jStr.c_str(), jStr.size(), fileName.c_str(), Adx::SaveType::OVERWRITE) != 0) {
        OP_LOGE(ACLNN_ERR_INNER, "DumpJsonToFile dump node[%s] json file failed!",
                opJson["op_type"].get<std::string>().c_str());
        return -1;
    }

    OP_LOGI("DumpJsonToFile json[%s] success!", fileName.c_str());
    return 0;
}

int32_t DumpJson(const std::set<nlohmann::json>& gOpInfoStatistics, const OpInfoType type)
{
    int32_t ret = 0;
    int32_t tmpRes = 0;
    for (const auto& item : gOpInfoStatistics) {
        try {
            tmpRes = DumpJsonToFile(item, type);
        } catch (nlohmann::json::exception& e) {
            ret = -1;
        }
        if (ret == 0 && tmpRes == -1) {
            ret = -1;
        }
    }
    return ret;
}

void AddPlatformInfoToJson(nlohmann::json& opJson, const gert::TilingContext* ctx)
{
    auto platformInfo = ctx->GetPlatformInfo();
    if (platformInfo != nullptr) {
        const std::string socInfoKey = "SoCInfo";
        const std::string vecCoreCntKey = "vector_core_cnt";
        const std::string cubeCoreCntKey = "cube_core_cnt";
        std::map<std::string, std::string> res;
        if (platformInfo->GetPlatformResWithLock(socInfoKey, res)) {
            // 不存在异常string转化int失败的场景, 依赖组件的fe::PlatFormInfos在保存值时已做约束
            opJson["platform_info"]["vector_core_cnt"] = std::stoi(res[vecCoreCntKey]);
            opJson["platform_info"]["cube_core_cnt"] = std::stoi(res[cubeCoreCntKey]);
            OP_LOGI("Save coreNum[%d, %d] to json.", opJson["platform_info"]["cube_core_cnt"].get<int32_t>(),
                    opJson["platform_info"]["vector_core_cnt"].get<int32_t>());
        }
    }
}

void AddJsonToOpInfoCompile(const nlohmann::json& opJson)
{
    if (g_opTypeBlackList.find(opJson["op_type"]) == g_opTypeBlackList.cend()) {
        g_opInfoStatisticsOpcompile.emplace(opJson);
    } else {
        OP_LOGD("TilingContextToJson node type[%s] is in black list!", opJson["op_type"].get<std::string>().c_str());
    }
}

void AddJsonToOpInfo(const nlohmann::json& opJson)
{
    if (g_opTypeWhiteList.find(opJson["op_type"]) == g_opTypeWhiteList.cend()) {
        OP_LOGD("TilingContextToJson node type[%s] not in white list!", opJson["op_type"].get<std::string>().c_str());
    } else {
        g_opInfoStatistics.emplace(opJson);
    }
}

void AddJsonToOpInfoDebug(const nlohmann::json& opJson) { g_opInfoStatisticsDebug.emplace(opJson); }

} // namespace

int32_t OpInfoDump(void)
{
    int32_t ret = 0;
    std::lock_guard<std::mutex> lck(g_opInfoStatisticsLck);
    OP_LOGI("Op info record start to dump %zu json!", g_opInfoStatistics.size());
    ret = DumpJson(g_opInfoStatistics, OpInfoType::OP_INFO);
    g_opInfoStatistics.clear();
    OP_LOGI("Op info record start to dump %zu op compile json!", g_opInfoStatisticsOpcompile.size());
    ret = DumpJson(g_opInfoStatisticsOpcompile, OpInfoType::OP_INFO_FOR_OPCOMPILE);
    g_opInfoStatisticsOpcompile.clear();
    OP_LOGI("Op info record start to dump %zu debug json!", g_opInfoStatisticsDebug.size());
    ret = DumpJson(g_opInfoStatisticsDebug, OpInfoType::OP_INFO_FOR_DEBUG);
    g_opInfoStatisticsDebug.clear();
    return ret;
}

/*
 * @brief: Extract Op Info from TilingContext and complete serialization.
 * @param [in] ctx: TilingContext
 * @return int32_t: 0 == SUCCESS, others is FAILED
 */
int32_t OpInfoSerialize(const gert::TilingContext* ctx, const aclnnOpInfoRecord::OpCompilerOption& opt,
                        const aclnnOpInfoRecord::OpKernelInfo* kernelInfo)
{
    OP_CHECK(ctx != nullptr, OP_LOGE(ACLNN_ERR_INNER, "OpInfoSerialize ctx is nullptr!"), return -1);
    OP_CHECK(kernelInfo != nullptr, OP_LOGE(ACLNN_ERR_INNER, "OpInfoSerialize kernelInfo is nullptr!"), return -1);
    IniParse iniInstance;
    std::map<std::string, std::string> iniConfigMap;
    if (iniInstance.GetIniParams(iniConfigMap) != 0) {
        OP_LOGE(ACLNN_ERR_INNER, "Get ini config map error.");
        return -1;
    }
    try {
        nlohmann::json builtInJsonConfig;
        const auto pos = kernelInfo->bin_info.rfind(FILE_TYPE_SPLIT_CHAR);
        if (pos == std::string::npos) {
            OP_LOGE(ACLNN_ERR_INNER, "Bin info is not a complete file path.");
            return -1;
        }
        std::string builtInJsonPath = kernelInfo->bin_info.substr(0, pos) + ".json";
        if (ReadJsonInfo(builtInJsonPath, builtInJsonConfig) != 0) {
            OP_LOGE(ACLNN_ERR_INNER, "Read json Info: %s failed.", builtInJsonPath.c_str());
            return -1;
        }
        if (!builtInJsonConfig.contains("supportInfo")) {
            OP_LOGE(ACLNN_ERR_INNER, "Json %s does not contains supportInfo keyword.", builtInJsonPath.c_str());
            return -1;
        }
        nlohmann::json jsonDebug = TilingContextToJson(ctx, iniConfigMap, builtInJsonConfig["supportInfo"],
                                                       kernelInfo->isMc2);
        if (jsonDebug.is_null()) {
            return 0;
        }
        AddPlatformInfoToJson(jsonDebug, ctx);
        jsonDebug["impl_mode"] = opt.impl_mode;
        jsonDebug["deterministic"] = (opt.deterministic == 0U) ? "false" : "true";
        jsonDebug["deterministic_level"] = opt.deterministic;
        std::lock_guard<std::mutex> lck(g_opInfoStatisticsLck);
        AddJsonToOpInfoCompile(jsonDebug); // dump json for compile, not needed "bin_type" and "bin_info"
        jsonDebug["bin_type"] = kernelInfo->bin_type;
        jsonDebug["bin_info"] = kernelInfo->bin_info;
        AddJsonToOpInfo(jsonDebug);
        AddJsonToOpInfoDebug(jsonDebug);
    } catch (nlohmann::json::exception& e) {
        OP_LOGE(ACLNN_ERR_INNER, "OpInfoSerialize json exception:%s!", e.what());
        return -1;
    }
    return 0;
}
} // namespace aclnnOpInfoRecord
