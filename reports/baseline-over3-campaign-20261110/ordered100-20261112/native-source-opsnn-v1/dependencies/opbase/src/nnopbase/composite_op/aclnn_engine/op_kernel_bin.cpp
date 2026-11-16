/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <sstream>
#include <iomanip>
#include <string>
#include "op_kernel.h"
#include "opdev/op_dfx.h"
#include "dump/adump_api.h"
#include "dump/adump_pub.h"
#include "utils/string_utils.h"
namespace op {
namespace internal {
constexpr const char* SIZE = "size";
constexpr const char* BLOCK_DIM = "blockDim";
constexpr const char* SUPPORT_INFO = "supportInfo";
constexpr const char* IMPL_MODE = "implMode";
constexpr const char* OP_DEBUG_CONFIG = "op_debug_config";
constexpr const char* ORI_OP_PARA_SIZE = "oriOpParaSize";
constexpr const char* DEBUG_OPTIONS = "debugOptions";
constexpr const char* DEBUG_BUF_SIZE = "debugBufSize";
constexpr const char* DEBUG_OPTIONS_ASSERT = "assert";
constexpr const char* DEBUG_OPTIONS_PRINTF = "printf";
constexpr const char* DYNAMIC_PARAM_MODE = "dynamicParamMode";
constexpr const char* FOLDED_WITH_DESC = "folded_with_desc";
constexpr const char* DYN_UBUF_SIZE = "localMemorySize";
constexpr int32_t MAX_BLOCK_DIM = 65535;
constexpr int32_t ONE_BYTE_FOR_HEX = 2;

aclnnStatus OpKernelBin::ParseStaticWorkSpace(const nlohmann::json& workspaceJson)
{
    try {
        auto sizeIter = workspaceJson.find(SIZE);
        if (sizeIter != workspaceJson.end()) {
            staticWorkspaceSize_ = sizeIter->get<FVector<size_t>>();
        } else {
            OP_LOGE(ACLNN_ERR_INNER_STATIC_WORKSPACE_INVALID, "workspace does not contain [size].");
            return ACLNN_ERR_INNER_STATIC_WORKSPACE_INVALID;
        }
    } catch (const nlohmann::json::exception& e) {
        OP_LOGE(ACLNN_ERR_INNER_STATIC_WORKSPACE_INVALID, "Parse [size] failed, exception:%s.", e.what());
        return ACLNN_ERR_INNER_STATIC_WORKSPACE_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus OpKernelBin::ParseStaticBlockdim(const nlohmann::json& objJson)
{
    try {
        auto blockDimIter = objJson.find(BLOCK_DIM);
        if (blockDimIter != objJson.end()) {
            staticBlockDim_ = blockDimIter->get<int64_t>();
        } else {
            OP_LOGE(ACLNN_ERR_INNER_STATIC_BLOCK_DIM_INVALID, "Static json does not contain [blockDim].");
            return ACLNN_ERR_INNER_STATIC_BLOCK_DIM_INVALID;
        }
    } catch (const nlohmann::json::exception& e) {
        OP_LOGE(ACLNN_ERR_INNER_STATIC_BLOCK_DIM_INVALID, "Parse [blockDim] failed, exception:%s.", e.what());
        return ACLNN_ERR_INNER_STATIC_BLOCK_DIM_INVALID;
    }
    if ((staticBlockDim_ < 0) || (staticBlockDim_ > MAX_BLOCK_DIM)) {
        OP_LOGE(ACLNN_ERR_INNER_STATIC_BLOCK_DIM_INVALID, "blockDim[%ld] is out of range[0, 65535].", staticBlockDim_);
        return ACLNN_ERR_INNER_STATIC_BLOCK_DIM_INVALID;
    }
    OP_LOGD("Static block dim is %ld", staticBlockDim_);
    return ACLNN_SUCCESS;
}

void OpKernelBin::ParseStaticImplMode(const nlohmann::json& objJson)
{
    auto supportInfoIter = objJson.find(SUPPORT_INFO);
    if (supportInfoIter == objJson.end()) {
        OP_LOGW("Static json can not find supportInfo in %s", jsonPath_.c_str());
        return;
    }
    try {
        auto implModeIter = supportInfoIter->find(IMPL_MODE);
        if (implModeIter != supportInfoIter->end()) {
            staticImplMode_ = implModeIter->get<std::string>();
        } else {
            OP_LOGW("Static json can not implMode in %s", jsonPath_.c_str());
            return;
        }
    } catch (const nlohmann::json::exception& e) {
        OP_LOGW("Static json parse implMode fail in %s", jsonPath_.c_str());
        return;
    }
}

void OpKernelBin::ParseStaticDevPtrMode(const nlohmann::json& objJson)
{
    auto dynamicParamMode = objJson.find(DYNAMIC_PARAM_MODE);
    if (dynamicParamMode == objJson.end()) {
        return;
    }
    const auto& value = dynamicParamMode->get<std::string>();
    if (value == FOLDED_WITH_DESC) {
        hasDevPtrArg_ = true;
    }
    OP_LOGI("parse static kernel dev ptr mode : %d", hasDevPtrArg_);
}

void OpKernelBin::ParseStaticDynUBufSize(const nlohmann::json& objJson)
{
    if (!objJson.contains(DYN_UBUF_SIZE)) {
        OP_LOGW("Static kernel json [%s] does not have localMemorySize", jsonPath_.c_str());
        staticKernelDynUBufSize_ = 0;
        return;
    }
    if (objJson[DYN_UBUF_SIZE].is_null()) {
        OP_LOGW("localMemorySize value is null in %s", jsonPath_.c_str());
        staticKernelDynUBufSize_ = 0;
        return;
    }
    try {
        staticKernelDynUBufSize_ = objJson[DYN_UBUF_SIZE].get<uint32_t>();
    } catch (const nlohmann::json::exception& e) {
        OP_LOGW("parse localMemorySize failed in %s, reason: %s", jsonPath_.c_str(), e.what());
        staticKernelDynUBufSize_ = 0;
        return;
    }
    OP_LOGI("Parse static kernel dynUBufSize(localMemorySize): %u", staticKernelDynUBufSize_);
}

void OpKernelBin::ParseOpDebugConfig(const nlohmann::json& objJson)
{
    static std::map<std::string, OpDebugConfig> opDebugConfigMap = {
        {"oom", OpDebugConfig::OOM},           {"dump_bin", OpDebugConfig::DUMP_BIN},
        {"dump_cce", OpDebugConfig::DUMP_CCE}, {"dump_loc", OpDebugConfig::DUMP_LOC},
        {"ccec_O0", OpDebugConfig::CCEC_O0},   {"ccec_g", OpDebugConfig::CCEC_G}};

    auto supportInfoIter = objJson.find(SUPPORT_INFO);
    if (supportInfoIter == objJson.end()) {
        OP_LOGW("Static json can not find supportInfo in %s", jsonPath_.c_str());
        return;
    }
    try {
        auto debugConfig = supportInfoIter->find(OP_DEBUG_CONFIG);
        if (debugConfig == supportInfoIter->end() || !debugConfig->is_string()) {
            OP_LOGW("json can not parse op_debug_config in %s", jsonPath_.c_str());
            return;
        }
        const std::string debugConfigStr = debugConfig->get<std::string>();
        std::stringstream ss(debugConfigStr);
        std::string token;
        while (std::getline(ss, token, ',')) {
            auto it = opDebugConfigMap.find(token);
            if (it != opDebugConfigMap.end()) {
                debugConfig_ |= static_cast<uint32_t>(it->second);
            }
        }
        OP_LOGI("%s debug config is %u", jsonPath_.c_str(), debugConfig_);
    } catch (const nlohmann::json::exception& e) {
        OP_LOGW("json can not parse op_debug_config in %s", jsonPath_.c_str());
        return;
    }
}

void OpKernelBin::ParseOriOpParaSize(const nlohmann::json& objJson)
{
    try {
        auto paraSize = objJson.find(ORI_OP_PARA_SIZE);
        if (paraSize != objJson.end() && paraSize->is_number()) {
            oriOpParaSize_ = paraSize->get<uint32_t>();
            OP_LOGI("%s oriOpParaSize is %u", jsonPath_.c_str(), oriOpParaSize_);
        } else {
            OP_LOGW("json can not parse oriOpParaSize in %s", jsonPath_.c_str());
            return;
        }
    } catch (const nlohmann::json::exception& e) {
        OP_LOGW("json can not parse oriOpParaSize in %s", jsonPath_.c_str());
        return;
    }
}

void OpKernelBin::ParseKernelDfxConfig(const nlohmann::json& objJson)
{
    auto debugOptionsConfig = objJson.find(DEBUG_OPTIONS);
    if (debugOptionsConfig == objJson.end() || !debugOptionsConfig->is_string()) {
        OP_LOGI("Dont enable kernel dfx.");
        return;
    }
    const std::string debugOptionsConfigStr = debugOptionsConfig->get<std::string>();
    if (debugOptionsConfigStr.find(DEBUG_OPTIONS_ASSERT) != string::npos) {
        kernelDfxType_ |= static_cast<uint32_t>(KernelDfxType::ASSERT);
    }
    if (debugOptionsConfigStr.find(DEBUG_OPTIONS_PRINTF) != string::npos) {
        kernelDfxType_ |= static_cast<uint32_t>(KernelDfxType::PRINTF);
    }

    auto debugOptionsBufSize = objJson.find(DEBUG_BUF_SIZE);
    if (debugOptionsBufSize == objJson.end() || !debugOptionsBufSize->is_number()) {
        OP_LOGW("Have debugOptions, but don't have debugBufSize.");
        kernelDfxType_ = static_cast<uint32_t>(KernelDfxType::NO_DFX);
        return;
    }
    kernelDfxBufSize_ = static_cast<uint64_t>(debugOptionsBufSize->get<uint32_t>());
    OP_LOGI("kernel dfx config is %u, buf size is %lu", kernelDfxType_, kernelDfxBufSize_);
}

bool OpKernelBin::IsAssertEnable() const { return kernelDfxType_ & static_cast<uint32_t>(KernelDfxType::ASSERT); }

bool OpKernelBin::IsPrintFEnable() const { return kernelDfxType_ & static_cast<uint32_t>(KernelDfxType::PRINTF); }

uint64_t OpKernelBin::GetKernelDfxBufSize() const { return kernelDfxBufSize_; }

void OpKernelBin::DumpWorkspaceData(aclrtStream stream, OpArgContext* args) const
{
    OP_CHECK(args != nullptr && args->ContainsOpArgType(op::OP_WORKSPACE_ARG), OP_LOGW("Dont has workspace"), return);
    auto& argList = *args->GetOpArg(op::OP_WORKSPACE_ARG);
    OP_CHECK((argList.count == 1), OP_LOGW("workspace must has only one value."), return);
    auto& arg = argList[0];

    OP_CHECK((arg.type == OpArgType::OPARG_ACLTENSOR_LIST), OP_LOGW("workspace should be tensor list."), return);
    aclTensorList* tensors = PtrCastTo<aclTensorList>(arg->pointer);

    OP_CHECK(tensors != nullptr && tensors->Size() > 0, OP_LOGW("workspace at least has one."), return);
    const aclTensor* firstWorkspace = (*tensors)[0];

    Adx::AdumpPrintWorkSpace(firstWorkspace->GetData(), kernelDfxBufSize_, stream,
                             op::OpTypeDict::ToString(opType_).GetString());
    OP_LOGI("AdumpPrintWorkspace for %p, %lu, %s", firstWorkspace->GetData(), kernelDfxBufSize_,
            op::OpTypeDict::ToString(opType_).GetString());
}

static void PrintHex(const uint8_t* p, size_t num, std::stringstream& ss)
{
    for (size_t i = 0; i < num; ++i) {
        // 通过std::setw设置输出位宽为2倍的sizeof(T)
        ss << "0x" << std::setfill('0') << std::setw(ONE_BYTE_FOR_HEX) << std::hex << +p[i] << ' ';
    }
}

void OpKernelBin::SetExceptionDumpInfo(uint32_t numBlocks, uint64_t tilingKey, void* tilingData, size_t tilingSize)
{
#if defined(NNOPBASE_UT) || defined(NNOPBASE_ST)
#else
    if (!IsExceptionDumpEnable()) {
        return;
    }
#endif
    GetThreadLocalContext().exceptionDumpInfo_.numBlocks_ = std::to_string(numBlocks);
    GetThreadLocalContext().exceptionDumpInfo_.tilingKey_ = std::to_string(tilingKey);
    GetThreadLocalContext().exceptionDumpInfo_.tilingData_ = "";
    if (tilingData != nullptr && tilingSize > 0) {
        std::stringstream tilingDataStr;
        PrintHex(PtrCastTo<uint8_t>(tilingData), tilingSize, tilingDataStr);
        GetThreadLocalContext().exceptionDumpInfo_.tilingData_ = tilingDataStr.str();
    }
    GetThreadLocalContext().exceptionDumpInfo_.magic_ = "";
    auto& opJson = binJson_.GetVar();
    if (opJson.contains("magic")) {
        GetThreadLocalContext().exceptionDumpInfo_.magic_ = opJson["magic"].get<std::string>();
    }
    GetThreadLocalContext().exceptionDumpInfo_.devFunc_ = "";
    std::vector<std::string> parts = SplitWith(binPath_, '/');
    if (parts.size() == 0) {
        return;
    }
    std::string kernelFile = parts.back();
    auto pos = kernelFile.find(BIN_SUFFIX);
    if (pos == string::npos) {
        return;
    }
    GetThreadLocalContext().exceptionDumpInfo_.devFunc_ = kernelFile.substr(0, pos);
}
} // namespace internal
} // namespace op