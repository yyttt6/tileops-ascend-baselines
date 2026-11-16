/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>
#include <limits>
#include <algorithm>
#include <nlohmann/json.hpp>

#include "register/op_impl_kernel_registry.h"
#include "platform/soc_spec.h"

#include "aclnn/acl_meta.h"
#include "opdev/format_utils.h"
#include "opdev/object.h"
#include "opdev/op_def.h"
#include "opdev/op_errno.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "platform/platform_info.h"
#include "block_pool.h"
#include "kernel_utils.h"
#include "tiling_parse_ctx_holder.h"
#include "thread_local_context.h"
#include "op_info_serialize.h"
#include "acl/acl_rt.h"
#include "nnopbase_error_msg.h"

namespace op::internal {

using Json = nlohmann::json;
using OpImplFunctions = gert::OpImplKernelRegistry::OpImplFunctions;

constexpr size_t K_DEFAULT_INPUT_SIZE = 3;
constexpr size_t K_DEFAULT_OUTPUT_SIZE = 1;

uint32_t CalcMixCoreNum(uint32_t cubeCoreNum, uint32_t vectorCoreNum, const Json& opJson)
{
    if (!opJson.contains("taskRation")) {
        OP_LOGW("MIX op has no taskRation");
        return vectorCoreNum;
    }

    uint32_t cubeRatio = 0;
    uint32_t vectorRatio = 0;
    constexpr int goodRes = 2;
    if (opJson["taskRation"].get<std::string>() != "tilingKey") {
        std::string ratio = opJson["taskRation"].get<std::string>();
        int rc = sscanf_s(ratio.c_str(), "%u:%u", &cubeRatio, &vectorRatio);
        if (rc != goodRes || (cubeRatio == 0 && vectorRatio == 0)) {
            OP_LOGW("MIX op taskRation incorrect. %s", ratio.c_str());
            return vectorCoreNum;
        }

        uint32_t cubeNum = (cubeRatio == 0) ? std::numeric_limits<uint32_t>::max() : (cubeCoreNum / cubeRatio);
        uint32_t vectorNum = (vectorRatio == 0) ? std::numeric_limits<uint32_t>::max() : (vectorCoreNum / vectorRatio);

        uint32_t ret = std::min(cubeNum, vectorNum);
        if (ret == 0) {
            OP_LOGW("MIX op wrong coreNum. ratio: %u:%u, cube:%u, vector:%u", cubeRatio, vectorRatio, cubeNum,
                    vectorNum);
            return 1;
        }

        return ret;
    } else {
        return vectorCoreNum;
    }
}

void UpdateThradLocalPlatformInfo(fe::PlatFormInfos* platformInfo, const uint32_t& coreNum, const uint32_t& cubeCoreNum,
                                  const uint32_t& vectorCoreNum)
{
    thread_local static uint32_t cubeCoreNumLast = -1;
    thread_local static uint32_t vectorCoreNumLast = -1;
    if ((cubeCoreNumLast != cubeCoreNum) || (vectorCoreNumLast != vectorCoreNum)) {
        const std::string socInfoKey = "SoCInfo";
        const std::string aicCntKey = "ai_core_cnt";
        const std::string vecCoreCntKey = "vector_core_cnt";
        const std::string cubeCoreCntKey = "cube_core_cnt";
        std::map<std::string, std::string> res;
        if (platformInfo->GetPlatformResWithLock(socInfoKey, res)) {
            res[aicCntKey] = std::to_string(cubeCoreNum);
            res[vecCoreCntKey] = std::to_string(vectorCoreNum);
            res[cubeCoreCntKey] = std::to_string(cubeCoreNum);
            platformInfo->SetPlatformResWithLock(socInfoKey, res);
            vectorCoreNumLast = vectorCoreNum;
            cubeCoreNumLast = cubeCoreNum;
        } else {
            OP_LOGD(
                "getPlatformResWithLock is false when updateThradLocalPlatformInfo, can't use thread control coreNum.");
        }
    }
    platformInfo->SetCoreNum(coreNum);
}

void SetCoreNum(const Json& opJson, fe::PlatFormInfos* platformInfo, uint32_t& coreNum)
{
    uint32_t cubeCoreNum = GetThreadLocalContext().opConfigInfo_.aicNum_;
    uint32_t vectorCoreNum = GetThreadLocalContext().opConfigInfo_.aivNum_;
    if (!opJson.contains("coreType")) {
        coreNum = cubeCoreNum;
        UpdateThradLocalPlatformInfo(platformInfo, coreNum, cubeCoreNum, vectorCoreNum);
        OP_LOGI("op json doesn't contain coreType, set core num: %u.", coreNum);
        return;
    }

    std::string coreType = opJson["coreType"].get<std::string>();
    NpuArch npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    OP_LOGI("current npu arch value: %u, core type: %s.", npuArch, coreType.c_str());
    if (coreType == "VectorCore") {
        coreNum = vectorCoreNum;
        OP_LOGI("Set VectorCore core num: %u", coreNum);
    } else if (coreType == "MIX") {
        coreNum = CalcMixCoreNum(cubeCoreNum, vectorCoreNum, opJson);
        OP_LOGI("Set MIX core num: %u", coreNum);
    } else if (npuArch == NpuArch::DAV_2002 && (coreType == "MIX_AIV" || coreType == "MIX_VECTOR_CORE")) {
        coreNum = vectorCoreNum + cubeCoreNum;
        OP_LOGI("Set MIX_AIV core num: %u", coreNum);
    } else {
        coreNum = cubeCoreNum;
        OP_LOGI("Set AiCore core num: %u. %s", coreNum, coreType.c_str());
    }
    UpdateThradLocalPlatformInfo(platformInfo, coreNum, cubeCoreNum, vectorCoreNum);
}

aclnnStatus TilingParseCtxHolder::BuildTilingParseCtx(uint32_t opType,
                                                      const gert::OpImplKernelRegistry::OpImplFunctions* tilingFuncs,
                                                      const Json& opJson, fe::PlatFormInfos* platformInfo,
                                                      const std::string& opImplModeStr,
                                                      const aclnnOpInfoRecord::OpKernelInfo& opKernelInfo)
{
    if (tilingFuncs == nullptr) {
        OP_LOGE_FOR_EXECUTION_TILING_ERROR("The tiling function does not exist");
        return ACLNN_ERR_RUNTIME_ERROR;
    }
    if (!opJson.contains("compileInfo")) {
        tilingParseInfo_.compileInfo_ = nullptr;
    } else {
        tilingParseInfo_.compileInfoStr_ = opJson["compileInfo"].dump();
        tilingParseInfo_.compileInfo_ = tilingParseInfo_.compileInfoStr_.c_str();
    }

    if (opJson.contains("binFileName")) {
        kernelName_ = opJson["binFileName"];
    }

    opTypeStr_ = OpTypeDict::ToString(opType).GetString();
    opImplModeStr_ = opImplModeStr;
    opKernelInfo_ = opKernelInfo;
    SetCoreNum(opJson, platformInfo, coreNum_);
    tilingParseInfo_.platformInfo_ = platformInfo;
    tilingParseInfo_.opType_ = opTypeStr_.c_str();

    std::size_t sz = sizeof(AsyncAnyValue) * static_cast<int>(kParseOutputNum);
    tilingParseCtxValue_ = static_cast<AsyncAnyValue*>(malloc(sz));
    CHECK_COND(tilingParseCtxValue_ != nullptr, ACLNN_ERR_INNER, "malloc failed. [%zu]", sz);

    (void)memset_s(tilingParseCtxValue_, sz, 0, sz);

    tilingParseCtxValue_[kCompileInfo].data.pointer = const_cast<char*>(tilingParseInfo_.compileInfo_);
    tilingParseCtxValue_[kPlatformInfo].data.pointer = tilingParseInfo_.platformInfo_;
    tilingParseCtxValue_[kOpType].data.pointer = const_cast<char*>(tilingParseInfo_.opType_);

    std::size_t tilingParseCtxSize = sizeof(AsyncAnyValue*) * static_cast<int>(kParseOutputNum) +
                                     sizeof(KernelRunContext);
    tilingParseCtx_ = static_cast<KernelRunContext*>(malloc(tilingParseCtxSize));
    CHECK_COND(tilingParseCtx_ != nullptr, ACLNN_ERR_INNER, "malloc failed. [%zu]", tilingParseCtxSize);

    (void)memset_s(tilingParseCtx_, tilingParseCtxSize, 0, tilingParseCtxSize);

    (void)memset_s(&dummyComputeNodeInfo_, sizeof(dummyComputeNodeInfo_), 0, sizeof(dummyComputeNodeInfo_));
    dummyComputeNodeInfo_.node_type_ = opTypeStr_.c_str();
    dummyComputeNodeInfo_.node_name_ = opTypeStr_.c_str();

    dummyKernelInfo_.kernel_name_ = kernelName_.c_str();
    dummyKernelInfo_.kernel_type_ = opTypeStr_.c_str();

    tilingParseCtx_->compute_node_info = &dummyComputeNodeInfo_;
    tilingParseCtx_->kernel_extend_info = &dummyKernelInfo_;
    tilingParseCtx_->input_size = K_DEFAULT_INPUT_SIZE;
    tilingParseCtx_->output_size = K_DEFAULT_OUTPUT_SIZE;
    tilingParseCtx_->values[kCompileInfo] = &tilingParseCtxValue_[kCompileInfo];
    tilingParseCtx_->values[kPlatformInfo] = &tilingParseCtxValue_[kPlatformInfo];
    tilingParseCtx_->values[kOpType] = &tilingParseCtxValue_[kOpType];
    tilingParseCtx_->values[kCompileInfoStruct] = &tilingParseCtxValue_[kCompileInfoStruct];
    tilingParseCtx_->output_start = tilingParseCtx_->values + tilingParseCtx_->input_size;

    auto compileInfoCreator = tilingFuncs->compile_info_creator;
    void* pStruct = nullptr;
    if (compileInfoCreator) {
        OP_LOGI("Op [%s] use compile info creator to alloc compile info struct", opTypeStr_.c_str());
        pStruct = compileInfoCreator();
        if (pStruct == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER, "compile_info_creator failed.");
            return ACLNN_ERR_INNER;
        }
        tilingParseInfoDeleter = tilingFuncs->compile_info_deleter;
    } else {
        pStruct = malloc(MAX_COMPILE_INFO_STRUCT_SIZE);
        CHECK_COND(pStruct != nullptr, ACLNN_ERR_INNER, "malloc failed");
        (void)memset_s(pStruct, MAX_COMPILE_INFO_STRUCT_SIZE, 0, MAX_COMPILE_INFO_STRUCT_SIZE);
    }
    tilingParseCtxValue_[kCompileInfoStruct].data.pointer = pStruct;
    tilingParseInfo_.compileInfoStruct_ = pStruct;

    OP_CHECK(tilingFuncs->tiling_parse != nullptr, OP_LOGW("Op [%s] has no tiling parse function", opTypeStr_.c_str()),
             return ACLNN_SUCCESS);

    auto rc = tilingFuncs->tiling_parse(PtrCastTo<gert::KernelContext>(tilingParseCtx_));
    CHECK_COND(rc == ACLNN_SUCCESS, ACLNN_ERR_INNER, "Op [%s] tiling compile info parse failed", opTypeStr_.c_str());
    OP_LOGI("Parse op [%s] tiling compile info succ", opTypeStr_.c_str());
    return ACLNN_SUCCESS;
}

const std::string& TilingParseCtxHolder::GetOpImplModeStr() const { return opImplModeStr_; }

const aclnnOpInfoRecord::OpKernelInfo* TilingParseCtxHolder::GetOpKernelInfo() const { return &opKernelInfo_; }

TilingParseCtxHolder::~TilingParseCtxHolder()
{
    if (tilingParseInfo_.compileInfoStruct_) {
        if (tilingParseInfoDeleter == nullptr) {
            free(tilingParseInfo_.compileInfoStruct_);
            tilingParseInfo_.compileInfoStruct_ = nullptr;
        }
    }
    FREE(tilingParseCtxValue_);
    FREE(tilingParseCtx_);
}

} // namespace op::internal
