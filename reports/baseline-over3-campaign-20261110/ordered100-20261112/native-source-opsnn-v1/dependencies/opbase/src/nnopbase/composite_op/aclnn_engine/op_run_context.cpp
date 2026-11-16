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
#include <iostream>
#include <memory>
#include <mutex>
#include <string>

#include "base/registry/op_impl_space_registry_v2.h"

#include "opp_resource_loader.h"
#include "kernel_utils.h"
#include "op_ctx_def.h"
#include "platform/platform_info.h"
#include "op_run_context.h"

namespace op::internal {

thread_local MemSetKernelContextHolder OpRunContextMgr::memSetCtx_;
thread_local OpRunContext OpRunContextMgr::opRunCtx_;
std::array<const gert::OpImplKernelRegistry::OpImplFunctions*, MAX_OP_TYPE_COUNT> OpRunContextMgr::opInferShapeFuncs_;
std::array<const gert::OpImplKernelRegistry::OpImplFunctions*, MAX_OP_TYPE_COUNT> OpRunContextMgr::opTilingFuncs_;

aclnnStatus OpRunContextMgr::InitOpFunctions(uint32_t opType)
{
    // This function may be called in phase 1 and phase 2 thread.
    op::opploader::LoadAllOppPackage();

    static std::once_flag flag[MAX_OP_TYPE_COUNT];
    aclnnStatus ret = ACLNN_SUCCESS;
    auto f = [&ret, &opType]() {
        opInferShapeFuncs_[0] = nullptr;
        opTilingFuncs_[0] = nullptr;
        ge::AscendString opTypeAscendStr = op::OpTypeDict::ToString(opType);
        const char* opTypeStr = opTypeAscendStr.GetString();
        gert::OppImplVersionTag oppVersionTag = GetOppImplVersion();
        auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry(oppVersionTag);
        OP_CHECK(spaceRegistry != nullptr,
                 OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "SpaceRegistry is nullptr, op type is %s", opTypeStr),
                 ret = ACLNN_ERR_INNER_NULLPTR;
                 return);

        auto funcs = spaceRegistry->GetOpImpl(opTypeStr);
        // ops infershape func
        opInferShapeFuncs_[opType] = funcs;
        if (funcs == nullptr || funcs->infer_shape == nullptr) {
            OP_LOGW("Op %s has no infer shape function, is funcs nullptr: %d", opTypeStr, funcs == nullptr);
        }
        // ops has tiling func
        OP_CHECK(funcs == nullptr || funcs->tiling_parse == nullptr || funcs->tiling == nullptr,
                 opTilingFuncs_[opType] = funcs;
                 OP_LOGI("Op %s has tiling and tiling parse funcs", opTypeStr), return);
        // ops use auto tiling func
        auto defaultFuncs = spaceRegistry->GetOpImpl("DefaultImpl");
        OP_CHECK(defaultFuncs == nullptr || defaultFuncs->tiling_parse == nullptr || defaultFuncs->tiling == nullptr,
                 opTilingFuncs_[opType] = defaultFuncs;
                 OP_LOGI("Op %s use auto tiling func", opTypeStr), return);

        OP_LOGE_FOR_EXECUTION_TILING_ERROR("The tiling function does not exist");
        ret = ACLNN_ERR_PARAM_NULLPTR;
    };

    std::call_once(flag[opType], f);
    return ret;
}

thread_local fe::PlatFormInfos SocContext::platformInfo_;

aclnnStatus SocContext::SetupPlatformInfo()
{
    int32_t deviceId;
    auto ret = aclrtGetDevice(&deviceId);
    CHECK_COND(ret == ACL_SUCCESS, ACLNN_ERR_INNER, "aclrtGetDevice failed, %d", ret);
    auto getPlatformRet = fe::PlatformInfoManager::GeInstance().GetRuntimePlatformInfosByDevice(deviceId, platformInfo_,
                                                                                                true);
    if (getPlatformRet != 0) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "Call PlatformInfoManager::GeInstance().GetPlatformInstanceByDevice failed.");
        return ACLNN_ERR_INNER;
    }
    return ACLNN_SUCCESS;
}

} // namespace op::internal
