/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __ACL_KERNEL_MGR_H__
#define __ACL_KERNEL_MGR_H__

#include <array>
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>
#include <mutex>

#include "nlohmann/json.hpp"
#include "acl/acl_base_rt.h"

#include "aclnn/acl_meta.h"
#include "opdev/op_arg_def.h"
#include "opdev/op_def.h"
#include "opdev/op_log.h"
#include "kernel_utils.h"
#include "common_utils.h"
#include "aclnn_engine/op_ctx_def.h"
#include "aclnn_engine/op_run_context.h"
#include "aclnn_engine/kernel_arg.h"
#include "aclnn_engine/op_kernel.h"
#include "aclnn_engine/tilingctx_builder.h"

namespace op {
namespace internal {
class KernelMgr {
public:
    static KernelMgr& getInstance()
    {
        // Warning: not thread-safe, use muxtex in multitheading environment
        static KernelMgr instance;
        return instance;
    }

    OpKernel* GetKernel(uint32_t opType);

    aclnnStatus Run(uint32_t opType, aclrtStream stream, OpArgContext* opArgCtx)
    {
        aclnnStatus rc = AclOpKernelInit(opType);
        if (rc != ACLNN_SUCCESS) {
            OP_LOGE(rc, "AclOpKernelInit failed, opType: %s.", op::OpTypeDict::ToString(opType).GetString());
            return rc;
        }

        OpKernel* kernel = GetKernel(opType);
        if (kernel == nullptr) {
            OP_LOGE(rc, "Kernel Not Found. opType: %u, %s", opType, op::OpTypeDict::ToString(opType).GetString());
            return ACLNN_ERR_INNER;
        }
        rc = kernel->Run(stream, opArgCtx);
        if (rc != ACLNN_SUCCESS) {
            OP_LOGE(rc, "Kernel Run failed. opType: %u, %s", opType, op::OpTypeDict::ToString(opType).GetString());
            return rc;
        }

        return ACLNN_SUCCESS;
    }

    aclnnStatus InferShape(uint32_t opType, OpArgList& inputs, OpArgList& outputs, OpArgList& attrs)
    {
        auto rc = OpRunContextMgr::InferShape(opType, inputs, outputs, attrs);
        CHECK_RET_CODE(rc, "Infer Shape failed.");
        return ACLNN_SUCCESS;
    }

    aclnnStatus GetWorkspace(uint32_t opType, size_t const*& size, size_t& num, OpArgList& inputs, OpArgList& outputs,
                             OpArgList& attrs)
    {
        OP_LOGI("Begin GetWorkspace, opType: %u", opType);
        aclnnStatus ret = AclOpKernelInit(opType);
        if (ret != ACLNN_SUCCESS) {
            OP_LOGE(ret, "AclOpKernelInit failed, opType: %s.", op::OpTypeDict::ToString(opType).GetString());
            return ret;
        }

        OpKernel* kernel = GetKernel(opType);
        if (kernel == nullptr) {
            OP_LOGE(ret, "Kernel Not Found. opType: %u", opType);
            return ACLNN_ERR_INNER;
        }
        ret = kernel->GetWorkspace(size, num, inputs, outputs, attrs);
        if (ret != ACLNN_SUCCESS) {
            OP_LOGE(ret, "Kernel GetWorkspace failed. opType: %u", opType);
            return ret;
        }
        return ACLNN_SUCCESS;
    }

    aclnnStatus AclOpKernelInit(uint32_t opType);

    aclnnStatus SelectMemsetOpBin(size_t inputNum, OpKernelBin*& opBin);

    void ReleaseTilingParse()
    {
        for (size_t i = 0; i < kernel_.size(); i++) {
            kernel_[i].ReleaseTilingParse();
        }
    }

    void ReloadStaticBinJson(const char* basePath = nullptr)
    {
        if (basePath != nullptr) {
            customStaticKernelBasePath_ = basePath;
        } else {
            customStaticKernelBasePath_.clear();
        }
        loadStaticBinJsonFlag_.Reset();
        for (size_t i = 0; i < kernel_.size(); i++) {
            initStaticKernelFlags_[i].Reset();
        }
    }

private:
    KernelMgr(){};

    // 由 aclnnReselectStaticKernelWithPath 入口 RealPath 校验后写入；空串表示用内置 OPP 路径；
    // LoadStaticBinJson 读取后立即清空（一次性消费）
    std::string customStaticKernelBasePath_;

    aclnnStatus ParseStaticKernelConfig(uint32_t opType);

    aclnnStatus ParseDynamicKernelInStaticLib(const string& configFileName);
    aclnnStatus ParseDynamicKernelConfig(const std::vector<std::string>& configFileNames);
    aclnnStatus ParseDynamicKernelConfig(const std::string& configDir, const std::string& binJsonDir,
                                         const std::string& configFileName, const std::string& opsRepoName, bool debug);
    aclnnStatus ParseDynamicKernelConfig(const std::vector<std::string>& configDir,
                                         const std::vector<std::string>& binJsonDir,
                                         const std::vector<std::string>& configFileNames);

    aclnnStatus ParseDynamicKernels(uint32_t opType);
    aclnnStatus ParseStaticKernels(uint32_t opType);

    aclnnStatus LoadStaticBinJson();
    aclnnStatus LoadDebugStaticBinJson();
    void GetDirPath();
    void GetConfigJsonOpsFolders();
    aclnnStatus Prepare();

    op::internal::ResettableOnceFlag loadStaticBinJsonFlag_;
    std::array<std::once_flag, MAX_OP_TYPE_COUNT> initDynKernelFlags_;
    std::array<op::internal::ResettableOnceFlag, MAX_OP_TYPE_COUNT> initStaticKernelFlags_;
    std::array<OpKernel, MAX_OP_TYPE_COUNT> kernel_;
    std::vector<std::string> customConfigDirs_;
    std::vector<std::string> customBinAndJsonDirs_;
    std::vector<std::string> configJsonOpsFolders_;
    std::string builtInConfigDir_;
    std::string builtInBinAndJsonDir_;
    std::string staticBinAndJsonDir_;
    nlohmann::json staticConfigJson_;

    std::string debugConfigDir_;
    std::string debugDynBinAndJsonDir_;
    std::string debugStaticBinAndJsonDir_;
    nlohmann::json debugStaticConfigJson_;

public:
    KernelMgr(KernelMgr const&) = delete;
    void operator=(KernelMgr const&) = delete;
};

extern internal::KernelMgr& gKernelMgr;
} // namespace internal
} // namespace op
#endif
