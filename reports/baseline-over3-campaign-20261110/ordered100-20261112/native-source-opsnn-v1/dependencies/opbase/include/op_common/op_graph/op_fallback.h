/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_OP_GRAPH_OP_FALLBACK_H_
#define INC_OP_GRAPH_OP_FALLBACK_H_

#include "aclnn/aclnn_base.h"
#include "exe_graph/runtime/op_execute_context.h"
#include "exe_graph/runtime/tensor.h"
#include "op_fallback_internal.h"
#include "register/op_impl_kernel_registry.h"
#include "register/op_impl_registry.h"

namespace fallback {

#define CANN_OPS_OPB_SYN_EXEC_ACLNN(ctx, aclnnApi, ...)                                                              \
    ({                                                                                                               \
        static auto ret = GRAPH_SUCCESS;                                                                             \
        do {                                                                                                         \
            static const auto ResetCacheThreadLocalAddr = GetOpApiFuncAddr("ResetCacheThreadLocal");                 \
            static const auto getWorkspaceSizeFuncAddr = GetOpApiFuncAddr(#aclnnApi "GetWorkspaceSize");             \
            static const auto opApiFuncAddr = GetOpApiFuncAddr(#aclnnApi);                                           \
            if (getWorkspaceSizeFuncAddr == nullptr || opApiFuncAddr == nullptr ||                                   \
                ResetCacheThreadLocalAddr == nullptr) {                                                              \
                OP_LOGE("aclnnfallback", "%s or %s not in  %s or %s  or ResetCacheThreadLocal not found.",           \
                        #aclnnApi "GetWorkspaceSize", #aclnnApi, GetOpApiLibName(), GetOpApiLibName());              \
                ret = GRAPH_FAILED;                                                                                  \
                break;                                                                                               \
            }                                                                                                        \
            auto ResetCacheThreadLocalFunc = reinterpret_cast<ResetCacheThreadLocal>(ResetCacheThreadLocalAddr);     \
            ResetCacheThreadLocalFunc();                                                                             \
            uint64_t workspace_size = 0;                                                                             \
            uint64_t* workspace_size_addr = &workspace_size;                                                         \
            aclOpExecutor* executor = nullptr;                                                                       \
            aclOpExecutor** executor_addr = &executor;                                                               \
            auto converted_params = ConvertTypes(__VA_ARGS__, workspace_size_addr, executor_addr);                   \
            static auto getWorkspaceSizeFunc = ConvertToOpApiFunc(converted_params, getWorkspaceSizeFuncAddr);       \
            auto workspace_status = call(getWorkspaceSizeFunc, converted_params);                                    \
            if (workspace_status != 0) {                                                                             \
                OP_LOGE("aclnnfallback", "call %s failed:", #aclnnApi);                                              \
                ret = GRAPH_FAILED;                                                                                  \
                break;                                                                                               \
            }                                                                                                        \
            void* workspace_addr = nullptr;                                                                          \
            if (workspace_size > 0) {                                                                                \
                workspace_addr = ctx->MallocWorkspace(workspace_size);                                               \
                if (workspace_addr == nullptr) {                                                                     \
                    OP_LOGE("aclnnfallback", "call %s allocate workspace failed", #aclnnApi);                        \
                    ret = GRAPH_FAILED;                                                                              \
                    break;                                                                                           \
                }                                                                                                    \
            }                                                                                                        \
            auto acl_stream = ctx->GetStream();                                                                      \
            auto acl_call = [converted_params, workspace_addr, workspace_size, ctx, acl_stream, executor]() -> int { \
                using OpApiFunc = int (*)(void*, uint64_t, aclOpExecutor*, const aclrtStream);                       \
                OpApiFunc opApiFunc = reinterpret_cast<OpApiFunc>(opApiFuncAddr);                                    \
                auto op_api_ret = opApiFunc(workspace_addr, workspace_size, executor, acl_stream);                   \
                ReleaseConvertTypes(converted_params);                                                               \
                ctx->FreeWorkspace();                                                                                \
                if (op_api_ret != 0) {                                                                               \
                    OP_LOGE("aclnnfallback", "call %s allocate workspace failed op_api_ret: %d", #aclnnApi,          \
                            op_api_ret);                                                                             \
                    return GRAPH_FAILED;                                                                             \
                }                                                                                                    \
                return op_api_ret;                                                                                   \
            };                                                                                                       \
                                                                                                                     \
            ret = acl_call();                                                                                        \
        } while (false);                                                                                             \
        (ret);                                                                                                       \
    })

#define EXEC_OPAPI_CMD(aclnn_api, ...) CANN_OPS_OPB_SYN_EXEC_ACLNN(host_api_ctx, aclnn_api, ##__VA_ARGS__)

#define CANN_OPS_OPB_ASYN_EXEC_ACLNN(ctx, aclnn_api, ...)                                                        \
    ({                                                                                                           \
        static auto ret = GRAPH_SUCCESS;                                                                         \
        do {                                                                                                     \
            static const auto ResetCacheThreadLocalAddr = GetOpApiFuncAddr("ResetCacheThreadLocal");             \
            static const auto opApiFuncAddr = GetOpApiFuncAddr(#aclnn_api);                                      \
            static const auto getWorkspaceSizeFuncAddr = GetOpApiFuncAddr(#aclnn_api "GetWorkspaceSize");        \
            if (getWorkspaceSizeFuncAddr == nullptr || opApiFuncAddr == nullptr ||                               \
                ResetCacheThreadLocalAddr == nullptr) {                                                          \
                OP_LOGE("aclnnfallback", "%s or %s not in  %s or %s  or ResetCacheThreadLocal not found.",       \
                        #aclnn_api "GetWorkspaceSize", #aclnn_api, GetOpApiLibName(), GetOpApiLibName());        \
                ret = GRAPH_FAILED;                                                                              \
                break;                                                                                           \
            }                                                                                                    \
            auto* op_api_params = new (std::nothrow) OpApiParams();                                              \
            auto ResetCacheThreadLocalFunc = reinterpret_cast<ResetCacheThreadLocal>(ResetCacheThreadLocalAddr); \
            ResetCacheThreadLocalFunc();                                                                         \
            op_api_params->op_api_func = reinterpret_cast<OpApiFunc2Stage>(opApiFuncAddr);                       \
            uint64_t workspace_size = 0;                                                                         \
            uint64_t* workspace_size_addr = &workspace_size;                                                     \
            aclOpExecutor** executor_addr = &op_api_params->executor;                                            \
            auto converted_params = ConvertTypes(__VA_ARGS__, workspace_size_addr, executor_addr);               \
            using TupleT = decltype(converted_params);                                                           \
            constexpr size_t tuple_size = std::tuple_size<TupleT>::value;                                        \
            op_api_params->converted_params.reserve(tuple_size);                                                 \
            CollectConvertedTypes(converted_params, op_api_params->converted_params);                            \
            ctx->SetOpApiParamsWithDefaultDeleter<OpApiParams>(op_api_params);                                   \
            static auto getWorkspaceSizeFunc = ConvertToOpApiFunc(converted_params, getWorkspaceSizeFuncAddr);   \
            auto workspace_status = call(getWorkspaceSizeFunc, converted_params);                                \
            if (workspace_status != 0) {                                                                         \
                OP_LOGE("aclnnfallback", "call %s failed:", #aclnn_api);                                         \
                ret = GRAPH_FAILED;                                                                              \
                break;                                                                                           \
            }                                                                                                    \
            auto set_workspace_status = ctx->SetWorkspaceSizes({workspace_size});                                \
            if (set_workspace_status != GRAPH_SUCCESS) {                                                         \
                ret = GRAPH_FAILED;                                                                              \
            }                                                                                                    \
        } while (false);                                                                                         \
        (ret);                                                                                                   \
    })

} // namespace fallback

#endif // INC_OP_GRAPH_OP_FALLBACK_H_
