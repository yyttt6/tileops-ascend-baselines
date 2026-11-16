/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_STUB_H_
#define OP_STUB_H_

#include "utils/indv_base.h"
#include "utils/indv_types.h"

#ifdef __cplusplus
extern "C" {
#endif
static inline uint64_t* NnopbaseTilingGetTilingKeyPointer(NnopbaseKernelRunContext* context)
{
    return (uint64_t*)(context->output_start[0]->data.inplace);
}
static inline uint32_t* NnopbaseTilingGetBlockDimPointer(NnopbaseKernelRunContext* context)
{
    return (uint32_t*)(context->output_start[1]->data.inplace);
}
static inline bool* NnopbaseTilingGetAtomicCleanPointer(NnopbaseKernelRunContext* context)
{
    return (bool*)(context->output_start[2]->data.inplace); // 2 for output
}
static inline NnopbaseTilingData* NnopbaseTilingGetTilingDataPointer(NnopbaseKernelRunContext* context)
{
    return (NnopbaseTilingData*)(context->output_start[3]->data.pointer); // 3 for tiling data
}
static inline NnopbaseWorkspaceSizes* NnopbaseTilingGetWorkspaceSizesPointer(NnopbaseKernelRunContext* context)
{
    return (NnopbaseWorkspaceSizes*)(context->output_start[4]->data.pointer); // 4 for workspace
}
static inline void* NnopbaseTilingGetInput(NnopbaseKernelRunContext* context, size_t index)
{
    if (index >= context->input_size) {
        return NULL;
    }
    return context->values[index]->data.pointer;
}

static inline uint32_t* NnopbaseTilingGetAicpuBlockDimPointer(NnopbaseKernelRunContext* context)
{
    return (uint32_t*)(context->output_start[8]->data.inplace);
}

#ifdef __cplusplus
}

template <typename Function, typename Tuple, size_t... I>
auto call(Function f, Tuple t, std::index_sequence<I...>)
{
    return f(std::get<I>(t)...);
}

template <typename Function, typename Tuple>
auto call(Function f, Tuple t)
{
    static constexpr auto size = std::tuple_size<Tuple>::value;
    return call(f, t, std::make_index_sequence<size>{});
}

#define RunOp(aclnn_api, ...)                                                                     \
    ({                                                                                            \
        uint64_t workspace_size = 0;                                                              \
        uint64_t* workspace_size_addr = &workspace_size;                                          \
        aclOpExecutor* executor = nullptr;                                                        \
        aclOpExecutor** executor_addr = &executor;                                                \
        auto converted_params = std::make_tuple(__VA_ARGS__, workspace_size_addr, executor_addr); \
        aclnnStatus _chk_stutus = call(aclnn_api##GetWorkspaceSize, converted_params);            \
        void* workspace_addr = nullptr;                                                           \
        if (workspace_size != 0) {                                                                \
            workspace_addr = malloc(workspace_size);                                              \
        }                                                                                         \
        _chk_stutus = aclnn_api(workspace_addr, workspace_size, executor, nullptr);               \
        if (workspace_addr != nullptr) {                                                          \
            free(workspace_addr);                                                                 \
        }                                                                                         \
        _chk_stutus;                                                                              \
    })

#endif
#endif
