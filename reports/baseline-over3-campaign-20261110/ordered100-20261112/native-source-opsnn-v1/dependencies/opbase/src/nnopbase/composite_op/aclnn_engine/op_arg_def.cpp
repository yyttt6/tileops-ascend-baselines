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
#include <string>
#include <tuple>
#include <vector>
#include <type_traits>
#include "opdev/common_types.h"
#include "opdev/op_arg_def.h"

#include "bridge_pool.h"

namespace op {
static void OpArgDeleter(void* ptr)
{
    if (ptr) {
        op::internal::DeAllocate(ptr);
    }
}

OpArgValue::OpArgValue(const std::string* value)
{
    if (!value) {
        data.pointer = nullptr;
        return;
    }
    size_t len = value->size();
    uint8_t* addr = static_cast<uint8_t*>(op::internal::Allocate(len + 1));
    OP_CHECK(addr != nullptr, OP_LOGE(ACLNN_ERR_INNER, "failed to allocate memory."), throw std::bad_alloc());
    if (len > 0) {
        OP_CHECK(memcpy_s(addr, len, value->c_str(), len) == EOK, OP_LOGE(ACLNN_ERR_INNER, "failed to memcpy."),
                 throw std::runtime_error("OpArgValue::OpArgValue memcpy runtime error."));
    }
    addr[len] = '\0';
    deleter = &OpArgDeleter;
    data.pointer = addr;
}
OpArgValue::OpArgValue(const std::string& value)
{
    size_t len = value.size();
    uint8_t* addr = static_cast<uint8_t*>(op::internal::Allocate(len + 1));
    OP_CHECK(addr != nullptr, OP_LOGE(ACLNN_ERR_INNER, "failed to allocate memory."), throw std::bad_alloc());
    if (len > 0) {
        OP_CHECK(memcpy_s(addr, len, value.c_str(), len) == EOK, OP_LOGE(ACLNN_ERR_INNER, "failed to memcpy."),
                 throw std::runtime_error("OpArgValue::OpArgValue memcpy runtime error."));
    }
    addr[len] = '\0';
    deleter = &OpArgDeleter;
    data.pointer = addr;
}

OpArgValue::OpArgValue(const char* value)
{
    if (!value) {
        data.pointer = nullptr;
        return;
    }
    size_t len = strlen(value);
    uint8_t* addr = static_cast<uint8_t*>(op::internal::Allocate(len + 1));
    OP_CHECK(addr != nullptr, OP_LOGE(ACLNN_ERR_INNER, "failed to allocate memory."), throw std::bad_alloc());
    if (len > 0) {
        OP_CHECK(memcpy_s(addr, len, value, len) == EOK, OP_LOGE(ACLNN_ERR_INNER, "failed to memcpy."),
                 throw std::runtime_error("OpArgValue::OpArgValue memcpy runtime error."));
    }
    addr[len] = '\0';
    deleter = &OpArgDeleter;
    data.pointer = addr;
}

void OpArgContext::AppendOpWorkspaceArg(aclTensorList* tensorList)
{
    if (ContainsOpArgType(OP_WORKSPACE_ARG)) {
        return;
    }

    OpArg* arg = reinterpret_cast<OpArg*>(this + 1);
    arg->type = OpArgType::OPARG_ACLTENSOR_LIST;
    arg->value = OpArgValue(tensorList);
    argLists[OP_WORKSPACE_ARG].args = arg;
    argLists[OP_WORKSPACE_ARG].count = 1;
}

void* Allocated(size_t size) { return op::internal::Allocate(size); }
void DeAllocated(void* addr) { op::internal::DeAllocate(addr); }

[[maybe_unused]] void DestroyOpArgContext(OpArgContext* ctx)
{
    if (ctx) {
        for (auto& argList : ctx->argLists) {
            argList.VisitByNoReturn([]([[maybe_unused]] size_t idx, OpArg& arg) {
                if (arg.value.deleter) {
                    arg.value.deleter(arg->pointer);
                }
            });
        }
        op::internal::DeAllocate(ctx);
    }
}
} // namespace op
