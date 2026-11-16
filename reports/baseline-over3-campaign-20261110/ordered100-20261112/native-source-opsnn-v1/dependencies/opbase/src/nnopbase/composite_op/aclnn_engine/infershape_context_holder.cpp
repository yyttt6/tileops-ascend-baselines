/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "infershape_context_holder.h"
#include "kernel_utils.h"

namespace op {
namespace internal {

constexpr size_t INFER_SHAPE_VALUES_OFFSET = 2;
constexpr size_t GROWTH_FACTOR = 2;

void InferShapeContextHolder::BuildInferShapeContext()
{
    auto ctxSize = sizeof(KernelRunContext) + sizeof(AsyncAnyValue*) * (inferShapeCtxCapacity_ + inferShapeValueNum_);
    inferShapeCtx_ = static_cast<KernelRunContext*>(malloc(ctxSize));
    OP_CHECK(inferShapeCtx_ != nullptr, OP_LOGE(ACLNN_ERR_INNER, "malloc failed. %zu", ctxSize), return);
    (void)memset_s(inferShapeCtx_, ctxSize, 0, ctxSize);
    size_t sz = sizeof(AsyncAnyValue) * inferShapeValueNum_;
    inferShapeValues_ = static_cast<AsyncAnyValue*>(malloc(sz));
    OP_CHECK(inferShapeValues_ != nullptr, OP_LOGE(ACLNN_ERR_INNER, "malloc failed. %zu", sz), return);
    (void)memset_s(inferShapeValues_, sz, 0, sz);
}

aclnnStatus InferShapeContextHolder::EnsureContextCapacity(size_t requiredCapacity)
{
    // The required capacity is inputNum_ + inferShapeValueNum_ from kernelCtx
    if (requiredCapacity <= inferShapeCtxCapacity_) {
        return ACLNN_SUCCESS;
    }

    // Calculate new capacity using growth factor 2x
    const size_t oldCapacity = inferShapeCtxCapacity_;
    size_t newCapacity = oldCapacity;
    while (newCapacity < requiredCapacity) {
        newCapacity *= GROWTH_FACTOR;
    }

    // Allocate new memory (do NOT use realloc)
    size_t newSize = sizeof(KernelRunContext) + sizeof(AsyncAnyValue*) * (newCapacity + inferShapeValueNum_);
    KernelRunContext* newCtx = static_cast<KernelRunContext*>(malloc(newSize));
    OP_CHECK(newCtx != nullptr, OP_LOGE(ACLNN_ERR_INNER, "failed to malloc inferShapeCtx, size %zu.", newSize),
             return ACLNN_ERR_INNER);

    // Calculate old size for copy
    size_t oldSize = sizeof(KernelRunContext) + sizeof(AsyncAnyValue*) * (inferShapeCtxCapacity_ + inferShapeValueNum_);

    // Copy existing data
    OP_CHECK(memcpy_s(newCtx, newSize, inferShapeCtx_, oldSize) == EOK,
             OP_LOGE(ACLNN_ERR_INNER, "failed to memcpy inferShapeCtx."), std::free(newCtx);
             return ACLNN_ERR_INNER);

    // Zero out the new portion
    size_t zeroSize = newSize - oldSize;
    (void)memset_s(PtrCastTo<uint8_t>(newCtx) + oldSize, zeroSize, 0, zeroSize);

    // Free old memory and update pointer
    std::free(inferShapeCtx_);
    inferShapeCtx_ = newCtx;
    inferShapeCtxCapacity_ = newCapacity;

    OP_LOGI("Expanded inferShapeCtx capacity from %zu to %zu.", oldCapacity, inferShapeCtxCapacity_);
    return ACLNN_SUCCESS;
}

aclnnStatus InferShapeContextHolder::UpdateInferShapeContext(const KernelContextHolder* kernelCtx) const
{
    CHECK_COND(kernelCtx != nullptr, ACLNN_ERR_RUNTIME_ERROR, "kernelCtx is NULL");

    // Ensure capacity before updating context
    size_t requiredCapacity = kernelCtx->inputNum_ + kernelCtx->outputNum_ + inferShapeValueNum_;
    CHECK_RET_CODE(const_cast<InferShapeContextHolder*>(this)->EnsureContextCapacity(requiredCapacity),
                   "EnsureContextCapacity failed.");

    inferShapeCtx_->compute_node_info = kernelCtx->computeNodeInfo_;
    inferShapeCtx_->kernel_extend_info = &kernelCtx->kernelExtendInfo_;

    inferShapeCtx_->input_size = kernelCtx->inputNum_ + inferShapeValueNum_;
    inferShapeCtx_->output_size = kernelCtx->outputNum_;
    inferShapeCtx_->output_start = inferShapeCtx_->values + inferShapeCtx_->input_size;

    for (size_t i = 0; i < kernelCtx->inputNum_; ++i) {
        inferShapeCtx_->values[i] = kernelCtx->opInArg_ + i;
    }
    inferShapeCtx_->values[kernelCtx->inputNum_ + 1] = inferShapeValues_;
    inferShapeCtx_->values[kernelCtx->inputNum_ + INFER_SHAPE_VALUES_OFFSET] = inferShapeValues_ + 1;

    for (size_t i = 0; i < kernelCtx->outputNum_; ++i) {
        inferShapeCtx_->values[i + inferShapeCtx_->input_size] = kernelCtx->opInArg_ + kernelCtx->inputNum_ + i;
    }
    return ACLNN_SUCCESS;
}

InferShapeContextHolder::~InferShapeContextHolder()
{
    FREE(inferShapeCtx_);
    FREE(inferShapeValues_);
}
} // namespace internal
} // namespace op
