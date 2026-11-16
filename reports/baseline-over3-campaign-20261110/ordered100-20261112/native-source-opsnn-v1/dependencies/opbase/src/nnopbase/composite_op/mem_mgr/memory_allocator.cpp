/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../common/inc/memory_allocator.h"

namespace op {
namespace mem {
constexpr uint64_t OFFSET_PLACEHOLDER = UINT64_MAX;

inline uint64_t Align(const uint64_t n)
{
    return (n + EXTRA_SIZE_IN_BYTE + ALIGN_SIZE_IN_BYTE - 1) & (~(ALIGN_SIZE_IN_BYTE - 1));
}

bool TensorBucket::IsReusable(const KernelTensor* tensor)
{
    if (refs_.empty()) {
        return true;
    }
    auto& back = refs_.back();
    return back->IsLifeTimeNotOverLap(tensor) && !back->IsInputOf(tensor);
}

void TensorBucket::AddRef(KernelTensor* tensor)
{
    const auto tensorSize = Align(static_cast<uint64_t>(tensor->GetSize()));
    size_ = std::max(tensorSize, size_);
    refs_.push_back(tensor);
}

void TensorBucket::UpdateOffset(const uint64_t offset)
{
    offset_ = offset;
    for (const auto ref : refs_) {
        ref->SetOffset(offset_);
        OP_LOGI(
            "MaxAllocator tensor index: %zu, offset: %lu, lifetime: [%ld, %ld], original size: %ld, align size: %lu.",
            ref->GetIndex(), offset_, ref->GetLifeTimeStart(), ref->GetLifeTimeEnd(), ref->GetSize(),
            Align(ref->GetSize()));
    }
}

uint64_t LinearAllocator::Allocate(const op::FVector<KernelTensor*, DEFAULT_TENSOR_NUM>& tensors)
{
    for (const auto tensor : tensors) {
        tensor->SetOffset(size_);
        size_ += Align(tensor->GetSize());
    }

    return size_;
}

uint64_t MaxAllocator::Allocate(const op::FVector<KernelTensor*, DEFAULT_TENSOR_NUM>& tensors)
{
    for (auto& tensor : tensors) {
        if (tensor->GetAclTensor() != nullptr && tensor->GetAclTensor()->GetWorkspaceOffset() == OFFSET_PLACEHOLDER) {
            continue;
        }
        TensorBucket& bucket = GetBestFitBucket(tensor);
        bucket.AddRef(tensor);
        if (tensor->GetAclTensor() != nullptr) {
            tensor->GetAclTensor()->SetWorkspaceOffset(OFFSET_PLACEHOLDER);
        }
    }

    for (auto& bucket : buckets_) {
        bucket.UpdateOffset(size_);
        size_ += bucket.GetSize();
    }
    return size_;
}

TensorBucket& MaxAllocator::GetBestFitBucket(const KernelTensor* tensor)
{
    int64_t firstFitIndex = INVALID_INDEX;
    for (int64_t index = 0; index != static_cast<int64_t>(buckets_.size()); ++index) {
        if (buckets_[static_cast<size_t>(index)].IsReusable(tensor)) {
            firstFitIndex = index;
        }
    }

    if (firstFitIndex == INVALID_INDEX) {
        buckets_.emplace_back();
        return buckets_.back();
    }
    return buckets_[static_cast<size_t>(firstFitIndex)];
}
} // namespace mem
} // namespace op
