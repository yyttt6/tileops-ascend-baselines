/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_OP_API_COMMON_SRC_MEM_MGR_MEMORY_ALLOCATOR_H
#define OP_API_OP_API_COMMON_SRC_MEM_MGR_MEMORY_ALLOCATOR_H
#include "kernel_tensor.h"

namespace op {
namespace mem {
constexpr uint64_t EXTRA_SIZE_IN_BYTE = 32;
constexpr uint64_t ALIGN_SIZE_IN_BYTE = 512;
constexpr int64_t INVALID_INDEX = -1;

class TensorBucket {
public:
    uint64_t GetSize() const { return size_; }

    bool IsReusable(const KernelTensor* tensor);

    void UpdateOffset(const uint64_t offset);

    void AddRef(KernelTensor* tensor);

    const op::FVector<KernelTensor*>& GetRefs() const { return refs_; }

private:
    uint64_t offset_{0};
    uint64_t size_{0};
    op::FVector<KernelTensor*> refs_;
};

class MemoryAllocator {
public:
    virtual ~MemoryAllocator() = default;
    virtual uint64_t Allocate(const op::FVector<KernelTensor*, DEFAULT_TENSOR_NUM>& tensors) = 0;
};

class LinearAllocator : public MemoryAllocator {
public:
    uint64_t Allocate(const op::FVector<KernelTensor*, DEFAULT_TENSOR_NUM>& tensors) override;

private:
    uint64_t size_{0};
};

class MaxAllocator : public MemoryAllocator {
public:
    uint64_t Allocate(const op::FVector<KernelTensor*, DEFAULT_TENSOR_NUM>& tensors) override;

private:
    TensorBucket& GetBestFitBucket(const KernelTensor* tensor);

    uint64_t size_{0};
    op::FVector<TensorBucket> buckets_;
};
} // namespace mem
} // namespace op
#endif
