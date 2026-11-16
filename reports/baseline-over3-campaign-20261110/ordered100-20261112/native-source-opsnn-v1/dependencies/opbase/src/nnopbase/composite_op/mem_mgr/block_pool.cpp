/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <array>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
#include <type_traits>
#include <vector>

#include "block_pool.h"

std::atomic<std::int64_t> object_num = 0;

namespace op {
namespace internal {

/**
 * @brief Initialize each BlockStore
 */
bool BlockPool::Init()
{
    const uint16_t tag = DEFAULT_TAG;
    hugeMemStart_ = nullptr;
    hugeMemEnd_ = nullptr;

    for (size_t i = 0; i < StoreIndex.size(); i++) {
        int rc = blockStoreArray_[i].Init(tag + i, StoreIndex[i].size, StoreIndex[i].count);
        if (rc != 0) {
            return false;
        }
    }

    auto* base = std::malloc(op::internal::kHugeBlockNum * op::internal::kHugeBlockSize);
    if (base == nullptr) {
        return false;
    }
    hugeMemStart_ = base;
    hugeMemEnd_ = (char*)base + op::internal::kHugeBlockNum * op::internal::kHugeBlockSize;
    for (int i = 0; i < op::internal::kHugeBlockNum; i++) {
        int64_t* offset = (int64_t*)((char*)base + i * op::internal::kHugeBlockSize);
        offset[0] = op::internal::kHugeBlockDefaultOffset;
        hugeMemArray_.push_back((char*)base + i * op::internal::kHugeBlockSize);
    }

    return true;
}

void BlockPool::UnInit()
{
    for (size_t i = 0; i < StoreIndex.size(); i++) {
        blockStoreArray_[i].UnInit();
    }
    if (hugeMemStart_) {
        std::free(hugeMemStart_);
    }
}

BlockPool globalPoolImpl__ __attribute__((init_priority(200)));
BlockPool& BlockPool::globalPool_ = globalPoolImpl__;

} // namespace internal
} // namespace op
