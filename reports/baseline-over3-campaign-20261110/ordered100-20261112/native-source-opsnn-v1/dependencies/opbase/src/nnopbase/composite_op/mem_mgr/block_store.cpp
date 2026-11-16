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
#include <limits>
#include <vector>
#include "kernel_utils.h"
#include "block_store.h"

namespace op {
namespace internal {

/**
 * @brief Initialize the BlockStore
 *
 * @param tag User-defined tag
 * @param block_size size of each block
 * @param block_count count of blocks_
 * @return int 0 for success, others for fail
 */
int BlockStore::Init(uint16_t tag, size_t block_size, size_t block_count)
{
    if ((block_size == 0) || (block_count == 0)) {
        return -1;
    }
    tag_ = tag;
    blockSize_ = AlignedBlockSize(block_size);
    blockCount_ = block_count;
    if (block_count > static_cast<size_t>(std::numeric_limits<BlockIdx>::max())) {
        return -1;
    }

    mem_ = std::malloc((blockSize_ + sizeof(BlockHeader)) * blockCount_);
    if (mem_ == nullptr) {
        return -1;
    }

    for (BlockIdx i = 0; i < static_cast<BlockIdx>(blockCount_); i++) {
        BlockHeader* head = GetHeader(i);
        new (head) BlockHeader;
        head->userTag_ = tag;
        head->blockIdx_ = i;
        if (i < static_cast<BlockIdx>(blockCount_) - 1) {
            head->next_ = i + 1;
        } else {
            head->next_ = -1;
        }
    }
    freeHead_ = 0;
    return 0;
}

void BlockStore::UnInit()
{
    FREE(mem_);
    tag_ = DEFAULT_TAG;
    blockSize_ = 0;
    blockCount_ = 0;
    freeHead_ = -1;
}

BlockStore::~BlockStore() { FREE(mem_); }

} // namespace internal
} // namespace op