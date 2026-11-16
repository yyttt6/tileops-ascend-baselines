/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bridge_pool.h"
#include <stddef.h>
#include <stdint.h>
#include "block_pool.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "thread_local_context.h"

namespace op {
namespace internal {

void* Allocate(size_t size)
{
    int32_t id = op::internal::GetThreadLocalContext().poolIndex_;
    if (id != op::kInvalidHugeMemIndexId) {
        return GetAddr(id, size);
    } else {
        return op::internal::BlockCache::CacheAlloc(size);
    }
}

void DeAllocate(void* addr)
{
    OP_CHECK(addr != nullptr, OP_LOGW("deAllocate addr is nullptr."), return);
    if (op::internal::BlockPool::InHugeMemRange(addr)) {
        // since huge mem pool use offset, so free just a dummy operation
    } else {
        op::internal::BlockCache::CacheFree(addr);
    }
}

int32_t GetPoolIndex() { return op::internal::GetThreadLocalContext().poolIndex_; }

void UpdateHugeMemIndex(int32_t id)
{
    OP_LOGI("Hugemem trace: update pool index: %d", id);
    op::internal::GetThreadLocalContext().poolIndex_ = id;
}

bool CheckDoubleFree(void* addr)
{
    if (addr == nullptr) {
        return false;
    }
    if (BlockPool::InHugeMemRange(addr)) {
        return false;
    }

    BlockStore::BlockHeader* head = BlockStore::GetBlockHeader(addr);
    if (head->magic_ != BlockStore::MAGIC) {
        return false;
    }

    uintptr_t ext = head->cacheExt_;
    if (ext == BlockStore::NOT_IN_CACHE) {
        return false;
    }
    if (ext == BlockCache::CurrentThreadCacheAddr()) {
        return false;
    }
    if (ext == reinterpret_cast<uintptr_t>(nullptr)) {
        OP_LOGW("cacheExt_ is nullptr, block has been returned to cache list (list head was empty).");
        return true;
    }
    // 区分"指向合法 BlockHeader（已归还到 cache 链表，cacheExt_ 存的是 cacheHead_[idx] 即 BlockHeader*）"
    // 与"指向活跃 block 的 BlockCache* this（含跨线程 free 场景）"
    // cacheExt_ 在 CacheFreeImpl/PoolAlloc 中写入的是 uintptr_t(cacheHead_[idx]) 或 uintptr_t(this)：
    //   - 已归还到链表：cacheExt_ = cacheHead_[idx]，类型为 BlockHeader*，直接强转即可读 magic_
    //   - 活跃态（本线程）：上面已通过 CurrentThreadCacheAddr() 短路返回 false
    //   - 跨线程 free：cacheExt_ 指向分配方线程的 BlockCache 实例对象（this），强转后落在该对象内部某偏移，
    //     4 字节偶然匹配 MAGIC 概率 1/2^32，可忽略
    // 注：block_pool.h 中 CacheFreeImpl/PoolAlloc 对 cacheExt_ 的写入均直接 reinterpret_cast 指针为 uintptr_t
    // （无 +1/-1 算术），故此处读 magic_ 也不做指针算术，保持与写入端对称
    BlockStore::BlockHeader* nextHead = reinterpret_cast<BlockStore::BlockHeader*>(ext); // NOLINT
    if (nextHead->magic_ == BlockStore::MAGIC) {
        OP_LOGW("cacheExt_ points to another block header in cache list, block has been returned already.");
        return true;
    }
    return false;
}

} // namespace internal
} // namespace op
