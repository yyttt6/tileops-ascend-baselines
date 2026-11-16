/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <mutex>
#include <vector>
#include "thread_local_context.h"
#include "opdev/op_log.h"
#include "opdev/op_cache.h"
#include "bridge_pool.h"
#include "huge_mem.h"

namespace op {
namespace internal {
static int32_t GetAvaiablePoolIndex();
}
} // namespace op

#ifdef __cplusplus
extern "C" {
#endif

int InitHugeMemThreadLocal([[maybe_unused]] void* arg, [[maybe_unused]] bool sync)
{
    op::internal::GetThreadLocalContext().poolIndex_ = op::internal::GetAvaiablePoolIndex();
    OP_LOGI("Hugemem trace: get avaiable pool index: %d", op::internal::GetThreadLocalContext().poolIndex_);
    return 0;
}

void UnInitHugeMemThreadLocal([[maybe_unused]] void* arg, [[maybe_unused]] bool sync)
{
    op::internal::GetThreadLocalContext().poolIndex_ = op::kInvalidHugeMemIndexId;
}

void ReleaseHugeMem([[maybe_unused]] void* arg, [[maybe_unused]] bool sync) { op::internal::FreeHugeMem(); }

#ifdef __cplusplus
}
#endif

namespace op {
namespace internal {

std::mutex g_hugeMemMutex;
constexpr int32_t kMaxHugeMemPoolArryNum = 5;

#if defined(NNOPBASE_UT) || defined(NNOPBASE_ST)
constexpr int32_t kMaxHugeMemObjectNum = 16;
#else
constexpr int32_t kMaxHugeMemObjectNum = 4096;
#endif

std::vector<int32_t> InitVecotr();
std::vector<int32_t> gHugeMemPoolIndex = InitVecotr();

std::vector<int32_t> InitVecotr()
{
    std::vector<int32_t> v;
    for (int i = 0; i < kMaxHugeMemObjectNum; i++) {
        v.push_back(i);
    }
    return v;
}

static int32_t GetAvaiablePoolIndex()
{
    const std::lock_guard<std::mutex> lock(g_hugeMemMutex);
    if (gHugeMemPoolIndex.empty()) {
        return op::kInvalidHugeMemIndexId;
    } else {
        int32_t res = gHugeMemPoolIndex.back();
        gHugeMemPoolIndex.pop_back();
        return res;
    }
}

struct BlockLink {
    BlockLink() { Init(); }

    void Init()
    {
        block_ = nullptr;
        next_ = nullptr;
    }

    void* block_;
    BlockLink* next_;
};

class HugeMemPool {
public:
    HugeMemPool() { Init(false); }

    HugeMemPool(const HugeMemPool& other) = delete;

    HugeMemPool& operator=(const HugeMemPool& rhs) noexcept
    {
        if (this != &rhs) {
            syncFlag_ = rhs.syncFlag_;
            head_ = rhs.head_;
            current_ = rhs.current_;
            currentArrayIndex_ = rhs.currentArrayIndex_;
            poolIndex_ = rhs.poolIndex_;
            for (int32_t i = 0; i < kMaxHugeMemPoolArryNum; i++) {
                baseArray_[i] = rhs.baseArray_[i];
            }
        }
        return *this;
    }

    void Init(const bool syncFlag)
    {
        syncFlag_ = syncFlag;
        head_ = nullptr;
        current_ = nullptr;
        currentArrayIndex_ = 0;
        poolIndex_ = op::kInvalidHugeMemIndexId;
        for (int32_t i = 0; i < kMaxHugeMemPoolArryNum; i++) {
            baseArray_[i] = nullptr;
        }
    }

    void UnInit(const bool syncFlag) { Init(syncFlag); }

    void* AddOneHugeBlockToLinkList(size_t size)
    {
        void* block = op::internal::BlockPool::GetOneHugeBlock();
        if (block == nullptr) {
            OP_LOGW("Hugemem trace: GetOneHugeBlock failed, use block cache to allocate memory!");
            return op::internal::BlockCache::CacheAlloc(size);
        }
        int64_t* offset = static_cast<int64_t*>(block);
        offset[0] = op::internal::kHugeBlockDefaultOffset;
        BlockLink* link = new (std::nothrow) BlockLink();
        if (link == nullptr) {
            return nullptr;
        }
        link->block_ = block;
        if (head_ == nullptr) {
            head_ = link;
            current_ = head_;
        } else {
            current_->next_ = link;
            current_ = link;
        }
        auto* res = static_cast<void*>(static_cast<uint8_t*>(block) + offset[0]);
        offset[0] += size;
        return res;
    }

    void* GetAddr(size_t size)
    {
        // 1. size is gt than kHugeMemorySize, pool can not provide memory, from libc to allocate.
        if (size > kHugeBlockSize - kHugeBlockDefaultOffset) {
            return op::internal::BlockPool::Malloc(size);
        }

        // 2. Fast path, GetBlock from array
        do {
            if (currentArrayIndex_ >= kMaxHugeMemPoolArryNum) {
                break; // array is full, goto Slow path
            }
            if (baseArray_[currentArrayIndex_] == nullptr) {
                void* block = op::internal::BlockPool::GetOneHugeBlock();
                if (block == nullptr) {
                    OP_LOGW("Hugemem trace: current array index: %d, GetOneHugeBlock failed, use block cache to "
                            "allocate memory!",
                            currentArrayIndex_);
                    return op::internal::BlockCache::CacheAlloc(size);
                }
                int64_t* offset = static_cast<int64_t*>(block);
                offset[0] = op::internal::kHugeBlockDefaultOffset;
                baseArray_[currentArrayIndex_] = block;
                auto* res = static_cast<void*>(static_cast<uint8_t*>(block) + offset[0]);
                offset[0] += size;
                return res;
            } else {
                int64_t* offset = (int64_t*)baseArray_[currentArrayIndex_];
                if (offset[0] + size <= op::internal::kHugeBlockSize) {
                    auto* res = static_cast<void*>(static_cast<uint8_t*>(baseArray_[currentArrayIndex_]) + offset[0]);
                    offset[0] += size;
                    return res;
                }
                currentArrayIndex_++;
            }
        } while (true);

        // 3. Slow path, need dynamic new a link node, GetBlock from link list
        if (current_ == nullptr) {
            return AddOneHugeBlockToLinkList(size);
        } else {
            int64_t* offset = static_cast<int64_t*>(current_->block_);
            if (offset[0] + size <= kHugeBlockSize) {
                auto* res = static_cast<void*>(static_cast<uint8_t*>(current_->block_) + offset[0]);
                offset[0] += size;
                return res;
            } else {
                return AddOneHugeBlockToLinkList(size);
            }
        }
    }

    void ReleaseAllAddr()
    {
        for (int i = 0; i < kMaxHugeMemPoolArryNum; i++) {
            op::internal::BlockPool::FreeOneHugeBlock(baseArray_[i]);
        }

        BlockLink* link = head_;
        while (link != nullptr) {
            op::internal::BlockPool::FreeOneHugeBlock(link->block_);
            link = link->next_;
        }

        link = head_;
        while (link != nullptr) {
            const BlockLink* tmp = link;
            link = link->next_;
            delete tmp;
        }
    }

    int32_t GetIndex() const { return currentArrayIndex_; }

    const void* GetHead() const { return head_; }

    const void* GetCurrent() const { return current_; }

private:
    char reserved1[64]{0};
    void* baseArray_[kMaxHugeMemPoolArryNum];
    int32_t currentArrayIndex_;
    BlockLink* head_;
    BlockLink* current_;
    bool syncFlag_;
    int32_t poolIndex_;
    char reserved2[64]{0}; // avoid false sharing
};

HugeMemPool gHugeMemPool[kMaxHugeMemObjectNum];

void FreeHugeMem()
{
    int32_t id = op::internal::GetThreadLocalContext().poolIndex_;
    OP_LOGI("Hugemem trace: get thread local context pool index: %d", id);
    op::internal::GetThreadLocalContext().poolIndex_ = op::kInvalidHugeMemIndexId;
    if (id == op::kInvalidHugeMemIndexId || id >= kMaxHugeMemObjectNum) {
        return;
    }
    gHugeMemPool[id].ReleaseAllAddr();
    gHugeMemPool[id].Init(false);
    const std::lock_guard<std::mutex> lock(g_hugeMemMutex);
    gHugeMemPoolIndex.push_back(id);
}

void* GetAddr(const int32_t id, size_t size)
{
    if (id == op::kInvalidHugeMemIndexId || id >= kMaxHugeMemObjectNum) {
        return nullptr;
    }
    return gHugeMemPool[id].GetAddr(size);
}

int32_t GetPoolCurrentArrayIndex(const int32_t id)
{
    if (id >= kMaxHugeMemObjectNum) {
        return -1;
    }
    return gHugeMemPool[id].GetIndex();
}

const void* GetPoolLinkHead(const int32_t id)
{
    if (id >= kMaxHugeMemObjectNum) {
        return nullptr;
    }
    return gHugeMemPool[id].GetHead();
}

const void* GetPoolLinkCurrent(const int32_t id)
{
    if (id >= kMaxHugeMemObjectNum) {
        return nullptr;
    }
    return gHugeMemPool[id].GetCurrent();
}

} // namespace internal
} // namespace op
