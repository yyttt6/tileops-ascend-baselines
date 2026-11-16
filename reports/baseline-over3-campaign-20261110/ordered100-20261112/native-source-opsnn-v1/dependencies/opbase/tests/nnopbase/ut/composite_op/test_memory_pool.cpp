/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include <array>
#include <memory>
#include <thread>

#include "acl/acl.h"
#include "opdev/make_op_executor.h"
#include "bridge_pool.h"
#include "thread_local_context.h"

using namespace op;

namespace op {
namespace internal {
void* Allocate(size_t size);
int32_t GetPoolCurrentArrayIndex(int32_t id);
void* GetPoolLinkHead(int32_t id);
void* GetPoolLinkCurrent(int32_t id);
} // namespace internal
} // namespace op

class C : public Object {
public:
    C() = default;
    ~C() = default;
};

extern "C" int InitHugeMemThreadLocal(void* arg, bool sync);
extern "C" void UnInitHugeMemThreadLocal(void* arg, bool sync);
extern "C" void ReleaseHugeMem(void* arg, bool sync);

class MemoryPoolUt : public testing::Test {
protected:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}
};

TEST_F(MemoryPoolUt, test_allocate_by_link)
{
    InitHugeMemThreadLocal(nullptr, false);

    int32_t id = op::internal::GetThreadLocalContext().poolIndex_;

    EXPECT_TRUE(id != op::kInvalidHugeMemIndexId);

    EXPECT_EQ(op::internal::GetPoolCurrentArrayIndex(id), 0);

    int half = op::internal::kHugeBlockSize / 2;
    int quda = op::internal::kHugeBlockSize / 4;
    void* addr1 = op::internal::Allocate(half);
    void* addr2 = op::internal::Allocate(quda); // array 0
    EXPECT_TRUE(addr1 != nullptr);
    EXPECT_TRUE(addr2 != nullptr);
    for (int i = 0; i < 5 - 1; i++) {              // kMaxHugeMemPoolArryNum = 5
        void* addr = op::internal::Allocate(half); // array[1],[2],[3],[4]
        EXPECT_TRUE(addr != nullptr);
    }

    EXPECT_EQ(op::internal::GetPoolCurrentArrayIndex(id), 4);

    EXPECT_TRUE(op::internal::GetPoolLinkHead(id) == nullptr);
    EXPECT_TRUE(op::internal::GetPoolLinkCurrent(id) == nullptr);

    void* addr7 = op::internal::Allocate(half); // link 0
    EXPECT_TRUE(addr7 != nullptr);
    void* addr8 = op::internal::Allocate(quda); // link 0
    EXPECT_TRUE(addr8 != nullptr);

    EXPECT_TRUE(op::internal::GetPoolLinkHead(id) != nullptr);
    EXPECT_TRUE(op::internal::GetPoolLinkHead(id) == op::internal::GetPoolLinkCurrent(id));

    void* addr9 = op::internal::Allocate(half); // link 1
    EXPECT_TRUE(addr9 != nullptr);

    EXPECT_TRUE(op::internal::GetPoolLinkHead(id) != nullptr);
    EXPECT_TRUE(op::internal::GetPoolLinkHead(id) != op::internal::GetPoolLinkCurrent(id));

    ReleaseHugeMem(nullptr, false);

    EXPECT_EQ(op::internal::GetPoolCurrentArrayIndex(id), 0);
    EXPECT_TRUE(op::internal::GetPoolLinkHead(id) == nullptr);
    EXPECT_TRUE(op::internal::GetPoolLinkCurrent(id) == nullptr);

    UnInitHugeMemThreadLocal(nullptr, false);
    id = op::internal::GetThreadLocalContext().poolIndex_;
    EXPECT_EQ(id, op::kInvalidHugeMemIndexId);

    C* c = new C[8];
    delete[] c;
    c = new (std::nothrow) C[8];
    delete[] c;
}

TEST_F(MemoryPoolUt, test_huge_memory)
{
    InitHugeMemThreadLocal(nullptr, false);

    int32_t id = op::internal::GetThreadLocalContext().poolIndex_;
    EXPECT_TRUE(id != op::kInvalidHugeMemIndexId);

    int large = op::internal::kHugeBlockSize * 2;
    int half = op::internal::kHugeBlockSize / 2;
    int quda = op::internal::kHugeBlockSize / 4;

    void* largeMemAddr = op::internal::Allocate(large);
    EXPECT_TRUE(largeMemAddr != nullptr);
    EXPECT_FALSE(op::internal::BlockPool::InHugeMemRange(largeMemAddr));

    std::array<void*, 16> addrs;
    for (int32_t i = 0; i < 16; i++) {
        addrs[i] = op::internal::Allocate(half);
        EXPECT_TRUE(addrs[i] != nullptr);
        EXPECT_TRUE(op::internal::BlockPool::InHugeMemRange(addrs[i]));
    }
    void* outofHugeMemAddr = op::internal::Allocate(half);
    EXPECT_TRUE(outofHugeMemAddr != nullptr);
    EXPECT_FALSE(op::internal::BlockPool::InHugeMemRange(outofHugeMemAddr));

    op::internal::DeAllocate(largeMemAddr);
    for (int32_t i = 0; i < 16; i++) {
        op::internal::DeAllocate(addrs[i]);
    }
    op::internal::DeAllocate(outofHugeMemAddr);

    ReleaseHugeMem(nullptr, false);

    EXPECT_EQ(op::internal::GetPoolCurrentArrayIndex(id), 0);
    EXPECT_TRUE(op::internal::GetPoolLinkHead(id) == nullptr);
    EXPECT_TRUE(op::internal::GetPoolLinkCurrent(id) == nullptr);

    UnInitHugeMemThreadLocal(nullptr, false);
    id = op::internal::GetThreadLocalContext().poolIndex_;
    EXPECT_EQ(id, op::kInvalidHugeMemIndexId);
}

static void allocHugeMemThread()
{
    InitHugeMemThreadLocal(nullptr, false);
    int32_t id = op::internal::GetThreadLocalContext().poolIndex_;
    EXPECT_TRUE(id != op::kInvalidHugeMemIndexId);

    int half = op::internal::kHugeBlockSize / 2;

    std::array<void*, 5> addrs;
    for (int32_t i = 0; i < 5; i++) {
        addrs[i] = op::internal::Allocate(half);
        EXPECT_TRUE(addrs[i] != nullptr);
    }

    for (int32_t i = 0; i < 5; i++) {
        op::internal::DeAllocate(addrs[i]);
    }
    ReleaseHugeMem(nullptr, false);

    EXPECT_EQ(op::internal::GetPoolCurrentArrayIndex(id), 0);
    EXPECT_TRUE(op::internal::GetPoolLinkHead(id) == nullptr);
    EXPECT_TRUE(op::internal::GetPoolLinkCurrent(id) == nullptr);

    UnInitHugeMemThreadLocal(nullptr, false);
    id = op::internal::GetThreadLocalContext().poolIndex_;
    EXPECT_EQ(id, op::kInvalidHugeMemIndexId);
}

TEST_F(MemoryPoolUt, test_huge_memory_2)
{
    InitHugeMemThreadLocal(nullptr, false);

    int32_t id = op::internal::GetThreadLocalContext().poolIndex_;
    EXPECT_TRUE(id != op::kInvalidHugeMemIndexId);

    int half = op::internal::kHugeBlockSize / 2;

    std::array<void*, 12> addrs;
    for (int32_t i = 0; i < 12; i++) {
        addrs[i] = op::internal::Allocate(half);
        EXPECT_TRUE(addrs[i] != nullptr);
        EXPECT_TRUE(op::internal::BlockPool::InHugeMemRange(addrs[i]));
    }

    std::thread t(allocHugeMemThread);
    t.join();

    for (int32_t i = 0; i < 12; i++) {
        op::internal::DeAllocate(addrs[i]);
    }
    ReleaseHugeMem(nullptr, false);

    EXPECT_EQ(op::internal::GetPoolCurrentArrayIndex(id), 0);
    EXPECT_TRUE(op::internal::GetPoolLinkHead(id) == nullptr);
    EXPECT_TRUE(op::internal::GetPoolLinkCurrent(id) == nullptr);

    UnInitHugeMemThreadLocal(nullptr, false);
    id = op::internal::GetThreadLocalContext().poolIndex_;
    EXPECT_EQ(id, op::kInvalidHugeMemIndexId);
}

TEST_F(MemoryPoolUt, test_block_store)
{
    op::internal::BlockStore blockStore;
    auto ret = blockStore.Init(0, 1024, 1);
    EXPECT_EQ(ret, 0);
}