/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include <cstdint>
#include <thread>

#include "acl/acl.h"
#include "opdev/object.h"
#include "block_pool.h"
#include "block_store.h"
#include "bridge_pool.h"
#include "thread_local_context.h"

using namespace op;
using namespace op::internal;

// 大页内存初始化接口（huge_mem.cpp 导出，extern "C"），供 HugeMemRange_NotReported 用例使用
extern "C" int InitHugeMemThreadLocal(void* arg, bool sync);
extern "C" void UnInitHugeMemThreadLocal(void* arg, bool sync);
extern "C" void ReleaseHugeMem(void* arg, bool sync);

class CheckDoubleFreeUt : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

// Task 1: CurrentThreadCacheAddr 应返回当前线程 BlockCache 实例地址（稳定且非零）
TEST_F(CheckDoubleFreeUt, CurrentThreadCacheAddr_ReturnsStableNonZeroAddr)
{
    uintptr_t addr1 = BlockCache::CurrentThreadCacheAddr();
    EXPECT_NE(addr1, 0u);
    uintptr_t addr2 = BlockCache::CurrentThreadCacheAddr();
    EXPECT_EQ(addr1, addr2); // thread_local 实例稳定
}

// nullptr 传入不报
TEST_F(CheckDoubleFreeUt, Nullptr_ReturnsFalse) { EXPECT_FALSE(CheckDoubleFree(nullptr)); }

// 野指针（栈上随机内存，magic 几乎不可能匹配）不报
TEST_F(CheckDoubleFreeUt, WildPointer_MagicMismatch_ReturnsFalse)
{
    uintptr_t buf[16] = {0}; // 全零，magic 不匹配
    void* wild = reinterpret_cast<void*>(buf);
    EXPECT_FALSE(CheckDoubleFree(wild));
}

// 同线程 alloc → delete 后再 CheckDoubleFree 应返回 true（block 已归还 cache 链表）
// 注：命中 nullptr 分支还是链表节点分支取决于此前同 size class 的 cache 链表状态
// （首次归还时链表为空命中 nullptr；非空命中链表节点）。两种分支均返回 true，断言不变。
// 分支级覆盖：实测 SameThreadDoubleFree 命中"链表节点"分支、CrossThreadDoubleFree 命中"nullptr"分支，
// 两个命中分支在本套件中均被实际触发；分支级日志文案断言依赖任务 3 的日志捕获框架，本任务不引入。
TEST_F(CheckDoubleFreeUt, SameThreadDoubleFree_Detected)
{
    void* addr = Allocate(64); // 走内存池场景（非大页）
    ASSERT_NE(addr, nullptr);
    DeAllocate(addr);          // 首次归还：cacheExt_=nullptr 或链表节点 head+1

    // 再次检测：应命中（具体分支见上方注释）
    EXPECT_TRUE(CheckDoubleFree(addr));
}

// 线程 A alloc，线程 B 检查：cacheExt_ 指向 A 的 BlockCache 实例，不应误报
// 注：子线程 .join() 必须在主线程 DeAllocate 之前，否则触发 spec 7.2 漏报场景（UAF）
TEST_F(CheckDoubleFreeUt, CrossThreadAccess_NoFalsePositive)
{
    void* addr = Allocate(64);
    ASSERT_NE(addr, nullptr);

    std::thread([addr]() {
        // 跨线程仅检查，不归还：cacheExt_=线程A的BlockCache地址，nextHead->magic_ 不匹配
        EXPECT_FALSE(CheckDoubleFree(addr));
    }).join();        // 必须先 join，确保子线程访问期间 block 仍活跃

    DeAllocate(addr); // 主线程归还（join 之后才安全）
}

// 线程 A alloc → 线程 B 归还 → 线程 B 再 CheckDoubleFree：应命中
// 注：命中分支（nullptr / 链表节点）取决于线程 B 此前在同 size class 的 cache 状态，均返回 true
TEST_F(CheckDoubleFreeUt, CrossThreadDoubleFree_Detected)
{
    void* addr = Allocate(64);
    ASSERT_NE(addr, nullptr);

    std::thread([addr]() {
        DeAllocate(addr);                   // 线程 B 归还：cacheExt_ 写入线程 B 的 cache 链表
        EXPECT_TRUE(CheckDoubleFree(addr)); // 线程 B 再检测：应命中
    }).join();
}

// 大页内存场景不报（InHugeMemRange 短路）
TEST_F(CheckDoubleFreeUt, HugeMemRange_NotReported)
{
    InitHugeMemThreadLocal(nullptr, false);
    void* addr = Allocate(64); // 大页场景
    ASSERT_NE(addr, nullptr);
    // 前导断言：确认已进入大页路径（避免用例名误导）
    ASSERT_TRUE(op::internal::BlockPool::InHugeMemRange(addr)) << "测试前提：已进入大页路径";
    EXPECT_FALSE(CheckDoubleFree(addr)); // 短路：大页 free 是 no-op
    // 仅调 ReleaseHugeMem 即可：其内部 FreeHugeMem 会读 poolIndex_ 归还 baseArray_ 后再置 -1。
    // 若先调 UnInitHugeMemThreadLocal 把 poolIndex_ 置 -1，FreeHugeMem 会因 id 无效早退，
    // 不归还 huge block，污染全局 hugeMemArray_ 配额，导致后续 MemoryPoolUt.test_huge_memory 失败。
    ReleaseHugeMem(nullptr, false);
}

// 冷路径（>64KB）block 释放后：SYS_TAG block 经 FreeImpl 直接 std::free（见 block_pool.h FreeImpl），
// CheckDoubleFree 读已释放内存（UB），magic 几乎不可能匹配 → magic 短路返回 false（漏报，符合设计）。
// best-effort 验证：依赖 OS 不立即复用该页，若复用且恰好填入 MAGIC 字节会 flaky
TEST_F(CheckDoubleFreeUt, ColdPath_Over64KB_NotReported)
{
    void* addr = Allocate(70 * 1024);    // >64KB，走冷路径（SYS_TAG）
    ASSERT_NE(addr, nullptr);
    DeAllocate(addr);                    // FreeImpl：SYS_TAG 直接 std::free，block 已归还系统
    EXPECT_FALSE(CheckDoubleFree(addr)); // magic 不匹配短路：漏报，符合设计
}

// 冷路径（>64KB）block 分配后未释放：cacheExt_ 仍为 MallocImpl 初始值 NOT_IN_CACHE（见 block_pool.h MallocImpl），
// magic_ == MAGIC（placement new 初始化），命中 NOT_IN_CACHE 短路分支
TEST_F(CheckDoubleFreeUt, ColdPath_NotInCacheShortCircuit)
{
    void* addr = Allocate(70 * 1024);    // >64KB 冷路径，MallocImpl 设 cacheExt_=NOT_IN_CACHE、magic_=MAGIC
    ASSERT_NE(addr, nullptr);
    EXPECT_FALSE(CheckDoubleFree(addr)); // ext==NOT_IN_CACHE → 短路返回 false
    DeAllocate(addr);                    // 清理（FreeImpl std::free）
}

// 活跃 block（同线程 alloc 后未释放）：cacheExt_ == CurrentThreadCacheAddr
// （CacheAllocImpl/PoolAlloc 分配时写入 this 指针），命中 CurrentThreadCacheAddr 短路分支（防活跃 block 误报）
TEST_F(CheckDoubleFreeUt, ActiveBlock_SameThread_NoFalsePositive)
{
    void* addr = Allocate(64);           // ≤64KB 走内存池，block 活跃
    ASSERT_NE(addr, nullptr);
    EXPECT_FALSE(CheckDoubleFree(addr)); // ext==CurrentThreadCacheAddr → 短路返回 false
    DeAllocate(addr);                    // 清理
}
