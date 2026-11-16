/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "acl/acl.h"
#include "opdev/object.h"
#include "bridge_pool.h"
#include "block_pool.h"
#include "thread_local_context.h"

using namespace op;
using namespace op::internal;

static int64_t dims[] = {2, 3, 4};
static int64_t strides[] = {12, 4, 1};

class BlockCacheLeakTest : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

static aclTensor* CreateTensor()
{
    return aclCreateTensor(dims, 3, ACL_FLOAT, strides, 0, ACL_FORMAT_ND, dims, 3, nullptr);
}

static void DestroyTensor(aclTensor* tensor) { aclDestroyTensor(tensor); }

static size_t GetProcessRss()
{
    FILE* f = fopen("/proc/self/status", "r");
    if (f == nullptr) {
        return 0;
    }
    char line[256];
    size_t rss = 0;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "VmRSS:", 6) == 0) {
            sscanf(line + 6, "%zu", &rss);
            rss *= 1024;
            break;
        }
    }
    fclose(f);
    return rss;
}

// ===== Test 1: Thread A alloc, Thread A free (baseline, should NOT leak) =====
TEST_F(BlockCacheLeakTest, test_alloc_free_same_thread_baseline)
{
    constexpr int32_t kIterCount = 10000;
    constexpr int32_t kBatchSize = 64;

    size_t rssBefore = GetProcessRss();

    for (int32_t round = 0; round < 5; round++) {
        std::vector<aclTensor*> tensors;
        tensors.reserve(kBatchSize);
        for (int32_t i = 0; i < kIterCount; i++) {
            aclTensor* t = CreateTensor();
            ASSERT_NE(t, nullptr);
            tensors.push_back(t);
            if (tensors.size() >= static_cast<size_t>(kBatchSize)) {
                for (auto* p : tensors) {
                    DestroyTensor(p);
                }
                tensors.clear();
            }
        }
        for (auto* p : tensors) {
            DestroyTensor(p);
        }
        tensors.clear();
    }

    size_t rssAfter = GetProcessRss();
    size_t rssDelta = (rssAfter > rssBefore) ? (rssAfter - rssBefore) : 0;
    printf("[Baseline] RSS before: %zu KB, after: %zu KB, delta: %zu KB\n", rssBefore / 1024, rssAfter / 1024,
           rssDelta / 1024);
}

// ===== Test 2: Thread A alloc, Thread B free (cross-thread, potential leak) =====
TEST_F(BlockCacheLeakTest, test_alloc_thread_a_free_thread_b_cross_thread)
{
    constexpr int32_t kTotalTensors = 50000;
    constexpr int32_t kQueueSize = 256;

    std::queue<aclTensor*> q;
    std::mutex mtx;
    std::condition_variable cv;
    std::atomic<bool> done{false};
    std::atomic<int32_t> freedCount{0};

    size_t rssBefore = GetProcessRss();

    auto producer = [&]() {
        for (int32_t i = 0; i < kTotalTensors; i++) {
            aclTensor* t = CreateTensor();
            if (t == nullptr) {
                printf("CreateTensor failed at %d\n", i);
                break;
            }
            {
                std::unique_lock<std::mutex> lock(mtx);
                cv.wait(lock, [&] { return q.size() < static_cast<size_t>(kQueueSize); });
                q.push(t);
            }
            cv.notify_one();
        }
        done = true;
        cv.notify_one();
    };

    auto consumer = [&]() {
        while (true) {
            aclTensor* t = nullptr;
            {
                std::unique_lock<std::mutex> lock(mtx);
                cv.wait(lock, [&] { return !q.empty() || done; });
                if (q.empty() && done) {
                    break;
                }
                if (!q.empty()) {
                    t = q.front();
                    q.pop();
                }
            }
            if (t != nullptr) {
                cv.notify_one();
                DestroyTensor(t);
                freedCount++;
            }
        }
    };

    std::thread tA(producer);
    std::thread tB(consumer);
    tA.join();
    tB.join();

    EXPECT_EQ(freedCount.load(), kTotalTensors);

    size_t rssAfter = GetProcessRss();
    size_t rssDelta = (rssAfter > rssBefore) ? (rssAfter - rssBefore) : 0;
    printf("[CrossThread A->B] Freed: %d, RSS before: %zu KB, after: %zu KB, delta: %zu KB\n", freedCount.load(),
           rssBefore / 1024, rssAfter / 1024, rssDelta / 1024);
}

// ===== Test 3: Thread A alloc, partial free in A, partial free in B =====
TEST_F(BlockCacheLeakTest, test_alloc_a_partial_free_a_partial_free_b)
{
    constexpr int32_t kTotalTensors = 50000;
    constexpr int32_t kFreeInARatio = 2; // every 3 tensors: 2 freed in A, 1 freed in B
    constexpr int32_t kBatchGroup = 3;

    std::queue<aclTensor*> q;
    std::mutex mtx;
    std::condition_variable cv;
    std::atomic<bool> done{false};
    std::atomic<int32_t> freedInA{0};
    std::atomic<int32_t> freedInB{0};

    size_t rssBefore = GetProcessRss();

    auto producer = [&]() {
        std::vector<aclTensor*> localFree;
        for (int32_t i = 0; i < kTotalTensors; i++) {
            aclTensor* t = CreateTensor();
            if (t == nullptr) {
                printf("CreateTensor failed at %d\n", i);
                break;
            }
            if (i % kBatchGroup < kFreeInARatio) {
                localFree.push_back(t);
                if (localFree.size() >= 32) {
                    for (auto* p : localFree) {
                        DestroyTensor(p);
                        freedInA++;
                    }
                    localFree.clear();
                }
            } else {
                std::unique_lock<std::mutex> lock(mtx);
                cv.wait(lock, [&] { return q.size() < 256; });
                q.push(t);
                cv.notify_one();
            }
        }
        for (auto* p : localFree) {
            DestroyTensor(p);
            freedInA++;
        }
        localFree.clear();
        done = true;
        cv.notify_one();
    };

    auto consumer = [&]() {
        while (true) {
            aclTensor* t = nullptr;
            {
                std::unique_lock<std::mutex> lock(mtx);
                cv.wait(lock, [&] { return !q.empty() || done; });
                if (q.empty() && done) {
                    break;
                }
                if (!q.empty()) {
                    t = q.front();
                    q.pop();
                }
            }
            if (t != nullptr) {
                cv.notify_one();
                DestroyTensor(t);
                freedInB++;
            }
        }
    };

    std::thread tA(producer);
    std::thread tB(consumer);
    tA.join();
    tB.join();

    int32_t totalFreed = freedInA.load() + freedInB.load();
    EXPECT_EQ(totalFreed, kTotalTensors);

    size_t rssAfter = GetProcessRss();
    size_t rssDelta = (rssAfter > rssBefore) ? (rssAfter - rssBefore) : 0;
    printf("[Mixed A+B] Freed in A: %d, Freed in B: %d, total: %d, "
           "RSS before: %zu KB, after: %zu KB, delta: %zu KB\n",
           freedInA.load(), freedInB.load(), totalFreed, rssBefore / 1024, rssAfter / 1024, rssDelta / 1024);
}

// ===== Test 4: Repeated thread create/destroy to accumulate leak =====
TEST_F(BlockCacheLeakTest, test_repeated_thread_create_destroy_leak)
{
    constexpr int32_t kRounds = 20;
    constexpr int32_t kTensorsPerRound = 1000;

    size_t rssBefore = GetProcessRss();

    for (int32_t round = 0; round < kRounds; round++) {
        std::vector<aclTensor*> tensors;
        tensors.reserve(kTensorsPerRound);

        // Thread A allocates
        auto allocFunc = [&]() {
            for (int32_t i = 0; i < kTensorsPerRound; i++) {
                aclTensor* t = CreateTensor();
                if (t != nullptr) {
                    tensors.push_back(t);
                }
            }
        };

        {
            std::thread tA(allocFunc);
            tA.join();
        }

        // Thread B frees all tensors, then thread B exits
        // This simulates: block allocated in A's BlockCache, freed in B's BlockCache
        // M2 fix: When B exits, B's BlockCache destructor frees cached blocks via std::free
        {
            auto freeFunc = [&]() {
                for (auto* p : tensors) {
                    DestroyTensor(p);
                }
            };
            std::thread tB(freeFunc);
            tB.join();
        }

        tensors.clear();
    }

    size_t rssAfter = GetProcessRss();
    size_t rssDelta = (rssAfter > rssBefore) ? (rssAfter - rssBefore) : 0;
    printf("[RepeatedThread] %d rounds x %d tensors, "
           "RSS before: %zu KB, after: %zu KB, delta: %zu KB\n",
           kRounds, kTensorsPerRound, rssBefore / 1024, rssAfter / 1024, rssDelta / 1024);

    // M2 fix ensures B's destructor frees cacheHead_, no accumulation across rounds.
    // Measured rssDelta after fix is approximately 5MB.
    // Threshold 15MB equals measured value times 3, leaving room for glibc per-thread arena noise (typical 5~10MB).
    constexpr size_t kMaxRssGrowthBytes = 15 * 1024 * 1024;
    EXPECT_LT(rssDelta, kMaxRssGrowthBytes);
}

// ===== Test 5: Multiple consumers free tensors from single producer =====
TEST_F(BlockCacheLeakTest, test_one_producer_multiple_consumers)
{
    constexpr int32_t kTotalTensors = 60000;
    constexpr int32_t kConsumerCount = 4;

    std::queue<aclTensor*> q;
    std::mutex mtx;
    std::condition_variable cv;
    std::atomic<bool> done{false};
    std::atomic<int32_t> freedCount{0};

    size_t rssBefore = GetProcessRss();

    auto producer = [&]() {
        for (int32_t i = 0; i < kTotalTensors; i++) {
            aclTensor* t = CreateTensor();
            if (t == nullptr) {
                printf("CreateTensor failed at %d\n", i);
                break;
            }
            {
                std::unique_lock<std::mutex> lock(mtx);
                cv.wait(lock, [&] { return q.size() < 512; });
                q.push(t);
            }
            cv.notify_one();
        }
        done = true;
        cv.notify_all();
    };

    auto consumer = [&]() {
        while (true) {
            aclTensor* t = nullptr;
            {
                std::unique_lock<std::mutex> lock(mtx);
                cv.wait(lock, [&] { return !q.empty() || done; });
                if (q.empty() && done) {
                    break;
                }
                if (!q.empty()) {
                    t = q.front();
                    q.pop();
                }
            }
            if (t != nullptr) {
                cv.notify_one();
                DestroyTensor(t);
                freedCount++;
            }
        }
    };

    std::thread tProd(producer);
    std::vector<std::thread> consumers;
    for (int32_t i = 0; i < kConsumerCount; i++) {
        consumers.emplace_back(consumer);
    }

    tProd.join();
    for (auto& c : consumers) {
        c.join();
    }

    EXPECT_EQ(freedCount.load(), kTotalTensors);

    size_t rssAfter = GetProcessRss();
    size_t rssDelta = (rssAfter > rssBefore) ? (rssAfter - rssBefore) : 0;
    printf("[1-Producer %d-Consumers] Freed: %d, "
           "RSS before: %zu KB, after: %zu KB, delta: %zu KB\n",
           kConsumerCount, freedCount.load(), rssBefore / 1024, rssAfter / 1024, rssDelta / 1024);
}

// ===== Test 6: Directly test BlockCache::CacheAlloc/CacheFree cross-thread =====
TEST_F(BlockCacheLeakTest, test_block_cache_direct_cross_thread)
{
    constexpr int32_t kIterCount = 50000;
    constexpr int32_t kBlockSize = 256;

    std::queue<void*> blockQ;
    std::mutex mtx;
    std::condition_variable cv;
    std::atomic<bool> done{false};
    std::atomic<int32_t> freedCount{0};

    size_t rssBefore = GetProcessRss();

    auto producer = [&]() {
        for (int32_t i = 0; i < kIterCount; i++) {
            void* p = BlockCache::CacheAlloc(kBlockSize);
            if (p == nullptr) {
                printf("CacheAlloc failed at %d\n", i);
                break;
            }
            {
                std::unique_lock<std::mutex> lock(mtx);
                cv.wait(lock, [&] { return blockQ.size() < 256; });
                blockQ.push(p);
            }
            cv.notify_one();
        }
        done = true;
        cv.notify_one();
    };

    auto consumer = [&]() {
        while (true) {
            void* p = nullptr;
            {
                std::unique_lock<std::mutex> lock(mtx);
                cv.wait(lock, [&] { return !blockQ.empty() || done; });
                if (blockQ.empty() && done) {
                    break;
                }
                if (!blockQ.empty()) {
                    p = blockQ.front();
                    blockQ.pop();
                }
            }
            if (p != nullptr) {
                cv.notify_one();
                BlockCache::CacheFree(p);
                freedCount++;
            }
        }
    };

    std::thread tA(producer);
    std::thread tB(consumer);
    tA.join();
    tB.join();

    EXPECT_EQ(freedCount.load(), kIterCount);

    size_t rssAfter = GetProcessRss();
    size_t rssDelta = (rssAfter > rssBefore) ? (rssAfter - rssBefore) : 0;
    printf("[DirectCacheCrossThread] Freed: %d, "
           "RSS before: %zu KB, after: %zu KB, delta: %zu KB\n",
           freedCount.load(), rssBefore / 1024, rssAfter / 1024, rssDelta / 1024);
}

// ===== Test 7: Stress test with alternating alloc/free from multiple threads =====
TEST_F(BlockCacheLeakTest, test_stress_multi_thread_alloc_free)
{
    constexpr int32_t kThreadCount = 8;
    constexpr int32_t kIterPerThread = 10000;
    constexpr int32_t kBatchSize = 32;

    size_t rssBefore = GetProcessRss();

    auto worker = [&](int32_t threadId) {
        for (int32_t round = 0; round < 10; round++) {
            std::vector<aclTensor*> tensors;
            tensors.reserve(kBatchSize);
            for (int32_t i = 0; i < kIterPerThread; i++) {
                aclTensor* t = CreateTensor();
                if (t != nullptr) {
                    tensors.push_back(t);
                }
                if (tensors.size() >= static_cast<size_t>(kBatchSize)) {
                    for (auto* p : tensors) {
                        DestroyTensor(p);
                    }
                    tensors.clear();
                }
            }
            for (auto* p : tensors) {
                DestroyTensor(p);
            }
        }
        printf("  Thread %d done\n", threadId);
    };

    std::vector<std::thread> threads;
    for (int32_t i = 0; i < kThreadCount; i++) {
        threads.emplace_back(worker, i);
    }
    for (auto& t : threads) {
        t.join();
    }

    size_t rssAfter = GetProcessRss();
    size_t rssDelta = (rssAfter > rssBefore) ? (rssAfter - rssBefore) : 0;
    printf("[Stress %d threads] RSS before: %zu KB, after: %zu KB, delta: %zu KB\n", kThreadCount, rssBefore / 1024,
           rssAfter / 1024, rssDelta / 1024);
}

// ===== Test 8: Repeated cross-thread with B exiting — definitive leak test =====
TEST_F(BlockCacheLeakTest, test_cross_thread_b_exit_accumulates_leak)
{
    constexpr int32_t kRounds = 30;
    constexpr int32_t kTensorsPerRound = 2000;

    size_t rssBefore = GetProcessRss();
    std::vector<size_t> rssSnapshots;
    rssSnapshots.push_back(rssBefore);

    for (int32_t round = 0; round < kRounds; round++) {
        std::vector<aclTensor*> tensors;
        tensors.reserve(kTensorsPerRound);

        // Allocate all in main thread (Thread A)
        for (int32_t i = 0; i < kTensorsPerRound; i++) {
            aclTensor* t = CreateTensor();
            if (t != nullptr) {
                tensors.push_back(t);
            }
        }

        // Free all in a short-lived thread (Thread B), B exits after freeing
        // M2 fix: B's BlockCache is thread_local, ~BlockCache() frees cached blocks via std::free
        {
            std::thread tB([&]() {
                for (auto* p : tensors) {
                    DestroyTensor(p);
                }
            });
            tB.join();
        }

        tensors.clear();

        // Sample RSS every 10 rounds
        if ((round + 1) % 10 == 0) {
            size_t rssNow = GetProcessRss();
            rssSnapshots.push_back(rssNow);
            printf("  Round %d/%d, RSS: %zu KB (delta from start: %zu KB)\n", round + 1, kRounds, rssNow / 1024,
                   (rssNow > rssBefore ? rssNow - rssBefore : 0) / 1024);
        }
    }

    size_t rssAfter = GetProcessRss();
    size_t rssDelta = (rssAfter > rssBefore) ? (rssAfter - rssBefore) : 0;
    printf("[DefinitiveLeakTest] %d rounds, RSS before: %zu KB, after: %zu KB, delta: %zu KB\n", kRounds,
           rssBefore / 1024, rssAfter / 1024, rssDelta / 1024);

    // Print growth trend
    printf("RSS growth trend:\n");
    for (size_t i = 0; i < rssSnapshots.size(); i++) {
        printf("  [%zu] %zu KB\n", i * 10, rssSnapshots[i] / 1024);
    }

    // M2 fix ensures B's destructor frees cacheHead_, no accumulation across rounds.
    // Measured rssDelta after fix is approximately 1MB; before fix it grew linearly to ~194MB across 30 rounds.
    // Threshold 30MB catches regression at approximately 15% of pre-fix baseline.
    // Larger margin than Test4/10 because 30 rounds of thread create/destroy amplifies glibc arena noise.
    constexpr size_t kMaxRssGrowthBytes = 30 * 1024 * 1024;
    EXPECT_LT(rssDelta, kMaxRssGrowthBytes);
}

// ===== Test 10: Repeated thread create/destroy with BlockCache direct API —
//               verifies memory is freed when thread exits, no accumulation across rounds =====
TEST_F(BlockCacheLeakTest, test_repeated_thread_destroy_frees_memory)
{
    constexpr int32_t kRounds = 30;
    constexpr int32_t kAllocsPerRound = 2000;
    constexpr int32_t kBlockSize = 256;

    size_t rssBefore = GetProcessRss();
    std::vector<size_t> rssSnapshots;
    rssSnapshots.push_back(rssBefore);

    for (int32_t round = 0; round < kRounds; round++) {
        std::vector<void*> blocks;
        blocks.reserve(kAllocsPerRound);

        {
            std::thread tAlloc([&]() {
                for (int32_t i = 0; i < kAllocsPerRound; i++) {
                    void* p = BlockCache::CacheAlloc(kBlockSize);
                    if (p != nullptr) {
                        blocks.push_back(p);
                    }
                }
            });
            tAlloc.join();
        }

        {
            std::thread tFree([&]() {
                for (auto* b : blocks) {
                    BlockCache::CacheFree(b);
                }
            });
            tFree.join();
        }

        blocks.clear();

        if ((round + 1) % 10 == 0) {
            size_t rssNow = GetProcessRss();
            rssSnapshots.push_back(rssNow);
            printf("  Round %d/%d, RSS: %zu KB (delta from start: %zu KB)\n", round + 1, kRounds, rssNow / 1024,
                   (rssNow > rssBefore ? rssNow - rssBefore : 0) / 1024);
        }
    }

    size_t rssAfter = GetProcessRss();
    size_t rssDelta = (rssAfter > rssBefore) ? (rssAfter - rssBefore) : 0;
    printf("[RepeatedThreadDestroy] %d rounds x %d blocks, "
           "RSS before: %zu KB, after: %zu KB, delta: %zu KB\n",
           kRounds, kAllocsPerRound, rssBefore / 1024, rssAfter / 1024, rssDelta / 1024);

    printf("RSS growth trend:\n");
    for (size_t i = 0; i < rssSnapshots.size(); i++) {
        printf("  [%zu] %zu KB\n", i * 10, rssSnapshots[i] / 1024);
    }

    // M2 fix: thread exit triggers ~BlockCache which frees cacheHead_ via std::free.
    // Test 10 directly exercises BlockCache::CacheAlloc/CacheFree API (bypassing aclTensor).
    // Measured rssDelta after fix is approximately 0MB.
    // Threshold 15MB equals measured value times 3, leaving room for glibc per-thread arena noise (typical 5~10MB).
    constexpr size_t kMaxRssGrowthBytes = 15 * 1024 * 1024;
    EXPECT_LT(rssDelta, kMaxRssGrowthBytes);
}

// ===== Test 11: Multiple threads alloc and free in same thread —
//               verifies no leak when each thread self-manages =====
TEST_F(BlockCacheLeakTest, test_multi_thread_self_manage_no_leak)
{
    constexpr int32_t kThreadCount = 8;
    constexpr int32_t kRoundsPerThread = 100;
    constexpr int32_t kAllocsPerRound = 100;
    constexpr int32_t kBlockSize = 128;

    size_t rssBefore = GetProcessRss();

    auto worker = [&]() {
        for (int32_t round = 0; round < kRoundsPerThread; round++) {
            std::vector<void*> blocks;
            blocks.reserve(kAllocsPerRound);
            for (int32_t i = 0; i < kAllocsPerRound; i++) {
                void* p = BlockCache::CacheAlloc(kBlockSize);
                if (p != nullptr) {
                    blocks.push_back(p);
                }
            }
            for (auto* b : blocks) {
                BlockCache::CacheFree(b);
            }
        }
    };

    std::vector<std::thread> threads;
    for (int32_t i = 0; i < kThreadCount; i++) {
        threads.emplace_back(worker);
    }
    for (auto& t : threads) {
        t.join();
    }

    size_t rssAfter = GetProcessRss();
    size_t rssDelta = (rssAfter > rssBefore) ? (rssAfter - rssBefore) : 0;
    printf("[MultiThreadSelfManage] %d threads, "
           "RSS before: %zu KB, after: %zu KB, delta: %zu KB\n",
           kThreadCount, rssBefore / 1024, rssAfter / 1024, rssDelta / 1024);
}

// ===== Test 12: High concurrency stress test —
//               verifies correctness under 200 threads x 1000 iterations =====
TEST_F(BlockCacheLeakTest, test_stress_high_concurrency)
{
    constexpr size_t kStressThreadCount = 200;
    constexpr size_t kStressIterCount = 1000;
    constexpr size_t kBlockSize = 256;

    std::atomic<size_t> successCount{0};
    std::atomic<size_t> failCount{0};

    auto worker = [&]() {
        for (size_t i = 0; i < kStressIterCount; i++) {
            void* p = BlockCache::CacheAlloc(kBlockSize);
            if (p != nullptr) {
                BlockCache::CacheFree(p);
                successCount++;
            } else {
                failCount++;
            }
        }
    };

    std::vector<std::thread> threads;
    for (size_t i = 0; i < kStressThreadCount; i++) {
        threads.emplace_back(worker);
    }
    for (auto& t : threads) {
        t.join();
    }

    OP_LOGI("[Test12] high concurrency stress test, threadCount [%zu], iterPerThread [%zu], "
            "success [%zu], fail [%zu].",
            kStressThreadCount, kStressIterCount, successCount.load(), failCount.load());
    EXPECT_EQ(successCount.load(), kStressThreadCount * kStressIterCount);
    EXPECT_EQ(failCount.load(), 0U);
}
