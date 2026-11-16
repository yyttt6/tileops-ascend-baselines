/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <array>
#include "gtest/gtest.h"
#include "wqmm_test_utils.h"
#include "blaze/gemm/block/block_scheduler_wqmm.h"

namespace {
using namespace WeightQuantBatchMatmulUT;
using Shape = asc::te::shape<uint64_t, uint64_t, uint64_t>;
using Scheduler = Blaze::Gemm::Block::BlockSchedulerWqmmTailResplit<Shape>;
using Tile = std::array<uint64_t, 7>; // range, local M/N index, M/N origin, M/N extent
constexpr uint32_t MAIN = 0, FIRST_TAIL = 1, SECOND_TAIL = 2;
constexpr uint64_t MAX_CORES = 8, CASES = 8;
struct ScheduleCase {
    uint64_t m, n, k, recordBase, capacity;
    Scheduler::Params params;
};
// Each core has its own bounded slice; unused slices retain poison. Output is
// compared exactly, not by hashes, with a global round-robin ownership oracle.
template <uint32_t Range>
void Append(const Scheduler& s, Tile* out, uint64_t capacity, uint64_t& count, uint64_t mi, uint64_t nCount)
{
    auto m = s.GetBlockCoordM(mi);
    for (uint64_t ni = 0; ni < nCount; ++ni) {
        auto n = s.GetBlockCoordN<Range>(ni);
        if (count < capacity)
            out[count] = {Range, mi, ni, m, n, s.GetBlockShapeM(m), s.GetBlockShapeN<Range>(n)};
        ++count;
    }
}
__global__ __aicore__ void SchedulerKernel(GM_ADDR configs, GM_ADDR records, GM_ADDR counts)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    uint64_t core, lane;
    if ASCEND_IS_AIC {
        core = AscendC::GetBlockIdx();
        lane = 0;
    } else {
        core = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        lane = 1 + AscendC::GetSubBlockIdx();
    }
    auto* cases = reinterpret_cast<const ScheduleCase*>(configs);
    auto* output = reinterpret_cast<Tile*>(records);
    auto* totals = reinterpret_cast<uint64_t*>(counts);
    for (uint64_t trial = 0; trial < CASES; ++trial) {
        const auto& c = cases[trial];
        if (core >= c.params.cubeNumBlocksM * c.params.cubeNumBlocksN)
            continue;
        Scheduler s(asc::te::make_shape(c.m, c.n, c.k), c.params);
        auto nums = s.GetBlockNums();
        auto* dst = output + c.recordBase + (lane * MAX_CORES + core) * c.capacity;
        uint64_t count = 0;
        for (uint64_t mi = 0; mi < asc::te::get<0>(nums); ++mi) {
            Append<MAIN>(s, dst, c.capacity, count, mi, asc::te::get<1>(nums));
            Append<FIRST_TAIL>(s, dst, c.capacity, count, mi, asc::te::get<2>(nums));
            Append<SECOND_TAIL>(s, dst, c.capacity, count, mi, asc::te::get<3>(nums));
        }
        uint64_t at = (trial * 3 * MAX_CORES + lane * MAX_CORES + core) * 2;
        totals[at] = count;
        totals[at + 1] = s.GetCoreNums();
    }
}
using Streams = std::vector<std::vector<Tile>>;
Streams Oracle(const ScheduleCase& c)
{
    const auto& p = c.params;
    Streams expected(p.cubeNumBlocksM * p.cubeNumBlocksN);
    const std::array<uint64_t, 3> counts{p.mainBlockCount, p.firstTailBlockCount, p.secondTailBlockCount};
    const std::array<uint64_t, 3> sizes{p.mainBlockSize, p.firstTailBlockSize, p.secondTailBlockSize};
    auto span = (c.m + p.cubeNumBlocksM - 1) / p.cubeNumBlocksM;
    for (uint64_t cm = 0; cm < p.cubeNumBlocksM; ++cm) {
        auto stop = std::min(c.m, (cm + 1) * span);
        uint64_t mi = 0;
        for (auto row = cm * span; row < stop; row += p.mL1Tile, ++mi) {
            uint64_t start = 0;
            for (uint32_t stage = 0; stage < 3; ++stage) {
                std::vector<uint64_t> ni(p.cubeNumBlocksN, 0);
                for (uint64_t j = 0; j < counts[stage]; ++j) {
                    auto owner = (j + (stage == SECOND_TAIL ? counts[FIRST_TAIL] : 0)) % p.cubeNumBlocksN;
                    auto col = start + j * sizes[stage];
                    expected[cm * p.cubeNumBlocksN + owner].push_back({stage, mi, ni[owner]++, row, col,
                                                                       std::min(stop - row, uint64_t(p.mL1Tile)),
                                                                       std::min(c.n - col, sizes[stage])});
                }
                start += counts[stage] * sizes[stage];
            }
        }
    }
    return expected;
}
TEST(WqmmSchedulerTest, BoundarySchedulesMatchOnAicAndBothAivSubcores)
{
    // Params: M tile, three N counts, three N sizes, M/N core counts.
    std::vector<ScheduleCase> cases{
        {1, 1, 1, 0, 0, {16, 1, 0, 0, 16, 8, 4, 1, 1}},       // singleton
        {33, 96, 256, 0, 0, {16, 3, 0, 0, 32, 16, 8, 2, 2}},  // main only, M tail
        {17, 29, 257, 0, 0, {16, 0, 2, 0, 32, 16, 8, 1, 4}},  // first tail only, idle N cores
        {1, 17, 64, 0, 0, {16, 0, 0, 3, 32, 16, 8, 1, 4}},    // second tail only
        {65, 101, 513, 0, 0, {16, 2, 1, 3, 32, 16, 8, 2, 4}}, // all ranges, N2 wraps
        {31, 157, 512, 0, 0, {16, 2, 5, 2, 32, 16, 8, 1, 4}}, // first tail spans a core round
        {33, 95, 128, 0, 0, {16, 2, 2, 0, 32, 16, 8, 2, 4}},  // no second tail
        {5, 65, 129, 0, 0, {2, 2, 0, 1, 32, 16, 8, 4, 2}},    // idle M core, empty first tail
    };
    ASSERT_EQ(cases.size(), CASES);
    uint64_t recordCount = 0;
    for (auto& c : cases) {
        auto oracle = Oracle(c);
        for (const auto& stream : oracle)
            c.capacity = std::max(c.capacity, uint64_t(stream.size()));
        ++c.capacity; // guard record after the longest stream
        c.recordBase = recordCount;
        recordCount += c.capacity * 3 * MAX_CORES;
    }
    GmBuffer configs(cases.size() * sizeof(ScheduleCase)), records(recordCount * sizeof(Tile)),
        counts(CASES * 3 * MAX_CORES * 2 * 8);
    configs.Set(cases);
    std::memset(records.Get(), 0xA5, recordCount * sizeof(Tile));
    std::memset(counts.Get(), 0xFF, CASES * 3 * MAX_CORES * 2 * 8);
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    ASSERT_TRUE(KERNEL_RUN_KF(SchedulerKernel, MAX_CORES, configs.Get(), records.Get(), counts.Get()));
    auto* actual = reinterpret_cast<Tile*>(records.Get());
    auto* total = reinterpret_cast<uint64_t*>(counts.Get());
    uint64_t mTails = 0, nTails = 0, emptyCores = 0;
    for (uint64_t trial = 0; trial < CASES; ++trial) {
        SCOPED_TRACE(trial);
        const auto& c = cases[trial];
        auto expected = Oracle(c);
        uint64_t area = 0;
        for (uint64_t core = 0; core < MAX_CORES; ++core) {
            for (uint64_t lane = 0; lane < 3; ++lane) {
                auto at = (trial * 3 * MAX_CORES + lane * MAX_CORES + core) * 2;
                auto* data = actual + c.recordBase + (lane * MAX_CORES + core) * c.capacity;
                uint64_t size = core < expected.size() ? expected[core].size() : 0;
                if (core < expected.size()) {
                    ASSERT_EQ(total[at], size);
                    ASSERT_EQ(total[at + 1], expected.size());
                    for (uint64_t j = 0; j < size; ++j)
                        ASSERT_EQ(data[j], expected[core][j]);
                } else {
                    ASSERT_EQ(total[at], UINT64_MAX);
                }
                for (uint64_t j = size; j < c.capacity; ++j)
                    for (auto word : data[j])
                        ASSERT_EQ(word, 0xA5A5A5A5A5A5A5A5ULL);
            }
            if (core >= expected.size())
                continue;
            emptyCores += expected[core].empty();
            for (const auto& t : expected[core]) {
                ASSERT_GT(t[5], 0U);
                ASSERT_GT(t[6], 0U);
                area += t[5] * t[6];
                mTails += t[5] < uint64_t(c.params.mL1Tile);
                auto tile = t[0] == MAIN ?
                                c.params.mainBlockSize :
                                (t[0] == FIRST_TAIL ? c.params.firstTailBlockSize : c.params.secondTailBlockSize);
                nTails += t[6] < tile;
            }
        }
        ASSERT_EQ(area, c.m * c.n);
    }
    EXPECT_GT(mTails, 0U);
    EXPECT_GT(nTails, 0U);
    EXPECT_GT(emptyCores, 0U);
}
} // namespace
