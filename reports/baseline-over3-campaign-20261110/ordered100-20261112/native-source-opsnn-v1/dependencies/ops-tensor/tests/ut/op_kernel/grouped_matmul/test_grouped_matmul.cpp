/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * Licensed under the CANN Open Software License Agreement Version 2.0 (the "License");
 * you may not use this file except in compliance with the License. You may obtain a copy of the
 * License at
 *
 * https://www.hiascend.com/software/ascend-cann-license
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the
 * License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 * express or implied. See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * \file test_grouped_matmul.cpp
 * \brief Non-quant grouped matmul kernel smoke tests, covering sparse groupListType=2 and
 *        dense regression paths. CPU-debug KERNEL_RUN_KF is smoke-only (fork semantics make
 *        child-side GM writes unobservable in the parent), so the assertions check that the
 *        kernel completes without crash/exit failures across the addressing paths.
 */
#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>
#include "gtest/gtest.h"
#include "kernel_ut_runner.h"
#include "tikicpulib.h"
#include "grouped_matmul_kernel_ut.h"

namespace {
constexpr int64_t K_DIM = 64;
constexpr int64_t N_DIM = 64;
constexpr uint32_t BLOCK_NUM = 1;
constexpr uint32_t GROUP_TYPE_SPLIT_M = 0;
constexpr int32_t GROUP_TYPE_NO_SPLIT = -1;
constexpr uint32_t GROUP_TYPE_SPLIT_K = 2;

class GmBuffer {
public:
    explicit GmBuffer(size_t bytes) : ptr_(static_cast<GM_ADDR>(AscendC::GmAlloc(bytes))), size_(bytes) {}
    ~GmBuffer()
    {
        if (ptr_ != nullptr) {
            AscendC::GmFree(ptr_);
        }
    }
    GmBuffer(const GmBuffer&) = delete;
    GmBuffer& operator=(const GmBuffer&) = delete;

    GM_ADDR Get() const { return ptr_; }
    size_t Size() const { return size_; }
    void Fill(uint16_t pattern)
    {
        auto* data = reinterpret_cast<uint16_t*>(ptr_);
        for (size_t i = 0; i < size_ / sizeof(uint16_t); ++i) {
            data[i] = pattern;
        }
    }

private:
    GM_ADDR ptr_{};
    size_t size_{0};
};

using Shape = std::vector<uint64_t>;

// Builds the ListTensorDesc-compatible descriptor list consumed by AscendC::ListTensorDecode /
// GetDataPtr / GetDesc: [dataPtrOffset, (dim | count<<32), shape descriptors..., tensor pointers...].
// Every tensor gets a shape descriptor so both the pointer-only and shape-reading paths work.
class TensorListBuilder {
public:
    static std::unique_ptr<GmBuffer> Build(const std::vector<Shape>& shapes, const std::vector<GM_ADDR>& tensors)
    {
        const auto count = static_cast<uint64_t>(shapes.size());
        const auto dim = shapes.empty() ? 0U : static_cast<uint64_t>(shapes.front().size());
        const uint64_t descStructSize = dim == 0 ? 2U : 1U + dim;
        const uint64_t dataPtrOffset = sizeof(uint64_t) + count * descStructSize * sizeof(uint64_t);
        const size_t total = (1 + count * descStructSize + count) * sizeof(uint64_t);
        auto buffer = std::make_unique<GmBuffer>(total);
        auto* data = reinterpret_cast<uint64_t*>(buffer->Get());
        data[0] = dataPtrOffset;
        for (uint64_t i = 0; i < count; ++i) {
            auto* desc = data + 1 + i * descStructSize;
            const uint64_t header = i == 0 ? (dim | (count << 32U)) : dim;
            desc[0] = header;
            for (uint64_t j = 0; j < dim; ++j) {
                desc[1 + j] = shapes[i][j];
            }
        }
        auto* ptrs = data + 1 + count * descStructSize;
        for (uint64_t i = 0; i < count; ++i) {
            ptrs[i] = reinterpret_cast<uint64_t>(tensors[i]);
        }
        return buffer;
    }
};

// Wraps a single packed tensor as a one-entry tensor list.
std::unique_ptr<GmBuffer> WrapSingle(GM_ADDR tensor, const Shape& shape)
{
    return TensorListBuilder::Build({shape}, {tensor});
}

struct GroupListDesc {
    std::vector<int64_t> entries; // flat [groupIdx, groupSize] pairs for sparse, values otherwise
    bool sparse{false};
};

std::unique_ptr<GmBuffer> BuildGroupList(const GroupListDesc& desc)
{
    const size_t items = desc.entries.size();
    auto buffer = std::make_unique<GmBuffer>(items * sizeof(int64_t));
    auto* data = reinterpret_cast<int64_t*>(buffer->Get());
    for (size_t i = 0; i < items; ++i) {
        data[i] = desc.entries[i];
    }
    return buffer;
}

GMMUT::GmmTilingData MakeTiling(uint32_t groupNum, int32_t groupType, uint32_t groupListType, uint64_t singleX,
                                uint64_t singleWeight, uint64_t singleY, uint32_t hasBias, int64_t m)
{
    GMMUT::GmmTilingData t{};
    t.groupNum = groupNum;
    t.groupType = groupType;
    t.groupListType = groupListType;
    t.singleX = singleX;
    t.singleWeight = singleWeight;
    t.singleY = singleY;
    t.hasBias = hasBias;
    t.weightNoL2Cache = 0;
    t.mTailCnt = 1;
    t.nTailCnt = 1;
    t.m = m;
    t.n = N_DIM;
    t.k = K_DIM;
    t.baseM = 16;
    t.baseN = 16;
    t.baseK = static_cast<uint32_t>(K_DIM);
    t.stepKa = 1;
    t.stepKb = 1;
    t.dbL0C = 1;
    return t;
}

// Sparse single-x / single-weight / single-y with bias and a trailing empty group.
// groupList = [[0, 16], [2, 24], [3, 16], [1, 0]]: group 1 is empty and back-loaded, so the
// kernel must stop at the first empty group and address weight/bias slices by the first column.
void RunSparseSingleTensorCase(bool withBias)
{
    constexpr uint32_t groupNum = 4;
    constexpr int64_t totalM = 16 + 24 + 16;
    constexpr int64_t groupTotal = 4;

    GmBuffer x(static_cast<size_t>(totalM * K_DIM) * sizeof(half));
    GmBuffer weight(static_cast<size_t>(groupTotal * K_DIM * N_DIM) * sizeof(half));
    GmBuffer bias(static_cast<size_t>(groupTotal * N_DIM) * sizeof(half));
    GmBuffer y(static_cast<size_t>(totalM * N_DIM) * sizeof(half));
    GmBuffer tiling(sizeof(GMMUT::GmmTilingData));
    x.Fill(0x3C00U);
    weight.Fill(0x3C00U);
    bias.Fill(0x3C00U);
    y.Fill(0U);

    auto xList = WrapSingle(x.Get(), {static_cast<uint64_t>(totalM), static_cast<uint64_t>(K_DIM)});
    auto wList = WrapSingle(weight.Get(), {static_cast<uint64_t>(K_DIM), static_cast<uint64_t>(N_DIM)});
    auto biasList = WrapSingle(bias.Get(), {static_cast<uint64_t>(N_DIM)});
    auto yList = WrapSingle(y.Get(), {static_cast<uint64_t>(totalM), static_cast<uint64_t>(N_DIM)});
    auto groupList = BuildGroupList({{0, 16, 2, 24, 3, 16, 1, 0}, true});

    auto t = MakeTiling(groupNum, GROUP_TYPE_SPLIT_M, 2U, 1U, 1U, 1U, withBias ? 1U : 0U, totalM);
    *reinterpret_cast<GMMUT::GmmTilingData*>(tiling.Get()) = t;

    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    auto fn = gmm_kernel_entry<half, half, half, half>;
    ASSERT_NE(xList->Get(), nullptr);
    ASSERT_TRUE(KERNEL_RUN_KF(fn, BLOCK_NUM, xList->Get(), wList->Get(),
                              withBias ? biasList->Get() : static_cast<GM_ADDR>(nullptr), groupList->Get(),
                              yList->Get(), tiling.Get()))
        << "Sparse single-tensor kernel execution failed";
}

// Sparse multi-weight: weight and y are tensor lists, the actual group index (first column)
// selects the weight tensor while x keeps sequential offsets and y is indexed by row order.
void RunSparseMultiWeightCase()
{
    constexpr uint32_t groupNum = 3;
    const std::vector<int64_t> groupSizes = {16, 24, 16};
    constexpr int64_t totalM = 16 + 24 + 16;

    GmBuffer x(static_cast<size_t>(totalM * K_DIM) * sizeof(half));
    x.Fill(0x3C00U);
    std::vector<std::unique_ptr<GmBuffer>> weightTensors;
    std::vector<std::unique_ptr<GmBuffer>> yTensors;
    std::vector<Shape> weightShapes;
    std::vector<Shape> yShapes;
    std::vector<GM_ADDR> weightAddrs;
    std::vector<GM_ADDR> yAddrs;
    for (uint32_t i = 0; i < groupNum; ++i) {
        weightTensors.emplace_back(std::make_unique<GmBuffer>(static_cast<size_t>(K_DIM * N_DIM) * sizeof(half)));
        weightTensors.back()->Fill(0x3C00U);
        weightAddrs.push_back(weightTensors.back()->Get());
        weightShapes.push_back({static_cast<uint64_t>(K_DIM), static_cast<uint64_t>(N_DIM)});
        yTensors.emplace_back(std::make_unique<GmBuffer>(static_cast<size_t>(groupSizes[i] * N_DIM) * sizeof(half)));
        yTensors.back()->Fill(0U);
        yAddrs.push_back(yTensors.back()->Get());
        yShapes.push_back({static_cast<uint64_t>(groupSizes[i]), static_cast<uint64_t>(N_DIM)});
    }

    auto xList = WrapSingle(x.Get(), {static_cast<uint64_t>(totalM), static_cast<uint64_t>(K_DIM)});
    auto wList = TensorListBuilder::Build(weightShapes, weightAddrs);
    auto yList = TensorListBuilder::Build(yShapes, yAddrs);
    // Actual group indices out of order: row i uses weight[groupIdx_i], y is addressed by row i.
    auto groupList = BuildGroupList({{2, 16, 0, 24, 1, 16}, true});

    GmBuffer tiling(sizeof(GMMUT::GmmTilingData));
    auto t = MakeTiling(groupNum, GROUP_TYPE_SPLIT_M, 2U, 1U, 0U, 0U, 0U, totalM);
    *reinterpret_cast<GMMUT::GmmTilingData*>(tiling.Get()) = t;

    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    auto fn = gmm_kernel_entry<half, half, half, half>;
    ASSERT_TRUE(
        KERNEL_RUN_KF(fn, BLOCK_NUM, xList->Get(), wList->Get(), nullptr, groupList->Get(), yList->Get(), tiling.Get()))
        << "Sparse multi-weight kernel execution failed";
}

// m-m-s with groupListType=2: tiling normalizes the groupType to -1 (no split), so the kernel
// must not parse the groupList at all — it reads each x's first dim as M and keeps dense
// sequential addressing. A leading empty x group must be skipped (continue), not truncate the
// remaining groups (sparse break contract does not apply here).
void RunNoSplitMultiMultiSingleSparseListCase()
{
    constexpr uint32_t groupNum = 3;
    const std::vector<int64_t> groupSizes = {0, 16, 24}; // first group is empty on purpose
    constexpr int64_t totalM = 16 + 24;

    std::vector<std::unique_ptr<GmBuffer>> xTensors;
    std::vector<std::unique_ptr<GmBuffer>> weightTensors;
    std::vector<Shape> xShapes;
    std::vector<Shape> weightShapes;
    std::vector<GM_ADDR> xAddrs;
    std::vector<GM_ADDR> weightAddrs;
    for (uint32_t i = 0; i < groupNum; ++i) {
        xTensors.emplace_back(std::make_unique<GmBuffer>(
            static_cast<size_t>(std::max<int64_t>(groupSizes[i], 1) * K_DIM) * sizeof(half)));
        xTensors.back()->Fill(0x3C00U);
        xAddrs.push_back(xTensors.back()->Get());
        xShapes.push_back({static_cast<uint64_t>(groupSizes[i]), static_cast<uint64_t>(K_DIM)});
        weightTensors.emplace_back(std::make_unique<GmBuffer>(static_cast<size_t>(K_DIM * N_DIM) * sizeof(half)));
        weightTensors.back()->Fill(0x3C00U);
        weightAddrs.push_back(weightTensors.back()->Get());
        weightShapes.push_back({static_cast<uint64_t>(K_DIM), static_cast<uint64_t>(N_DIM)});
    }
    GmBuffer y(static_cast<size_t>(totalM * N_DIM) * sizeof(half));
    y.Fill(0U);

    auto xList = TensorListBuilder::Build(xShapes, xAddrs);
    auto wList = TensorListBuilder::Build(weightShapes, weightAddrs);
    auto yList = WrapSingle(y.Get(), {static_cast<uint64_t>(totalM), static_cast<uint64_t>(N_DIM)});
    // groupList content is irrelevant for groupType=-1; [E, 2] layout with a zero leading row.
    auto groupList = BuildGroupList({{0, 0, 1, 16, 2, 24}, true});

    GmBuffer tiling(sizeof(GMMUT::GmmTilingData));
    auto t = MakeTiling(groupNum, GROUP_TYPE_NO_SPLIT, 2U, 0U, 0U, 1U, 0U, totalM);
    *reinterpret_cast<GMMUT::GmmTilingData*>(tiling.Get()) = t;

    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    auto fn = gmm_kernel_entry<half, half, half, half>;
    ASSERT_TRUE(
        KERNEL_RUN_KF(fn, BLOCK_NUM, xList->Get(), wList->Get(), nullptr, groupList->Get(), yList->Get(), tiling.Get()))
        << "m-m-s sparse-list kernel execution failed";
}

// s-s-s with groupType=-1 and groupListType=2: no-split single-tensor path must keep the dense
// sequential weight offsets from the scheduler instead of the sparse direct positioning.
void RunNoSplitSingleTensorSparseListCase()
{
    constexpr uint32_t groupNum = 2;
    constexpr int64_t totalM = 40;

    GmBuffer x(static_cast<size_t>(totalM * K_DIM) * sizeof(half));
    GmBuffer weight(static_cast<size_t>(groupNum * K_DIM * N_DIM) * sizeof(half));
    GmBuffer y(static_cast<size_t>(totalM * N_DIM) * sizeof(half));
    GmBuffer tiling(sizeof(GMMUT::GmmTilingData));
    x.Fill(0x3C00U);
    weight.Fill(0x3C00U);
    y.Fill(0U);

    auto xList = WrapSingle(x.Get(), {static_cast<uint64_t>(totalM), static_cast<uint64_t>(K_DIM)});
    auto wList = WrapSingle(weight.Get(), {static_cast<uint64_t>(K_DIM), static_cast<uint64_t>(N_DIM)});
    auto yList = WrapSingle(y.Get(), {static_cast<uint64_t>(totalM), static_cast<uint64_t>(N_DIM)});
    auto groupList = BuildGroupList({{0, 20, 1, 20}, true});

    auto t = MakeTiling(groupNum, GROUP_TYPE_NO_SPLIT, 2U, 1U, 1U, 1U, 0U, totalM);
    *reinterpret_cast<GMMUT::GmmTilingData*>(tiling.Get()) = t;

    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    auto fn = gmm_kernel_entry<half, half, half, half>;
    ASSERT_TRUE(
        KERNEL_RUN_KF(fn, BLOCK_NUM, xList->Get(), wList->Get(), nullptr, groupList->Get(), yList->Get(), tiling.Get()))
        << "s-s-s no-split sparse-list kernel execution failed";
}

// Dense groupListType regression: 0 = cumsum offsets, 1 = per-group counts.
void RunDenseCase(uint32_t groupListType, const std::vector<int64_t>& groupListValues)
{
    const auto groupNum = static_cast<uint32_t>(groupListValues.size());
    int64_t totalM = 0;
    if (groupListType == 0) {
        totalM = groupListValues.back(); // cumulative offsets: the last value is the total M.
    } else {
        for (auto value : groupListValues) {
            totalM += value;
        }
    }

    GmBuffer x(static_cast<size_t>(totalM * K_DIM) * sizeof(half));
    GmBuffer weight(static_cast<size_t>(groupNum * K_DIM * N_DIM) * sizeof(half));
    GmBuffer y(static_cast<size_t>(totalM * N_DIM) * sizeof(half));
    GmBuffer tiling(sizeof(GMMUT::GmmTilingData));
    x.Fill(0x3C00U);
    weight.Fill(0x3C00U);
    y.Fill(0U);

    auto xList = WrapSingle(x.Get(), {static_cast<uint64_t>(totalM), static_cast<uint64_t>(K_DIM)});
    auto wList = WrapSingle(weight.Get(), {static_cast<uint64_t>(K_DIM), static_cast<uint64_t>(N_DIM)});
    auto yList = WrapSingle(y.Get(), {static_cast<uint64_t>(totalM), static_cast<uint64_t>(N_DIM)});
    auto groupList = BuildGroupList({groupListValues, false});

    auto t = MakeTiling(groupNum, GROUP_TYPE_SPLIT_M, groupListType, 1U, 1U, 1U, 0U, totalM);
    *reinterpret_cast<GMMUT::GmmTilingData*>(tiling.Get()) = t;

    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    auto fn = gmm_kernel_entry<half, half, half, half>;
    ASSERT_TRUE(
        KERNEL_RUN_KF(fn, BLOCK_NUM, xList->Get(), wList->Get(), nullptr, groupList->Get(), yList->Get(), tiling.Get()))
        << "Dense groupListType=" << groupListType << " kernel execution failed";
}
} // namespace

class GroupedMatmulKernelTest : public testing::Test {};

TEST_F(GroupedMatmulKernelTest, SparseSingleTensorWithBias)
{
    ASSERT_NO_FATAL_FAILURE(RunSparseSingleTensorCase(true));
}

TEST_F(GroupedMatmulKernelTest, SparseSingleTensorNoBias) { ASSERT_NO_FATAL_FAILURE(RunSparseSingleTensorCase(false)); }

TEST_F(GroupedMatmulKernelTest, SparseMultiWeight) { ASSERT_NO_FATAL_FAILURE(RunSparseMultiWeightCase()); }

TEST_F(GroupedMatmulKernelTest, NoSplitMultiMultiSingleSparseList)
{
    ASSERT_NO_FATAL_FAILURE(RunNoSplitMultiMultiSingleSparseListCase());
}

TEST_F(GroupedMatmulKernelTest, NoSplitSingleTensorSparseList)
{
    ASSERT_NO_FATAL_FAILURE(RunNoSplitSingleTensorSparseListCase());
}

TEST_F(GroupedMatmulKernelTest, SparseNzWeight)
{
    // NZ weight: per-group storage is CeilAlign(n, C0) * CeilAlign(k, 16) elements; the sparse
    // offset formula must use the NZ-aligned group size from the scheduler.
    constexpr uint32_t groupNum = 2;
    constexpr int64_t totalM = 64;
    constexpr size_t c0 = 16; // C0_ELEMENT<half>
    constexpr size_t groupWeightElems = static_cast<size_t>(N_DIM) * static_cast<size_t>(K_DIM);

    GmBuffer x(static_cast<size_t>(totalM * K_DIM) * sizeof(half));
    GmBuffer weight(static_cast<size_t>(groupNum) * groupWeightElems * sizeof(half));
    GmBuffer y(static_cast<size_t>(totalM * N_DIM) * sizeof(half));
    GmBuffer tiling(sizeof(GMMUT::GmmTilingData));
    x.Fill(0x3C00U);
    weight.Fill(0x3C00U);
    y.Fill(0U);

    auto xList = WrapSingle(x.Get(), {static_cast<uint64_t>(totalM), static_cast<uint64_t>(K_DIM)});
    auto wList = WrapSingle(weight.Get(), {static_cast<uint64_t>(K_DIM), static_cast<uint64_t>(N_DIM)});
    auto yList = WrapSingle(y.Get(), {static_cast<uint64_t>(totalM), static_cast<uint64_t>(N_DIM)});
    auto groupList = BuildGroupList({{0, 32, 1, 32}, true});

    auto t = MakeTiling(groupNum, GROUP_TYPE_SPLIT_M, 2U, 1U, 1U, 1U, 0U, totalM);
    *reinterpret_cast<GMMUT::GmmTilingData*>(tiling.Get()) = t;

    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    auto fn = gmm_kernel_entry<half, half, half, half, AscendC::Te::NDExtLayoutPtn, AscendC::Te::NZLayoutPtn>;
    ASSERT_TRUE(
        KERNEL_RUN_KF(fn, BLOCK_NUM, xList->Get(), wList->Get(), nullptr, groupList->Get(), yList->Get(), tiling.Get()))
        << "Sparse NZ-weight kernel execution failed";
}

TEST_F(GroupedMatmulKernelTest, DenseOffsetRegression) { ASSERT_NO_FATAL_FAILURE(RunDenseCase(0U, {16, 40})); }

TEST_F(GroupedMatmulKernelTest, DenseCountRegression) { ASSERT_NO_FATAL_FAILURE(RunDenseCase(1U, {16, 24})); }

TEST_F(GroupedMatmulKernelTest, InplaceAddRejectsSparseGroupList)
{
    // The inplace-add path only implements dense offset/count semantics; groupListType=2 must be
    // rejected by IsValidGroupParams so the kernel returns without touching any buffer.
    constexpr uint32_t groupNum = 2;
    constexpr int64_t totalM = 32;

    GmBuffer x(static_cast<size_t>(totalM * K_DIM) * sizeof(half));
    GmBuffer weight(static_cast<size_t>(groupNum * K_DIM * N_DIM) * sizeof(half));
    GmBuffer y(static_cast<size_t>(totalM * N_DIM) * sizeof(float));
    GmBuffer tiling(sizeof(GMMUT::GmmTilingData));
    auto groupList = BuildGroupList({{0, 16, 1, 16}, true});

    auto xList = WrapSingle(x.Get(), {static_cast<uint64_t>(totalM), static_cast<uint64_t>(K_DIM)});
    auto wList = WrapSingle(weight.Get(), {static_cast<uint64_t>(K_DIM), static_cast<uint64_t>(N_DIM)});
    auto yList = WrapSingle(y.Get(), {static_cast<uint64_t>(totalM), static_cast<uint64_t>(N_DIM)});

    auto t = MakeTiling(groupNum, GROUP_TYPE_SPLIT_K, 2U, 1U, 1U, 1U, 0U, totalM);
    *reinterpret_cast<GMMUT::GmmTilingData*>(tiling.Get()) = t;

    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    auto fn = gmm_kernel_entry<half, half, float, float, AscendC::Te::NDExtLayoutPtn, AscendC::Te::NDExtLayoutPtn,
                               Blaze::Gemm::MatmulOutputMode::INPLACE_ADD>;
    ASSERT_TRUE(
        KERNEL_RUN_KF(fn, BLOCK_NUM, xList->Get(), wList->Get(), nullptr, groupList->Get(), yList->Get(), tiling.Get()))
        << "Inplace-add sparse rejection path failed";
}
