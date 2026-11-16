/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file block_scheduler_wqmm.h
 * \brief 3-segment N-axis scheduler: mainBlock + firstTailBlock + secondTailBlock with tail resplit.
 */

#pragma once

#include "blaze/gemm/utils/common_utils.h"

namespace Blaze {
namespace Gemm {
namespace Block {

// Partition M across cores and distribute precomputed N main/first-tail/second-tail ranges.
// Coordinates and valid lengths use elements; K traversal belongs to the kernel and BlockMmad.
template <class ProblemShape_>
class BlockSchedulerWqmmTailResplit {
private:
    using BlockNums = asc::te::shape<uint64_t, uint64_t, uint64_t, uint64_t>;
    static constexpr uint32_t N_RANGE_MAIN = 0;        // N0: main blocks.
    static constexpr uint32_t N_RANGE_FIRST_TAIL = 1;  // N1: first tail blocks.
    static constexpr uint32_t N_RANGE_SECOND_TAIL = 2; // N2: second tail blocks after resplitting.

public:
    struct Params {
        int64_t mL1Tile{0};
        uint64_t mainBlockCount{0};
        uint64_t firstTailBlockCount{0};
        uint64_t secondTailBlockCount{0};
        uint64_t mainBlockSize{0};
        uint64_t firstTailBlockSize{0};
        uint64_t secondTailBlockSize{0};
        uint64_t cubeNumBlocksM{0};
        uint64_t cubeNumBlocksN{0};
    };

    __aicore__ inline BlockSchedulerWqmmTailResplit(const ProblemShape_& problemShape, const Params& params)
    {
        uint64_t mSize = static_cast<uint64_t>(asc::te::get<0>(problemShape));
        nSize_ = static_cast<uint64_t>(asc::te::get<1>(problemShape));
        if ASCEND_IS_AIC {
            blockIdx_ = static_cast<uint64_t>(AscendC::GetBlockIdx());
        } else {
            blockIdx_ = static_cast<uint64_t>(AscendC::GetBlockIdx()) / AscendC::GetSubBlockNum();
        }

        auto singleCoreM = CeilDiv(mSize, params.cubeNumBlocksM);
        mStart_ = blockIdx_ / params.cubeNumBlocksN * singleCoreM;
        mTile_ = params.mL1Tile;
        mStop_ = Min(mStart_ + singleCoreM, mSize);
        coreNums_ = params.cubeNumBlocksM * params.cubeNumBlocksN;

        nDimIdx_ = blockIdx_ % params.cubeNumBlocksN;
        cubeNumBlocksN_ = params.cubeNumBlocksN;
        mainBlockCount_ = params.mainBlockCount;
        firstTailBlockCount_ = params.firstTailBlockCount;
        secondTailBlockCount_ = params.secondTailBlockCount;

        n0Tile_ = params.mainBlockSize;
        n0Stop_ = Min(mainBlockCount_ * n0Tile_, nSize_);

        n1Tile_ = params.firstTailBlockSize;
        n1Stop_ = Min(mainBlockCount_ * n0Tile_ + firstTailBlockCount_ * n1Tile_, nSize_);

        n2Tile_ = params.secondTailBlockSize;
        firstN2BlockIdx_ = (nDimIdx_ + cubeNumBlocksN_ - firstTailBlockCount_ % cubeNumBlocksN_) % cubeNumBlocksN_;
    }

    __aicore__ inline BlockNums GetBlockNums() const
    {
        uint64_t mBlockNum = mStart_ < mStop_ ? CeilDiv(mStop_ - mStart_, mTile_) : 0UL;
        uint64_t n0BlockNum = GetDistributedBlockNum(mainBlockCount_, nDimIdx_);
        uint64_t n1BlockNum = GetDistributedBlockNum(firstTailBlockCount_, nDimIdx_);
        uint64_t n2BlockNum = GetDistributedBlockNum(secondTailBlockCount_, firstN2BlockIdx_);
        return asc::te::make_shape(mBlockNum, n0BlockNum, n1BlockNum, n2BlockNum);
    }

    __aicore__ inline uint64_t GetCoreNums() const { return coreNums_; }

    // Return a scalar because only the M-axis coordinate is needed; no Coord tuple is necessary.
    __aicore__ inline uint64_t GetBlockCoordM(uint64_t mIdx) const { return mStart_ + mIdx * mTile_; }

    // Return a scalar because only the M-axis length is needed; no Shape tuple is necessary.
    __aicore__ inline uint64_t GetBlockShapeM(uint64_t coordM) const
    {
        uint64_t tileM = coordM < mStop_ ? Min(mStop_ - coordM, mTile_) : 0UL;
        return tileM;
    }

    // Return a scalar because only the N-axis coordinate is needed; no Coord tuple is necessary.
    template <uint32_t N_RANGE>
    __aicore__ inline uint64_t GetBlockCoordN(uint64_t nIdx) const
    {
        static_assert(N_RANGE <= N_RANGE_SECOND_TAIL, "N range must be main, first tail or second tail");
        uint64_t coordN;
        if constexpr (N_RANGE == N_RANGE_MAIN) {
            coordN = (nDimIdx_ + nIdx * cubeNumBlocksN_) * n0Tile_;
        } else if constexpr (N_RANGE == N_RANGE_FIRST_TAIL) {
            coordN = n0Stop_ + (nDimIdx_ + nIdx * cubeNumBlocksN_) * n1Tile_;
        } else {
            coordN = n1Stop_ + (firstN2BlockIdx_ + nIdx * cubeNumBlocksN_) * n2Tile_;
        }
        return coordN;
    }

    // Return a scalar because only the N-axis length is needed; no Shape tuple is necessary.
    template <uint32_t N_RANGE>
    __aicore__ inline uint64_t GetBlockShapeN(uint64_t coordN) const
    {
        static_assert(N_RANGE <= N_RANGE_SECOND_TAIL, "N range must be main, first tail or second tail");
        uint64_t rangeStop;
        uint64_t rangeTile;
        if constexpr (N_RANGE == N_RANGE_MAIN) {
            rangeStop = n0Stop_;
            rangeTile = n0Tile_;
        } else if constexpr (N_RANGE == N_RANGE_FIRST_TAIL) {
            rangeStop = n1Stop_;
            rangeTile = n1Tile_;
        } else {
            rangeStop = nSize_;
            rangeTile = n2Tile_;
        }
        uint64_t tileN = coordN < rangeStop ? Min(rangeStop - coordN, rangeTile) : 0UL;
        return tileN;
    }

private:
    __aicore__ inline uint64_t GetDistributedBlockNum(uint64_t totalBlockNum, uint64_t firstBlockIdx) const
    {
        if (firstBlockIdx >= totalBlockNum) {
            return 0UL;
        }
        return CeilDiv(totalBlockNum - firstBlockIdx, cubeNumBlocksN_);
    }

    uint64_t blockIdx_;
    uint64_t coreNums_;
    uint64_t nDimIdx_;
    uint64_t cubeNumBlocksN_;
    uint64_t nSize_;

    uint64_t mainBlockCount_;
    uint64_t firstTailBlockCount_;
    uint64_t secondTailBlockCount_;
    uint64_t firstN2BlockIdx_;

    uint64_t mStart_;
    uint64_t mStop_;
    uint64_t mTile_;

    uint64_t n0Stop_;
    uint64_t n0Tile_;

    uint64_t n1Stop_;
    uint64_t n1Tile_;

    uint64_t n2Tile_;
};

} // namespace Block
} // namespace Gemm
} // namespace Blaze
