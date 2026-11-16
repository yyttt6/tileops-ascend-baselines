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
 * \file block_scheduler_matmul_swat_with_tail_split.h
 * \brief SWAT block scheduler with edge merging and compact tail splitting.
 */
#pragma once

#include "blaze/gemm/utils/common_utils.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Block {

template <class ProblemShape_ = asc::te::shape<int64_t, int64_t, int64_t>>
class BlockSchedulerMatmulSwatWithTailSplit {
public:
    using ProblemShape = ProblemShape_;
    using BlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = asc::te::coord<int64_t, int64_t, int64_t, int64_t>;

    struct Params {
        uint64_t baseM{0};
        uint64_t baseN{0};
        uint64_t mTailTile{1};
        uint64_t nTailTile{1};
        uint64_t mBaseTailSplitCnt{1};
        uint64_t nBaseTailSplitCnt{1};
        uint64_t mTailMain{0};
        uint64_t nTailMain{0};
    };

    __aicore__ inline BlockSchedulerMatmulSwatWithTailSplit(const ProblemShape& problemShape, const Params& params)
    {
        mSize_ = static_cast<uint64_t>(asc::te::get<MNK_M>(problemShape));
        nSize_ = static_cast<uint64_t>(asc::te::get<MNK_N>(problemShape));
        kSize_ = static_cast<uint64_t>(asc::te::get<MNK_K>(problemShape));
        baseM_ = params.baseM;
        baseN_ = params.baseN;
        if (baseM_ == 0U || baseN_ == 0U) {
            return;
        }
        mCnt_ = CeilDiv(mSize_, baseM_);
        nCnt_ = CeilDiv(nSize_, baseN_);
        totalCnt_ = mCnt_ * nCnt_;
        blockNum_ = AscendC::GetBlockNum();
        tailBlockCnt_ = blockNum_ == 0U ? 0U : totalCnt_ % blockNum_;
        mCoreNum_ = Min(static_cast<uint64_t>(WINDOW_LEN), mCnt_);
        mainRow_ = mCoreNum_ == 0U ? 0U : mCnt_ / mCoreNum_ - 1U;
        mTailCoreNum_ = mCoreNum_ == 0U ? 0U : mCnt_ - mCoreNum_ * mainRow_;
        InitTailParams(params);
    }

    __aicore__ inline uint64_t GetTileCount() const { return totalTileNum_; }

    __aicore__ inline BlockCoord GetBlockCoord(uint64_t tileIdx) const
    {
        uint64_t logicalTileIdx = tileIdx;
        int64_t splitIdx = NO_SPLIT;
        if (tailBlockCnt_ > 0U && tileIdx >= baseRoundTileNum_) {
            ResolveCompactTailTile(tileIdx - baseRoundTileNum_, logicalTileIdx, splitIdx);
        }

        uint64_t mTileIdx = 0;
        uint64_t nTileIdx = 0;
        GetLogicalTileCoord(logicalTileIdx, mTileIdx, nTileIdx);

        uint64_t mOffset = GetMOrigin(mTileIdx);
        uint64_t nOffset = GetNOrigin(nTileIdx);
        if (splitIdx >= 0) {
            uint64_t singleCoreM = GetSingleCoreM(mTileIdx);
            uint64_t singleCoreN = GetSingleCoreN(nTileIdx);
            uint64_t singleCoreMSplit = Align16(CeilDiv(singleCoreM, mTailTile_));
            uint64_t singleCoreNSplit = Align16(CeilDiv(singleCoreN, nTailTile_));
            uint64_t mSplitIdx = static_cast<uint64_t>(splitIdx) % mTailTile_;
            uint64_t nSplitIdx = static_cast<uint64_t>(splitIdx) / mTailTile_;
            mOffset += mSplitIdx * singleCoreMSplit;
            nOffset += nSplitIdx * singleCoreNSplit;
        }
        return asc::te::make_coord(static_cast<int64_t>(mOffset), static_cast<int64_t>(nOffset),
                                   static_cast<int64_t>(logicalTileIdx), splitIdx);
    }

    __aicore__ inline BlockShape GetBlockShape(const BlockCoord& blockCoord) const
    {
        uint64_t logicalTileIdx = static_cast<uint64_t>(asc::te::get<2>(blockCoord));
        int64_t splitIdx = asc::te::get<3>(blockCoord);
        uint64_t mTileIdx = 0;
        uint64_t nTileIdx = 0;
        GetLogicalTileCoord(logicalTileIdx, mTileIdx, nTileIdx);

        uint64_t singleCoreM = GetSingleCoreM(mTileIdx);
        uint64_t singleCoreN = GetSingleCoreN(nTileIdx);
        if (splitIdx >= 0) {
            uint64_t singleCoreMSplit = Align16(CeilDiv(singleCoreM, mTailTile_));
            uint64_t singleCoreNSplit = Align16(CeilDiv(singleCoreN, nTailTile_));
            uint64_t mSplitIdx = static_cast<uint64_t>(splitIdx) % mTailTile_;
            uint64_t nSplitIdx = static_cast<uint64_t>(splitIdx) / mTailTile_;
            uint64_t mSplitOffset = mSplitIdx * singleCoreMSplit;
            uint64_t nSplitOffset = nSplitIdx * singleCoreNSplit;
            singleCoreM = Min(singleCoreM - mSplitOffset, singleCoreMSplit);
            singleCoreN = Min(singleCoreN - nSplitOffset, singleCoreNSplit);
        }
        return asc::te::make_shape(static_cast<int64_t>(singleCoreM), static_cast<int64_t>(singleCoreN),
                                   static_cast<int64_t>(kSize_), 1L);
    }

private:
    __aicore__ inline void InitTailParams(const Params& params)
    {
        mTailTile_ = Max(static_cast<uint64_t>(1), params.mTailTile);
        nTailTile_ = Max(static_cast<uint64_t>(1), params.nTailTile);
        totalTailTile_ = mTailTile_ * nTailTile_;
        baseRoundTileNum_ = totalCnt_ - tailBlockCnt_;

        mBaseTailSplitCnt_ = Min(Max(static_cast<uint64_t>(1), params.mBaseTailSplitCnt), mCnt_);
        nBaseTailSplitCnt_ = Min(Max(static_cast<uint64_t>(1), params.nBaseTailSplitCnt), nCnt_);
        mBaseNormCnt_ = mCnt_ - mBaseTailSplitCnt_;
        nBaseNormCnt_ = nCnt_ - nBaseTailSplitCnt_;
        uint64_t mMergeSize = mSize_ - mBaseNormCnt_ * baseM_;
        uint64_t nMergeSize = nSize_ - nBaseNormCnt_ * baseN_;
        mBaseTailMain_ = mBaseTailSplitCnt_ == 1U ? mMergeSize : params.mTailMain;
        nBaseTailMain_ = nBaseTailSplitCnt_ == 1U ? nMergeSize : params.nTailMain;
        mBaseTailLast_ = mMergeSize - (mBaseTailSplitCnt_ - 1U) * mBaseTailMain_;
        nBaseTailLast_ = nMergeSize - (nBaseTailSplitCnt_ - 1U) * nBaseTailMain_;

        compactTailTileNum_ = CalcCompactTailTileNum(mTailTile_, nTailTile_);
        if (tailBlockCnt_ == 0U || compactTailTileNum_ > blockNum_) {
            mTailTile_ = 1U;
            nTailTile_ = 1U;
            totalTailTile_ = 1U;
            compactTailTileNum_ = CalcCompactTailTileNum(mTailTile_, nTailTile_);
        }
        totalTileNum_ = baseRoundTileNum_ + compactTailTileNum_;
    }

    __aicore__ inline void ResolveCompactTailTile(uint64_t tailTileIdx, uint64_t& logicalTileIdx,
                                                  int64_t& splitIdx) const
    {
        uint64_t remain = tailTileIdx;
        for (uint64_t tailIdx = 0; tailIdx < tailBlockCnt_; ++tailIdx) {
            uint64_t curLogicalTileIdx = baseRoundTileNum_ + tailIdx;
            uint64_t mTileIdx = 0;
            uint64_t nTileIdx = 0;
            GetLogicalTileCoord(curLogicalTileIdx, mTileIdx, nTileIdx);
            uint64_t validM = 1U;
            uint64_t validN = 1U;
            CalcValidSplit(GetSingleCoreM(mTileIdx), GetSingleCoreN(nTileIdx), mTailTile_, nTailTile_, validM, validN);
            uint64_t validCount = validM * validN;
            if (remain < validCount) {
                uint64_t mSplitIdx = remain % validM;
                uint64_t nSplitIdx = remain / validM;
                logicalTileIdx = curLogicalTileIdx;
                splitIdx = static_cast<int64_t>(nSplitIdx * mTailTile_ + mSplitIdx);
                return;
            }
            remain -= validCount;
        }
        logicalTileIdx = baseRoundTileNum_;
        splitIdx = 0;
    }

    __aicore__ inline uint64_t CalcCompactTailTileNum(uint64_t mTile, uint64_t nTile) const
    {
        uint64_t tileNum = 0U;
        for (uint64_t tailIdx = 0; tailIdx < tailBlockCnt_; ++tailIdx) {
            uint64_t logicalTileIdx = baseRoundTileNum_ + tailIdx;
            uint64_t mTileIdx = 0;
            uint64_t nTileIdx = 0;
            GetLogicalTileCoord(logicalTileIdx, mTileIdx, nTileIdx);
            uint64_t validM = 1U;
            uint64_t validN = 1U;
            CalcValidSplit(GetSingleCoreM(mTileIdx), GetSingleCoreN(nTileIdx), mTile, nTile, validM, validN);
            tileNum += validM * validN;
        }
        return tileNum;
    }

    __aicore__ inline static void CalcValidSplit(uint64_t singleCoreM, uint64_t singleCoreN, uint64_t mTile,
                                                 uint64_t nTile, uint64_t& validM, uint64_t& validN)
    {
        uint64_t singleCoreMSplit = Align16(CeilDiv(singleCoreM, mTile));
        uint64_t singleCoreNSplit = Align16(CeilDiv(singleCoreN, nTile));
        validM = Min(mTile, CeilDiv(singleCoreM, singleCoreMSplit));
        validN = Min(nTile, CeilDiv(singleCoreN, singleCoreNSplit));
    }

    __aicore__ inline void GetLogicalTileCoord(uint64_t tileIdx, uint64_t& mTileIdx, uint64_t& nTileIdx) const
    {
        uint64_t rowIdx = tileIdx / (mCoreNum_ * nCnt_);
        if (rowIdx < mainRow_) {
            uint64_t localTileIdx = tileIdx - rowIdx * mCoreNum_ * nCnt_;
            mTileIdx = rowIdx * mCoreNum_ + localTileIdx % mCoreNum_;
            nTileIdx = (localTileIdx / mCoreNum_) % nCnt_;
        } else {
            rowIdx = mainRow_;
            uint64_t tailIdx = tileIdx - mainRow_ * mCoreNum_ * nCnt_;
            mTileIdx = mainRow_ * mCoreNum_ + tailIdx % mTailCoreNum_;
            nTileIdx = (tailIdx / mTailCoreNum_) % nCnt_;
        }
        if (rowIdx & 1U) {
            nTileIdx = nCnt_ - 1U - nTileIdx;
        }
    }

    __aicore__ inline uint64_t GetSingleCoreM(uint64_t mTileIdx) const
    {
        if (mTileIdx >= mBaseNormCnt_) {
            return mTileIdx < mCnt_ - 1U ? mBaseTailMain_ : mBaseTailLast_;
        }
        return baseM_;
    }

    __aicore__ inline uint64_t GetSingleCoreN(uint64_t nTileIdx) const
    {
        if (nTileIdx >= nBaseNormCnt_) {
            return nTileIdx < nCnt_ - 1U ? nBaseTailMain_ : nBaseTailLast_;
        }
        return baseN_;
    }

    __aicore__ inline uint64_t GetMOrigin(uint64_t mTileIdx) const
    {
        if (mTileIdx > mBaseNormCnt_) {
            return mTileIdx * baseM_ - (mTileIdx - mBaseNormCnt_) * (baseM_ - mBaseTailMain_);
        }
        return mTileIdx * baseM_;
    }

    __aicore__ inline uint64_t GetNOrigin(uint64_t nTileIdx) const
    {
        if (nTileIdx > nBaseNormCnt_) {
            return nTileIdx * baseN_ - (nTileIdx - nBaseNormCnt_) * (baseN_ - nBaseTailMain_);
        }
        return nTileIdx * baseN_;
    }

    uint64_t mSize_{0};
    uint64_t nSize_{0};
    uint64_t kSize_{0};
    uint64_t baseM_{0};
    uint64_t baseN_{0};
    uint64_t mCnt_{0};
    uint64_t nCnt_{0};
    uint64_t totalCnt_{0};
    uint64_t totalTileNum_{0};
    uint64_t compactTailTileNum_{0};
    uint64_t blockNum_{1};
    uint64_t tailBlockCnt_{0};
    uint64_t baseRoundTileNum_{0};
    uint64_t mTailTile_{1};
    uint64_t nTailTile_{1};
    uint64_t totalTailTile_{1};
    uint64_t mBaseTailSplitCnt_{1};
    uint64_t nBaseTailSplitCnt_{1};
    uint64_t mBaseNormCnt_{0};
    uint64_t nBaseNormCnt_{0};
    uint64_t mBaseTailMain_{0};
    uint64_t nBaseTailMain_{0};
    uint64_t mBaseTailLast_{0};
    uint64_t nBaseTailLast_{0};
    uint64_t mCoreNum_{0};
    uint64_t mTailCoreNum_{0};
    uint64_t mainRow_{0};

    static constexpr int64_t NO_SPLIT = -1;
};

} // namespace Block
} // namespace Gemm
} // namespace Blaze
