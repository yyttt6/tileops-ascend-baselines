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
 * \file block_scheduler_wqmm_block_split.h
 * \brief Matmul block scheduler with fixed-core partition and in-core M/N swizzle:
 *        core partitioning is resolved at construction from blockIdx, each core owns
 *        a singleCoreM x singleCoreN rectangle and iterates it with ORDER_M/ORDER_N
 *        swizzle traversal. Relies on the host tiling invariant stepM == stepN == 1
 *        (L1 group == L0 tile).
 */

#pragma once

#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Block {

template <class ProblemShape_, class LayoutB_, class AType_>
class BlockSchedulerWqmmBlockSplit {
public:
    using ProblemShape = ProblemShape_;
    using BlockCoord = asc::te::coord<int64_t, int64_t, int64_t, int64_t>;
    using BlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;

    struct Params {
        uint32_t cubeNumBlocksM{0};
        uint32_t cubeNumBlocksN{0};
        uint32_t baseM{0};
        uint32_t baseN{0};
        uint32_t iterateOrder{0};
    };

    __aicore__ inline BlockSchedulerWqmmBlockSplit(const ProblemShape& problemShape, const Params& params)
    {
        mSize_ = asc::te::get<MNK_M>(problemShape);
        nSize_ = asc::te::get<MNK_N>(problemShape);
        kSize_ = asc::te::get<MNK_K>(problemShape);
        baseM_ = static_cast<int64_t>(params.baseM);
        baseN_ = static_cast<int64_t>(params.baseN);
        iterateOrder_ = params.iterateOrder;
        if (baseM_ <= 0 || baseN_ <= 0) {
            blockNums_ = 0;
            return;
        }
        const uint64_t logicalBlockIdx = GetCurrentBlockIdx();
        if (logicalBlockIdx >=
            static_cast<uint64_t>(params.cubeNumBlocksM) * static_cast<uint64_t>(params.cubeNumBlocksN)) {
            blockNums_ = 0;
            return;
        }
        mDimIdx_ = static_cast<int64_t>(logicalBlockIdx % params.cubeNumBlocksM);
        nDimIdx_ = static_cast<int64_t>(logicalBlockIdx / params.cubeNumBlocksM);

        constexpr int64_t M_CORE_ALIGN = MATMUL_MNK_ALIGN;
        constexpr int64_t N_CORE_ALIGN = (IsWeightNz<LayoutB_>::value && !IsTrans<LayoutB_>::value) ?
                                             static_cast<int64_t>(asc::te::c0_element<AType_>) :
                                             static_cast<int64_t>(BLOCK_CUBE);
        singleMAligned_ = CeilAlign(CeilDiv(mSize_, static_cast<int64_t>(params.cubeNumBlocksM)), M_CORE_ALIGN);
        singleNAligned_ = CeilAlign(CeilDiv(nSize_, static_cast<int64_t>(params.cubeNumBlocksN)), N_CORE_ALIGN);

        coreMOffset_ = mDimIdx_ * singleMAligned_;
        coreNOffset_ = nDimIdx_ * singleNAligned_;
        if (coreMOffset_ >= mSize_ || coreNOffset_ >= nSize_) {
            blockNums_ = 0;
            return;
        }
        singleCoreM_ = Min(singleMAligned_, mSize_ - coreMOffset_);
        singleCoreN_ = Min(singleNAligned_, nSize_ - coreNOffset_);

        mBlockNums_ = CeilDiv(singleCoreM_, baseM_);
        nBlockNums_ = CeilDiv(singleCoreN_, baseN_);
        blockNums_ = mBlockNums_ * nBlockNums_;
    }

    // Tile count of the current core's responsibility rectangle.
    __aicore__ inline uint64_t GetBlockNums() const { return blockNums_ <= 0 ? 0 : static_cast<uint64_t>(blockNums_); }

    // tileIdx is a local index within the current core; returns the absolute GM element
    // coordinates of the tile origin. ORDER_N traverses M-fast, ORDER_M traverses N-fast.
    __aicore__ inline BlockCoord GetBlockCoord(uint64_t tileIdx) const
    {
        const int64_t idx = static_cast<int64_t>(tileIdx);
        int64_t localMIdx = 0;
        int64_t localNIdx = 0;
        if (iterateOrder_ == static_cast<uint32_t>(IterateOrder::ORDER_N)) {
            localMIdx = idx % mBlockNums_;
            localNIdx = idx / mBlockNums_;
        } else {
            localMIdx = idx / nBlockNums_;
            localNIdx = idx % nBlockNums_;
        }
        return asc::te::make_coord(coreMOffset_ + localMIdx * baseM_, coreNOffset_ + localNIdx * baseN_,
                                   static_cast<int64_t>(0), static_cast<int64_t>(0));
    }

    // Returns the real tile size (validM, validN, kSize, 1) at blockCoord, shrinking the
    // base block on the current core's M/N tails.
    __aicore__ inline BlockShape GetBlockShape(const BlockCoord& blockCoord) const
    {
        const int64_t mOffset = asc::te::get<MNK_M>(blockCoord);
        const int64_t nOffset = asc::te::get<MNK_N>(blockCoord);
        const int64_t validM = Min(baseM_, coreMOffset_ + singleCoreM_ - mOffset);
        const int64_t validN = Min(baseN_, coreNOffset_ + singleCoreN_ - nOffset);
        return asc::te::make_shape(validM, validN, kSize_, static_cast<int64_t>(1));
    }

private:
    enum class IterateOrder : uint32_t { ORDER_M = 0, ORDER_N, UNDEF };

    int64_t mSize_{0};
    int64_t nSize_{0};
    int64_t kSize_{0};
    int64_t baseM_{0};
    int64_t baseN_{0};
    uint32_t iterateOrder_{0};
    int64_t mDimIdx_{0};
    int64_t nDimIdx_{0};
    int64_t singleMAligned_{0};
    int64_t singleNAligned_{0};
    int64_t coreMOffset_{0};
    int64_t coreNOffset_{0};
    int64_t singleCoreM_{0};
    int64_t singleCoreN_{0};
    int64_t mBlockNums_{0};
    int64_t nBlockNums_{0};
    int64_t blockNums_{0};
};

} // namespace Block
} // namespace Gemm
} // namespace Blaze
