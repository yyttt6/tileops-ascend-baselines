/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS PROGRAM IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file block_scheduler_flat_quant.h
 * \brief
 */

#pragma once

#include "blaze/gemm/utils/common_utils.h"

namespace Blaze {
namespace Attention {
namespace Block {

template <class ProblemShape_>
class BlockSchedulerFlatQuant {
public:
    using BlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockL1L0Shape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = ProblemShape_;

    struct Params {
        int64_t iterBatch = 1;
        float dstTypeMax = 0.0f;
        float invDstTypeMax = 0.0f;
    };

    __aicore__ inline BlockSchedulerFlatQuant(const ProblemShape& shape, int64_t blockNum, const Params& params)
    {
        m_ = asc::te::get<Gemm::MNK_M>(shape);
        n_ = asc::te::get<Gemm::MNK_N>(shape);
        k_ = asc::te::get<Gemm::MNK_B>(shape);
        mL1_ = m_;
        kL1_ = n_;
        nL1_ = n_;
        iterBatch_ = params.iterBatch;
        blockNum_ = blockNum;
        mL1_ *= iterBatch_;
        mainBatchLoop_ = k_ / iterBatch_ / blockNum_;
        int64_t remainderBatch = k_ - mainBatchLoop_ * blockNum_ * iterBatch_;
        mainTailBatch_ = Gemm::CeilDiv(remainderBatch, blockNum_);
        mainTailBlock_ = remainderBatch % blockNum_;
    }

    __aicore__ inline int64_t GetBlockNums() { return Gemm::CeilAlign(Gemm::CeilDiv(k_, iterBatch_), blockNum_); }

    __aicore__ inline int64_t GetCoreNums(int64_t blockNum)
    {
        if (k_ < blockNum) {
            return k_;
        }
        return blockNum;
    }

    __aicore__ inline BlockL1L0Shape GetBlockShape(int64_t tileIdx)
    {
        int64_t blkM = mL1_;
        int64_t blkN = nL1_;
        int64_t blkK = kL1_;
        int64_t curLoopIdx = tileIdx / blockNum_;
        if (curLoopIdx < mainBatchLoop_) {
            return {blkM, blkN, blkK, iterBatch_};
        } else if (mainTailBatch_ > 0) {
            int64_t mainTailIdx = tileIdx % blockNum_;
            if (mainTailBlock_ > 0 && mainTailIdx >= mainTailBlock_) {
                blkM = blkM / iterBatch_ * (mainTailBatch_ - 1);
                return {blkM, blkN, blkK, mainTailBatch_ - 1};
            } else {
                blkM = blkM / iterBatch_ * mainTailBatch_;
                return {blkM, blkN, blkK, mainTailBatch_};
            }
        }
        return {blkM, blkN, blkK, iterBatch_};
    }

    __aicore__ inline int64_t GetBlockCoord(int64_t tileIdx, int64_t curBlockIdx)
    {
        const int64_t curLoopIdx = tileIdx / blockNum_;
        const int64_t mainBatchTotal = mainBatchLoop_ * blockNum_ * iterBatch_;
        if (curLoopIdx < mainBatchLoop_) {
            return tileIdx * iterBatch_;
        }
        if (mainTailBatch_ > 0) {
            if (mainTailBlock_ > 0 && curBlockIdx >= mainTailBlock_) {
                return mainBatchTotal + mainTailBlock_ * mainTailBatch_ +
                       (curBlockIdx - mainTailBlock_) * (mainTailBatch_ - 1);
            }
            return mainBatchTotal + curBlockIdx * mainTailBatch_;
        }
        return tileIdx * iterBatch_;
    }

private:
    int64_t k_{0};
    int64_t m_{0};
    int64_t n_{0};
    int64_t mL1_{0};
    int64_t kL1_{0};
    int64_t nL1_{0};
    int64_t iterBatch_{1};
    int64_t blockNum_{1};
    int64_t mainBatchLoop_{1};
    int64_t mainTailBatch_{1};
    int64_t mainTailBlock_{1};
};

} // namespace Block
} // namespace Attention
} // namespace Blaze
