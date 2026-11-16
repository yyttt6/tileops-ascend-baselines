/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License).
 * You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the License in the root of the software repository for the full text of the License.
 */

/*!
 * \file block_scheduler_matmul_iterbatch.h
 * \brief Scheduler for the IterBatch path.
 *        - blockNums = CeilDiv(totalBatch, iterBatchL1); block blockIdx covers batches [blockIdx*L1, ...)
 *        - cores gate on blockNums and loop blocks with stride blockNum (strided blocks)
 *        - tail block covers totalBatch - blockIdx*L1 batches
 */

#pragma once

#include "blaze/gemm/utils/common_utils.h"

namespace Blaze {
namespace Gemm {
namespace Block {

template <class ProblemShape_>
class BlockSchedulerMatmulIterBatch {
public:
    using BlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = asc::te::coord<int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = ProblemShape_;

    struct Params {
        uint32_t baseM = 1;
        uint32_t baseN = 1;
        uint32_t baseK = 1;
        uint32_t iterBatchL1 = 1;
        uint32_t iterBatchL0 = 1;
        uint8_t isHf32 = 0;
        uint32_t l2CacheDisable = L2_CACHE_DEFAULT;
    };

public:
    __aicore__ inline BlockSchedulerMatmulIterBatch(const ProblemShape& shape, const Params& params)
    {
        m_ = asc::te::get<MNK_M>(shape);
        n_ = asc::te::get<MNK_N>(shape);
        k_ = asc::te::get<MNK_K>(shape);
        b_ = asc::te::get<MNK_B>(shape);
        iterBatchL1_ = params.iterBatchL1;
    }

    __aicore__ inline int64_t GetBlockNums() const { return CeilDiv(b_, iterBatchL1_); }

    __aicore__ inline int64_t GetCoreNums(int64_t blockNum) const
    {
        int64_t blockNums = GetBlockNums();
        return (blockNums < blockNum) ? blockNums : blockNum;
    }

    // batches covered by block blockIdx (tail block may be smaller than iterBatchL1)
    __aicore__ inline BlockShape GetBlockShape(int64_t blockIdx) const
    {
        int64_t next = (blockIdx + 1) * iterBatchL1_;
        int64_t curIterBatchL1 = (next > b_) ? (b_ - blockIdx * iterBatchL1_) : iterBatchL1_;
        return {m_, n_, k_, curIterBatchL1};
    }

    __aicore__ inline BlockCoord GetBlockCoord(int64_t blockIdx) const { return {0, 0, 0, blockIdx * iterBatchL1_}; }

private:
    int64_t m_{0};
    int64_t n_{0};
    int64_t k_{0};
    int64_t b_{0};
    int64_t iterBatchL1_{1};
};

} // namespace Block
} // namespace Gemm
} // namespace Blaze
