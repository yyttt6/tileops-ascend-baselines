/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * \file qbmm_ut_fill_helpers.h
 * \brief QBMM Kernel UT 公共 qbmmParams 填充助手：供 fixpipe(qbmm_cube.h) 与 MIX(qbmm_mix.h) wrapper 复用，
 *        消除 batch 维度 / tile 配置字段赋值块的重复代码。
 */

#pragma once

#include "qbmm_tiling_data.h"

namespace QBMMUT {

// 填充 batch 广播维度字段（A/B/C 各 4 维）+ bias/量化模式标志。
// QbmmParams 通过鸭子类型匹配 GemmUniversal 各特化的 QBMMTiling（fixpipe / MIX 多 batch）。
template <typename QbmmParams>
__aicore__ inline void FillQbmmBatchParams(QbmmParams& qbmmParams, const QBMMV3TilingData& tilingData)
{
    qbmmParams.batchA1 = tilingData.batchA1;
    qbmmParams.batchA2 = tilingData.batchA2;
    qbmmParams.batchA3 = tilingData.batchA3;
    qbmmParams.batchA4 = tilingData.batchA4;
    qbmmParams.batchB1 = tilingData.batchB1;
    qbmmParams.batchB2 = tilingData.batchB2;
    qbmmParams.batchB3 = tilingData.batchB3;
    qbmmParams.batchB4 = tilingData.batchB4;
    qbmmParams.batchC1 = tilingData.batchC1;
    qbmmParams.batchC2 = tilingData.batchC2;
    qbmmParams.batchC3 = tilingData.batchC3;
    qbmmParams.batchC4 = tilingData.batchC4;
    qbmmParams.biasThreeDim = tilingData.biasThreeDim;
    qbmmParams.x1QuantMode = tilingData.x1QuantMode;
    qbmmParams.x2QuantMode = tilingData.x2QuantMode;
}

// 填充 L1 载入 / L0C tile 配置字段（三种 QBMMTiling 均含这些字段）。
template <typename QbmmParams>
__aicore__ inline void FillQbmmTileParams(QbmmParams& qbmmParams, const QBMMV3TilingData& tilingData)
{
    qbmmParams.kAL1 = tilingData.kAL1;
    qbmmParams.kBL1 = tilingData.kBL1;
    qbmmParams.nBufferNum = tilingData.nBufferNum;
    qbmmParams.baseM = tilingData.baseM_qbmm;
    qbmmParams.baseN = tilingData.baseN_qbmm;
    qbmmParams.baseK = tilingData.baseK_qbmm;
    qbmmParams.isBias = tilingData.isBias;
    qbmmParams.dbL0C = tilingData.dbL0C;
    qbmmParams.bMustHitL2 = tilingData.weightMustHitL2;
}

template <typename SchParams>
__aicore__ inline void FillQbmmSchParams(SchParams& schedulerParams, const QBMMV3TilingData& tilingData)
{
    schedulerParams.baseM = tilingData.baseM;
    schedulerParams.baseN = tilingData.baseN;
    schedulerParams.mTailTile = tilingData.mTailTile;
    schedulerParams.nTailTile = tilingData.nTailTile;
    schedulerParams.mBaseTailSplitCnt = tilingData.mBaseTailSplitCnt;
    schedulerParams.nBaseTailSplitCnt = tilingData.nBaseTailSplitCnt;
    schedulerParams.mTailMain = tilingData.mTailMain;
    schedulerParams.nTailMain = tilingData.nTailMain;
}

} // namespace QBMMUT
