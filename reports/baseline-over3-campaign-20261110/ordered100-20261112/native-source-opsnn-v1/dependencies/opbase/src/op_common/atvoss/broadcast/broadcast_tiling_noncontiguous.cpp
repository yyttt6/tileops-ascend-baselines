/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file broadcast_tiling_noncontiguous.cpp
 * \brief atvoss broadcast template tiling
 */
#include "op_common/atvoss/broadcast/broadcast_tiling_noncontiguous.h"

namespace Ops {
namespace Base {
uint64_t GetBlockSplitFactorLastTranspose(BroadcastTilingData& broadcastTilingData, ubSplitInfo& ubInfo,
                                          uint64_t maxElemNum, const BroadcastComputeParams& computeParams)
{
    int64_t minDtypeBits = computeParams.minDtypeBits;
    int64_t minDtypeBlockAlignSize = BLOCK_LENGTH * BROADCAST_BITS_NUM / minDtypeBits;

    // 做ub切分
    uint64_t curProduct = 1;
    uint64_t ubSplitAxes = 0;
    bool flag = true;
    int32_t shapeLen = broadcastTilingData.dims.back().size();
    int64_t lastAxis = broadcastTilingData.dims.back()[shapeLen - 1];
    int64_t axisBeforelastAxis = broadcastTilingData.dims.back()[shapeLen - 2];

    int64_t ubOuter = 1;
    int64_t ubTail = 1;
    int64_t ubOuterLastAxis = 1;
    int64_t ubTailLastAxis = 1;

    int64_t axisBeforelastAxisSplitFactor = 128 / minDtypeBlockAlignSize;
    // -2轴切分至少128B, last轴转置至少有两维，所以不用考虑一维场景
    int64_t ubFormer = axisBeforelastAxis > axisBeforelastAxisSplitFactor ? axisBeforelastAxisSplitFactor :
                                                                            axisBeforelastAxis;
    int64_t ubFormerLastAxis = maxElemNum / ubFormer;

    if (lastAxis > ubFormerLastAxis) {
        // 如果-2轴切分完之后，-1轴还能切分则直接切-1轴，-2轴切分直接应用
        ubSplitAxes = shapeLen - 2;
        ubOuterLastAxis = (lastAxis + ubFormerLastAxis - 1) / ubFormerLastAxis;
        ubTailLastAxis = lastAxis - (ubOuterLastAxis - 1) * ubFormerLastAxis;

        ubOuter = (axisBeforelastAxis + ubFormer - 1) / ubFormer;
        ubTail = axisBeforelastAxis - (ubOuter - 1) * ubFormer;
    } else {
        // 如果 -1轴不够切分，则-2轴往前切分，尽可能多的切分-2轴，甚至到-3轴。
        ubOuterLastAxis = 1;
        ubFormerLastAxis = lastAxis;
        ubTailLastAxis = lastAxis;

        // curProduct需要将补维能力加入，确保后续ub切分的结果是带pad的。
        for (int64_t i = shapeLen - 1; i >= 0; i--) {
            // 遇到倒数第三轴的时候，需要对前两维补pad。
            if (i == shapeLen - 3) {
                curProduct = (curProduct + minDtypeBlockAlignSize - 1) / minDtypeBlockAlignSize *
                             minDtypeBlockAlignSize;
            }
            curProduct *= broadcastTilingData.dims.back()[i];
            if (curProduct > maxElemNum) {
                curProduct = curProduct / broadcastTilingData.dims.back()[i];
                ubSplitAxes = i;
                flag = false;
                break;
            }
        }

        if (flag) {
            curProduct = curProduct / broadcastTilingData.dims.back()[0];
        }

        // ubFormer 不可能在最后一根轴
        ubFormer = maxElemNum / curProduct;
        ubOuter = (broadcastTilingData.dims.back()[ubSplitAxes] + ubFormer - 1) / ubFormer;
        ubTail = broadcastTilingData.dims.back()[ubSplitAxes] - (ubOuter - 1) * ubFormer;
    }

    // 计算ub外轴乘积
    uint64_t fusedProduct = ubOuter;
    for (uint64_t i = 0; i < ubSplitAxes; i++) {
        fusedProduct *= broadcastTilingData.dims.back()[i];
    }

    ubInfo.ubSplitAxis = ubSplitAxes;
    ubInfo.ubFormer = ubFormer;
    ubInfo.ubOuter = ubOuter;
    ubInfo.ubTail = ubTail;
    ubInfo.ubFormerLastAxis = ubFormerLastAxis;
    ubInfo.ubOuterLastAxis = ubOuterLastAxis;
    ubInfo.ubTailLastAxis = ubTailLastAxis;

    return fusedProduct;
}

ge::graphStatus DoBrodcastTilingLastTranspose(const BroadcastTilingParams& broadcastTilingParams,
                                              BroadcastTilingData& broadcastTilingData)
{
    uint64_t computeKey = BroadcastGetComputeKey();
    auto iter = broadcastTilingParams.computeMap.find(computeKey);
    BroadcastComputeParams computeParams;
    if (iter != broadcastTilingParams.computeMap.end()) {
        computeParams = iter->second;
    } else {
        OP_LOGE("BroadcastTiling", "can not find computeKey");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(broadcastTilingParams.ubSize < computeParams.extraSize[0],
                OP_LOGE("BroadcastTiling", "ubSize is smaller than extra size."), return ge::GRAPH_FAILED);

    // 获取最大存活空间大小
    uint64_t maxElemNum = BroadcastGetMaxElemNum(broadcastTilingParams.ubSize, computeParams);
    OP_LOGI("Broadcast", "Broadcast DoBrodcastTiling. origin maxElemNum: %lu ubSize: %ld", maxElemNum,
            broadcastTilingParams.ubSize);
    OP_CHECK_IF(broadcastTilingParams.ubSize <= 0, OP_LOGE("BroadcastTiling", "ubSize can not be 0"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(broadcastTilingParams.coreNum <= 0, OP_LOGE("BroadcastTiling", "coreNum can not be 0"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(maxElemNum == 0, OP_LOGE("BroadcastTiling", "maxElemNum can not be 0"), return ge::GRAPH_FAILED);

    ubSplitInfo ubInfo;
    if (broadcastTilingData.dims.back().size() < 2) {
        OP_LOGE("BroadcastTiling", "The last transpose shape at least 2 dims.");
        return ge::GRAPH_FAILED;
    }
    uint64_t fusedProduct = GetBlockSplitFactorLastTranspose(broadcastTilingData, ubInfo, maxElemNum, computeParams);
    uint64_t blockFormer = (fusedProduct + broadcastTilingParams.coreNum - 1) / broadcastTilingParams.coreNum;
    uint64_t blockNum = (fusedProduct + blockFormer - 1) / blockFormer;

    // 当preferMultiCore为true且当前使用的核数少于总核数，尝试降低UB分块以充分利用更多核心
    if (broadcastTilingParams.preferMultiCore && blockNum < static_cast<uint64_t>(broadcastTilingParams.coreNum)) {
        // dstFusedProduct为尽量切多核时的理想多核切分因子
        uint64_t dstFusedProduct = blockFormer * broadcastTilingParams.coreNum;
        uint64_t tmpFusedProduct = fusedProduct;
        while (tmpFusedProduct < dstFusedProduct) {
            maxElemNum = maxElemNum - CACHE_LINE;
            if (maxElemNum <= CACHE_LINE_512) {
                break;
            }
            tmpFusedProduct = GetBlockSplitFactorLastTranspose(broadcastTilingData, ubInfo, maxElemNum, computeParams);
        }
        // 计算最终调整后的多核切分因子
        maxElemNum = (maxElemNum + CACHE_LINE + CACHE_LINE - 1) / CACHE_LINE * CACHE_LINE;
        fusedProduct = GetBlockSplitFactorLastTranspose(broadcastTilingData, ubInfo, maxElemNum, computeParams);
        // 更新blockFormer和blockNum
        blockFormer = (fusedProduct + broadcastTilingParams.coreNum - 1) / broadcastTilingParams.coreNum;
        blockNum = (fusedProduct + blockFormer - 1) / blockFormer;
    }

    uint64_t blockTail = fusedProduct - (blockNum - 1) * blockFormer;
    uint64_t dimProductBeforeUbInner = fusedProduct;
    OP_LOGI("Broadcast", "Broadcast DoBrodcastTiling. maxElemNum: %lu fusedProduct: %lu ubFormer: %ld ", maxElemNum,
            fusedProduct, ubInfo.ubFormer);

    broadcastTilingData.ubSplitAxis = ubInfo.ubSplitAxis;
    broadcastTilingData.ubFormer = ubInfo.ubFormer;
    broadcastTilingData.ubOuter = ubInfo.ubOuter;
    broadcastTilingData.ubTail = ubInfo.ubTail;
    broadcastTilingData.ubFormerLastAxis = ubInfo.ubFormerLastAxis;
    broadcastTilingData.ubOuterLastAxis = ubInfo.ubOuterLastAxis;
    broadcastTilingData.ubTailLastAxis = ubInfo.ubTailLastAxis;

    broadcastTilingData.blockFormer = blockFormer;
    broadcastTilingData.blockNum = blockNum;
    broadcastTilingData.blockTail = blockTail;
    broadcastTilingData.dimProductBeforeUbInner = dimProductBeforeUbInner;
    broadcastTilingData.elemNum = maxElemNum;

    int64_t minDtypeBits = computeParams.minDtypeBits;
    int64_t minDtypeBlockAlignSize = BLOCK_LENGTH * BROADCAST_BITS_NUM / minDtypeBits;
    broadcastTilingData.minDtypeBlockAlignSize = minDtypeBlockAlignSize;

    uint64_t scheduleKey = BroadcastGetScheduleKey(broadcastTilingData.shapeLen - broadcastTilingData.ubSplitAxis);
    broadcastTilingData.innerKey = computeKey * BROADCAST_COMPUTE_KEY_OFFSET + scheduleKey;
    return ge::GRAPH_SUCCESS;
}

uint64_t GetBlockSplitFactorNLastTranspose(BroadcastTilingData& broadcastTilingData, ubSplitInfo& ubInfo,
                                           uint64_t maxElemNum, const BroadcastComputeParams& computeParams,
                                           bool isUbBroadcast)
{
    int64_t minDtypeBits = computeParams.minDtypeBits;
    int64_t minDtypeBlockAlignSize = BLOCK_LENGTH * BROADCAST_BITS_NUM / minDtypeBits;

    // 做ub切分
    uint64_t curProduct = 1;
    uint64_t ubSplitAxes = 0;
    bool flag = true;

    int64_t boundaryNum = 2;
    if (isUbBroadcast) {
        boundaryNum = 1;
    }

    for (int64_t i = broadcastTilingData.dims.back().size() - 1; i >= 0; i--) {
        curProduct *= broadcastTilingData.dims.back()[i];
        if (i == static_cast<int64_t>(broadcastTilingData.dims.back().size() - boundaryNum)) {
            curProduct = ((curProduct + minDtypeBlockAlignSize - 1) / minDtypeBlockAlignSize) * minDtypeBlockAlignSize;
        }
        if (curProduct > maxElemNum) {
            curProduct = curProduct / broadcastTilingData.dims.back()[i];
            ubSplitAxes = i;
            flag = false;
            break;
        }
    }

    // 全部能放下，则去掉第一个维度
    if (flag) {
        curProduct = curProduct / broadcastTilingData.dims.back()[0];
    }

    uint32_t ubFormer = 0; // 表示当前切分轴的切分因子
    if (broadcastTilingData.dims.back().size() == 1) {
        ubFormer = maxElemNum;
    } else {
        ubFormer = maxElemNum / curProduct;
    }

    uint64_t ubOuter = (broadcastTilingData.dims.back()[ubSplitAxes] + ubFormer - 1) / ubFormer;
    uint64_t ubTail = broadcastTilingData.dims.back()[ubSplitAxes] - (ubOuter - 1) * ubFormer;

    // 计算ub外轴乘积
    uint64_t fusedProduct = ubOuter;
    for (uint64_t i = 0; i < ubSplitAxes; i++) {
        fusedProduct *= broadcastTilingData.dims.back()[i];
    }

    ubInfo.ubFormer = ubFormer;
    ubInfo.ubSplitAxis = ubSplitAxes;
    ubInfo.ubOuter = ubOuter;
    ubInfo.ubTail = ubTail;

    return fusedProduct;
}

ge::graphStatus BroadcastTilingNLastTranspose(const BroadcastTilingParams& broadcastTilingParams,
                                              BroadcastTilingData& broadcastTilingData, bool isUbBroadcast)
{
    uint64_t computeKey = BroadcastGetComputeKey();
    auto iter = broadcastTilingParams.computeMap.find(computeKey);
    BroadcastComputeParams computeParams;
    if (iter != broadcastTilingParams.computeMap.end()) {
        computeParams = iter->second;
    } else {
        OP_LOGE("BroadcastTiling", "can not find computeKey");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(broadcastTilingParams.ubSize < computeParams.extraSize[0],
                OP_LOGE("BroadcastTiling", "ubSize is smaller than extra size."), return ge::GRAPH_FAILED);

    // 获取最大存活空间大小
    uint64_t maxElemNum = BroadcastGetMaxElemNum(broadcastTilingParams.ubSize, computeParams);
    OP_LOGI("Broadcast", "Broadcast DoBrodcastTiling. origin maxElemNum: %lu ubSize: %ld", maxElemNum,
            broadcastTilingParams.ubSize);
    OP_CHECK_IF(broadcastTilingParams.ubSize <= 0, OP_LOGE("BroadcastTiling", "ubSize can not be 0"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(broadcastTilingParams.coreNum <= 0, OP_LOGE("BroadcastTiling", "coreNum can not be 0"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(maxElemNum == 0, OP_LOGE("BroadcastTiling", "maxElemNum can not be 0"), return ge::GRAPH_FAILED);

    ubSplitInfo ubInfo;
    uint64_t fusedProduct = GetBlockSplitFactorNLastTranspose(broadcastTilingData, ubInfo, maxElemNum, computeParams,
                                                              isUbBroadcast);
    uint64_t blockFormer = (fusedProduct + broadcastTilingParams.coreNum - 1) / broadcastTilingParams.coreNum;
    uint64_t blockNum = (fusedProduct + blockFormer - 1) / blockFormer;

    // 当preferMultiCore为true且当前使用的核数少于总核数，尝试降低UB分块以充分利用更多核心
    if (broadcastTilingParams.preferMultiCore && blockNum < static_cast<uint64_t>(broadcastTilingParams.coreNum)) {
        // dstFusedProduct为尽量切多核时的理想多核切分因子
        uint64_t dstFusedProduct = blockFormer * broadcastTilingParams.coreNum;
        uint64_t tmpFusedProduct = fusedProduct;
        while (tmpFusedProduct < dstFusedProduct) {
            maxElemNum = maxElemNum - CACHE_LINE;
            if (maxElemNum <= CACHE_LINE_512) {
                break;
            }
            tmpFusedProduct = GetBlockSplitFactorNLastTranspose(broadcastTilingData, ubInfo, maxElemNum, computeParams,
                                                                isUbBroadcast);
        }
        // 计算最终调整后的多核切分因子
        maxElemNum = (maxElemNum + CACHE_LINE + CACHE_LINE - 1) / CACHE_LINE * CACHE_LINE;
        fusedProduct = GetBlockSplitFactorNLastTranspose(broadcastTilingData, ubInfo, maxElemNum, computeParams,
                                                         isUbBroadcast);
        // 更新blockFormer和blockNum
        blockFormer = (fusedProduct + broadcastTilingParams.coreNum - 1) / broadcastTilingParams.coreNum;
        blockNum = (fusedProduct + blockFormer - 1) / blockFormer;
    }

    uint64_t blockTail = fusedProduct - (blockNum - 1) * blockFormer;
    uint64_t dimProductBeforeUbInner = fusedProduct;
    OP_LOGI("Broadcast", "Broadcast DoBrodcastTiling. maxElemNum: %lu fusedProduct: %lu ubFormer: %ld ", maxElemNum,
            fusedProduct, ubInfo.ubFormer);

    broadcastTilingData.ubSplitAxis = ubInfo.ubSplitAxis;
    broadcastTilingData.ubFormer = ubInfo.ubFormer;
    broadcastTilingData.ubOuter = ubInfo.ubOuter;
    broadcastTilingData.ubTail = ubInfo.ubTail;

    broadcastTilingData.blockFormer = blockFormer;
    broadcastTilingData.blockNum = blockNum;
    broadcastTilingData.blockTail = blockTail;
    broadcastTilingData.dimProductBeforeUbInner = dimProductBeforeUbInner;
    broadcastTilingData.elemNum = maxElemNum;
    int64_t minDtypeBits = computeParams.minDtypeBits;
    int64_t minDtypeBlockAlignSize = BLOCK_LENGTH * BROADCAST_BITS_NUM / minDtypeBits;
    broadcastTilingData.minDtypeBlockAlignSize = minDtypeBlockAlignSize;
    uint64_t scheduleKey = BroadcastGetScheduleKey(broadcastTilingData.shapeLen - broadcastTilingData.ubSplitAxis);
    broadcastTilingData.innerKey = computeKey * BROADCAST_COMPUTE_KEY_OFFSET + scheduleKey;
    return ge::GRAPH_SUCCESS;
}

} // namespace Base
} // namespace Ops
