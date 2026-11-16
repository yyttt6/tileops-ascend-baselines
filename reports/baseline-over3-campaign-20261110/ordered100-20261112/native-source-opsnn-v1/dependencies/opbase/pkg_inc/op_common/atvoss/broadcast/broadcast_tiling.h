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
 * \file broadcast_tiling.h
 * \brief atvoss broadcast template tiling
 */
#ifndef BROADCAST_TILING_H_
#define BROADCAST_TILING_H_

#include "broadcast_tiling_base.h"
#include "broadcast_tiling_noncontiguous.h"

namespace Ops {
namespace Base {

OPBASE_API uint64_t GetBlockSplitFactor(BroadcastTilingData& broadcastTilingData, ubSplitInfo& ubInfo,
                                        uint64_t maxElemNum);

OPBASE_API ge::graphStatus DoBrodcastTiling(const BroadcastTilingParams& broadcastTilingParams,
                                            BroadcastTilingData& broadcastTilingData);

OPBASE_API ge::graphStatus BroadcastTiling(const BroadcastTilingParams& broadcastTilingParams,
                                           BroadcastTilingData& broadcastTilingData);

OPBASE_API ge::graphStatus DoDimensionCollapse(const BroadcastTilingParams& broadcastTilingParams,
                                               BroadcastTilingData& broadcastTilingData);

OPBASE_API ge::graphStatus NonContiguousDimensionCollapse(const std::vector<gert::Shape>& inShapes,
                                                          const std::vector<gert::Stride>& inStrides,
                                                          const gert::Shape& outShapes,
                                                          std::vector<std::vector<int64_t>>& dims,
                                                          std::vector<std::vector<int64_t>>& strides);

OPBASE_API ge::graphStatus DimensionCollapse(const std::vector<gert::Shape>& inShapes, const gert::Shape& outShapes,
                                             std::vector<std::vector<int64_t>>& dims,
                                             std::vector<std::vector<int64_t>>& strides);

OPBASE_API ge::graphStatus IsTensorContiguous(const gert::Shape& viewShape, const gert::Stride* viewStride,
                                              bool& isContiguous);

const gert::Shape g_vec_1_shape = {1};
/**
 * Ensure that the returned shape is non-scalar.
 * When the dim num of shape is 0, this shape is considered to express a scalar.
 * This function returns the original shape when it receives a non-scalar shape,
 * and returns the vector shape that returns a {1} when it receives a scalar shape
 * @param inShape input shape
 * @return non-scalar shape
 */
inline const gert::Shape& EnsureNotScalar(const gert::Shape& inShape)
{
    if (inShape.IsScalar()) {
        return g_vec_1_shape;
    }
    return inShape;
}

template <typename OpDag>
class OPBASE_API BroadcastBaseTiling {
public:
    /**
     *  构造函数
     * @param context tiling上下文
     * @param kernelType kernel选择的类型，1,NDDMA 2 UB BRC，默认为0，都选择
     */
    explicit BroadcastBaseTiling(gert::TilingContext* context,
                                 uint32_t kernelType = static_cast<uint32_t>(BROADCAST_KERNEL_TYPE::KERNEL_TYPE_BOTH))
        : context_(context), kernelType_(kernelType)
    {}

    /**
     *  tiling入口
     *  @param extraSize      额外存活空间
     *  @param extraBufferNum 额外存活节点
     *  @param preferMultiCore 是否优先切多核
     */
    ge::graphStatus DoTiling(int64_t extraSize = 0, int64_t extraBufferNum = 0, bool preferMultiCore = false)
    {
        OP_LOGI(context_->GetNodeName(),
                "Broadcast DoTiling. extraSize: %ld extraBufNum: %ld "
                "Broadcast dag param InputSize is %u, CopyBrcSize is %u, VecBrcSize is %u ",
                tmpExtraSize, tmpExtraBufferNum, OpDag::InputSize, OpDag::CopyBrcSize, OpDag::VecBrcSize);

        tmpExtraSize = extraSize;
        tmpExtraBufferNum = extraBufferNum;
        BroadcastTilingParams broadcastTilingParams;

        // 1. 获取平台信息
        auto status = GetPlatformInfo();
        if (status != ge::GRAPH_SUCCESS) {
            OP_LOGE(context_, "Get platform info failed.");
            return ge::GRAPH_FAILED;
        }

        // 2. 获取原始输入输出shape信息
        status = GetShapeInfo();
        if (status != ge::GRAPH_SUCCESS) {
            OP_LOGE(context_, "Get shape info failed.");
            return ge::GRAPH_FAILED;
        }

        BrcPrintShapes(inShapes, "inShapes");
        BrcPrintShape(outShape, "outShape");
        BrcPrintStrides(inStrides, "inStrides");

        broadcastTilingParams.inShape = inShapes;
        broadcastTilingParams.inStride = inStrides;
        broadcastTilingParams.outShape = outShape;
        broadcastTilingParams.coreNum = coreNum;
        broadcastTilingParams.ubSize = ubSize;
        broadcastTilingParams.inputAllContiguous = inputAllContiguous;
        broadcastTilingParams.preferMultiCore = preferMultiCore;

        OP_LOGI(context_->GetNodeName(), "coreNum: %llu, ubSize: %llu, inputAllContiguous:%d", coreNum, ubSize,
                inputAllContiguous);

        // 3. 合轴
        status = DoDimensionCollapse(broadcastTilingParams, tilingData);
        if (status != ge::GRAPH_SUCCESS) {
            OP_LOGE(context_, "dimension collapse failed.");
            return ge::GRAPH_FAILED;
        }

        BrcPrintVectors(tilingData.dims, "after DoDimensionCollapse in&out dims");
        BrcPrintVectors(tilingData.strides, "after DoDimensionCollapse in&out strides");

        // 4. 根据轴信息判断是否走onedim分支
        OP_CHECK_IF((tilingData.dims.back().size() == 0),
                    OP_LOGE(context_->GetNodeName(), "tensor check is empty, check failed"), return ge::GRAPH_FAILED);
        if (tilingData.dims.back().size() == 1 && broadcastTilingParams.inputAllContiguous) {
            if constexpr ((OpDag::InputSize + OpDag::OutputSize <= MAX_IN_OUT_ARGS_NUMBER) && OpDag::VarSize <= 0) {
                status = DoOneDimOpTilingAdvance();
            } else {
                status = DoOneDimOpTiling();
            }
        } else if (broadcastTilingParams.inputAllContiguous) {
            status = DoBroadcastOpTiling(broadcastTilingParams);
        } else {
            if constexpr (OpDag::VecBrcSize > 0) {
                OP_LOGE(context_, "Do broadcast tiling failed, VecBrc noncontigunous not supported.");
                return ge::GRAPH_FAILED;
            }
            status = DoNonContiguousBroadcastOpTiling(broadcastTilingParams);
        }

        if (status != ge::GRAPH_SUCCESS) {
            OP_LOGE(context_, "Do broadcast tiling failed.");
            return ge::GRAPH_FAILED;
        }

        // set workspace
        size_t* workspaces = context_->GetWorkspaceSizes(1);
        OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
        workspaces[0] = DEFAULT_TILING_WORKSPACE_SIZE;

        return ge::GRAPH_SUCCESS;
    }

    /**
     *  提供给算子获取schMode的接口
     * @return schMode
     */
    uint64_t GetSchMode() { return schMode; }

    /**
     *  提供给算子设置scalar的接口
     * @tparam T    scalar数据类型
     * @param value scalar值
     */
    template <typename T>
    void SetScalar(T value)
    {
        if (brcBaseTilingData != nullptr) {
            *reinterpret_cast<T*>(&brcBaseTilingData->scalarData[offset]) = value;
        } else if (brcOneDimTilingData != nullptr) {
            *reinterpret_cast<T*>(&brcOneDimTilingData->scalarData[offset]) = value;
        } else if (brcNLastTransposeTilingData != nullptr) {
            *reinterpret_cast<T*>(&brcNLastTransposeTilingData->scalarData[offset]) = value;
        } else if (brcLastTransposeTilingData != nullptr) {
            *reinterpret_cast<T*>(&brcLastTransposeTilingData->scalarData[offset]) = value;
        } else {
            *reinterpret_cast<T*>(&scalarData[offset]) = value;
        }
        offset += sizeof(T);
    }

    /**
     *  提供给算子设置输入shape的接口
     * @param opInputStorageShapes 算子自定义的输入shape
     * @return
     */
    ge::graphStatus SetOpInputStorageShapes(const std::vector<gert::Shape>& opInputStorageShapes)
    {
        OP_CHECK_IF((opInputStorageShapes.size() <= 0),
                    OP_LOGE(context_->GetNodeName(), "The size of the parameter opInputStorageShapes for the method "
                                                     "SetOpInputStorageShapes must be greater than 0."),
                    return ge::GRAPH_FAILED);
        OP_LOGI(context_->GetNodeName(), "Size of opInputStorageShapes is %zu", opInputStorageShapes.size());
        inShapes = opInputStorageShapes;

        return ge::GRAPH_SUCCESS;
    }

private:
    /**
     *  获取coreNum和ubSize
     * @return
     */
    ge::graphStatus GetPlatformInfo()
    {
        auto platformInfo = context_->GetPlatformInfo();
        if (platformInfo == nullptr) {
            auto compileInfoPtr = reinterpret_cast<const BroadcastCompileInfo*>(context_->GetCompileInfo());
            OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_, "compile info is null"), return ge::GRAPH_FAILED);
            coreNum = compileInfoPtr->coreNum;
            ubSize = compileInfoPtr->ubSize;

            OP_CHECK_IF((coreNum == 0),
                        OP_LOGE(context_->GetNodeName(), "compileInfo core num check is 0, check failed"),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF((ubSize == 0), OP_LOGE(context_->GetNodeName(), "compileInfo ubSize check is 0, check failed"),
                        return ge::GRAPH_FAILED);
        } else {
            auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
            coreNum = ascendcPlatform.GetCoreNumAiv();
            uint64_t ubSizePlatForm = 0;
            ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
            ubSize = ubSizePlatForm;

            OP_CHECK_IF((coreNum == 0),
                        OP_LOGE(context_->GetNodeName(), "ascendcPlatform core num check is 0, check failed"),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF((ubSize == 0),
                        OP_LOGE(context_->GetNodeName(), "ascendcPlatform ubSize check is 0, check failed"),
                        return ge::GRAPH_FAILED);
        }

        OP_LOGI(context_->GetNodeName(), "Broadcast GetPlatformInfo ubSize is %lu, coreNum is %lu", ubSize, coreNum);
        return ge::GRAPH_SUCCESS;
    }

    /**
     *  获取输入输出shape
     * @return
     */
    ge::graphStatus GetShapeInfo()
    {
        auto outStorageShape = context_->GetOutputShape(0);
        OP_CHECK_NULL_WITH_CONTEXT(context_, outStorageShape);
        outShape = Ops::Base::EnsureNotScalar(outStorageShape->GetStorageShape());
        // If inShapes is not empty, it indicates that an operator has already performed custom settings.
        if (!inShapes.empty()) {
            return ge::GRAPH_SUCCESS;
        }
        for (uint64_t i = 0; i < context_->GetComputeNodeInputNum(); i++) {
            inStrides.push_back(gert::Stride());
            bool isInContiguous = true;
            auto inputStorageShape = context_->GetInputShape(i);
            OP_CHECK_NULL_WITH_CONTEXT(context_, inputStorageShape);
            auto inCheckRet = IsTensorContiguous(inputStorageShape->GetShape(), context_->GetInputStride(i),
                                                 isInContiguous);
            OP_CHECK_IF((inCheckRet != ge::GRAPH_SUCCESS),
                        OP_LOGE(context_->GetNodeName(), "check input %lu IsTensorContiguous failed", i),
                        return ge::GRAPH_FAILED);
            if (!isInContiguous) {
                inShapes.push_back(inputStorageShape->GetShape());
                inStrides[i] = *(context_->GetInputStride(i));
                inputAllContiguous = false;
            } else {
                gert::Shape inputShape;
                // aclnn 使用tensorV2构造的tensor，没有经过Contiguous处理，通过GetShape获取
                if (context_->InputIsView(i)) {
                    inputShape = inputStorageShape->GetShape();
                } else {
                    inputShape = inputStorageShape->GetStorageShape();
                }
                auto& storageShape = Ops::Base::EnsureNotScalar(inputShape);
                inShapes.push_back(storageShape);
            }
        }
        return ge::GRAPH_SUCCESS;
    }

    /**
     *  根据kernel类型设置schmode
     * @param isUbBroadcast 是否选择了UB BRC模版
     * @return
     */
    void SetSchMode(bool isUbBroadcast)
    {
        if (isUbBroadcast) {
            int32_t runningRank = tilingData.shapeLen - tilingData.ubSplitAxis;
            int32_t rank = runningRank > HIGH_PERF_RANK ? DEFAULT_ADD_VALUE : runningRank;
            schMode = SCH_MODE_UB_BROADCAST + rank;
        } else {
            schMode = tilingData.shapeLen - tilingData.ubSplitAxis > MAX_NNDMA_DIM ?
                          SCH_MODE_NDDMA_WITH_LOOP_CACHE_BRC :
                          SCH_MODE_NDDMA_WITHOUT_LOOP_CACHE_BRC;
        }
    }

    /**
     *  适配tilingData
     * @return
     */
    void AdaptBroadcastBaseTilingData()
    {
        brcBaseTilingData->blockFormer = tilingData.blockFormer;
        brcBaseTilingData->ubFormer = tilingData.ubFormer;
        brcBaseTilingData->ubOuter = tilingData.ubOuter;
        brcBaseTilingData->ubTail = tilingData.ubTail;
        brcBaseTilingData->blockTail = tilingData.blockTail;
        brcBaseTilingData->shapeLen = tilingData.shapeLen;
        brcBaseTilingData->ubSplitAxis = tilingData.ubSplitAxis;
        brcBaseTilingData->dimProductBeforeUbInner = tilingData.dimProductBeforeUbInner;
        brcBaseTilingData->elemNum = tilingData.elemNum;
        brcBaseTilingData->blockNum = tilingData.blockNum;
        std::copy(tilingData.dims.back().begin(), tilingData.dims.back().end(), brcBaseTilingData->outputDims);
        std::copy(tilingData.strides.back().begin(), tilingData.strides.back().end(), brcBaseTilingData->outputStrides);

        // 手动循环逐个字节拷贝
        for (uint64_t i = 0; i < BROADCAST_MAX_SCALAR_BYTES; i++) {
            brcBaseTilingData->scalarData[i] = scalarData[i];
        }
    }
    /**
     *  获取copyInBrc和VecBrc的位置，并按顺序将对应节点的输入shape信息设置到tilingdata中
     * @tparam pos 按顺序执行递归的位置
     * @return
     */
    template <int pos = 0>
    void SetInputNddmaParams()
    {
        if constexpr (OpDag::CopyBrcSize != 0) {
            using Op = typename OpDag::CopyBrcNodes::template At<pos>;
            using Input = typename Op::InHolders::template At<0>;
            int64_t idx = Input::Pos;
            std::copy(tilingData.dims[idx].begin(), tilingData.dims[idx].end(),
                      *(brcBaseTilingData->inputBrcDims + pos));
            std::copy(tilingData.strides[idx].begin(), tilingData.strides[idx].end(),
                      *(brcBaseTilingData->inputBrcStrides + pos));
            if constexpr (pos + 1 < OpDag::CopyBrcSize) {
                SetInputNddmaParams<pos + 1>();
            }
        }
    }

    /**
     *  获取copyInBrc的位置
     * @tparam pos 按顺序执行递归的位置
     * @return
     */
    template <int pos = 0>
    void GetCopyInBrcPosList()
    {
        if constexpr (OpDag::CopyBrcSize != 0) {
            using Op = typename OpDag::CopyBrcNodes::template At<pos>;
            using Input = typename Op::InHolders::template At<0>;
            int64_t idx = Input::Pos;
            copyInBrcPos.insert(idx);
            if constexpr (pos + 1 < OpDag::CopyBrcSize) {
                GetCopyInBrcPosList<pos + 1>();
            }
        }
    }

    /**
     *  获取VecBrc的位置，并按顺序将对应节点的输入shape信息设置到tilingdata中
     *  @tparam pos 按顺序执行递归的位置
     */
    template <int pos = 0>
    void SetInputUbBrcParams()
    {
        if constexpr (OpDag::VecBrcSize > 0) {
            using Op = typename OpDag::VecBrcNodes::template At<pos>;
            using Input = typename Op::SourceInHolders::template At<0>;
            int64_t idx = Input::Pos;

            std::copy(tilingData.dims[idx].begin(), tilingData.dims[idx].end(),
                      *(brcBaseTilingData->inputVecBrcDims + pos));

            brcBaseTilingData->inputVecBrcStrides[pos] = tilingData.strides[idx][tilingData.ubSplitAxis];
            if constexpr (pos + 1 < OpDag::VecBrcSize) {
                SetInputUbBrcParams<pos + 1>();
            }
        }
    }

    /**
     *  根据切分结果，设置CopyIn节点对应的切分大小
     * @return
     */
    void SetInputParams()
    {
        for (uint32_t i = 0; i < OpDag::InputSize; i++) {
            if (copyInBrcPos.count(i)) {
                continue;
            }
            std::copy(tilingData.strides[i].begin(), tilingData.strides[i].end(),
                      *(brcBaseTilingData->inputStrides + i));

            int64_t formerLength = 1;
            int64_t tailLength = 1;
            if (tilingData.dims[i][tilingData.ubSplitAxis] != 1) {
                formerLength = tilingData.ubFormer;
                tailLength = tilingData.ubTail;
            }
            for (int64_t j = tilingData.ubSplitAxis + 1; j < tilingData.shapeLen; j++) {
                formerLength *= tilingData.dims[i][j];
                tailLength *= tilingData.dims[i][j];
            }

            brcBaseTilingData->inputDims[i][0] = formerLength;
            brcBaseTilingData->inputDims[i][1] = tailLength;
        }
    }
    /**
     *  判断是否为nlast brc并且尾轴大于4096场景， 该场景走ub brc。
     *  1. 该输入必须是copyInBrc节点
     *  2. 该输入不可以是last brc节点
     *  3. 该输入不可以是纯搬运节点
     *  4. 该输入满足尾轴字节数大于等于nddma dcache的一半
     * @return
     */
    bool IsBigNLastBrc()
    {
        for (uint64_t i = 0; i < tilingData.strides.size() - 1; i++) {
            if (!copyInBrcPos.count(i)) { // 该输入不是copyInBrc
                continue;
            }
            if (tilingData.strides[i].back() == 0) { // 该输入最后一个轴为broadcast轴
                continue;
            }

            int64_t broadcastIdx = -1;
            // 检查该输入除尾轴之外，是否存在broadcast轴
            for (uint64_t j = 0; j < tilingData.strides[i].size() - 1; j++) {
                if (tilingData.strides[i][j] == 0 && tilingData.strides.back()[j] != 0) {
                    broadcastIdx = j;
                    break;
                }
            }
            if (broadcastIdx != -1) { // 该输入除尾轴之外，存在broadcast轴
                auto inputDesc = context_->GetInputDesc(i);
                OP_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);
                auto inputDtype = inputDesc->GetDataType();
                int64_t dTypeSize = ge::GetSizeByDataType(inputDtype);
                int64_t nddmaDcacheSize = static_cast<int64_t>(Ops::Base::GetNddmaDcacheSize(context_));
                // 判断尾轴是否大于nddma dcache size 的一半
                if (tilingData.dims[i].back() * dTypeSize >= nddmaDcacheSize / VALUE_TWO) {
                    return true;
                }
            }
        }
        return false;
    }
    /**
     *  判断是否满足走UB BRC的条件
     *  1. 如果设置了kernelType，则以kernelType为准
     *  2. 如果所有输入都不需要做brc，则走nddma
     *  3. 如果没有CopyInBrc节点，则走nddma
     *  4. 如果是nlastBig场景，则走ub brc
     *  5. 如果是B8 B16，并且尾轴对齐场景则走ub brc
     * @return
     */
    bool IsUbBroadcast()
    {
        OP_LOGI(context_->GetNodeName(), "kernelType is %d", kernelType_);
        int64_t lastDim = tilingData.dims.back().back();
        if (kernelType_ == static_cast<uint32_t>(BROADCAST_KERNEL_TYPE::KERNEL_TYPE_NDDMA)) {
            return false;
        } else if (kernelType_ == static_cast<uint32_t>(BROADCAST_KERNEL_TYPE::KERNEL_TYPE_UB_BROADCAST)) {
            return true;
        }

        int64_t broadcastIdx = -1;
        for (uint64_t i = 0; i < tilingData.strides.size() - 1; i++) {
            for (uint64_t j = 0; j < tilingData.strides.back().size(); j++) {
                if (tilingData.strides[i][j] == 0 && tilingData.strides.back()[j] != 0) {
                    broadcastIdx = i;
                    break;
                }
            }
            if (broadcastIdx != -1) {
                break;
            }
        }
        if (broadcastIdx == -1) { // 纯elewise场景走NDDMA
            return false;
        }
        if constexpr (OpDag::CopyBrcSize == 0) { // 非elewise且不存在CopyInBrc节点场景走UBBRC
            return true;
        }
        if (IsBigNLastBrc()) { // NLast场景，尾轴大于4096B时走UBBRC
            return true;
        }
        auto inputDesc = context_->GetInputDesc(broadcastIdx);
        OP_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);
        auto inputDtype = inputDesc->GetDataType();
        int64_t dTypeSize = ge::GetSizeByDataType(inputDtype);
        if (dTypeSize != 0 && lastDim % (BLOCK_LENGTH / dTypeSize) == 0 && ubBroadcastDtypes.count(inputDtype)) {
            return true;
        }
        return false;
    }

    /**
     *  获取存活节点个数
     * @param isUbBroadcast 是否ub brc kernel
     * @return
     */
    int64_t GetBufferNum(bool isUbBroadcast)
    {
        if (isUbBroadcast) {
            return OpDag::template GetBufferNum<false, true>();
        } else {
            return OpDag::template GetBufferNum<true, true>();
        }
    }

    ge::graphStatus DoBroadcastOpTiling(BroadcastTilingParams& broadcastTilingParams)
    {
        brcBaseTilingData = context_->GetTilingData<BroadcastBaseTilingData<OpDag>>();

        OP_CHECK_IF((brcBaseTilingData == nullptr),
                    OP_LOGE(context_->GetNodeName(), "Get brcBaseTilingData from GE context failed"),
                    return ge::GRAPH_FAILED);
        // 1、判断ubbrc用到 2、设置copyIn参数用到
        GetCopyInBrcPosList();

        bool isUbBroadcast = IsUbBroadcast();
        int64_t bufferNum = GetBufferNum(isUbBroadcast);
        OP_LOGI(context_->GetNodeName(), "Broadcast DoBroadcastOpTiling. isUbBroadcast: %d bufferNum: %ld ",
                isUbBroadcast, bufferNum);

        BroadcastComputeParams params;
        params.maxDtypeBits = OpDag::MaxDtypeBytes * BROADCAST_BITS_NUM;
        params.minDtypeBits = OpDag::MinDtypeBytes * BROADCAST_BITS_NUM;
        params.extraSize = {tmpExtraSize};
        params.bufferDivisor = {(bufferNum + tmpExtraBufferNum) * params.maxDtypeBits};
        broadcastTilingParams.computeMap = {{1, params}};

        auto status = DoBrodcastTiling(broadcastTilingParams, tilingData);
        if (status != ge::GRAPH_SUCCESS) {
            OP_LOGE(context_, "Do broadcast tiling failed.");
            return ge::GRAPH_FAILED;
        }

        SetSchMode(isUbBroadcast);
        // adapt
        AdaptBroadcastBaseTilingData();

        SetInputNddmaParams<0>();
        SetInputUbBrcParams<0>();
        SetInputParams();

        // set block dim
        context_->SetBlockDim(brcBaseTilingData->blockNum);

        return ge::GRAPH_SUCCESS;
    }

    static bool isOnlyLastTwoAxesTransposed(const std::vector<int64_t>& viewShape, const std::vector<int64_t>& strides)
    {
        size_t shapeDim = viewShape.size();
        size_t stridesDim = strides.size();
        if (shapeDim < static_cast<size_t>(DIM_TWO) || shapeDim != stridesDim) {
            return false;
        }
        size_t lastDim = shapeDim - 1;
        size_t secondLastDim = stridesDim - static_cast<size_t>(DIM_TWO);
        bool transposedStride = (strides[lastDim] == viewShape[secondLastDim]) && (strides[secondLastDim] == 1);
        bool othersContiguous = true;
        if (shapeDim > DIM_TWO) {
            int64_t expectedStride = viewShape[lastDim] * viewShape[secondLastDim];
            for (int64_t i = shapeDim - DIM_THREE; i >= 0; i--) {
                if (strides[i] != expectedStride) {
                    othersContiguous = false;
                    break;
                }
                expectedStride = expectedStride * viewShape[i];
            }
        }
        return transposedStride && othersContiguous;
    }

    // 如果是隐式的broadcast场景，走兜底模版
    bool isBroadcastTo(std::vector<int64_t> dims, std::vector<int64_t> strides)
    {
        for (size_t i = 0; i < dims.size(); i++) {
            if (dims[i] != 1 && strides[i] == 0) {
                return true;
            }
        }
        return false;
    }

    //
    ge::graphStatus isBroadcastTemplateNonContiguousSupport(bool& isSupported, bool& isLastTranspose,
                                                            bool& isLastAxisBrc)
    {
        isSupported = true;
        isLastTranspose = false;
        bool isLastAxisContigunous = true;
        isLastAxisBrc = false;
        for (int64_t i = 0; i < static_cast<int64_t>(inShapes.size()); ++i) {
            auto shapeDim = tilingData.dims[i].size();
            auto strideDim = tilingData.strides[i].size();

            OP_CHECK_IF((shapeDim != strideDim),
                        OP_LOGE(context_->GetNodeName(), "shapeDim %lu != strideDim %lu", shapeDim, strideDim),
                        return ge::GRAPH_FAILED);

            // DimNum为0 表示连续场景
            if (inStrides[i].GetDimNum() == 0) {
                // 尾轴brc的话，NLast UbBrc模板不支持
                if (tilingData.strides[i][strideDim - 1] == 0) {
                    isLastAxisBrc = true;
                }
                continue;
            }

            // 如果是隐式的broadcast场景，走兜底模板
            if (isBroadcastTo(tilingData.dims[i], tilingData.strides[i])) {
                isSupported = false;
                return ge::GRAPH_SUCCESS;
            }

            // 尾轴跳读不连续的话，NLast模板都不支持
            if (tilingData.strides[i][strideDim - 1] > 1) {
                isLastAxisContigunous = false;
            }

            // 尾轴brc的话，NLast UbBrc模板不支持
            if (tilingData.strides[i][strideDim - 1] == 0) {
                isLastAxisBrc = true;
            }

            // 不支持大于4维
            if (shapeDim > 4) {
                isSupported = false;
                return ge::GRAPH_SUCCESS;
            }
            // 转置场景
            if (isOnlyLastTwoAxesTransposed(tilingData.dims[i], tilingData.strides[i]) && shapeDim <= 3) {
                isLastTranspose = true;
                continue;
            }

            auto inputDesc = context_->GetInputDesc(i);
            OP_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);
            auto inputDtype = inputDesc->GetDataType();
            int64_t dTypeSize = ge::GetSizeByDataType(inputDtype);
            int64_t dimLimit = CACHE_LINE / dTypeSize;
            int64_t sizeLimit = SINGLE_CORE_SIZE_LIMIT * coreNum;

            // last轴128B连续场景
            if ((tilingData.dims[i][shapeDim - 1] >= dimLimit) && (tilingData.strides[i][strideDim - 1] == 1)) {
                continue;
            }

            // 小包场景
            if (shapeDim > 1 && tilingData.dims[i][shapeDim - 1] < dimLimit &&
                tilingData.strides[i][strideDim - DIM_TWO] > dimLimit) {
                continue;
            }

            // 小shape场景
            if (tilingData.dims[i][shapeDim - 1] < dimLimit && inShapes[i].GetShapeSize() * dTypeSize <= sizeLimit) {
                continue;
            }

            isSupported = false;
            return ge::GRAPH_SUCCESS;
        }

        // nlast场景不支持尾轴非连续（尾轴stride大于1）
        if (isLastTranspose == false && isLastAxisContigunous == false) {
            isSupported = false;
        }

        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus DoNonContiguousBroadcastOpTiling(BroadcastTilingParams broadcastTilingParams)
    {
        bool isSupport = false;
        bool isLastTranspose = false;
        bool isLastAxisBrc = false;
        auto status = isBroadcastTemplateNonContiguousSupport(isSupport, isLastTranspose, isLastAxisBrc);

        OP_CHECK_IF((status != ge::GRAPH_SUCCESS),
                    OP_LOGE(context_->GetNodeName(), "isBroadcastTemplateNonContiguousSupport failed"),
                    return ge::GRAPH_FAILED);

        // 添加判断条件
        if (isSupport) {
            if (isLastTranspose) {
                // Last轴转置
                OP_LOGI(context_->GetNodeName(),
                        "Broadcast DoNonContiguousBroadcastOpTiling: enter last transpose schmode (303). Next to run "
                        "DoBroadcastOpTilingLastTranspose");
                status = DoBroadcastOpTilingLastTranspose(broadcastTilingParams);
            } else {
                // NLast轴转置 (尾轴需要连续(stride == 1))
                OP_LOGI(
                    context_->GetNodeName(),
                    "Broadcast DoNonContiguousBroadcastOpTiling: enter nlast transpose schmode (301 or 302). Next to "
                    "run "
                    "DoBroadcastOpTilingNLastTranspose");
                status = DoBroadcastOpTilingNLastTranspose(broadcastTilingParams, isLastAxisBrc);
            }
        } else {
            // 非连续兜底模板
            OP_LOGI(
                context_->GetNodeName(),
                "Broadcast DoNonContiguousBroadcastOpTiling: enter non_contigunous base schmode (304 or 305). Next to "
                "run "
                "DoBroadcastOpTilingNonContiguousBase");
            status = DoBroadcastOpTilingNonContiguousBase(broadcastTilingParams);
        }

        if (status != ge::GRAPH_SUCCESS) {
            OP_LOGE(context_, "Do broadcast tiling failed.");
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus DoBroadcastOpTilingNonContiguousBase(BroadcastTilingParams& broadcastTilingParams)
    {
        auto status = DoBroadcastOpTiling(broadcastTilingParams);

        if (status != ge::GRAPH_SUCCESS) {
            OP_LOGE(context_, "Do broadcast tiling failed.");
            return ge::GRAPH_FAILED;
        }

        schMode = tilingData.shapeLen - tilingData.ubSplitAxis > MAX_NNDMA_DIM ?
                      SCH_MODE_NON_CONTIGUOUS_NDDMA_WITH_LOOP_CACHE_BRC :
                      SCH_MODE_NON_CONTIGUOUS_NDDMA_WITHOUT_LOOP_CACHE_BRC;

        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus DoBroadcastOpTilingLastTranspose(BroadcastTilingParams& broadcastTilingParams)
    {
        brcLastTransposeTilingData = context_->GetTilingData<BroadcastLastTransposeTilingData<OpDag>>();

        OP_CHECK_IF((brcLastTransposeTilingData == nullptr),
                    OP_LOGE(context_->GetNodeName(), "Get brcLastTransposeTilingData from GE context failed"),
                    return ge::GRAPH_FAILED);
        int64_t bufferNum = GetBufferNum(false);

        BroadcastComputeParams params;
        params.maxDtypeBits = OpDag::MaxDtypeBytes * BROADCAST_BITS_NUM;
        params.minDtypeBits = OpDag::MinDtypeBytes * BROADCAST_BITS_NUM;
        params.extraSize = {tmpExtraSize};
        params.bufferDivisor = {(bufferNum + tmpExtraBufferNum) * params.maxDtypeBits};
        broadcastTilingParams.computeMap = {{1, params}};

        auto status = DoBroadcastTilingLastTranspose<OpDag, BroadcastLastTransposeTilingData<OpDag>>(
            broadcastTilingParams, brcLastTransposeTilingData, tilingData);
        if (status != ge::GRAPH_SUCCESS) {
            OP_LOGE(context_, "Do broadcast tiling failed.");
            return ge::GRAPH_FAILED;
        }

        schMode = SCH_MODE_NONCONTIGUOUS_LAST_TRANSPOSE;
        brcLastTransposeTilingData->scheMode = schMode;
        // 手动循环逐个字节拷贝
        for (uint64_t i = 0; i < BROADCAST_MAX_SCALAR_BYTES; i++) {
            brcLastTransposeTilingData->scalarData[i] = scalarData[i];
        }

        // set block dim
        context_->SetBlockDim(brcLastTransposeTilingData->blockNum);

        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus DoBroadcastOpTilingNLastTranspose(BroadcastTilingParams& broadcastTilingParams, bool isLastAxisBrc)
    {
        brcNLastTransposeTilingData = context_->GetTilingData<BroadcastNlastTransposeTilingData<OpDag>>();

        OP_CHECK_IF((brcNLastTransposeTilingData == nullptr),
                    OP_LOGE(context_->GetNodeName(), "Get brcNLastTransposeTilingData from GE context failed"),
                    return ge::GRAPH_FAILED);

        // 1、判断ubbrc用到 2、设置copyIn参数用到
        GetCopyInBrcPosList();
        bool isUbBroadcast = isLastAxisBrc ? false : IsUbBroadcast();
        int64_t bufferNum = GetBufferNum(isUbBroadcast);

        BroadcastComputeParams params;
        params.maxDtypeBits = OpDag::MaxDtypeBytes * BROADCAST_BITS_NUM;
        params.minDtypeBits = OpDag::MinDtypeBytes * BROADCAST_BITS_NUM;
        params.extraSize = {tmpExtraSize};
        params.bufferDivisor = {(bufferNum + tmpExtraBufferNum) * params.maxDtypeBits};
        broadcastTilingParams.computeMap = {{1, params}};

        auto status = DoBroadcastTilingNLastTranspose<OpDag, BroadcastNlastTransposeTilingData<OpDag>>(
            broadcastTilingParams, brcNLastTransposeTilingData, tilingData, isUbBroadcast);
        OP_CHECK_IF((status != ge::GRAPH_SUCCESS),
                    OP_LOGE(context_->GetNodeName(), "DoBroadcastTilingNLastTranspose failed"),
                    return ge::GRAPH_FAILED);

        schMode = isUbBroadcast ? SCH_MODE_NONCONTIGUOUS_NLAST_TRANSPOSE_UB_BRC :
                                  SCH_MODE_NONCONTIGUOUS_NLAST_TRANSPOSE_NDDMA;
        brcNLastTransposeTilingData->scheMode = schMode;

        OP_LOGI(context_->GetNodeName(),
                "Broadcast DoBroadcastOpTilingNLastTranspose: now enter nlast transpose schmode (%ld). ", schMode);

        // 手动循环逐个字节拷贝
        for (uint64_t i = 0; i < BROADCAST_MAX_SCALAR_BYTES; i++) {
            brcNLastTransposeTilingData->scalarData[i] = scalarData[i];
        }

        // set block dim
        context_->SetBlockDim(brcNLastTransposeTilingData->blockNum);

        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus GetOneDimScalarFlag()
    {
        int32_t scalarFlag = 0;
        for (uint64_t i = 0; i < tilingData.dims.size() - 1; i++) {
            int32_t isScalar = tilingData.dims[i][0] == 1 ? ONE_DIM_INPUT_IS_SCALAR : ONE_DIM_INPUT_IS_NOT_SCALAR;
            int32_t input_index = (1 << i) * isScalar;
            scalarFlag = scalarFlag + input_index;
        }
        brcOneDimTilingData->scalarFlag = scalarFlag;

        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus GetOneDimUBFormer(int32_t aliveNum, int64_t actualUbSize)
    {
        OP_CHECK_IF((aliveNum == 0),
                    OP_LOGE(context_->GetNodeName(), "aliveNum check is 0, check failed, aliveNum is %d", aliveNum),
                    return ge::GRAPH_FAILED);

        // 先分ub，再分核
        // 开DB，CACHE_LINE 对齐
        const uint32_t dTypeSize = OpDag::MaxDtypeBytes;
        OP_CHECK_IF((dTypeSize == 0), OP_LOGE(context_->GetNodeName(), "OpDag::MaxDtypeBytes check is 0, check failed"),
                    return ge::GRAPH_FAILED);

        OP_LOGI(context_->GetNodeName(), "Broadcast GetOneDimUBFormer. aliveNum: %d dTypeSize: %ld ", aliveNum,
                dTypeSize);

        int64_t ubFormerByte = actualUbSize / aliveNum;
        int64_t ubFormerByteFloorAlign = (ubFormerByte / CACHE_LINE) * CACHE_LINE;
        brcOneDimTilingData->ubFormer = ubFormerByteFloorAlign / dTypeSize;
        OP_CHECK_IF((brcOneDimTilingData->ubFormer == 0),
                    OP_LOGE(context_->GetNodeName(),
                            "ub size or CACHE_LINE check failed. actualUbSize: %ld CACHE_LINE: %ld dTypeSize: %u",
                            actualUbSize, CACHE_LINE, dTypeSize),
                    return ge::GRAPH_FAILED);

        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus AdaptOneDimTilingData(int64_t blockNum, int64_t ubFormer, int64_t ubTail, int64_t blockFormer,
                                          int64_t blockTail)
    {
        brcOneDimTilingData->blockNum = blockNum;
        brcOneDimTilingData->ubFormer = ubFormer;
        brcOneDimTilingData->ubTail = ubTail;
        brcOneDimTilingData->blockFormer = blockFormer;
        brcOneDimTilingData->blockTail = blockTail;

        for (uint64_t i = 0; i < BROADCAST_MAX_SCALAR_BYTES; i++) {
            brcOneDimTilingData->scalarData[i] = scalarData[i];
        }

        return ge::GRAPH_SUCCESS;
    }
    ge::graphStatus DoOneDimFormerTiling(int32_t aliveNum)
    {
        OP_CHECK_IF((aliveNum == 0),
                    OP_LOGE(context_->GetNodeName(), "aliveNum check is 0, check failed, aliveNum is %d", aliveNum),
                    return ge::GRAPH_FAILED);

        const uint32_t dTypeSize = OpDag::MaxDtypeBytes;
        OP_CHECK_IF((tilingData.dims.back().size() == 0),
                    OP_LOGE(context_->GetNodeName(), "tilingData.dims.back check is empty, check failed"),
                    return ge::GRAPH_FAILED);

        int64_t dimLength = tilingData.dims.back()[0];
        OP_CHECK_IF((dimLength == 0), OP_LOGE(context_->GetNodeName(), "dimLength is zero, check failed"),
                    return ge::GRAPH_FAILED);

        // 为方便kernel计算，减少kernel scalar，对于整除无尾场景，也认为有个整数的尾块
        int64_t ubFormer = brcOneDimTilingData->ubFormer;
        int64_t ubOuter = (dimLength + ubFormer - 1) / ubFormer;
        int64_t ubTail = dimLength % ubFormer;
        ubTail = (ubTail == 0) ? ubFormer : ubTail;
        int64_t blockFormer = (ubOuter + coreNum - 1) / coreNum;
        int64_t blockTail = ubOuter % blockFormer;
        blockTail = (blockTail == 0) ? blockFormer : blockTail;
        int64_t blockNum = (ubOuter + blockFormer - 1) / blockFormer;

        // 如果blockNum小于核数一半，则尝试分配到核数的一半
        if (static_cast<uint64_t>(blockNum) < (coreNum / HALF_CORE_NUM_DIVIDE) &&
            (ubFormer * dTypeSize * aliveNum) > PER_CORE_MIN_UB_BYTE) {
            int64_t dimPerCore = dimLength * 2 / coreNum; // 1/2 的coreNum进行分配
            // 向上对齐到128Byte，理论上不会超过原来计算的ubFormer
            int64_t alignDimPerCore = ((((dimPerCore * dTypeSize) + CACHE_LINE - 1) / CACHE_LINE) * CACHE_LINE) /
                                      dTypeSize;
            ubFormer = (ubFormer > alignDimPerCore) ? alignDimPerCore : ubFormer;

            // 如果分配到核数一半后，小于8k(开DB后)，则直接按照 8k
            // 计算。进入该分支，说明前面已经判断过开db后ub大小大于8KB
            int64_t lowestUbFormer = (((PER_CORE_MIN_UB_BYTE / aliveNum) / CACHE_LINE) * CACHE_LINE) / dTypeSize;
            if (ubFormer < lowestUbFormer) {
                ubFormer = lowestUbFormer;
            }

            // 重新计算分核参数
            ubOuter = (dimLength + ubFormer - 1) / ubFormer;
            ubTail = dimLength % ubFormer;
            ubTail = (ubTail == 0) ? ubFormer : ubTail;
            blockFormer = (ubOuter + coreNum - 1) / coreNum;
            blockTail = ubOuter % blockFormer;
            blockTail = (blockTail == 0) ? blockFormer : blockTail;
            blockNum = (ubOuter + blockFormer - 1) / blockFormer;
        }

        AdaptOneDimTilingData(blockNum, ubFormer, ubTail, blockFormer, blockTail);

        OP_LOGI(context_->GetNodeName(),
                "Broadcast one dim do tiling finish. coreNum: %ld CACHE_LINE: %ld "
                "blockNum: %ld ubFormer: %ld ubTail: %ld blockFormer: %ld blockTail: %ld "
                "dimLength: %ld, scalarFlag: %d",
                coreNum, CACHE_LINE, blockNum, ubFormer, ubTail, blockFormer, blockTail, dimLength,
                brcOneDimTilingData->scalarFlag);

        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus DoOneDimOpTiling()
    {
        brcOneDimTilingData = context_->GetTilingData<BroadcastOneDimTilingData>();

        OP_CHECK_IF((brcOneDimTilingData == nullptr),
                    OP_LOGE(context_->GetNodeName(), "Get brcOneDimTilingData from GE context failed"),
                    return ge::GRAPH_FAILED);

        int32_t aliveNum = OpDag::template GetBufferNum<true, true>();
        aliveNum = aliveNum + tmpExtraBufferNum;
        int64_t actualUbSize = ubSize - tmpExtraSize;

        OP_CHECK_IF((GetOneDimScalarFlag() == ge::GRAPH_FAILED),
                    OP_LOGE(context_->GetNodeName(), "GetOneDimScalarFlag failed"), return ge::GRAPH_FAILED);

        OP_CHECK_IF((GetOneDimUBFormer(aliveNum, actualUbSize) == ge::GRAPH_FAILED),
                    OP_LOGE(context_->GetNodeName(), "GetUBFormer failed"), return ge::GRAPH_FAILED);

        OP_CHECK_IF((DoOneDimFormerTiling(aliveNum) == ge::GRAPH_FAILED),
                    OP_LOGE(context_->GetNodeName(), "DoOneDimFormerTiling failed"), return ge::GRAPH_FAILED);

        schMode = SCH_MODE_ONE_DIM;
        context_->SetBlockDim(brcOneDimTilingData->blockNum);

        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus GetOneDimScalarFlagAdvance()
    {
        int16_t scalarFlag = 0;
        for (uint64_t i = 0; i < tilingData.dims.size() - 1; i++) {
            int32_t isScalar = tilingData.dims[i][0] == 1 ? ONE_DIM_INPUT_IS_SCALAR : ONE_DIM_INPUT_IS_NOT_SCALAR;
            int32_t input_index = (1 << i) * isScalar;
            scalarFlag = scalarFlag + input_index;
        }
        brcOneDimTilingDataAdvance->scalarFlag = scalarFlag;

        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus GetOneDimUBFormerAdvance(int32_t aliveNum, int64_t actualUbSize)
    {
        OP_CHECK_IF((aliveNum == 0),
                    OP_LOGE(context_->GetNodeName(), "aliveNum check is 0, check failed, aliveNum is %d", aliveNum),
                    return ge::GRAPH_FAILED);

        // 先分ub，再分核
        // 开DB，CACHE_LINE 对齐
        const uint32_t dTypeSize = OpDag::MaxDtypeBytes;
        OP_CHECK_IF((dTypeSize == 0), OP_LOGE(context_->GetNodeName(), "OpDag::MaxDtypeBytes check is 0, check failed"),
                    return ge::GRAPH_FAILED);

        OP_LOGI(context_->GetNodeName(), "Broadcast GetOneDimUBFormerAdvance. aliveNum: %d dTypeSize: %u ", aliveNum,
                dTypeSize);

        int64_t ubFormerByte = actualUbSize / aliveNum;
        int64_t ubFormerByteFloorAlign = (ubFormerByte / CACHE_LINE_512) * CACHE_LINE_512;
        brcOneDimTilingDataAdvance->tileNum = ubFormerByteFloorAlign / dTypeSize;
        OP_CHECK_IF(
            (brcOneDimTilingDataAdvance->tileNum == 0),
            OP_LOGE(context_->GetNodeName(),
                    "ub size or CACHE_LINE_512 check failed. actualUbSize: %ld CACHE_LINE_512: %ld dTypeSize: %u",
                    actualUbSize, CACHE_LINE_512, dTypeSize),
            return ge::GRAPH_FAILED);

        OP_LOGI(context_->GetNodeName(), "Broadcast GetOneDimUBFormerAdvance finish. tileNum is: %d ",
                brcOneDimTilingDataAdvance->tileNum);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus DoOneDimFormerTilingAdvance()
    {
        OP_LOGI(context_->GetNodeName(), "Broadcast DoOneDimFormerTilingAdvance. tilingData.dims.back().size is: %zu ",
                tilingData.dims.back().size());
        const uint32_t inputDTypeSize = OpDag::MinDtypeBytes;
        const uint32_t dTypeSize = OpDag::MaxDtypeBytes;
        OP_CHECK_IF((tilingData.dims.back().size() == 0),
                    OP_LOGE(context_->GetNodeName(), "tilingData.dims.back check is empty, check failed"),
                    return ge::GRAPH_FAILED);

        uint64_t dimLength = tilingData.dims.back()[0];
        OP_CHECK_IF((dimLength == 0), OP_LOGE(context_->GetNodeName(), "dimLength is zero, check failed"),
                    return ge::GRAPH_FAILED);

        int32_t tileNum = brcOneDimTilingDataAdvance->tileNum;
        uint64_t blockNum = coreNum;
        int32_t inputcacheLineNum = CACHE_LINE_512 / inputDTypeSize;
        int32_t cacheLineNum = CACHE_LINE_512 / dTypeSize;
        OP_LOGI(context_->GetNodeName(),
                "Broadcast DoOneDimFormerTilingAdvance begin. blockNum: %lu, inputDTypeSize: %u, inputcacheLineNum %d, "
                "cacheLineNum %d, dimLength %lu",
                blockNum, inputDTypeSize, inputcacheLineNum, cacheLineNum, dimLength);
        if (coreNum * inputcacheLineNum > dimLength) {
            // 如果dimLength不足 coreNum * 512, 则减少核数。
            tileNum = inputcacheLineNum;
        } else if (coreNum * tileNum > dimLength) {
            // 如果dimLength不足 coreNum * tileNum, 则减少tileNum
            tileNum = (dimLength / blockNum) / cacheLineNum * cacheLineNum;
        }

        // 重新计算分核参数
        uint64_t ubOuter = (dimLength + tileNum - 1) / tileNum;
        uint64_t blockFormer = (ubOuter + coreNum - 1) / coreNum;
        // 将核数再减少一半，减少核启动开销，并开启核内doublebuffer流水
        if (blockFormer == 1) {
            blockFormer = DB_LOOP;
        }
        blockNum = (ubOuter + blockFormer - 1) / blockFormer;

        brcOneDimTilingDataAdvance->tileNum = tileNum;
        brcOneDimTilingDataAdvance->blockNum = blockNum;
        brcOneDimTilingDataAdvance->dimLen = dimLength;

        OP_LOGI(context_->GetNodeName(),
                "Broadcast one dim advance do tiling finish. coreNum: %ld, CACHE_LINE: %ld, "
                "tileNum: %d, blockNum: %lu, dimLength: %ld, scalarFlag: %d",
                coreNum, CACHE_LINE_512, tileNum, blockNum, dimLength, brcOneDimTilingDataAdvance->scalarFlag);

        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus DoOneDimOpTilingAdvance()
    {
        brcOneDimTilingDataAdvance = context_->GetTilingData<BroadcastOneDimTilingDataAdvance>();

        OP_CHECK_IF((brcOneDimTilingDataAdvance == nullptr),
                    OP_LOGE(context_->GetNodeName(), "Get brcOneDimTilingDataAdvance from GE context failed"),
                    return ge::GRAPH_FAILED);

        int32_t aliveNum = OpDag::template GetBufferNum<true, true>();
        aliveNum = aliveNum + tmpExtraBufferNum;
        int64_t actualUbSize = ubSize - tmpExtraSize;

        OP_CHECK_IF((GetOneDimScalarFlagAdvance() == ge::GRAPH_FAILED),
                    OP_LOGE(context_->GetNodeName(), "GetOneDimScalarFlag failed"), return ge::GRAPH_FAILED);

        OP_CHECK_IF((GetOneDimUBFormerAdvance(aliveNum, actualUbSize) == ge::GRAPH_FAILED),
                    OP_LOGE(context_->GetNodeName(), "GetUBFormer failed"), return ge::GRAPH_FAILED);

        OP_CHECK_IF((DoOneDimFormerTilingAdvance() == ge::GRAPH_FAILED),
                    OP_LOGE(context_->GetNodeName(), "DoOneDimFormerTiling failed"), return ge::GRAPH_FAILED);

        schMode = SCH_MODE_ONE_DIM_ADVANCE;
        context_->SetBlockDim(brcOneDimTilingDataAdvance->blockNum);

        return ge::GRAPH_SUCCESS;
    }

    gert::TilingContext* context_{nullptr};
    uint64_t coreNum{0};
    uint64_t ubSize{0};
    std::vector<gert::Shape> inShapes;
    std::vector<bool> inputIsContiguous;
    std::vector<gert::Stride> inStrides;
    bool inputAllContiguous = true;
    gert::Shape outShape;
    gert::Stride outStride;
    BroadcastTilingData tilingData;

    uint64_t schMode;
    uint32_t offset = 0;
    char scalarData[BROADCAST_MAX_SCALAR_BYTES] = {0};

    BroadcastBaseTilingData<OpDag>* brcBaseTilingData = nullptr;
    BroadcastOneDimTilingData* brcOneDimTilingData = nullptr;
    BroadcastOneDimTilingDataAdvance* brcOneDimTilingDataAdvance = nullptr;
    BroadcastLastTransposeTilingData<OpDag>* brcLastTransposeTilingData = nullptr;
    BroadcastNlastTransposeTilingData<OpDag>* brcNLastTransposeTilingData = nullptr;

    // kernelType default both:0  nddma:1  ub_brc: 2
    uint32_t kernelType_ = 0;
    int64_t tmpExtraSize = 0;
    int64_t tmpExtraBufferNum = 0;
    std::set<int64_t> copyInBrcPos;
    std::set<ge::DataType> ubBroadcastDtypes = {ge::DT_INT8, ge::DT_UINT8, ge::DT_FLOAT16,
                                                ge::DT_BF16, ge::DT_INT16, ge::DT_UINT16};
};

} // namespace Base
} // namespace Ops
#endif // BROADCAST_TILING_H_
