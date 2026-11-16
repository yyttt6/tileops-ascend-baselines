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
 * \file reduce_tiling.h
 * \brief tiling for reduce up
 */

#ifndef _ATVOSS_REDUCE_TILING_H_
#define _ATVOSS_REDUCE_TILING_H_

#include <memory>
#include "exe_graph/runtime/tiling_context.h"
#include "ascendc/host_api/tiling/template_argument.h"
#include "op_common/atvoss/util/dag.h"
#include "op_common/atvoss/reduce/reduce_tiling_data.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "op_common/op_host/util/opbase_export.h"
#include "register/tilingdata_base.h"

namespace Ops {
namespace Base {
using namespace ReduceOpTmpl;
using namespace optiling;

constexpr uint64_t WORKSPACE_SIZE = 16 * 1024 * 1024; // fixed workspace size for ascendc
constexpr uint64_t BASIC_BLOCK = 64 * 1024UL;
constexpr uint64_t POST_BUF_SIZE = 8 * 1024UL;        // post reduce size for ub reduce
constexpr uint64_t CACHE_BUF_SIZE = 16 * 1024UL;      // cache for binary reduce

struct ReduceTilingUnit {
    int32_t idx = -1;   // ub cut axis
    uint64_t inner = 1; // record the cache line aSize's num or rSize's num in ub
    uint64_t outer = 1; // outer size of ub
    uint64_t step = 1;  // step of cacheline
};

struct CacheLineBlock {
    int32_t axis = -1;           // cacheline cut axis
    uint64_t cacheLineStep = 1;  // step len in cacheline axis
    uint64_t cacheLineOuter = 1; // relative to cacheLineStep, out size of cacheline cut axis
    uint64_t aSize = 1;          // A axis size in cacheline
    uint64_t rSize = 1;          // R axis size in cacheline
};

struct ReduceOpCompileInfo {
    uint64_t vectorCoreNum = 0;
    uint64_t ubSize = 0;
    uint64_t cacheLineSize = 0;
    uint64_t ubBlockSize = 0;
    uint64_t vRegSize = 0;
};

struct ReduceOpDagParam {
    uint64_t maxInputBytes = 0; // dag图上节点最大字节
    uint64_t minInputBytes = 0; // dag图上节点最小字节
    uint64_t preInput = 1;      // dag图reduce 前的输入个数
    uint64_t preOutput = 0;     // dag图reduce 前的输出个数
    uint64_t preTempCalc = 0;   // dag图reduce 前的临时计算节点个数
    uint64_t postInput = 0;     // dag图reduce 后的输入个数
    uint64_t postOutput = 1;    // dag图reduce 后的输出个数
    uint64_t postTempCalc = 0;  // dag图reduce 后的临时计算节点个数
    int32_t reduceOpPos = 0;    // dag图reduce节点索引
    MemLevel memLevel = MemLevel::LEVEL_2;
};

struct ReduceOpInputParam {
    uint32_t reservedSize = 0;
    uint32_t reservedNode = 0;
    bool isTailAOne = 0;
    ge::DataType inputDtype = ge::DT_UNDEFINED;
    ge::DataType promoteDtpye = ge::DT_UNDEFINED;
    std::vector<int64_t> axes;
    std::vector<int64_t> shape;
    std::vector<int64_t> dimStrides;
};

struct ReduceTilingKey {
    uint32_t patternID = 0;
    uint32_t loopARCount = 0;
    uint32_t loopInnerARCount = 0;
    bool isContiguous = true;
    bool batchInvariant = false;
};

namespace ReduceOpTmpl {
template <typename T>
ge::graphStatus GetConstInputData(gert::TilingContext* context, int32_t idx, std::vector<int64_t>& axes)
{
    auto axesInput = context->GetInputTensor(idx);
    OP_CHECK_NULL_WITH_CONTEXT(context, axesInput);
    auto size = axesInput->GetShapeSize();
    OP_CHECK_IF((size >= static_cast<int64_t>(MAX_DIM)),
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context->GetNodeName(), "axes", std::to_string(size).c_str(),
                                                          "axes dim size is over max dim, cannot support"),
                return ge::GRAPH_FAILED);

    if (size == 0) {
        return ge::GRAPH_SUCCESS;
    }
    axes.resize(size);
    auto axesData = axesInput->GetData<T>();
    OP_CHECK_NULL_WITH_CONTEXT(context, axesData);
    for (int64_t i = 0; i < size; i++) {
        axes[i] = static_cast<int64_t>(axesData[i]);
    }
    return ge::GRAPH_SUCCESS;
}

template <typename T>
std::string VectorToString(const std::vector<T>& s)
{
    std::stringstream ss;
    for (auto iter = s.begin(); iter != s.end(); ++iter) {
        ss << *iter;
        if (iter != s.end() - 1) {
            ss << ", ";
        }
    }
    return ss.str();
}

template <typename T, typename U>
std::string VectorToString(const T* s, U size)
{
    std::stringstream ss;
    for (U i = 0; i < size; i++) {
        ss << s[i];
        if (i != size - 1) {
            ss << ", ";
        }
    }
    return ss.str();
}

/*
 * \brief get input storage shape with input idx
 * @param context
 *  ge tiling context
 *
 * @param idx
 *  reduce operator input idx
 *
 * @param shape
 *  return value, operator input shape
 */
OPBASE_API ge::graphStatus GetInputStorageShape(gert::TilingContext* context, int32_t idx, std::vector<int64_t>& shape);

/*
 * \brief get input view shape with input idx
 * @param context
 *  ge tiling context
 *
 * @param idx
 *  reduce operator input idx
 *
 * @param shape
 *  return value, operator input shape
 */
OPBASE_API ge::graphStatus GetInputShape(gert::TilingContext* context, int32_t idx, std::vector<int64_t>& shape);

/*
 * \brief get input stride with input idx
 * @param context
 *  ge tiling context
 *
 * @param idx
 *  reduce operator input idx
 *
 * @param dimStrides
 *  return value, operator input dimStrides
 */
OPBASE_API ge::graphStatus GetInputStride(const gert::TilingContext* context, int32_t idx,
                                          std::vector<int64_t>& dimStrides);

/*
 * \brief get input dtype with input idx
 * @param context
 *  ge tiling context
 *
 * @param idx
 *  reduce operator input idx
 *
 * @param dtype
 *  return value, operator input dtype
 */
OPBASE_API ge::graphStatus GetInputDtype(gert::TilingContext* context, int32_t idx, ge::DataType& dtype);

/*
 * \brief get operator input param with input idx, if axes idx is not constant input, use this interface
 * @param context
 *  ge tiling context
 *
 * @param opInput
 *  return value, operator input param
 *
 * @param inputIdx
 *  reduce operator input idx
 */
OPBASE_API ge::graphStatus GetInputParam(gert::TilingContext* context, ReduceOpInputParam& opInput, int32_t inputIdx);

/*
 * \brief get operator input param with input idx, if axes idx is not constant input and support batchinvariant, use
 * this interface
 * @param context
 *  ge tiling context
 *
 * @param opInput
 *  return value, operator input param
 *
 * @param inputIdx
 *  reduce operator input idx
 */
OPBASE_API ge::graphStatus GetInputParam(gert::TilingContext* context, ReduceOpInputParam& opInput, int32_t inputIdx,
                                         const ReduceTilingKey& key);

/*
 * \brief get operator input param with input idx, if axes idx is constant input, use this interface
 * @param context
 *  ge tiling context
 *
 * @param opInput
 *  return value, operator input param
 *
 * @param inputIdx
 *  reduce operator input idx
 *
 * @param axesIdx
 *  reduce operator axes input idx
 *
 * @param outIdx
 *  reduce operator reduce output idx
 */
OPBASE_API ge::graphStatus GetInputParam(gert::TilingContext* context, ReduceOpInputParam& opInput, int32_t inputIdx,
                                         int32_t axesIdx, int32_t outIdx);
/*
 * \brief get operator input param with input idx, if axes idx is constant input and support batchinvariant, use this
 * interface
 * @param context
 *  ge tiling context
 *
 * @param opInput
 *  return value, operator input param
 *
 * @param inputIdx
 *  reduce operator input idx
 *
 * @param axesIdx
 *  reduce operator axes input idx
 *
 * @param outIdx
 *  reduce operator reduce output idx
 * @param key
 *  reduce operator tilingkey
 */
OPBASE_API ge::graphStatus GetInputParam(gert::TilingContext* context, ReduceOpInputParam& opInput, int32_t inputIdx,
                                         int32_t axesIdx, int32_t outIdx, const ReduceTilingKey& key);
} // namespace ReduceOpTmpl

class OPBASE_API ReduceOpTiling {
public:
    /*
     * @param context
     *  ge tiling context
     *
     * @param compileInfo
     *  reduce op compileInfo struct, if nullptr, template will get from tiling context
     *
     * @param tilingData
     *  reduec op tilingData struct, if nullptr, template will get from tiling context
     */
    ReduceOpTiling(gert::TilingContext* context, ReduceOpCompileInfo* compileInfo = nullptr,
                   ReduceOpTilingData* tilingData = nullptr)
        : context_(context), compileInfo_(compileInfo), tilingData_(tilingData){};

    virtual ~ReduceOpTiling() {}
    /*
     * \brief reduce template do tiling with input shape and axis
     *
     * @param opInput
     *  operator input, aquired from GetInputParam interface
     *
     * @param key
     *  return value, reduce template tiling key
     */
    template <class OpDag = void>
    ge::graphStatus DoTiling(ReduceOpInputParam& opInput, ReduceTilingKey& key)
    {
        OP_CHECK_IF((ParamCheck(opInput) != ge::GRAPH_SUCCESS),
                    OP_LOGE(context_->GetNodeName(), "Do tiling param check failed"), return ge::GRAPH_FAILED);

        PreProcessInput<OpDag>(opInput);

        OP_CHECK_IF((PreProcessOptionalParam() != ge::GRAPH_SUCCESS),
                    OP_LOGE(context_->GetNodeName(), "Do tiling preprocess optional param failed"),
                    return ge::GRAPH_FAILED);

        DoReduceTiling(key);

        return ge::GRAPH_SUCCESS;
    }

protected:
    virtual void CalcUserBasicBlock([[maybe_unused]] bool patternA){};

    virtual void CalcUserWorkSpace();

protected:
    template <class OpDag = void>
    void PreProcessInput(ReduceOpInputParam& opInput)
    {
        if constexpr (!std::is_same<OpDag, void>::value) {
            opDag_.preInput = OpDag::template GetMte2NumImpl<typename OpDag::PreReduceNodeInfo, true, false>();
            opDag_.preOutput = OpDag::template GetMte3NumImpl<typename OpDag::PreReduceNodeInfo, true, false>();
            opDag_.preTempCalc = OpDag::template GetTempBufNumImpl<typename OpDag::PreReduceNodeInfo, true, false>();
            opDag_.postInput = OpDag::template GetMte2NumImpl<typename OpDag::PostReduceNodeInfo, true, false>();
            opDag_.postOutput = OpDag::template GetMte3NumImpl<typename OpDag::PostReduceNodeInfo, true, false>();
            opDag_.postTempCalc = OpDag::template GetTempBufNumImpl<typename OpDag::PostReduceNodeInfo, true, false>();
            opDag_.maxInputBytes = OpDag::MaxDtypeBytes;
            opDag_.minInputBytes = OpDag::MinDtypeBytes;
            opDag_.reduceOpPos = OpDag::ReduceOpPos;
            opDag_.memLevel = OpDag::BufLevel;
        }
        OP_LOGI(
            context_,
            "PreDAG Input:%ld, Output:%ld, TempCalc:%ld, PostDAG Input:%ld, Output:%ld, TempCalc:%ld, maxBtypes:%ld, "
            "minBtypes:%ld",
            opDag_.preInput, opDag_.preOutput, opDag_.preTempCalc, opDag_.postInput, opDag_.postOutput,
            opDag_.postTempCalc, opDag_.maxInputBytes, opDag_.minInputBytes);

        opInput_ = opInput;
        if (opDag_.maxInputBytes == 0UL && opInput_.promoteDtpye != ge::DT_UNDEFINED) {
            opDag_.maxInputBytes = ge::GetSizeByDataType(opInput_.promoteDtpye);
        }
    }

    ge::graphStatus PreProcessOptionalParam();

    template <class Pattern>
    bool IsAxisA(int32_t idx);

    uint64_t Ratio();

    ge::graphStatus ParamCheck(ReduceOpInputParam& opInput);

    ge::graphStatus AxesCheck(const std::vector<int64_t>& shape, const std::vector<int64_t>& axes);

    void EliminateOne(uint64_t* viewShape, int64_t* viewStride, int32_t& shapeSize);

    void PadDimForNonContiguous(uint64_t* viewShape, int64_t* viewStride, int32_t& shapeSize);

    void MergeAxis(uint64_t* viewShape, int64_t* viewStride, int32_t& shapeSize);

    void DoReduceTiling(ReduceTilingKey& key);

    void TransformShape(uint64_t* shape, int64_t* newStride, int32_t& shapeSize);

    template <class Pattern>
    void PadDimOne(uint64_t* shape);

    template <class Pattern>
    uint64_t CaculateReduceSize(const uint64_t* shape);

    template <class Pattern>
    ge::graphStatus CalcBasicBlock(const uint64_t* shape);

    template <class Pattern>
    uint64_t TryGetReduceBlock(const uint64_t* shape, uint64_t preInputBufferNum, uint64_t preRestBufferNum,
                               uint64_t postBufferNum);

    template <class Pattern>
    bool IsEmtpyTensor(const uint64_t* shape);

    template <class Pattern>
    void InitUnit(const uint64_t* shape);

    template <class Pattern>
    void ComputeCacheLineBlock(const uint64_t* shape);

    template <class Pattern>
    void ComputeCacheLineBlockAndUnit(const uint64_t* shape);

    template <class Pattern>
    void ComputeUnitA(const uint64_t* shape);

    template <class Pattern>
    void ComputeUnitR(const uint64_t* shape);

    template <class Pattern>
    void ComputeProgressUnitA(const uint64_t* shape);

    template <class Pattern>
    void ComputeRFirst(const uint64_t* shape);

    template <class Pattern>
    void ComputeAWithRFullLoad(const uint64_t* shape);

    template <class Pattern>
    void ComputeAWithRCannotFullLoad(const uint64_t* shape);

    template <class Pattern>
    void SetTilingDataBatchInvariant(const uint64_t* shape);

    template <class Pattern>
    ge::graphStatus ComputeEmptyTiling(uint64_t* shape);

    template <class Pattern>
    ge::graphStatus ComputeTiling(uint64_t* shape);

    template <class Pattern>
    void ComputeStride(const uint64_t* shape);

    template <class Pattern>
    int32_t IsUseNddma(const uint64_t* shape);

    template <class Pattern>
    void SetTilingData(const uint64_t* shape);

    ge::graphStatus DoTilingMatchPattern(uint64_t* shape, int32_t shapeSize);

    template <class Pattern>
    void SetTilingKey();

    void PrintTilingData();

    void GetTilingKey(ReduceTilingKey& key);

    bool isPrime(uint64_t num) const;

protected:
    gert::TilingContext* context_{nullptr};
    ReduceOpCompileInfo* compileInfo_{nullptr};
    std::unique_ptr<ReduceOpCompileInfo> compileInfoPtr_;
    ReduceOpTilingData* tilingData_{nullptr};
    uint64_t basicBlock_{0};  // 算子搬入的buffer大小
    uint64_t resultBlock_{0}; // reduce计算后的buffer大小
    int32_t dimNum_{0};
    bool needReserveA_ = 0;
    size_t workSpaceSize_{0};
    CacheLineBlock cBlock_;
    ReduceTilingUnit unitA_;
    ReduceTilingUnit unitR_;
    ReduceTilingKey tilingKey_;
    ReduceOpInputParam opInput_;
    ReduceOpDagParam opDag_;
    int64_t viewStride_[MAX_DIM] = {0};   // 每个轴slice切片的个数
    uint64_t sliceNum_[MAX_DIM] = {0};    // 每个轴slice切片的个数
    uint64_t sliceShape_[MAX_DIM] = {0};  // 每个轴slice切片后的shape大小
    uint64_t sliceStride_[MAX_DIM] = {0}; // 每个轴slice切片后的stride长度
};                                        // class ReduceOpTiling

/*
 * \brief reduce template tiling interface
 *
 * @tparam OpDag
 *  operator compute graph
 */
template <class OpDag = void>
OPBASE_API ge::graphStatus Tiling4ReduceOp(gert::TilingContext* context, ReduceOpInputParam& opInput,
                                           ReduceTilingKey& key)
{
    ReduceOpTiling tiling(context, nullptr, nullptr);
    return tiling.DoTiling<OpDag>(opInput, key);
}

template <class OpDag = void>
OPBASE_API ge::graphStatus Tiling4ReduceOp(gert::TilingContext* context, ReduceOpInputParam& opInput,
                                           ReduceTilingKey& key, const ReduceOpCompileInfo* compileInfo)
{
    (void)compileInfo;
    ReduceOpTiling tiling(context, nullptr, nullptr);
    return tiling.DoTiling<OpDag>(opInput, key);
}

template <class OpDag = void>
OPBASE_API ge::graphStatus Tiling4ReduceOp(gert::TilingContext* context, ReduceOpInputParam& opInput,
                                           ReduceTilingKey& key, ReduceOpTilingData* tilingData)
{
    ReduceOpTiling tiling(context, nullptr, tilingData);
    return tiling.DoTiling<OpDag>(opInput, key);
}

template <class OpDag = void>
OPBASE_API ge::graphStatus Tiling4ReduceOp(gert::TilingContext* context, ReduceOpInputParam& opInput,
                                           ReduceTilingKey& key, const ReduceOpCompileInfo* compileInfo,
                                           ReduceOpTilingData* tilingData)
{
    (void)compileInfo;
    ReduceOpTiling tiling(context, nullptr, tilingData);
    return tiling.DoTiling<OpDag>(opInput, key);
}
} // namespace Base
} // namespace Ops

#define GEN_REDUCE_TILING_KEY(result, reduceTilingKey, ...)                                   \
    result = GET_TPL_TILING_KEY(reduceTilingKey.isContiguous, reduceTilingKey.batchInvariant, \
                                reduceTilingKey.patternID, reduceTilingKey.loopARCount,       \
                                reduceTilingKey.loopInnerARCount, __VA_ARGS__)
#endif