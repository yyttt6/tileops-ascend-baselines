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
 * \file broadcast_tiling_base.h
 * \brief atvoss broadcast template tiling
 */
#ifndef BROADCAST_TILING_BASE_H_
#define BROADCAST_TILING_BASE_H_

#include <vector>
#include "broadcast_base_struct.h"
#include "op_common/log/log.h"
#include "exe_graph/runtime/tiling_context.h"
#include "tiling/platform/platform_ascendc.h"
#include "op_common/op_host/util/platform_util.h"
#include "op_common/op_host/util/opbase_export.h"

namespace Ops {
namespace Base {
constexpr uint64_t BROADCAST_OP_KEY_OFFSET = 100000000000000;
constexpr uint64_t BROADCAST_MAX_DIMS = 8;
constexpr int SCH_MODE_UB_BROADCAST = 100;
constexpr int SCH_MODE_NDDMA_WITHOUT_LOOP_CACHE_BRC = 1;
constexpr int SCH_MODE_NDDMA_WITH_LOOP_CACHE_BRC = 2;
constexpr int SCH_MODE_ONE_DIM = 201;
constexpr int SCH_MODE_ONE_DIM_ADVANCE = 202;
constexpr int SCH_MODE_NONCONTIGUOUS_NLAST_TRANSPOSE_NDDMA = 301;
constexpr int SCH_MODE_NONCONTIGUOUS_NLAST_TRANSPOSE_UB_BRC = 302;
constexpr int SCH_MODE_NONCONTIGUOUS_LAST_TRANSPOSE = 303;
constexpr int SCH_MODE_NON_CONTIGUOUS_NDDMA_WITHOUT_LOOP_CACHE_BRC = 304;
constexpr int SCH_MODE_NON_CONTIGUOUS_NDDMA_WITH_LOOP_CACHE_BRC = 305;
constexpr int ONE_DIM_INPUT_IS_NOT_SCALAR = 0;
constexpr int ONE_DIM_INPUT_IS_SCALAR = 1;
constexpr int MAX_NNDMA_DIM = 5;
constexpr int32_t HIGH_PERF_RANK = 4;
constexpr int32_t DEFAULT_ADD_VALUE = 9;
constexpr int64_t VALUE_TWO = 2;
constexpr int64_t MAX_IN_OUT_ARGS_NUMBER = 4;
constexpr uint64_t DEFAULT_TILING_WORKSPACE_SIZE = 16 * 1024 * 1024;
constexpr int64_t PER_CORE_MIN_UB_BYTE = 8 * 1024;
constexpr uint64_t HALF_CORE_NUM_DIVIDE = 2;
constexpr int64_t CACHE_LINE = 128;
constexpr int64_t CACHE_LINE_512 = 512;
constexpr int64_t DB_LOOP = 2;
constexpr uint64_t DIM_TWO = 2;
constexpr uint64_t DIM_THREE = 3;
constexpr int64_t SINGLE_CORE_SIZE_LIMIT = 8 * 1024;

static constexpr uint32_t BLOCK_LENGTH = 32;
static constexpr uint64_t BROADCAST_COMPUTE_KEY_OFFSET = 1000000;

enum class BROADCAST_BITS_SIZE { BITS1_SIZE = 1, BITS8_SIZE = 8, BITS16_SIZE = 16, BITS32_SIZE = 32, BITS64_SIZE = 64 };
enum class BROADCAST_KERNEL_TYPE : uint32_t {
    KERNEL_TYPE_BOTH = 0,
    KERNEL_TYPE_NDDMA = 1,
    KERNEL_TYPE_UB_BROADCAST = 2,
};
/**
 * @brief 计算图中用于做tiling计算的信息
 *
 * - maxDtypeBits  单位bit 计算图中的最大Dtype bit数，用于计算切分块大小
 * - minDtypeBits  单位bit 计算图中的最小Dtype bit数，用于计算对齐点
 * - extraSize     单位byte 计算图需要的额外空间大小
 * - bufferDivisor 存活节点大小
 */
struct BroadcastComputeParams {
    int64_t maxDtypeBits;
    int64_t minDtypeBits;
    std::vector<int64_t> extraSize;
    std::vector<int64_t> bufferDivisor;
};

/**
 *  @brief tiling切分参数
 * - coreNum 核数大小
 * - inShape 输入shape大小
 * - outShape 输出shape大小
 * - ubSize ub空间大小
 * - computeMap 不同数据类型对应的compute参数
 */
struct BroadcastTilingParams {
    int64_t coreNum;
    std::vector<gert::Shape> inShape;
    std::vector<gert::Stride> inStride;
    gert::Shape outShape;
    int64_t ubSize;
    bool preferMultiCore = false;
    bool inputAllContiguous = true;
    std::map<uint64_t, BroadcastComputeParams> computeMap;
};

/**
 *  @brief 编译参数信息
 * - dslCompileInfo dsl分支场景的compileInfo信息
 * - isAscendC 是否走Acendc分支标记
 * - coreNum 核数大小
 * - ubSize ub空间大小
 */
struct BroadcastCompileInfo {
    // std::shared_ptr<AutoTilingCompileInfo> dslCompileInfo;
    bool isAscendC{false};
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
};

/**
 *  @brief 公共切分tiling结构体
 * - dims 输入shape信息
 * - strides 输入shape轴的stride信息
 * - outputDims 输出dim大小
 * - innerKey 工具化场景下的key值
 * - shapeLen 输出shape长度
 * - ubSplitAxis ub切分轴
 * - ubFormer ub切分大小
 * - blockFormer 多核切分大小
 * - ubOuter ub切分外循环轴
 * - ubTail ub切分尾块大小
 * - blockNum 多核切分大小
 * - blockTail 多核切分尾块大小
 * - dimProductBeforeUbInner ub切分外所有轴乘积，计算偏移用
 * - elemNum 存活空间大小
 */
struct BroadcastTilingData {
    std::vector<std::vector<int64_t>> dims;
    std::vector<std::vector<int64_t>> strides;
    std::vector<int64_t> outputDims;
    std::vector<int64_t> outputStrides;
    uint64_t innerKey;
    int64_t shapeLen;
    int64_t ubSplitAxis;
    int64_t ubFormer;
    int64_t blockFormer;
    int64_t ubOuter;
    int64_t ubTail;
    int64_t ubFormerLastAxis;
    int64_t ubOuterLastAxis;
    int64_t ubTailLastAxis;
    int64_t minDtypeBlockAlignSize;
    int64_t blockNum;
    int64_t blockTail;
    int64_t dimProductBeforeUbInner;
    int64_t elemNum;
};

struct ubSplitInfo {
    int64_t ubSplitAxis = 0;
    int64_t ubFormer = 0;
    int64_t ubOuter = 0;
    int64_t ubTail = 0;
    int64_t ubFormerLastAxis = 0;
    int64_t ubOuterLastAxis = 0;
    int64_t ubTailLastAxis = 0;
};

OPBASE_API void BrcPrintShape(const gert::Shape& shape, const std::string& prefix);

OPBASE_API void BrcPrintShapes(const std::vector<gert::Shape>& shapes, const std::string& prefix);

OPBASE_API void BrcPrintStride(const gert::Stride& stride, const std::string& prefix);

OPBASE_API void BrcPrintStrides(const std::vector<gert::Stride>& strides, const std::string& prefix);

OPBASE_API void BrcPrintVector(const std::vector<int64_t>& dims, const std::string& prefix);

OPBASE_API void BrcPrintVectors(const std::vector<std::vector<int64_t>>& allDims, const std::string& prefix);

uint64_t BroadcastGetComputeKey();
uint64_t BroadcastGetScheduleKey(uint32_t axisInsideUB);
int64_t BroadcastGetMaxElemNum(int64_t ubSize, const BroadcastComputeParams& computeParams);

} // namespace Base
} // namespace Ops
#endif // BROADCAST_TILING_BASE_H_
