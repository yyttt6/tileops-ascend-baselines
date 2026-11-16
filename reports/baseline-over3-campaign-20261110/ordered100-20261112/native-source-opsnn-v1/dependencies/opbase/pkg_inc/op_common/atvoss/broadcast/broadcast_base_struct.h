/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BROADCAST_BASE_STRUCT_H_
#define BROADCAST_BASE_STRUCT_H_

/*!
 * \file broadcast_base_struct.h
 * \brief
 */
#include <cstdint>
#include "ascendc/host_api/tiling/template_argument.h"

namespace Ops {
namespace Base {
constexpr uint64_t BROADCAST_MAX_DIMS_NUM = 8;
constexpr uint64_t BROADCAST_NON_CONTIGIOUS_MAX_DIMS_NUM = 4;
constexpr int64_t BROADCAST_BITS_NUM = 8;
constexpr uint64_t BROADCAST_MAX_COPYIN_NUM = 8;
constexpr uint64_t BROADCAST_MAX_COPYINBRC_NUM = 8;
constexpr uint64_t BROADCAST_MAX_VECBRC_NUM = 8;
constexpr uint64_t BROADCAST_MAX_SCALAR_BYTES = 16;
constexpr uint64_t BROADCAST_MAX_TILINGDATA_BYTES = 4096;

struct BroadcastFakeTilingData {
    char scalarData[BROADCAST_MAX_TILINGDATA_BYTES];
};

struct BroadcastOneDimTilingData {
    int32_t scalarFlag;
    int32_t ubSplitAxis;
    int32_t ubFormer;
    int32_t ubTail;
    int32_t blockNum;
    int64_t blockFormer;
    int64_t blockTail;

    char scalarData[BROADCAST_MAX_SCALAR_BYTES];
};

struct BroadcastOneDimTilingDataAdvance {
    int64_t dimLen;
    int32_t tileNum;
    int16_t blockNum;
    int16_t scalarFlag;
};

template <class BrcDag>
struct BroadcastBaseTilingData {
    int32_t scheMode;
    int32_t shapeLen;
    int32_t ubSplitAxis;
    int32_t ubFormer;
    int32_t ubTail;
    int64_t ubOuter;
    int64_t blockFormer;
    int64_t blockTail;
    int64_t dimProductBeforeUbInner;
    int64_t elemNum;
    int64_t blockNum;

    int64_t outputDims[BROADCAST_MAX_DIMS_NUM];
    int64_t outputStrides[BROADCAST_MAX_DIMS_NUM];
    int64_t inputDims[BrcDag::InputSize][2]; // 整块 + 尾块
    int64_t inputBrcDims[BrcDag::CopyBrcSize][BROADCAST_MAX_DIMS_NUM];
    int64_t inputVecBrcDims[BrcDag::VecBrcSize][BROADCAST_MAX_DIMS_NUM];
    int64_t inputStrides[BrcDag::InputSize][BROADCAST_MAX_DIMS_NUM];
    int64_t inputBrcStrides[BrcDag::CopyBrcSize][BROADCAST_MAX_DIMS_NUM];
    int64_t inputVecBrcStrides[BrcDag::VecBrcSize];

    char scalarData[BROADCAST_MAX_SCALAR_BYTES];
};

template <class BrcDag>
struct BroadcastLastTransposeTilingData {
    int32_t scheMode;
    int32_t shapeLen;
    int32_t ubSplitAxis;
    int32_t ubFormer;
    int32_t ubTail;
    int32_t ubFormerLastAxis;
    int32_t ubTailLastAxis;
    int32_t minDtypeBlockAlignSize;
    int64_t ubOuter;
    int64_t ubOuterLastAxis;
    int64_t blockFormer;
    int64_t blockTail;
    int64_t dimProductBeforeUbInner;
    int64_t elemNum;
    int64_t blockNum;

    int64_t outputDims[BROADCAST_NON_CONTIGIOUS_MAX_DIMS_NUM];
    int64_t outputStrides[BROADCAST_NON_CONTIGIOUS_MAX_DIMS_NUM];
    int64_t outputStridesWithPad[BROADCAST_NON_CONTIGIOUS_MAX_DIMS_NUM];
    int64_t inputDims[BrcDag::InputSize][BROADCAST_NON_CONTIGIOUS_MAX_DIMS_NUM];
    int64_t inputBrcDims[BrcDag::CopyBrcSize][BROADCAST_NON_CONTIGIOUS_MAX_DIMS_NUM];
    int64_t inputVecBrcDims[BrcDag::VecBrcSize][BROADCAST_NON_CONTIGIOUS_MAX_DIMS_NUM];
    int64_t inputStrides[BrcDag::InputSize][BROADCAST_NON_CONTIGIOUS_MAX_DIMS_NUM];
    int64_t inputBrcStrides[BrcDag::CopyBrcSize][BROADCAST_NON_CONTIGIOUS_MAX_DIMS_NUM];
    int64_t inputVecBrcStrides[BrcDag::VecBrcSize];

    char scalarData[BROADCAST_MAX_SCALAR_BYTES];
};

template <class BrcDag>
struct BroadcastNlastTransposeTilingData {
    int32_t scheMode;
    int32_t shapeLen;
    int32_t ubSplitAxis;
    int32_t ubFormer;
    int32_t ubTail;
    int64_t ubOuter;
    int64_t blockFormer;
    int64_t blockTail;
    int64_t dimProductBeforeUbInner;
    int64_t elemNum;
    int64_t blockNum;
    int32_t minDtypeBlockAlignSize;

    int64_t outputDims[BROADCAST_NON_CONTIGIOUS_MAX_DIMS_NUM];
    int64_t outputStrides[BROADCAST_NON_CONTIGIOUS_MAX_DIMS_NUM];
    int64_t outputStridesWithPad[BROADCAST_NON_CONTIGIOUS_MAX_DIMS_NUM];
    int64_t inputDims[BrcDag::InputSize][BROADCAST_NON_CONTIGIOUS_MAX_DIMS_NUM];
    int64_t inputBrcDims[BrcDag::CopyBrcSize][BROADCAST_NON_CONTIGIOUS_MAX_DIMS_NUM];
    int64_t inputVecBrcDims[BrcDag::VecBrcSize][BROADCAST_NON_CONTIGIOUS_MAX_DIMS_NUM];
    int64_t inputStrides[BrcDag::InputSize][BROADCAST_NON_CONTIGIOUS_MAX_DIMS_NUM];
    int64_t inputBrcStrides[BrcDag::CopyBrcSize][BROADCAST_NON_CONTIGIOUS_MAX_DIMS_NUM];
    int64_t inputVecBrcStrides[BrcDag::VecBrcSize];

    char scalarData[BROADCAST_MAX_SCALAR_BYTES];
};

#define BRC_NDDMA_SCH_MODE_KEY_DECL(schMode) \
    ASCENDC_TPL_UINT_DECL(schMode, 16, ASCENDC_TPL_UI_LIST, 1, 2, 201, 202, 301, 303, 304, 305, 999)
#define BRC_NDDMA_SCH_MODE_KEY_SEL(schMode) \
    ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, 1, 2, 201, 202, 301, 303, 304, 305)
#define BRC_NDDMA_SCH_CUSTOM_MODE_KEY_SEL(schMode) ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, 999)
#define BRC_VEC_SCH_CUSTOM_MODE_KEY_DECL(schMode)                                                                  \
    ASCENDC_TPL_UINT_DECL(schMode, 16, ASCENDC_TPL_UI_LIST, 101, 102, 103, 104, 109, 201, 202, 301, 302, 303, 304, \
                          305, 999)
#define BRC_VEC_SCH_MODE_KEY_SEL(schMode) \
    ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, 101, 102, 103, 104, 109, 201, 202, 301, 302, 303, 304, 305)
#define BRC_VEC_SCH_CUSTOM_MODE_KEY_SEL(schMode) ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, 999)
#define BRC_TEMP_SCH_MODE_KEY_DECL(schMode)                                                                         \
    ASCENDC_TPL_UINT_DECL(schMode, 16, ASCENDC_TPL_UI_LIST, 1, 2, 101, 102, 103, 104, 109, 201, 202, 301, 302, 303, \
                          304, 305, 999)
#define BRC_TEMP_SCH_MODE_KEY_SEL(schMode) \
    ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, 1, 2, 101, 102, 103, 104, 109, 201, 202, 301, 302, 303, 304, 305)
#define BRC_TEMP_SCH_CUSTOM_MODE_KEY_SEL(schMode) ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, 999)
} // namespace Base
} // namespace Ops
#endif // BROADCAST_BASE_STRUCT_H_
