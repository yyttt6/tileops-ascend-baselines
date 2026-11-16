/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INDV_CACHE_KEY_BUILDER_H_
#define INDV_CACHE_KEY_BUILDER_H_
#include "indv_executor.h"
#include "utils/indv_types.h"
#include "graph/types.h"
namespace Indv {
class CacheKeyBuilder {
public:
    static void AppendTensor(NnopbaseExecutorArgs* args, const aclTensor* tensor);
    static void AppendTensorList(NnopbaseExecutorArgs* args, const aclTensorList* tensorList);
    static NnopbaseUChar* AppendPlaceHolder(NnopbaseExecutorArgs* args, NnopbaseUChar* key);
    static NnopbaseUChar* AppendCoreNum(NnopbaseExecutorArgs* args, const NnopbaseCoreNum* coreNum);
    static NnopbaseUChar* AppendMc2RankId(NnopbaseExecutorArgs* args, const uint32_t* rankId);
    static NnopbaseUChar* AppendDeterministicLevel(NnopbaseExecutorArgs* args, const uint8_t* deterministicLevel);
    static void AppendValueDependTensor(NnopbaseExecutorArgs* args, const void* addr, const uint64_t dim,
                                        const uint64_t dataLen, ge::DataType dType);
    static void AppendScalar(NnopbaseExecutorArgs* args, const aclScalar* scalar, const uint32_t index,
                             const int32_t srcIndex, const ge::DataType dtype);
    static void AppendScalarList(NnopbaseExecutorArgs* args, const aclScalarList* scalarList, const uint32_t index,
                                 const int32_t srcIndex, const ge::DataType dtype);
    static void GenerateCacheArgsKeyV1(NnopbaseExecutor* executor);

private:
    static NnopbaseUChar* AppendShapeInfo(NnopbaseExecutorArgs* args, const aclTensor* tensor);
    static NnopbaseUChar* AppendScalarInfo(NnopbaseExecutorArgs* args, const ge::DataType dtype);
    static NnopbaseUChar* AppendScalarListInfo(NnopbaseExecutorArgs* args, const ge::DataType dtype,
                                               const uint64_t size);
    static void EnsureCapacity(NnopbaseExecutorArgs* args, size_t len);
    static size_t CalculateCacheKeyLenV1(NnopbaseExecutor* executor);

    static constexpr size_t BASE_BYTES = 3U;
    static constexpr size_t SHAPE_BYTES = 8U;
    static constexpr size_t OFFSET_BYTES = 8U;
    static constexpr size_t STRIDE_NUM_BYTES = 1U;
};

} // namespace Indv

#endif