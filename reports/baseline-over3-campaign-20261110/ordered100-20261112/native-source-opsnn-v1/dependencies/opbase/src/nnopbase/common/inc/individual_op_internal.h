/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_OP_API_COMMON_INC_OPDEV_NNOPBASE_H
#define OP_API_OP_API_COMMON_INC_OPDEV_NNOPBASE_H

#include "opdev/common_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const void* addr;
    size_t size;
    bool isOptional;
    bool isVector;
    size_t elementSize;
} NnopbaseAttrAddr;

struct NnopbaseStaticTensorNumInfo {
    int64_t numTensors;
    int64_t numDynamic;
    int64_t numAttrs;
    int64_t numValueDepend;
};

struct NnopbaseStaticRuntimeInfo {
    std::string opType;
    uint32_t aicNum;
    uint32_t aivNum;
    int64_t implMode;
    int64_t deterMode;
};

/**
 * @description: find static kernel path for hostapi
 * @param [in] tensors input and output tensor
 * @param [in] attrs attributes
 * @param [in] valueDepend valueDepend input index
 * @param [in] tensorNumInfo number description of operator
 * @param [in] staticRuntimeInfo runtime infomation of operator
 * @return if find static kernel return path else return nullptr
 */
const char* NnopbaseFindStaticKernel(const aclTensor* tensors[], const NnopbaseAttrAddr* attrs[],
                                     const int64_t valueDepend[],
                                     const NnopbaseStaticTensorNumInfo* const tensorNumInfo,
                                     const NnopbaseStaticRuntimeInfo* const staticRuntimeInfo);

/**
 * @description: 获取从stream和event
 * @param [in] stream 用户传入的主stream
 * @param [in] subStream 创建的从stream
 * @param [in] evtA 创建的eventA
 * @param [in] evtB 创建的eventB
 * @param [in] streamLckPtr 返回的stream级别的锁
 * @return OK for ok
 * @return ACLNN_ERR_RUNTIME_ERROR for error input
 */
aclnnStatus NnopbaseGetStreamAndEvent(const aclrtStream stream, aclrtStream* subStream, aclrtEvent* evtA,
                                      aclrtEvent* evtB, std::shared_ptr<std::mutex>& streamLckPtr);
#ifdef __cplusplus
}
#endif
#endif
