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
 * \file broadcast_template_util.h
 * \brief
 */
#ifndef BROADCAST_TMPLATE_UTILS_H_
#define BROADCAST_TMPLATE_UTILS_H_

#include "broadcast_base_struct.h"

namespace Ops {
namespace Base {
template <class ElemDag, size_t N>
__aicore__ inline void GetUbBroadcastShapeInfo(const int64_t (&oriShape)[N], int64_t inputUblength[2],
                                               uint32_t (&ubFormershape)[N], uint32_t (&ubTailshape)[N],
                                               const BroadcastBaseTilingData<ElemDag>* tilingData)
{
    int64_t ubFormerLength = 1;
    int64_t ubTailLength = 1;
    if (oriShape[tilingData->ubSplitAxis] == 1) {
        ubFormershape[0] = 1;
        ubTailshape[0] = 1;
    } else {
        ubTailshape[0] = tilingData->ubTail;
        ubTailLength = tilingData->ubTail;
        ubFormershape[0] = tilingData->ubFormer;
        ubFormerLength = tilingData->ubFormer;
    }

    int j = 1;
    for (uint64_t i = tilingData->ubSplitAxis + 1; i < tilingData->shapeLen; i++) {
        ubFormershape[j] = oriShape[i];
        ubTailshape[j] = oriShape[i];
        ubFormerLength *= oriShape[i];
        ubTailLength *= oriShape[i];
        j = j + 1;
    }

    inputUblength[0] = ubFormerLength;
    inputUblength[1] = ubTailLength;
}
} // namespace Base
} // namespace Ops
#endif