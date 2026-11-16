/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __INDV_ARGS_H_
#define __INDV_ARGS_H_

#include <vector>
#include "acl/acl_rt.h"
#include "utils/indv_base.h"

struct NnopbaseRTArgsExt {
    void* args;                             // args host mem addr
    aclrtPlaceHolderInfo* hostInputInfoPtr; // nullptr means no host mem input
    uint32_t argsSize;                      // input + output + tiling addr size + tiling data size + host mem
    uint32_t tilingAddrOffset;              // tiling addr offset
    uint32_t tilingDataOffset;              // tiling data offset
    uint8_t hasTiling;                      // if has tiling: 0 means no tiling
    uint16_t hostInputInfoNum;              // hostInputInfo num
};

std::vector<aclrtPlaceHolderInfo> NnopbaseGetRTSPlaceHolder(NnopbaseRTArgsExt* const argsExt);

void NnopbaseGetIrIndex(const NnopbaseParamDesc& paramDesc, const size_t index, size_t& irIndex, size_t& relativeIndex);

#endif
