/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "wqmm_copy_observer.h"

// GNU ld --wrap observes this one CCE intrinsic. It does not mock layouts,
// routing, Copy arithmetic or the SDK call, and stays inert outside a capture.
extern "C" void __real__Z17copy_ubuf_to_cbufPvS_htttt(void*, void*, uint8_t, uint16_t, uint16_t, uint16_t, uint16_t);
extern "C" void __wrap__Z17copy_ubuf_to_cbufPvS_htttt(void* dst, void* src, uint8_t sid, uint16_t count,
                                                      uint16_t length, uint16_t srcGap, uint16_t dstGap)
{
    if (auto* record = WeightQuantBatchMatmulUT::observedCopy) {
        ++record->calls;
        record->src = reinterpret_cast<uintptr_t>(src);
        record->dst = reinterpret_cast<uintptr_t>(dst);
        record->sid = sid;
        record->count = count;
        record->length = length;
        record->srcGap = srcGap;
        record->dstGap = dstGap;
    }
    __real__Z17copy_ubuf_to_cbufPvS_htttt(dst, src, sid, count, length, srcGap, dstGap);
}
