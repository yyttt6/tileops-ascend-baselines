/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INDV_HASH_H_
#define INDV_HASH_H_

#include <string>
#include "indv_base.h"

#ifdef __cplusplus
extern "C" {
#endif
static constexpr uint64_t NNOPBASE_HASH_SEED = 0x9e3779b9U;
size_t NnopbaseHashBinary(const NnopbaseUChar* const addr, const size_t len);

static inline size_t NnopbaseHashInt(size_t& seed, int32_t v)
{
    const std::hash<int32_t> hasher;
    seed ^= hasher(v) + NNOPBASE_HASH_SEED + (seed << 6U) + (seed >> 2U);
    return seed;
}

#ifdef __cplusplus
}
#endif
#endif
