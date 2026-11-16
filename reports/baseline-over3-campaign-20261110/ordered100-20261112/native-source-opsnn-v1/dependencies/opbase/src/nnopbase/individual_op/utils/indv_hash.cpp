/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "indv_hash.h"
#include "opdev/op_dfx.h"

#ifdef __cplusplus
extern "C" {
#endif

size_t NnopbaseHashBinary(const NnopbaseUChar* const addr, const size_t len)
{
    constexpr std::hash<uint64_t> hasher;
    const size_t size = len / sizeof(uint64_t);
    size_t rem = len % sizeof(uint64_t);
    size_t i = 0U;
    size_t seed = 0U;
    const uint64_t* ptr = op::internal::PtrCastTo<const uint64_t>(addr);

    while (i < size) {
        seed ^= hasher(*ptr) + NNOPBASE_HASH_SEED + (seed << 6U) + (seed >> 2U);
        ptr++;
        i++;
    }
    if (rem != 0) {
        const NnopbaseUChar* p = op::internal::PtrCastTo<const NnopbaseUChar>(ptr);
        uint64_t val = 0U;
        for (size_t j = 0U; j < rem; ++j) {
            val |= (*p);
            val = val << 8U;
            p++;
        }
        seed ^= hasher(val) + NNOPBASE_HASH_SEED + (seed << 6U) + (seed >> 2U);
    }
    return seed;
}
#ifdef __cplusplus
}
#endif