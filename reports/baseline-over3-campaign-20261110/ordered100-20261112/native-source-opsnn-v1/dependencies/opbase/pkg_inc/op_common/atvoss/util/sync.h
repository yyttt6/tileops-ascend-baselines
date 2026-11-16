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
 * \file sync.h
 * \brief
 */
#ifndef UTIL_SYNC_H_
#define UTIL_SYNC_H_

#include "atvoss/util/kernel_vec_intf_apt.h"
namespace Ops {
namespace Base {
using namespace AscendC;
/**
 * 调用get_buf函数
 * @tparam p TPosition的类型
 * @param bufID 当前处理的buf的真是ID
 */
template <TPosition p>
__aicore__ inline void GetBuf(uint8_t bufID)
{
    if constexpr (p == TPosition::VECIN) {
        Mutex::Lock<PIPE_MTE2>(bufID);
    } else if constexpr (p == TPosition::VECCALC) {
        Mutex::Lock<PIPE_V>(bufID);
    } else if constexpr (p == TPosition::VECOUT) {
        Mutex::Lock<PIPE_MTE3>(bufID);
    }
}

/**
 * 调用rls_buf函数
 * @tparam p TPosition的类型
 * @param bufID 当前处理的buf的真实ID
 */
template <TPosition p>
__aicore__ inline void RlsBuf(uint8_t bufID)
{
    if constexpr (p == TPosition::VECIN) {
        Mutex::Unlock<PIPE_MTE2>(bufID);
    } else if constexpr (p == TPosition::VECCALC) {
        Mutex::Unlock<PIPE_V>(bufID);
    } else if constexpr (p == TPosition::VECOUT) {
        Mutex::Unlock<PIPE_MTE3>(bufID);
    }
}

/**
 * 插入GetBuf同步函数
 * @tparam p TPosition的类型
 * @param bufID 当前处理的buf的真实ID
 */
template <TPosition p>
__aicore__ void inline GetTensor(uint8_t bufID)
{
    RUN_LOG("GetTensor: ID: %d, Position = %d", bufID, p);
    GetBuf<p>(bufID);
}

/**
 * 插入RlsBuf同步函数
 * @tparam p TPosition的类型
 * @param bufID 当前处理的buf的真实ID
 */
template <TPosition p>
__aicore__ void inline ReleaseTensor(uint8_t bufID)
{
    RUN_LOG("ReleaseTensor: ID: %d, Position = %d", bufID, p);
    RlsBuf<p>(bufID);
}
} // namespace Base
} // namespace Ops
#endif // UTIL_SYNC_H_