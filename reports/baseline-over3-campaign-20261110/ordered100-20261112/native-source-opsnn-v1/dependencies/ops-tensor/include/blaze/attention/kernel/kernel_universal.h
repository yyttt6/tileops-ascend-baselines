/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS PROGRAM IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file kernel_universal.h
 * \brief Primary template for Attention kernels assembled from block components.
 */

#pragma once

#include "blaze/gemm/utils/common_utils.h"

namespace Blaze {
namespace Attention {
namespace Kernel {

/**
 * @class AttentionUniversal
 * @brief Common compile-time dispatch entry for Attention kernels.
 */
template <class ProblemShape_, class BlockMmad_, class BlockEpilogue_, class BlockScheduler_, typename Enable_ = void>
class AttentionUniversal {
    static_assert(Gemm::always_false_v<BlockEpilogue_> && Gemm::always_false_v<BlockMmad_>,
                  "AttentionUniversal is not implemented for this BlockEpilogue or BlockMmad");
};

} // namespace Kernel
} // namespace Attention
} // namespace Blaze

#include "blaze/attention/kernel/kernel_flat_quant.h"
#include "blaze/attention/kernel/kernel_block_attn_res_prepare.h"
