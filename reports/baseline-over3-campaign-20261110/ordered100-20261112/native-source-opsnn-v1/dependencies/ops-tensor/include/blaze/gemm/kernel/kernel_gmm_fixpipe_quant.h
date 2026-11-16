/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file kernel_gmm_fixpipe_quant.h
 * \deprecated This header has been renamed to kernel_qgmm_mix_fixpipe_quant.h. Include the new header instead.
 *
 * This compatibility header is retained for downstream repositories through Q2 2027 and may be removed afterward.
 */
#pragma once

#pragma message( \
    "kernel_gmm_fixpipe_quant.h has been renamed to kernel_qgmm_mix_fixpipe_quant.h and is supported through Q2 2027. Please update your #include.")

#include "blaze/gemm/kernel/kernel_qgmm_mix_fixpipe_quant.h"
