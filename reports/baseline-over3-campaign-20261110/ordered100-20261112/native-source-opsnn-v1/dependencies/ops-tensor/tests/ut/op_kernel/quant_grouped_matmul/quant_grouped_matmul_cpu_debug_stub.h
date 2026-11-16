/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * \file quant_grouped_matmul_cpu_debug_stub.h
 * \brief CPU-debug compatibility for QGMM MX padding.
 */

#pragma once

#if defined(ASCENDC_CPU_DEBUG)
#include "lib/matmul_intf.h"

// tikicpulib does not provide the out-of-line half integer conversion used by half(0). Function-like macro
// replacement leaves the half type untouched and selects asc_fill_l1's uint32_t bit-pattern overload instead.
#define half(...) static_cast<uint32_t>(0)
#endif
