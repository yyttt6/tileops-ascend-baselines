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
 * \file qbmm_cpu_debug_stub.h
 * \brief CPU-debug 下引入 Matmul 公共类型与辅助函数定义。
 *        复用 CANN 提供的实现，避免与其他 Matmul 头文件重复定义。
 */

#pragma once

#if defined(ASCENDC_CPU_DEBUG)
#include "lib/matmul_intf.h"
#endif
