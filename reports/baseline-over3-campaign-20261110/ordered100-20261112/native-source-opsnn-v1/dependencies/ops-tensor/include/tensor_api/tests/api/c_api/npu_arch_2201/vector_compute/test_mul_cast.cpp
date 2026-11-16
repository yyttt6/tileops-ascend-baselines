/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../utils/test_binary_instr_diff_types.h"

// ==========asc_mul_cast_half2int8(dst:int8, src:half)==========
TEST_VECTOR_COMPUTE_BINARY_INSTR_DIFF_TYPES(MulCastCApi, asc_mul_cast_half2int8, vmulconv_f162s8, int8_t, half);

// ==========asc_mul_cast_half2uint8(dst:uint8, src:half)==========
TEST_VECTOR_COMPUTE_BINARY_INSTR_DIFF_TYPES(MulCastCApi, asc_mul_cast_half2uint8, vmulconv_f162u8, uint8_t, half);
