/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../utils/test_conv_instr_utils.h"

// ==========asc_float2bf16(r/a/f/c/z)==========
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvF2Bf16r, asc_float2bfloat16_rn, vconv_f322bf16r, bfloat16_t, float);
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvF2Bf16f, asc_float2bfloat16_rd, vconv_f322bf16f, bfloat16_t, float);
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvF2Bf16c, asc_float2bfloat16_ru, vconv_f322bf16c, bfloat16_t, float);
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvF2Bf16z, asc_float2bfloat16_rz, vconv_f322bf16z, bfloat16_t, float);
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvF2Bf16a, asc_float2bfloat16_rna, vconv_f322bf16a, bfloat16_t, float);
