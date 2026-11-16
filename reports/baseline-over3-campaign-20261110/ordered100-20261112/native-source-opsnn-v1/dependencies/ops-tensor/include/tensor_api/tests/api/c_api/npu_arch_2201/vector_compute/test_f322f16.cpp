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

// ==========asc_float2half(NA/r/a/f/c/z/o)==========
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvF2H, asc_float2half, vconv_f322f16, half, float);
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvF2Hr, asc_float2half_rn, vconv_f322f16r, half, float);
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvF2Hf, asc_float2half_rd, vconv_f322f16f, half, float);
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvF2Hc, asc_float2half_ru, vconv_f322f16c, half, float);
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvF2Hz, asc_float2half_rz, vconv_f322f16z, half, float);
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvF2Ho, asc_float2half_ro, vconv_f322f16o, half, float);
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvF2Ha, asc_float2half_rna, vconv_f322f16a, half, float);
