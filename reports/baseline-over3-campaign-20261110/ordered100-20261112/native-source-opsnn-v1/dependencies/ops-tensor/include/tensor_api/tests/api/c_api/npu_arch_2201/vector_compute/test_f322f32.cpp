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

// ==========asc_float2float(r/f/c/a/z)==========
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvF2Fr, asc_float2float_rn, vconv_f322f32r, float, float);
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvF2Ff, asc_float2float_rd, vconv_f322f32f, float, float);
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvF2Fc, asc_float2float_ru, vconv_f322f32c, float, float);
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvF2Fa, asc_float2float_rna, vconv_f322f32a, float, float);
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvF2Fz, asc_float2float_rz, vconv_f322f32z, float, float);
