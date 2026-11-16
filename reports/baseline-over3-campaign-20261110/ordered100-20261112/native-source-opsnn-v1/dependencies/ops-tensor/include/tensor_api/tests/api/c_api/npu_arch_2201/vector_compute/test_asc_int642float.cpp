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

// ==========asc_int642float32(a/c/f/r/z)==========
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvI642Fa, asc_int642float_rna, vconv_s642f32a, float, int64_t);
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvI642Fc, asc_int642float_ru, vconv_s642f32c, float, int64_t);
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvI642Ff, asc_int642float_rd, vconv_s642f32f, float, int64_t);
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvI642Fr, asc_int642float_rn, vconv_s642f32r, float, int64_t);
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvI642Fz, asc_int642float_rz, vconv_s642f32z, float, int64_t);
