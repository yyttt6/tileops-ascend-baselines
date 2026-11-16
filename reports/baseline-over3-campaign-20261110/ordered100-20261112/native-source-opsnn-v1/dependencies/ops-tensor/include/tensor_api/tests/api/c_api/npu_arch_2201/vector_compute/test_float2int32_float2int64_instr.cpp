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

// ==========asc_float2int32(a/c/f/r/z)==========
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvF2I32a, asc_float2int32_rna, vconv_f322s32a, int32_t, float);
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvF2I32c, asc_float2int32_ru, vconv_f322s32c, int32_t, float);
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvF2I32f, asc_float2int32_rd, vconv_f322s32f, int32_t, float);
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvF2I32r, asc_float2int32_rn, vconv_f322s32r, int32_t, float);
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvF2I32z, asc_float2int32_rz, vconv_f322s32z, int32_t, float);
// ==========asc_float2int64(a/c/f/r/z)==========
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvF2I64a, asc_float2int64_rna, vconv_f322s64a, int64_t, float);
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvF2I64c, asc_float2int64_ru, vconv_f322s64c, int64_t, float);
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvF2I64f, asc_float2int64_rd, vconv_f322s64f, int64_t, float);
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvF2I64r, asc_float2int64_rn, vconv_f322s64r, int64_t, float);
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvF2I64z, asc_float2int64_rz, vconv_f322s64z, int64_t, float);
