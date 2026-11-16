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

// ==========asc_half2int8(NA/r/a/f/c/z)==========
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvH2S8, asc_half2int8, vconv_f162s8, int8_t, half);
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvH2S8r, asc_half2int8_rn, vconv_f162s8r, int8_t, half);
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvH2S8f, asc_half2int8_rd, vconv_f162s8f, int8_t, half);
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvH2S8c, asc_half2int8_ru, vconv_f162s8c, int8_t, half);
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvH2S8z, asc_half2int8_rz, vconv_f162s8z, int8_t, half);
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvH2S8a, asc_half2int8_rna, vconv_f162s8a, int8_t, half);
