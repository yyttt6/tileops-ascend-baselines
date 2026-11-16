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

// ==========asc_half2int16(a/c/f/r/z)==========
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvH2I16a, asc_half2int16_rna, vconv_f162s16a, int16_t, half);
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvH2I16c, asc_half2int16_ru, vconv_f162s16c, int16_t, half);
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvH2I16f, asc_half2int16_rd, vconv_f162s16f, int16_t, half);
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvH2I16r, asc_half2int16_rn, vconv_f162s16r, int16_t, half);
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvH2I16z, asc_half2int16_rz, vconv_f162s16z, int16_t, half);
