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

// ==========asc_half2uint8(NA/a/c/f/r/z)==========
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvH2U8CAPI, asc_half2uint8, vconv_f162u8, uint8_t, half);
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvH2U8aCAPI, asc_half2uint8_rna, vconv_f162u8a, uint8_t, half);
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvH2U8cCAPI, asc_half2uint8_ru, vconv_f162u8c, uint8_t, half);
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvH2U8fCAPI, asc_half2uint8_rd, vconv_f162u8f, uint8_t, half);
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvH2U8rCAPI, asc_half2uint8_rn, vconv_f162u8r, uint8_t, half);
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvH2U8zCAPI, asc_half2uint8_rz, vconv_f162u8z, uint8_t, half);
