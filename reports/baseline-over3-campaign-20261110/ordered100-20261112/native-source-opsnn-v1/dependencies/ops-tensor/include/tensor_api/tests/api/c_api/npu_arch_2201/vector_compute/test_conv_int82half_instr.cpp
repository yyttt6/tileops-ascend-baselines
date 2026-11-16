/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "c_api/npu_arch_2201/utils/test_conv_instr_utils.h"

// ==========asc_int82half==========
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvI82HCApi, asc_int82half, vconv_s82f16, half, int8_t);
// ==========asc_uint82half==========
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvU82HCApi, asc_uint82half, vconv_u82f16, half, uint8_t);
// ==========asc_int162float==========
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvI162FCApi, asc_int162float, vconv_s162f32, float, int16_t);
// ==========asc_int322int16==========
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvI322I16CApi, asc_int322int16, vconv_s322s16, int16_t, int32_t);
// ==========asc_int322int64==========
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvI322I64CApi, asc_int322int64, vconv_s322s64, int64_t, int32_t);
// ==========asc_int642int32==========
TEST_VECTOR_COMPUTE_CONV_INSTR(ConvI642I32CApi, asc_int642int32, vconv_s642s32, int32_t, int64_t);
