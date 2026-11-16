/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tests/api/c_api/npu_arch_3510/utils/test_vcvt_instr_utils.h"

TEST_VECTOR_COMPUTE_VCVT_INSTR_5(Vcvt_ru, asc_float2half_ru, vcvt, vector_half, vector_float, 1);
TEST_VECTOR_COMPUTE_VCVT_INSTR_5(Vcvt_ru_sat, asc_float2half_ru_sat, vcvt, vector_half, vector_float, 2);
TEST_VECTOR_COMPUTE_VCVT_INSTR_5(Vcvt_ru_v2, asc_float2half_ru_v2, vcvt, vector_half, vector_float, 3);
TEST_VECTOR_COMPUTE_VCVT_INSTR_5(Vcvt_ru_sat_v2, asc_float2half_ru_sat_v2, vcvt, vector_half, vector_float, 4);
