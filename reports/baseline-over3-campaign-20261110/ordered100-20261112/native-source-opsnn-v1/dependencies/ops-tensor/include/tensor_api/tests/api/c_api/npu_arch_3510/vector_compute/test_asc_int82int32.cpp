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

TEST_VECTOR_COMPUTE_VCVT_INSTR_0(Int82int32, asc_int82int32, vcvt, vector_int32_t, vector_int8_t, 0);
TEST_VECTOR_COMPUTE_VCVT_INSTR_0(Int82int32_v2, asc_int82int32_v2, vcvt, vector_int32_t, vector_int8_t, 1);
TEST_VECTOR_COMPUTE_VCVT_INSTR_0(Int82int32_v3, asc_int82int32_v3, vcvt, vector_int32_t, vector_int8_t, 2);
TEST_VECTOR_COMPUTE_VCVT_INSTR_0(Int82int32_v4, asc_int82int32_v4, vcvt, vector_int32_t, vector_int8_t, 3);
