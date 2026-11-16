/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../utils/test_vcmp_instr_utils.h"
#include "../utils/test_vcmpv_instr_utils.h"

// ==========asc_eq(half/float)==========
TEST_VECTOR_COMPUTE_COMPARE_VCMP_INSTR(VCmpEQCApi, asc_eq, vcmp_eq, half);
TEST_VECTOR_COMPUTE_COMPARE_VCMP_INSTR(VCmpEQCApi, asc_eq, vcmp_eq, float);

// ==========asc_lt(half/float)==========
TEST_VECTOR_COMPUTE_COMPARE_VCMPV_INSTR(VCmpVEQCApi, asc_eq, vcmpv_eq, uint8_t, half);
TEST_VECTOR_COMPUTE_COMPARE_VCMPV_INSTR(VCmpVEQCApi, asc_eq, vcmpv_eq, uint8_t, float);
