/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tests/api/c_api/npu_arch_3510/utils/test_loadalign_vlds_instr_utils.h"

// vlds, norm
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignNorm, asc_loadalign, vlds, vector_int8_t, int8_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignNorm, asc_loadalign, vlds, vector_uint8_t, uint8_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignNorm, asc_loadalign, vlds, vector_fp4x2_e2m1_t, fp4x2_e2m1_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignNorm, asc_loadalign, vlds, vector_fp4x2_e1m2_t, fp4x2_e1m2_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignNorm, asc_loadalign, vlds, vector_fp8_e8m0_t, fp8_e8m0_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignNorm, asc_loadalign, vlds, vector_fp8_e5m2_t, fp8_e5m2_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignNorm, asc_loadalign, vlds, vector_fp8_e4m3fn_t, fp8_e4m3fn_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR_HIF8(LoadAlignNorm, asc_loadalign, vlds, vector_hifloat8_t, hifloat8_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignNorm, asc_loadalign, vlds, vector_int16_t, int16_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignNorm, asc_loadalign, vlds, vector_uint16_t, uint16_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignNorm, asc_loadalign, vlds, vector_half, half);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignNorm, asc_loadalign, vlds, vector_bfloat16_t, bfloat16_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignNorm, asc_loadalign, vlds, vector_int32_t, int32_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignNorm, asc_loadalign, vlds, vector_uint32_t, uint32_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignNorm, asc_loadalign, vlds, vector_float, float);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignNorm, asc_loadalign, vlds, vector_int64_t, int64_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignNorm, asc_loadalign, vlds, vector_uint64_t, uint64_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR_INT4(LoadAlignNorm, asc_loadalign, vlds, vector_int4x2_t, int4b_t);

// vlds, brc
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignBrc, asc_loadalign_brc, vlds, vector_int8_t, int8_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignBrc, asc_loadalign_brc, vlds, vector_uint8_t, uint8_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignBrc, asc_loadalign_brc, vlds, vector_fp4x2_e2m1_t, fp4x2_e2m1_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignBrc, asc_loadalign_brc, vlds, vector_fp4x2_e1m2_t, fp4x2_e1m2_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignBrc, asc_loadalign_brc, vlds, vector_fp8_e8m0_t, fp8_e8m0_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignBrc, asc_loadalign_brc, vlds, vector_fp8_e5m2_t, fp8_e5m2_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignBrc, asc_loadalign_brc, vlds, vector_fp8_e4m3fn_t, fp8_e4m3fn_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR_HIF8(LoadAlignBrc, asc_loadalign_brc, vlds, vector_hifloat8_t, hifloat8_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignBrc, asc_loadalign_brc, vlds, vector_int16_t, int16_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignBrc, asc_loadalign_brc, vlds, vector_uint16_t, uint16_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignBrc, asc_loadalign_brc, vlds, vector_half, half);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignBrc, asc_loadalign_brc, vlds, vector_bfloat16_t, bfloat16_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignBrc, asc_loadalign_brc, vlds, vector_int32_t, int32_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignBrc, asc_loadalign_brc, vlds, vector_uint32_t, uint32_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignBrc, asc_loadalign_brc, vlds, vector_float, float);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR_INT4(LoadAlignBrc, asc_loadalign_brc, vlds, vector_int4x2_t, int4b_t);

// vlds, upsample
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignUpsample, asc_loadalign_upsample, vlds, vector_int8_t, int8_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignUpsample, asc_loadalign_upsample, vlds, vector_uint8_t, uint8_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(
    LoadAlignUpsample, asc_loadalign_upsample, vlds, vector_fp4x2_e2m1_t, fp4x2_e2m1_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(
    LoadAlignUpsample, asc_loadalign_upsample, vlds, vector_fp4x2_e1m2_t, fp4x2_e1m2_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignUpsample, asc_loadalign_upsample, vlds, vector_fp8_e8m0_t, fp8_e8m0_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignUpsample, asc_loadalign_upsample, vlds, vector_fp8_e5m2_t, fp8_e5m2_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(
    LoadAlignUpsample, asc_loadalign_upsample, vlds, vector_fp8_e4m3fn_t, fp8_e4m3fn_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR_HIF8(
    LoadAlignUpsample, asc_loadalign_upsample, vlds, vector_hifloat8_t, hifloat8_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignUpsample, asc_loadalign_upsample, vlds, vector_int16_t, int16_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignUpsample, asc_loadalign_upsample, vlds, vector_uint16_t, uint16_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignUpsample, asc_loadalign_upsample, vlds, vector_half, half);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignUpsample, asc_loadalign_upsample, vlds, vector_bfloat16_t, bfloat16_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR_INT4(LoadAlignUpsample, asc_loadalign_upsample, vlds, vector_int4x2_t, int4b_t);

// vlds, downsample
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignDownsample, asc_loadalign_downsample, vlds, vector_int8_t, int8_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignDownsample, asc_loadalign_downsample, vlds, vector_uint8_t, uint8_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(
    LoadAlignDownsample, asc_loadalign_downsample, vlds, vector_fp4x2_e2m1_t, fp4x2_e2m1_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(
    LoadAlignDownsample, asc_loadalign_downsample, vlds, vector_fp4x2_e1m2_t, fp4x2_e1m2_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(
    LoadAlignDownsample, asc_loadalign_downsample, vlds, vector_fp8_e8m0_t, fp8_e8m0_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(
    LoadAlignDownsample, asc_loadalign_downsample, vlds, vector_fp8_e5m2_t, fp8_e5m2_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(
    LoadAlignDownsample, asc_loadalign_downsample, vlds, vector_fp8_e4m3fn_t, fp8_e4m3fn_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR_HIF8(
    LoadAlignDownsample, asc_loadalign_downsample, vlds, vector_hifloat8_t, hifloat8_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignDownsample, asc_loadalign_downsample, vlds, vector_int16_t, int16_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignDownsample, asc_loadalign_downsample, vlds, vector_uint16_t, uint16_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignDownsample, asc_loadalign_downsample, vlds, vector_half, half);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(
    LoadAlignDownsample, asc_loadalign_downsample, vlds, vector_bfloat16_t, bfloat16_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR_INT4(
    LoadAlignDownsample, asc_loadalign_downsample, vlds, vector_int4x2_t, int4b_t);

// vlds, unpack
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignUnpack, asc_loadalign_unpack, vlds, vector_int8_t, int8_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignUnpack, asc_loadalign_unpack, vlds, vector_uint8_t, uint8_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignUnpack, asc_loadalign_unpack, vlds, vector_fp4x2_e2m1_t, fp4x2_e2m1_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignUnpack, asc_loadalign_unpack, vlds, vector_fp4x2_e1m2_t, fp4x2_e1m2_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignUnpack, asc_loadalign_unpack, vlds, vector_fp8_e8m0_t, fp8_e8m0_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignUnpack, asc_loadalign_unpack, vlds, vector_fp8_e5m2_t, fp8_e5m2_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignUnpack, asc_loadalign_unpack, vlds, vector_fp8_e4m3fn_t, fp8_e4m3fn_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR_HIF8(LoadAlignUnpack, asc_loadalign_unpack, vlds, vector_hifloat8_t, hifloat8_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignUnpack, asc_loadalign_unpack, vlds, vector_int16_t, int16_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignUnpack, asc_loadalign_unpack, vlds, vector_uint16_t, uint16_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignUnpack, asc_loadalign_unpack, vlds, vector_half, half);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignUnpack, asc_loadalign_unpack, vlds, vector_bfloat16_t, bfloat16_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignUnpack, asc_loadalign_unpack, vlds, vector_int32_t, int32_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignUnpack, asc_loadalign_unpack, vlds, vector_uint32_t, uint32_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignUnpack, asc_loadalign_unpack, vlds, vector_float, float);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR_INT4(LoadAlignUnpack, asc_loadalign_unpack, vlds, vector_int4x2_t, int4b_t);

// vlds, unpack_v2
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignUnpackV2, asc_loadalign_unpack_v2, vlds, vector_int8_t, int8_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignUnpackV2, asc_loadalign_unpack_v2, vlds, vector_uint8_t, uint8_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(
    LoadAlignUnpackV2, asc_loadalign_unpack_v2, vlds, vector_fp4x2_e2m1_t, fp4x2_e2m1_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(
    LoadAlignUnpackV2, asc_loadalign_unpack_v2, vlds, vector_fp4x2_e1m2_t, fp4x2_e1m2_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignUnpackV2, asc_loadalign_unpack_v2, vlds, vector_fp8_e8m0_t, fp8_e8m0_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignUnpackV2, asc_loadalign_unpack_v2, vlds, vector_fp8_e5m2_t, fp8_e5m2_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(
    LoadAlignUnpackV2, asc_loadalign_unpack_v2, vlds, vector_fp8_e4m3fn_t, fp8_e4m3fn_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR_HIF8(
    LoadAlignUnpackV2, asc_loadalign_unpack_v2, vlds, vector_hifloat8_t, hifloat8_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR_INT4(LoadAlignUnpackV2, asc_loadalign_unpack_v2, vlds, vector_int4x2_t, int4b_t);

// vlds, brc_v2
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignBrcV2, asc_loadalign_brc_v2, vlds, vector_int8_t, int8_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignBrcV2, asc_loadalign_brc_v2, vlds, vector_uint8_t, uint8_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignBrcV2, asc_loadalign_brc_v2, vlds, vector_fp4x2_e2m1_t, fp4x2_e2m1_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignBrcV2, asc_loadalign_brc_v2, vlds, vector_fp4x2_e1m2_t, fp4x2_e1m2_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignBrcV2, asc_loadalign_brc_v2, vlds, vector_fp8_e8m0_t, fp8_e8m0_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignBrcV2, asc_loadalign_brc_v2, vlds, vector_fp8_e5m2_t, fp8_e5m2_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignBrcV2, asc_loadalign_brc_v2, vlds, vector_fp8_e4m3fn_t, fp8_e4m3fn_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR_HIF8(LoadAlignBrcV2, asc_loadalign_brc_v2, vlds, vector_hifloat8_t, hifloat8_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignBrcV2, asc_loadalign_brc_v2, vlds, vector_int16_t, int16_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignBrcV2, asc_loadalign_brc_v2, vlds, vector_uint16_t, uint16_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignBrcV2, asc_loadalign_brc_v2, vlds, vector_half, half);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignBrcV2, asc_loadalign_brc_v2, vlds, vector_bfloat16_t, bfloat16_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignBrcV2, asc_loadalign_brc_v2, vlds, vector_int32_t, int32_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignBrcV2, asc_loadalign_brc_v2, vlds, vector_uint32_t, uint32_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignBrcV2, asc_loadalign_brc_v2, vlds, vector_float, float);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR_INT4(LoadAlignBrcV2, asc_loadalign_brc_v2, vlds, vector_int4x2_t, int4b_t);

// vlds, brc_v3
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignBrcV3, asc_loadalign_brc_v3, vlds, vector_int16_t, int16_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignBrcV3, asc_loadalign_brc_v3, vlds, vector_uint16_t, uint16_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignBrcV3, asc_loadalign_brc_v3, vlds, vector_half, half);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignBrcV3, asc_loadalign_brc_v3, vlds, vector_bfloat16_t, bfloat16_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignBrcV3, asc_loadalign_brc_v3, vlds, vector_int32_t, int32_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignBrcV3, asc_loadalign_brc_v3, vlds, vector_uint32_t, uint32_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_INSTR(LoadAlignBrcV3, asc_loadalign_brc_v3, vlds, vector_float, float);

// vlds, deintlv
TEST_VECTOR_DATAMOVE_LOADALIGN_DEINTLV_INSTR(LoadAlignDeintlv, asc_loadalign_deintlv, vlds, vector_int8_t, int8_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_DEINTLV_INSTR(LoadAlignDeintlv, asc_loadalign_deintlv, vlds, vector_uint8_t, uint8_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_DEINTLV_INSTR(
    LoadAlignDeintlv, asc_loadalign_deintlv, vlds, vector_fp4x2_e2m1_t, fp4x2_e2m1_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_DEINTLV_INSTR(
    LoadAlignDeintlv, asc_loadalign_deintlv, vlds, vector_fp4x2_e1m2_t, fp4x2_e1m2_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_DEINTLV_INSTR(
    LoadAlignDeintlv, asc_loadalign_deintlv, vlds, vector_fp8_e8m0_t, fp8_e8m0_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_DEINTLV_INSTR(
    LoadAlignDeintlv, asc_loadalign_deintlv, vlds, vector_fp8_e5m2_t, fp8_e5m2_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_DEINTLV_INSTR(
    LoadAlignDeintlv, asc_loadalign_deintlv, vlds, vector_fp8_e4m3fn_t, fp8_e4m3fn_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_DEINTLV_INSTR_HIF8(
    LoadAlignDeintlv, asc_loadalign_deintlv, vlds, vector_hifloat8_t, hifloat8_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_DEINTLV_INSTR(LoadAlignDeintlv, asc_loadalign_deintlv, vlds, vector_int16_t, int16_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_DEINTLV_INSTR(LoadAlignDeintlv, asc_loadalign_deintlv, vlds, vector_uint16_t, uint16_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_DEINTLV_INSTR(LoadAlignDeintlv, asc_loadalign_deintlv, vlds, vector_half, half);
TEST_VECTOR_DATAMOVE_LOADALIGN_DEINTLV_INSTR(
    LoadAlignDeintlv, asc_loadalign_deintlv, vlds, vector_bfloat16_t, bfloat16_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_DEINTLV_INSTR(LoadAlignDeintlv, asc_loadalign_deintlv, vlds, vector_int32_t, int32_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_DEINTLV_INSTR(LoadAlignDeintlv, asc_loadalign_deintlv, vlds, vector_uint32_t, uint32_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_DEINTLV_INSTR(LoadAlignDeintlv, asc_loadalign_deintlv, vlds, vector_float, float);
TEST_VECTOR_DATAMOVE_LOADALIGN_DEINTLV_INSTR_INT4(
    LoadAlignDeintlv, asc_loadalign_deintlv, vlds, vector_int4x2_t, int4b_t);
