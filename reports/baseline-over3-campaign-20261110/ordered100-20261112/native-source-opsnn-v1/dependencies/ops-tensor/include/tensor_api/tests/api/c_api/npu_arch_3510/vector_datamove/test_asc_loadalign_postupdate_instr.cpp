/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tests/api/c_api/npu_arch_3510/utils/test_loadalign_postupdate_vlds_instr_utils.h"

// vlds, norm
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateNorm, asc_loadalign_postupdate, vlds, vector_int8_t, int8_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateNorm, asc_loadalign_postupdate, vlds, vector_uint8_t, uint8_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateNorm, asc_loadalign_postupdate, vlds, vector_fp4x2_e2m1_t, fp4x2_e2m1_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateNorm, asc_loadalign_postupdate, vlds, vector_fp4x2_e1m2_t, fp4x2_e1m2_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateNorm, asc_loadalign_postupdate, vlds, vector_fp8_e8m0_t, fp8_e8m0_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateNorm, asc_loadalign_postupdate, vlds, vector_fp8_e5m2_t, fp8_e5m2_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateNorm, asc_loadalign_postupdate, vlds, vector_fp8_e4m3fn_t, fp8_e4m3fn_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR_HIF8(
    LoadAlignPostupdateNorm, asc_loadalign_postupdate, vlds, vector_hifloat8_t, hifloat8_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateNorm, asc_loadalign_postupdate, vlds, vector_int16_t, int16_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateNorm, asc_loadalign_postupdate, vlds, vector_uint16_t, uint16_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateNorm, asc_loadalign_postupdate, vlds, vector_half, half);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateNorm, asc_loadalign_postupdate, vlds, vector_bfloat16_t, bfloat16_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateNorm, asc_loadalign_postupdate, vlds, vector_int32_t, int32_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateNorm, asc_loadalign_postupdate, vlds, vector_uint32_t, uint32_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateNorm, asc_loadalign_postupdate, vlds, vector_float, float);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateNorm, asc_loadalign_postupdate, vlds, vector_int64_t, int64_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateNorm, asc_loadalign_postupdate, vlds, vector_uint64_t, uint64_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR_INT4(
    LoadAlignPostupdateNorm, asc_loadalign_postupdate, vlds, vector_int4x2_t, int4b_t);

// vlds, brc
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateBrc, asc_loadalign_brc_postupdate, vlds, vector_int8_t, int8_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateBrc, asc_loadalign_brc_postupdate, vlds, vector_uint8_t, uint8_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateBrc, asc_loadalign_brc_postupdate, vlds, vector_fp4x2_e2m1_t, fp4x2_e2m1_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateBrc, asc_loadalign_brc_postupdate, vlds, vector_fp4x2_e1m2_t, fp4x2_e1m2_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateBrc, asc_loadalign_brc_postupdate, vlds, vector_fp8_e8m0_t, fp8_e8m0_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateBrc, asc_loadalign_brc_postupdate, vlds, vector_fp8_e5m2_t, fp8_e5m2_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateBrc, asc_loadalign_brc_postupdate, vlds, vector_fp8_e4m3fn_t, fp8_e4m3fn_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR_HIF8(
    LoadAlignPostupdateBrc, asc_loadalign_brc_postupdate, vlds, vector_hifloat8_t, hifloat8_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateBrc, asc_loadalign_brc_postupdate, vlds, vector_int16_t, int16_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateBrc, asc_loadalign_brc_postupdate, vlds, vector_uint16_t, uint16_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateBrc, asc_loadalign_brc_postupdate, vlds, vector_half, half);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateBrc, asc_loadalign_brc_postupdate, vlds, vector_bfloat16_t, bfloat16_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateBrc, asc_loadalign_brc_postupdate, vlds, vector_int32_t, int32_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateBrc, asc_loadalign_brc_postupdate, vlds, vector_uint32_t, uint32_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateBrc, asc_loadalign_brc_postupdate, vlds, vector_float, float);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR_INT4(
    LoadAlignPostupdateBrc, asc_loadalign_brc_postupdate, vlds, vector_int4x2_t, int4b_t);

// vlds, upsample
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateUpsample, asc_loadalign_upsample_postupdate, vlds, vector_int8_t, int8_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateUpsample, asc_loadalign_upsample_postupdate, vlds, vector_uint8_t, uint8_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateUpsample, asc_loadalign_upsample_postupdate, vlds, vector_fp4x2_e2m1_t, fp4x2_e2m1_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateUpsample, asc_loadalign_upsample_postupdate, vlds, vector_fp4x2_e1m2_t, fp4x2_e1m2_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateUpsample, asc_loadalign_upsample_postupdate, vlds, vector_fp8_e8m0_t, fp8_e8m0_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateUpsample, asc_loadalign_upsample_postupdate, vlds, vector_fp8_e5m2_t, fp8_e5m2_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateUpsample, asc_loadalign_upsample_postupdate, vlds, vector_fp8_e4m3fn_t, fp8_e4m3fn_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR_HIF8(
    LoadAlignPostupdateUpsample, asc_loadalign_upsample_postupdate, vlds, vector_hifloat8_t, hifloat8_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateUpsample, asc_loadalign_upsample_postupdate, vlds, vector_int16_t, int16_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateUpsample, asc_loadalign_upsample_postupdate, vlds, vector_uint16_t, uint16_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateUpsample, asc_loadalign_upsample_postupdate, vlds, vector_half, half);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateUpsample, asc_loadalign_upsample_postupdate, vlds, vector_bfloat16_t, bfloat16_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR_INT4(
    LoadAlignPostupdateUpsample, asc_loadalign_upsample_postupdate, vlds, vector_int4x2_t, int4b_t);

// vlds, downsample
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateDownsample, asc_loadalign_downsample_postupdate, vlds, vector_int8_t, int8_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateDownsample, asc_loadalign_downsample_postupdate, vlds, vector_uint8_t, uint8_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateDownsample, asc_loadalign_downsample_postupdate, vlds, vector_fp4x2_e2m1_t, fp4x2_e2m1_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateDownsample, asc_loadalign_downsample_postupdate, vlds, vector_fp4x2_e1m2_t, fp4x2_e1m2_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateDownsample, asc_loadalign_downsample_postupdate, vlds, vector_fp8_e8m0_t, fp8_e8m0_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateDownsample, asc_loadalign_downsample_postupdate, vlds, vector_fp8_e5m2_t, fp8_e5m2_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateDownsample, asc_loadalign_downsample_postupdate, vlds, vector_fp8_e4m3fn_t, fp8_e4m3fn_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR_HIF8(
    LoadAlignPostupdateDownsample, asc_loadalign_downsample_postupdate, vlds, vector_hifloat8_t, hifloat8_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateDownsample, asc_loadalign_downsample_postupdate, vlds, vector_int16_t, int16_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateDownsample, asc_loadalign_downsample_postupdate, vlds, vector_uint16_t, uint16_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateDownsample, asc_loadalign_downsample_postupdate, vlds, vector_half, half);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateDownsample, asc_loadalign_downsample_postupdate, vlds, vector_bfloat16_t, bfloat16_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR_INT4(
    LoadAlignPostupdateDownsample, asc_loadalign_downsample_postupdate, vlds, vector_int4x2_t, int4b_t);

// vlds, unpack
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateUnpack, asc_loadalign_unpack_postupdate, vlds, vector_int8_t, int8_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateUnpack, asc_loadalign_unpack_postupdate, vlds, vector_uint8_t, uint8_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateUnpack, asc_loadalign_unpack_postupdate, vlds, vector_fp4x2_e2m1_t, fp4x2_e2m1_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateUnpack, asc_loadalign_unpack_postupdate, vlds, vector_fp4x2_e1m2_t, fp4x2_e1m2_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateUnpack, asc_loadalign_unpack_postupdate, vlds, vector_fp8_e8m0_t, fp8_e8m0_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateUnpack, asc_loadalign_unpack_postupdate, vlds, vector_fp8_e5m2_t, fp8_e5m2_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateUnpack, asc_loadalign_unpack_postupdate, vlds, vector_fp8_e4m3fn_t, fp8_e4m3fn_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR_HIF8(
    LoadAlignPostupdateUnpack, asc_loadalign_unpack_postupdate, vlds, vector_hifloat8_t, hifloat8_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateUnpack, asc_loadalign_unpack_postupdate, vlds, vector_int16_t, int16_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateUnpack, asc_loadalign_unpack_postupdate, vlds, vector_uint16_t, uint16_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateUnpack, asc_loadalign_unpack_postupdate, vlds, vector_half, half);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateUnpack, asc_loadalign_unpack_postupdate, vlds, vector_bfloat16_t, bfloat16_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateUnpack, asc_loadalign_unpack_postupdate, vlds, vector_int32_t, int32_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateUnpack, asc_loadalign_unpack_postupdate, vlds, vector_uint32_t, uint32_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateUnpack, asc_loadalign_unpack_postupdate, vlds, vector_float, float);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR_INT4(
    LoadAlignPostupdateUnpack, asc_loadalign_unpack_postupdate, vlds, vector_int4x2_t, int4b_t);

// vlds, unpack_v2
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateUnpackV2, asc_loadalign_unpack_postupdate_v2, vlds, vector_int8_t, int8_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateUnpackV2, asc_loadalign_unpack_postupdate_v2, vlds, vector_uint8_t, uint8_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateUnpackV2, asc_loadalign_unpack_postupdate_v2, vlds, vector_fp4x2_e2m1_t, fp4x2_e2m1_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateUnpackV2, asc_loadalign_unpack_postupdate_v2, vlds, vector_fp4x2_e1m2_t, fp4x2_e1m2_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateUnpackV2, asc_loadalign_unpack_postupdate_v2, vlds, vector_fp8_e8m0_t, fp8_e8m0_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateUnpackV2, asc_loadalign_unpack_postupdate_v2, vlds, vector_fp8_e5m2_t, fp8_e5m2_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateUnpackV2, asc_loadalign_unpack_postupdate_v2, vlds, vector_fp8_e4m3fn_t, fp8_e4m3fn_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR_HIF8(
    LoadAlignPostupdateUnpackV2, asc_loadalign_unpack_postupdate_v2, vlds, vector_hifloat8_t, hifloat8_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR_INT4(
    LoadAlignPostupdateUnpackV2, asc_loadalign_unpack_postupdate_v2, vlds, vector_int4x2_t, int4b_t);

// vlds, brc_v2
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateBrcV2, asc_loadalign_brc_postupdate_v2, vlds, vector_int8_t, int8_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateBrcV2, asc_loadalign_brc_postupdate_v2, vlds, vector_uint8_t, uint8_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateBrcV2, asc_loadalign_brc_postupdate_v2, vlds, vector_fp4x2_e2m1_t, fp4x2_e2m1_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateBrcV2, asc_loadalign_brc_postupdate_v2, vlds, vector_fp4x2_e1m2_t, fp4x2_e1m2_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateBrcV2, asc_loadalign_brc_postupdate_v2, vlds, vector_fp8_e8m0_t, fp8_e8m0_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateBrcV2, asc_loadalign_brc_postupdate_v2, vlds, vector_fp8_e5m2_t, fp8_e5m2_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateBrcV2, asc_loadalign_brc_postupdate_v2, vlds, vector_fp8_e4m3fn_t, fp8_e4m3fn_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR_HIF8(
    LoadAlignPostupdateBrcV2, asc_loadalign_brc_postupdate_v2, vlds, vector_hifloat8_t, hifloat8_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateBrcV2, asc_loadalign_brc_postupdate_v2, vlds, vector_int16_t, int16_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateBrcV2, asc_loadalign_brc_postupdate_v2, vlds, vector_uint16_t, uint16_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateBrcV2, asc_loadalign_brc_postupdate_v2, vlds, vector_half, half);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateBrcV2, asc_loadalign_brc_postupdate_v2, vlds, vector_bfloat16_t, bfloat16_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateBrcV2, asc_loadalign_brc_postupdate_v2, vlds, vector_int32_t, int32_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateBrcV2, asc_loadalign_brc_postupdate_v2, vlds, vector_uint32_t, uint32_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateBrcV2, asc_loadalign_brc_postupdate_v2, vlds, vector_float, float);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR_INT4(
    LoadAlignPostupdateBrcV2, asc_loadalign_brc_postupdate_v2, vlds, vector_int4x2_t, int4b_t);

// vlds, brc_v3
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateBrcV3, asc_loadalign_brc_postupdate_v3, vlds, vector_int16_t, int16_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateBrcV3, asc_loadalign_brc_postupdate_v3, vlds, vector_uint16_t, uint16_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateBrcV3, asc_loadalign_brc_postupdate_v3, vlds, vector_half, half);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateBrcV3, asc_loadalign_brc_postupdate_v3, vlds, vector_bfloat16_t, bfloat16_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateBrcV3, asc_loadalign_brc_postupdate_v3, vlds, vector_int32_t, int32_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateBrcV3, asc_loadalign_brc_postupdate_v3, vlds, vector_uint32_t, uint32_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_INSTR(
    LoadAlignPostupdateBrcV3, asc_loadalign_brc_postupdate_v3, vlds, vector_float, float);

// vlds, deintlv
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_DEINTLV_INSTR(
    LoadAlignPostupdateDeintlv, asc_loadalign_deintlv_postupdate, vlds, vector_int8_t, int8_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_DEINTLV_INSTR(
    LoadAlignPostupdateDeintlv, asc_loadalign_deintlv_postupdate, vlds, vector_uint8_t, uint8_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_DEINTLV_INSTR(
    LoadAlignPostupdateDeintlv, asc_loadalign_deintlv_postupdate, vlds, vector_fp4x2_e2m1_t, fp4x2_e2m1_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_DEINTLV_INSTR(
    LoadAlignPostupdateDeintlv, asc_loadalign_deintlv_postupdate, vlds, vector_fp4x2_e1m2_t, fp4x2_e1m2_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_DEINTLV_INSTR(
    LoadAlignPostupdateDeintlv, asc_loadalign_deintlv_postupdate, vlds, vector_fp8_e8m0_t, fp8_e8m0_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_DEINTLV_INSTR(
    LoadAlignPostupdateDeintlv, asc_loadalign_deintlv_postupdate, vlds, vector_fp8_e5m2_t, fp8_e5m2_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_DEINTLV_INSTR(
    LoadAlignPostupdateDeintlv, asc_loadalign_deintlv_postupdate, vlds, vector_fp8_e4m3fn_t, fp8_e4m3fn_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_DEINTLV_INSTR_HIF8(
    LoadAlignPostupdateDeintlv, asc_loadalign_deintlv_postupdate, vlds, vector_hifloat8_t, hifloat8_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_DEINTLV_INSTR(
    LoadAlignPostupdateDeintlv, asc_loadalign_deintlv_postupdate, vlds, vector_int16_t, int16_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_DEINTLV_INSTR(
    LoadAlignPostupdateDeintlv, asc_loadalign_deintlv_postupdate, vlds, vector_uint16_t, uint16_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_DEINTLV_INSTR(
    LoadAlignPostupdateDeintlv, asc_loadalign_deintlv_postupdate, vlds, vector_half, half);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_DEINTLV_INSTR(
    LoadAlignPostupdateDeintlv, asc_loadalign_deintlv_postupdate, vlds, vector_bfloat16_t, bfloat16_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_DEINTLV_INSTR(
    LoadAlignPostupdateDeintlv, asc_loadalign_deintlv_postupdate, vlds, vector_int32_t, int32_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_DEINTLV_INSTR(
    LoadAlignPostupdateDeintlv, asc_loadalign_deintlv_postupdate, vlds, vector_uint32_t, uint32_t);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_DEINTLV_INSTR(
    LoadAlignPostupdateDeintlv, asc_loadalign_deintlv_postupdate, vlds, vector_float, float);
TEST_VECTOR_DATAMOVE_LOADALIGN_POSTUPDATE_DEINTLV_INSTR_INT4(
    LoadAlignPostupdateDeintlv, asc_loadalign_deintlv_postupdate, vlds, vector_int4x2_t, int4b_t);
