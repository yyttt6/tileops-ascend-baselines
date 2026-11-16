/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file reduce_tiling_key_sel_non_contiguous.h
 * \brief reduce tiling key select non contiguous
 */

#ifndef _REDUCE_TILING_KEY_SEL_NON_CONTIGUOUS_H_
#define _REDUCE_TILING_KEY_SEL_NON_CONTIGUOUS_H_

#include "reduce_tiling_key_decl_non_contiguous.h"

ASCENDC_TPL_SEL(
    // Empty
    ASCENDC_TPL_ARGS_SEL(REDUCE_TPL_KEY_SEL_EMPTY_NON_CONTIGUOUS()),
    // A
    ASCENDC_TPL_ARGS_SEL(REDUCE_TPL_KEY_SEL_A_NON_CONTIGUOUS()),
    // AR nonContiguous
    ASCENDC_TPL_ARGS_SEL(REDUCE_TPL_KEY_SEL_AR_NORMAL_NON_CONTIGUOUS()),
    ASCENDC_TPL_ARGS_SEL(REDUCE_TPL_KEY_SEL_AR_GROUP_NON_CONTIGUOUS()),
    // ARA nonContiguous
    ASCENDC_TPL_ARGS_SEL(REDUCE_TPL_KEY_SEL_ARA_NORMAL_NON_CONTIGUOUS()),
    ASCENDC_TPL_ARGS_SEL(REDUCE_TPL_KEY_SEL_ARA_GROUP_NON_CONTIGUOUS()),
    // ARAR nonContiguous
    ASCENDC_TPL_ARGS_SEL(REDUCE_TPL_KEY_SEL_ARAR_NORMAL_NON_CONTIGUOUS()),
    ASCENDC_TPL_ARGS_SEL(REDUCE_TPL_KEY_SEL_ARAR_GROUP_NON_CONTIGUOUS()),
    // ARARARAR nonContiguous
    ASCENDC_TPL_ARGS_SEL(REDUCE_TPL_KEY_SEL_ARARARAR_NORMAL_NON_CONTIGUOUS()),
    ASCENDC_TPL_ARGS_SEL(REDUCE_TPL_KEY_SEL_ARARARAR_GROUP_NON_CONTIGUOUS()),
    // ARARARARA nonContiguous
    ASCENDC_TPL_ARGS_SEL(REDUCE_TPL_KEY_SEL_ARARARARA_NORMAL_NON_CONTIGUOUS()),
    ASCENDC_TPL_ARGS_SEL(REDUCE_TPL_KEY_SEL_ARARARARA_GROUP_NON_CONTIGUOUS()));

#endif