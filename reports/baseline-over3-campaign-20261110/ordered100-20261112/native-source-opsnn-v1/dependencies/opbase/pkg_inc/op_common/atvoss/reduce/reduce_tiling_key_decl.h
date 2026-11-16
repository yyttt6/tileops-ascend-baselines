/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file reduce_tiling_key_decl.h
 * \brief reduce tiling key declare
 */

#ifndef _REDUCE_TILING_KEY_DECL_H_
#define _REDUCE_TILING_KEY_DECL_H_
#include "ascendc/host_api/tiling/template_argument.h"

#define A1R0 10
#define A2R0 20
#define A3R0 30
#define A4R0 40
#define A5R0 50
#define A1R2 12
#define A1R3 13
#define A1R4 14
#define A1R5 15
#define A2R3 23
#define A2R4 24
#define A2R5 25
#define A2R6 26
#define A3R4 34
#define A3R5 35
#define A3R6 36
#define A3R7 37
#define A4R5 45
#define A4R6 46
#define A4R7 47
#define A4R8 48
#define A5R6 56
#define A5R7 57
#define A5R8 58
#define A5R9 59
#define AR_PATTERN 11
#define ARA_PATTERN 20
#define ARAR_PATTERN 31
#define ARARA_PATTERN 40
#define ARARAR_PATTERN 51
#define ARARARA_PATTERN 60
#define ARARARAR_PATTERN 71
#define ARARARARA_PATTERN 80

#define BIT_WIDTH 8

#define C0 0
#define C1 1
#define C2 2
#define C3 3
#define C4 4
#define C100 100

#define REDUCE_TPL_PARAM \
    bool IsContiguous, bool BatchInvariant, uint32_t PatternID, uint32_t LoopARCount, uint32_t LoopInnerARCount
#define REDUCE_TPL_VALUE IsContiguous, BatchInvariant, PatternID, LoopARCount, LoopInnerARCount

#define REDUCE_TPL_KEY_DECL()                                                             \
    ASCENDC_TPL_BOOL_DECL(IsContiguous, C1), ASCENDC_TPL_BOOL_DECL(BatchInvariant, C0),   \
        ASCENDC_TPL_UINT_DECL(PatternID, BIT_WIDTH, ASCENDC_TPL_UI_RANGE, 1, C0, C100),   \
        ASCENDC_TPL_UINT_DECL(LoopARCount, BIT_WIDTH, ASCENDC_TPL_UI_RANGE, 1, C0, C100), \
        ASCENDC_TPL_UINT_DECL(LoopInnerARCount, BIT_WIDTH, ASCENDC_TPL_UI_RANGE, 1, C0, C100)

#define REDUCE_TPL_KEY_SEL_EMPTY()                                                                         \
    ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY), ASCENDC_TPL_BOOL_SEL(IsContiguous, C1),             \
        ASCENDC_TPL_BOOL_SEL(BatchInvariant, C0), ASCENDC_TPL_UINT_SEL(PatternID, ASCENDC_TPL_UI_LIST, 0), \
        ASCENDC_TPL_UINT_SEL(LoopARCount, ASCENDC_TPL_UI_LIST, 0),                                         \
        ASCENDC_TPL_UINT_SEL(LoopInnerARCount, ASCENDC_TPL_UI_LIST, 0)

#define REDUCE_TPL_KEY_SEL_A()                                                                                \
    ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY), ASCENDC_TPL_BOOL_SEL(IsContiguous, C1),                \
        ASCENDC_TPL_BOOL_SEL(BatchInvariant, C0), ASCENDC_TPL_UINT_SEL(PatternID, ASCENDC_TPL_UI_LIST, C100), \
        ASCENDC_TPL_UINT_SEL(LoopARCount, ASCENDC_TPL_UI_LIST, A1R0),                                         \
        ASCENDC_TPL_UINT_SEL(LoopInnerARCount, ASCENDC_TPL_UI_LIST, C0)

#define REDUCE_TPL_KEY_SEL_AR_NORMAL()                                                                              \
    ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY), ASCENDC_TPL_BOOL_SEL(IsContiguous, C1),                      \
        ASCENDC_TPL_BOOL_SEL(BatchInvariant, C0), ASCENDC_TPL_UINT_SEL(PatternID, ASCENDC_TPL_UI_LIST, AR_PATTERN), \
        ASCENDC_TPL_UINT_SEL(LoopARCount, ASCENDC_TPL_UI_LIST, A1R0),                                               \
        ASCENDC_TPL_UINT_SEL(LoopInnerARCount, ASCENDC_TPL_UI_RANGE, 1, C0, C1)

#define REDUCE_TPL_KEY_SEL_AR_GROUP()                                                                               \
    ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIV_1_0), ASCENDC_TPL_BOOL_SEL(IsContiguous, C1),                   \
        ASCENDC_TPL_BOOL_SEL(BatchInvariant, C0), ASCENDC_TPL_UINT_SEL(PatternID, ASCENDC_TPL_UI_LIST, AR_PATTERN), \
        ASCENDC_TPL_UINT_SEL(LoopARCount, ASCENDC_TPL_UI_LIST, A1R2),                                               \
        ASCENDC_TPL_UINT_SEL(LoopInnerARCount, ASCENDC_TPL_UI_LIST, C0)

#define REDUCE_TPL_KEY_SEL_ARA_NORMAL()                                                                              \
    ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY), ASCENDC_TPL_BOOL_SEL(IsContiguous, C1),                       \
        ASCENDC_TPL_BOOL_SEL(BatchInvariant, C0), ASCENDC_TPL_UINT_SEL(PatternID, ASCENDC_TPL_UI_LIST, ARA_PATTERN), \
        ASCENDC_TPL_UINT_SEL(LoopARCount, ASCENDC_TPL_UI_LIST, A1R0, A2R0),                                          \
        ASCENDC_TPL_UINT_SEL(LoopInnerARCount, ASCENDC_TPL_UI_RANGE, 1, C0, C1)

#define REDUCE_TPL_KEY_SEL_ARA_GROUP()                                                                               \
    ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIV_1_0), ASCENDC_TPL_BOOL_SEL(IsContiguous, C1),                    \
        ASCENDC_TPL_BOOL_SEL(BatchInvariant, C0), ASCENDC_TPL_UINT_SEL(PatternID, ASCENDC_TPL_UI_LIST, ARA_PATTERN), \
        ASCENDC_TPL_UINT_SEL(LoopARCount, ASCENDC_TPL_UI_LIST, A1R2, A2R3),                                          \
        ASCENDC_TPL_UINT_SEL(LoopInnerARCount, ASCENDC_TPL_UI_LIST, C0)

#define REDUCE_TPL_KEY_SEL_ARAR_NORMAL()                                                                              \
    ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY), ASCENDC_TPL_BOOL_SEL(IsContiguous, C1),                        \
        ASCENDC_TPL_BOOL_SEL(BatchInvariant, C0), ASCENDC_TPL_UINT_SEL(PatternID, ASCENDC_TPL_UI_LIST, ARAR_PATTERN), \
        ASCENDC_TPL_UINT_SEL(LoopARCount, ASCENDC_TPL_UI_LIST, A1R0, A2R0),                                           \
        ASCENDC_TPL_UINT_SEL(LoopInnerARCount, ASCENDC_TPL_UI_RANGE, 1, C0, C2)

#define REDUCE_TPL_KEY_SEL_ARAR_GROUP()                                                                               \
    ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIV_1_0), ASCENDC_TPL_BOOL_SEL(IsContiguous, C1),                     \
        ASCENDC_TPL_BOOL_SEL(BatchInvariant, C0), ASCENDC_TPL_UINT_SEL(PatternID, ASCENDC_TPL_UI_LIST, ARAR_PATTERN), \
        ASCENDC_TPL_UINT_SEL(LoopARCount, ASCENDC_TPL_UI_LIST, A1R2, A1R3, A2R3, A2R4),                               \
        ASCENDC_TPL_UINT_SEL(LoopInnerARCount, ASCENDC_TPL_UI_LIST, C0)

#define REDUCE_TPL_KEY_SEL_ARARARAR_NORMAL()                                                   \
    ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY), ASCENDC_TPL_BOOL_SEL(IsContiguous, C1), \
        ASCENDC_TPL_BOOL_SEL(BatchInvariant, C0),                                              \
        ASCENDC_TPL_UINT_SEL(PatternID, ASCENDC_TPL_UI_LIST, ARARARAR_PATTERN),                \
        ASCENDC_TPL_UINT_SEL(LoopARCount, ASCENDC_TPL_UI_LIST, A1R0, A2R0, A3R0, A4R0),        \
        ASCENDC_TPL_UINT_SEL(LoopInnerARCount, ASCENDC_TPL_UI_RANGE, 1, C0, C4)

#define REDUCE_TPL_KEY_SEL_ARARARAR_GROUP()                                                                           \
    ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIV_1_0), ASCENDC_TPL_BOOL_SEL(IsContiguous, C1),                     \
        ASCENDC_TPL_BOOL_SEL(BatchInvariant, C0),                                                                     \
        ASCENDC_TPL_UINT_SEL(PatternID, ASCENDC_TPL_UI_LIST, ARARARAR_PATTERN),                                       \
        ASCENDC_TPL_UINT_SEL(                                                                                         \
            LoopARCount, ASCENDC_TPL_UI_LIST, A1R2, A1R3, A1R4, A1R5, A2R3, A2R4, A2R5, A2R6, A3R4, A3R5, A3R6, A3R7, \
            A4R5, A4R6, A4R7, A4R8),                                                                                  \
        ASCENDC_TPL_UINT_SEL(LoopInnerARCount, ASCENDC_TPL_UI_LIST, C0)

#define REDUCE_TPL_KEY_SEL_ARARARARA_NORMAL()                                                  \
    ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY), ASCENDC_TPL_BOOL_SEL(IsContiguous, C1), \
        ASCENDC_TPL_BOOL_SEL(BatchInvariant, C0),                                              \
        ASCENDC_TPL_UINT_SEL(PatternID, ASCENDC_TPL_UI_LIST, ARARARARA_PATTERN),               \
        ASCENDC_TPL_UINT_SEL(LoopARCount, ASCENDC_TPL_UI_LIST, A1R0, A2R0, A3R0, A4R0, A5R0),  \
        ASCENDC_TPL_UINT_SEL(LoopInnerARCount, ASCENDC_TPL_UI_RANGE, 1, C0, C4)

#define REDUCE_TPL_KEY_SEL_ARARARARA_GROUP()                                                                          \
    ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIV_1_0), ASCENDC_TPL_BOOL_SEL(IsContiguous, C1),                     \
        ASCENDC_TPL_BOOL_SEL(BatchInvariant, C0),                                                                     \
        ASCENDC_TPL_UINT_SEL(PatternID, ASCENDC_TPL_UI_LIST, ARARARARA_PATTERN),                                      \
        ASCENDC_TPL_UINT_SEL(                                                                                         \
            LoopARCount, ASCENDC_TPL_UI_LIST, A1R2, A1R3, A1R4, A1R5, A2R3, A2R4, A2R5, A2R6, A3R4, A3R5, A3R6, A3R7, \
            A4R5, A4R6, A4R7, A4R8, A5R6, A5R7, A5R8, A5R9),                                                          \
        ASCENDC_TPL_UINT_SEL(LoopInnerARCount, ASCENDC_TPL_UI_LIST, C0)

#endif