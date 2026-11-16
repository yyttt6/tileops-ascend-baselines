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
 * \file cann_ops_tensor_types.h
 * \brief ops-tensor 类型定义头文件
 */

#ifndef CANN_OPS_TENSOR_TYPES_H
#define CANN_OPS_TENSOR_TYPES_H

#include <stdint.h>

/*============================================================================
 * 1. 状态码定义
 *============================================================================*/
typedef enum {
    ACLTENSOR_STATUS_SUCCESS = 0,
    ACLTENSOR_STATUS_NOT_INITIALIZED = 1,
    ACLTENSOR_STATUS_ALLOC_FAILED = 3,
    ACLTENSOR_STATUS_INVALID_VALUE = 7,
    ACLTENSOR_STATUS_ARCH_MISMATCH = 8,
    ACLTENSOR_STATUS_EXECUTION_FAILED = 13,
    ACLTENSOR_STATUS_INTERNAL_ERROR = 14,
    ACLTENSOR_STATUS_NOT_SUPPORTED = 15,
    ACLTENSOR_STATUS_INSUFFICIENT_WORKSPACE = 19,
    ACLTENSOR_STATUS_INSUFFICIENT_DRIVER = 20,
    ACLTENSOR_STATUS_IO_ERROR = 21,
} acltensorStatus_t;

/*============================================================================
 * 2. 数据类型枚举
 *============================================================================*/
typedef enum {
    ACLTENSOR_R_16F = 0,     /* FP16 半精度浮点 */
    ACLTENSOR_R_32F = 1,     /* FP32 单精度浮点 - 阶段一支持 */
    ACLTENSOR_R_64F = 2,     /* FP64 双精度浮点 */
    ACLTENSOR_R_16BF = 3,    /* BF16 BFloat16 */
    ACLTENSOR_R_8I = 4,      /* INT8 有符号8位整数 */
    ACLTENSOR_R_8U = 5,      /* UINT8 无符号8位整数 */
    ACLTENSOR_R_32I = 6,     /* INT32 有符号32位整数 */
    ACLTENSOR_R_32U = 7,     /* UINT32 无符号32位整数 */
    ACLTENSOR_C_32F = 8,     /* Complex FP32 */
    ACLTENSOR_C_64F = 9,     /* Complex FP64 */
} acltensorDataType_t;

/*============================================================================
 * 3. 计算精度描述符
 *============================================================================*/
typedef enum {
    ACLTENSOR_COMPUTE_DESC_32F = (1U << 2),   /* FP32 计算 - 阶段一支持，bit[2] */
    ACLTENSOR_COMPUTE_DESC_NONE = 0,
    ACLTENSOR_COMPUTE_DESC_16F = (1U << 0),   /* FP16 计算，bit[0] */
    ACLTENSOR_COMPUTE_DESC_64F = (1U << 4),   /* FP64 计算，bit[4] */
    ACLTENSOR_COMPUTE_DESC_16BF = (1U << 10), /* BF16 计算，bit[10] */
    ACLTENSOR_COMPUTE_DESC_C32F = (1U << 11), /* 复数FP32 计算，bit[11] */
    ACLTENSOR_COMPUTE_DESC_C64F = (1U << 12), /* 复数FP64 计算，bit[12] */
} acltensorComputeDescriptor_t;

/*============================================================================
 * 4. 操作符枚举
 *============================================================================*/
typedef enum {
    /* 一元操作符 */
    ACLTENSOR_OP_IDENTITY = 1,  /* 恒等: y = x - 阶段一支持 */
    ACLTENSOR_OP_SQRT = 2,      /* 平方根 */
    ACLTENSOR_OP_RELU = 8,      /* ReLU */
    ACLTENSOR_OP_CONJ = 9,      /* 共轭 */
    ACLTENSOR_OP_RCP = 10,      /* 倒数 */
    ACLTENSOR_OP_SIGMOID = 11,  /* Sigmoid */
    ACLTENSOR_OP_TANH = 12,     /* Tanh */
    ACLTENSOR_OP_EXP = 22,      /* 指数 */
    ACLTENSOR_OP_LOG = 23,      /* 对数 */
    ACLTENSOR_OP_ABS = 24,      /* 绝对值 */
    ACLTENSOR_OP_NEG = 25,      /* 取负 */
    ACLTENSOR_OP_SIN = 26,      /* 正弦 */
    ACLTENSOR_OP_COS = 27,      /* 余弦 */
    ACLTENSOR_OP_TAN = 28,      /* 正切 */
    ACLTENSOR_OP_SINH = 29,     /* 双曲正弦 */
    ACLTENSOR_OP_COSH = 30,     /* 双曲余弦 */
    ACLTENSOR_OP_ASIN = 31,     /* 反正弦 */
    ACLTENSOR_OP_ACOS = 32,     /* 反余弦 */
    ACLTENSOR_OP_ATAN = 33,     /* 反正切 */
    ACLTENSOR_OP_CEIL = 37,     /* 向上取整 */
    ACLTENSOR_OP_FLOOR = 38,    /* 向下取整 */
    ACLTENSOR_OP_GELU = 39,     /* GELU 激活函数 */
    ACLTENSOR_OP_SILU = 40,     /* SiLU/Swish 激活函数 */

    /* 二元操作符 */
    ACLTENSOR_OP_ADD = 3,       /* 加法: y = a + b - 阶段一支持 */
    ACLTENSOR_OP_MUL = 5,       /* 乘法: y = a * b */
    ACLTENSOR_OP_MAX = 6,       /* 最大值: y = max(a, b) */
    ACLTENSOR_OP_MIN = 7,       /* 最小值: y = min(a, b) */
    ACLTENSOR_OP_SUB = 50,      /* 减法: y = a - b */
    ACLTENSOR_OP_DIV = 51,      /* 除法: y = a / b */
    ACLTENSOR_OP_POW = 52,      /* 幂运算: y = a^b */

    ACLTENSOR_OP_UNKNOWN = 126,
} acltensorOperator_t;

/*============================================================================
 * 5. 算法选择
 *============================================================================*/
typedef enum {
    ACLTENSOR_ALGO_DEFAULT = -1,          /* 启发式自动选择 - 阶段一支持 */
    ACLTENSOR_ALGO_DEFAULT_PATIENT = -6,  /* 更精确但耗时的选择 */
} acltensorAlgo_t;

/*============================================================================
 * 6. 工作空间偏好
 *============================================================================*/
typedef enum {
    ACLTENSOR_WORKSPACE_MIN = 1,
    ACLTENSOR_WORKSPACE_DEFAULT = 2,      /* 阶段一支持 */
    ACLTENSOR_WORKSPACE_MAX = 3,
} acltensorWorksizePreference_t;

/*============================================================================
 * 12. 不透明句柄类型
 *============================================================================*/
typedef struct acltensorHandle* acltensorHandle_t;
typedef struct acltensorTensorDescriptor* acltensorTensorDescriptor_t;
typedef struct acltensorOperationDescriptor* acltensorOperationDescriptor_t;
typedef struct acltensorPlanPreference* acltensorPlanPreference_t;
typedef struct acltensorPlan* acltensorPlan_t;

#endif /* CANN_OPS_TENSOR_TYPES_H */
