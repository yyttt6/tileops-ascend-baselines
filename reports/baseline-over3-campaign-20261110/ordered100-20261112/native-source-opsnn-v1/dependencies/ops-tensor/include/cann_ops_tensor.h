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
 * \file cann_ops_tensor.h
 * \brief ops-tensor 公共 API 头文件
 */

#ifndef CANN_OPS_TENSOR_H
#define CANN_OPS_TENSOR_H

#include "cann_ops_tensor_types.h"
#include "acl/acl.h"

/* 导出宏定义 */
#if defined(_WIN32) || defined(__CYGWIN__)
    #define ACLTENSOR_API __declspec(dllexport)
#else
    #define ACLTENSOR_API __attribute__((visibility("default")))
#endif

/*============================================================================
 *                        1. 句柄管理 (Handle Management)
 *============================================================================*/

/**
 * @brief Tensor Add operation: C = A + B (element-wise)
 *
 * @param x1    Input tensor A (device pointer, float type)
 * @param x2    Input tensor B (device pointer, float type)
 * @param y     Output tensor (device pointer, float type)
 * @param n     Number of elements in each tensor
 * @param stream ACL stream for execution
 * @return      0 (ACL_SUCCESS) on success, error code otherwise
 */
ACLTENSOR_API acltensorStatus_t acltensorCreate(acltensorHandle_t* handle);

/**
 * @brief 销毁 ops-tensor 库句柄
 * @param[in] handle  要销毁的句柄
 * @return ACLTENSOR_STATUS_SUCCESS 成功
 */
ACLTENSOR_API acltensorStatus_t acltensorDestroy(acltensorHandle_t handle);

/*============================================================================
 *                        2. 张量描述符管理 (Tensor Descriptor)
 *============================================================================*/

/**
 * @brief 创建张量描述符
 * @param[in] handle       库句柄
 * @param[out] desc        返回的张量描述符指针
 * @param[in] rank         张量秩（维度数量）
 * @param[in] dimSizes     各维度长度数组
 * @param[in] stridesIn    各维度步长数组（NULL 表示连续内存）
 * @param[in] dType        数据类型
 * @param[in] alignReq     内存对齐要求（字节）
 * @return ACLTENSOR_STATUS_SUCCESS 成功
 */
ACLTENSOR_API acltensorStatus_t acltensorCreateTensorDescriptor(
    const acltensorHandle_t      handle,
    acltensorTensorDescriptor_t* desc,
    const uint32_t               rank,
    const int64_t                dimSizes[],
    const int64_t                stridesIn[],
    acltensorDataType_t          dType,
    uint32_t                     alignReq);

/**
 * @brief 销毁张量描述符
 * @param[in] desc  要销毁的张量描述符
 * @return ACLTENSOR_STATUS_SUCCESS 成功
 */
ACLTENSOR_API acltensorStatus_t acltensorDestroyTensorDescriptor(acltensorTensorDescriptor_t desc);

/*============================================================================
 *                        3. 操作描述符管理 (Operation Descriptor)
 *============================================================================*/

/**
 * @brief 创建 Elementwise Binary 操作描述符
 *        D = opAC(alpha * opA(A), gamma * opC(C))
 * @param[in] handle       库句柄
 * @param[out] desc       返回的操作描述符指针
 * @param[in] descA       输入张量 A 的描述符
 * @param[in] modeA       输入张量 A 的模式数组
 * @param[in] opA         对张量 A 的操作符
 * @param[in] descC       输入张量 C 的描述符
 * @param[in] modeC       输入张量 C 的模式数组
 * @param[in] opC         对张量 C 的操作符
 * @param[in] descD       输出张量 D 的描述符
 * @param[in] modeD       输出张量 D 的模式数组
 * @param[in] opAC        A 和 C 之间的二元操作符
 * @param[in] descCompute 计算精度描述符
 * @return ACLTENSOR_STATUS_SUCCESS 成功
 */
ACLTENSOR_API acltensorStatus_t acltensorCreateElementwiseBinary(
    const acltensorHandle_t            handle,
    acltensorOperationDescriptor_t*    desc,
    const acltensorTensorDescriptor_t  descA,
    const int32_t                      modeA[],
    acltensorOperator_t                opA,
    const acltensorTensorDescriptor_t  descC,
    const int32_t                      modeC[],
    acltensorOperator_t                opC,
    const acltensorTensorDescriptor_t  descD,
    const int32_t                      modeD[],
    acltensorOperator_t                opAC,
    const acltensorComputeDescriptor_t descCompute);

/**
 * @brief 销毁操作描述符
 * @param[in] desc  要销毁的操作描述符
 * @return ACLTENSOR_STATUS_SUCCESS 成功
 */
ACLTENSOR_API acltensorStatus_t acltensorDestroyOperationDescriptor(acltensorOperationDescriptor_t desc);

/*============================================================================
 *                        4. Plan 和 PlanPreference 管理
 *============================================================================*/

/**
 * @brief 创建 Plan 偏好
 * @param[in] handle  库句柄
 * @param[out] pref   返回的 Plan 偏好指针
 * @param[in] algo    算法选择
 * @return ACLTENSOR_STATUS_SUCCESS 成功
 */
ACLTENSOR_API acltensorStatus_t acltensorCreatePlanPreference(
    const acltensorHandle_t    handle,
    acltensorPlanPreference_t* pref,
    acltensorAlgo_t            algo);

/**
 * @brief 销毁 Plan 偏好
 * @param[in] pref  要销毁的 Plan 偏好
 * @return ACLTENSOR_STATUS_SUCCESS 成功
 */
ACLTENSOR_API acltensorStatus_t acltensorDestroyPlanPreference(acltensorPlanPreference_t pref);

/**
 * @brief 创建 Plan
 * @param[in] handle             库句柄
 * @param[out] plan             返回的 Plan 指针
 * @param[in] desc              操作描述符
 * @param[in] pref              Plan 偏好
 * @param[in] workspaceSizeLimit 工作空间大小限制
 * @return ACLTENSOR_STATUS_SUCCESS 成功
 */
ACLTENSOR_API acltensorStatus_t acltensorCreatePlan(
    const acltensorHandle_t              handle,
    acltensorPlan_t*                     plan,
    const acltensorOperationDescriptor_t desc,
    const acltensorPlanPreference_t      pref,
    uint64_t                             workspaceSizeLimit);

/**
 * @brief 销毁 Plan
 * @param[in] plan  要销毁的 Plan
 * @return ACLTENSOR_STATUS_SUCCESS 成功
 */
ACLTENSOR_API acltensorStatus_t acltensorDestroyPlan(acltensorPlan_t plan);

/*============================================================================
 *                        5. 执行函数 (Execution)
 *============================================================================*/

/**
 * @brief 执行 Elementwise Binary 操作
 *        D = opAC(alpha * opA(A), gamma * opC(C))
 * @param[in] handle  库句柄
 * @param[in] plan    执行计划
 * @param[in] alpha   缩放因子 alpha
 * @param[in] A       输入张量 A 的设备指针
 * @param[in] gamma   缩放因子 gamma
 * @param[in] C       输入张量 C 的设备指针
 * @param[out] D      输出张量 D 的设备指针
 * @param[in] stream  ACL 流
 * @return ACLTENSOR_STATUS_SUCCESS 成功
 */
ACLTENSOR_API acltensorStatus_t acltensorElementwiseBinaryExecute(
    const acltensorHandle_t handle,
    const acltensorPlan_t   plan,
    const void*             alpha,
    const void*             A,
    const void*             gamma,
    const void*             C,
    void*                   D,
    aclrtStream             stream);

/*============================================================================
 *                        6. 辅助工具函数 (Utilities)
 *============================================================================*/

/**
 * @brief 获取错误码对应的错误字符串
 * @param[in] error  错误码
 * @return 错误字符串
 */
ACLTENSOR_API const char* acltensorGetErrorString(const acltensorStatus_t error);

/**
 * @brief 获取库版本号
 * @return 版本号
 */
ACLTENSOR_API size_t acltensorGetVersion(void);

#endif /* CANN_OPS_TENSOR_H */
