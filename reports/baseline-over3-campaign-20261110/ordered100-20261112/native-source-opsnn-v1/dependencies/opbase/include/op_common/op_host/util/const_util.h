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
 * \file const_util.h
 * \brief
 */

#ifndef OP_COMMON_OP_HOST_UTIL_CONST_UTIL_H
#define OP_COMMON_OP_HOST_UTIL_CONST_UTIL_H

#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "op_common/log/log.h"

namespace Ops {
namespace Base {
using namespace ge;

template <typename T1, typename T2>
static bool inline GetValueToInteger(const gert::Tensor* constTensor, T2& value)
{
    const T1* constValuePtr = constTensor->GetData<T1>();
    OP_CHECK_IF(constValuePtr == nullptr, OP_LOGE("GetValueToInteger", "const tensor GetData is nullptr."),
                return false);
    OP_CHECK_IF(constTensor->GetSize() <= 0,
                OP_LOGE("GetValueToInteger", "const tensor GetSize is not larger than zero."), return false);
    value = static_cast<T2>(constValuePtr[0]);
    return true;
}

/*
 * @brief: read constvalue from paras store into value
 * @param [in] context: gert::InferShapeContext or gert::TilingContext
 * @param [in] inputIdx: constvalue axes index
 * @param [out] value: integer to store return value.
 * @return bool: flag of success or not
 */
template <typename ContextType, typename T>
bool GetConstInt(const ContextType* context, int64_t inputIdx, T& value)
{
    static_assert(std::is_same<typename std::remove_const<ContextType>::type, gert::InferShapeContext>::value ||
                      std::is_same<typename std::remove_const<ContextType>::type, gert::TilingContext>::value,
                  "ContextType should be gert::InferShapeContext or gert::TilingContext");
    OP_CHECK_IF(context == nullptr, OP_LOGE("GetConstInt", "context is nullptr"), return false);
    const gert::Tensor* constTensor = context->GetInputTensor(inputIdx);
    OP_CHECK_IF(constTensor == nullptr, OP_LOGE("GetConstInt", "the constTensor is nullptr"), return false);

    ge::DataType dtype = constTensor->GetDataType();
    switch (dtype) {
        case ge::DT_UINT64: {
            OP_CHECK_IF(!(GetValueToInteger<uint64_t, T>(constTensor, value)),
                        OP_LOGE("GetConstInt", "Get value uint64_t failed."), return false);
            break;
        }
        case ge::DT_INT64: {
            OP_CHECK_IF(!(GetValueToInteger<int64_t, T>(constTensor, value)),
                        OP_LOGE("GetConstInt", "Get value int64_t failed."), return false);
            break;
        }
        case ge::DT_UINT32: {
            OP_CHECK_IF(!(GetValueToInteger<uint32_t, T>(constTensor, value)),
                        OP_LOGE("GetConstInt", "Get value uint32_t failed."), return false);
            break;
        }
        case ge::DT_INT32: {
            OP_CHECK_IF(!(GetValueToInteger<int32_t, T>(constTensor, value)),
                        OP_LOGE("GetConstInt", "Get value int32_t failed."), return false);
            break;
        }
        default: {
            OP_LOGW(context->GetNodeName(), "GetConstInt only support [int32, int64, uint64, uint32]. but is %s",
                    ToString(dtype).c_str());
            return false;
        }
    }
    OP_LOGI("GetConstInt", "GetConstInt of value is %ld, inputIdx is %ld", static_cast<int64_t>(value), inputIdx);
    return true;
}

template <typename T>
static bool inline GetValueToShape(const gert::Tensor* constTensor, gert::Shape& shape)
{
    const T* constValue = constTensor->GetData<T>();
    OP_CHECK_IF(constValue == nullptr, OP_LOGE("GetValueToShape", "const tensor GetData is nullptr"), return false);
    const size_t constNum = static_cast<size_t>(constTensor->GetShapeSize());
    shape.SetDimNum(0);
    for (size_t i = 0; i < constNum; ++i) {
        shape.AppendDim(constValue[i]);
    }
    return true;
}

template <typename ContextType>
bool GetConstIntToShape(const ContextType* context, int64_t inputIdx, gert::Shape& shape)
{
    static_assert(std::is_same<typename std::remove_const<ContextType>::type, gert::InferShapeContext>::value ||
                      std::is_same<typename std::remove_const<ContextType>::type, gert::TilingContext>::value,
                  "ContextType should be gert::InferShapeContext or gert::TilingContext");
    OP_CHECK_IF(context == nullptr, OP_LOGE("GetConstIntToShape", "context is nullptr"), return false);
    const gert::Tensor* constTensor = context->GetInputTensor(inputIdx);
    OP_CHECK_IF(constTensor == nullptr, OP_LOGE("GetConstIntToShape", "the constTensor is nullptr, return false"),
                return false);

    ge::DataType constDtype = constTensor->GetDataType();
    switch (constDtype) {
        case ge::DT_INT32: {
            OP_CHECK_IF(!(GetValueToShape<int32_t>(constTensor, shape)),
                        OP_LOGE("GetConstIntToShape", "Get value to shape int32_t failed."), return false);
            break;
        }
        case ge::DT_INT64: {
            OP_CHECK_IF(!(GetValueToShape<int64_t>(constTensor, shape)),
                        OP_LOGE("GetConstIntToShape", "Get value to shape int64_t failed."), return false);
            break;
        }
        case ge::DT_UINT64: {
            OP_CHECK_IF(!(GetValueToShape<uint64_t>(constTensor, shape)),
                        OP_LOGE("GetConstIntToShape", "Get value to shape uint64_t failed."), return false);
            break;
        }
        case ge::DT_UINT32: {
            OP_CHECK_IF(!(GetValueToShape<uint32_t>(constTensor, shape)),
                        OP_LOGE("GetConstIntToShape", "Get value to shape uint32_t failed."), return false);
            break;
        }
        default: {
            OP_LOGW(context->GetNodeName(), "GetConstIntToShape only support [int32, int64, uint64, uint32]. but is %s",
                    ToString(constDtype).c_str());
            return false;
        }
    }

    OP_LOGI(context->GetNodeName(), "GetConstIntToShape: output shape is %s", ToString(shape).c_str());
    return true;
}
} // namespace Base
} // namespace Ops
#endif // OP_COMMON_OP_HOST_UTIL_CONST_UTIL_H
