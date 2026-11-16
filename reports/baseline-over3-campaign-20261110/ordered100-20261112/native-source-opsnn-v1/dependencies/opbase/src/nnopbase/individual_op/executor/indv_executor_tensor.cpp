/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <set>
#include "indv_executor.h"
#include "indv_tilingcontext_builder.h"
#include "indv_args_pool.h"
#include "indv_cache_key_builder.h"
#include "op_dfx_internal.h"

namespace nnopbase {
static const std::set<ge::Format> FormatsLikeND{ge::FORMAT_NCL, ge::FORMAT_NCDHW, ge::FORMAT_DHWCN, ge::FORMAT_NHWC,
                                                ge::FORMAT_NCHW};

template <typename T, typename S>
void NnopbaseExecutorConvertTensorType(GertTensor* tensor, ge::DataType dataType, std::vector<uint8_t>& scalarValue)
{
    scalarValue.resize(tensor->GetShapeSize() * sizeof(T));
    if (dataType == ge::DataType::DT_FLOAT) {
        if constexpr (std::is_same<bool, typename std::decay<T>::type>::value) {
            for (int64_t i = 0; i < tensor->GetShapeSize(); i++) {
                (op::internal::PtrCastTo<bool>(
                    scalarValue.data()))[i] = (std::abs(op::internal::PtrCastTo<bool>(tensor->MutableTensorData()
                                                                                          .GetAddr())[i]) >=
                                               std::numeric_limits<float>::epsilon());
            }
        } else {
            for (int64_t i = 0; i < tensor->GetShapeSize(); i++) {
                op::internal::PtrCastTo<T>(scalarValue.data())[i] = static_cast<T>(
                    op::internal::PtrCastTo<float>(tensor->MutableTensorData().GetAddr())[i]);
            }
        }
        tensor->MutableTensorData().SetAddr(scalarValue.data(), nullptr);
        return;
    }

    if (dataType == ge::DataType::DT_FLOAT16) {
        if constexpr (std::is_same<bool, typename std::decay<T>::type>::value) {
            for (int64_t i = 0; i < tensor->GetShapeSize(); i++) {
                op::internal::PtrCastTo<bool>(
                    scalarValue
                        .data())[i] = (std::abs(op::fp16_t(op::internal::PtrCastTo<uint16_t>(tensor->MutableTensorData()
                                                                                                 .GetAddr())[i])
                                                    .toFloat()) >= std::numeric_limits<float>::epsilon());
            }
        } else {
            for (int64_t i = 0; i < tensor->GetShapeSize(); i++) {
                op::internal::PtrCastTo<T>(scalarValue.data())[i] = static_cast<T>(
                    op::fp16_t(op::internal::PtrCastTo<uint16_t>(tensor->MutableTensorData().GetAddr())[i]).toDouble());
            }
        }
        tensor->MutableTensorData().SetAddr(scalarValue.data(), nullptr);
        return;
    }

    if constexpr (!(std::is_same<std::complex<float>, T>::value) && !(std::is_same<std::complex<double>, T>::value)) {
        if (dataType == ge::DataType::DT_BF16) {
            for (int64_t i = 0; i < tensor->GetShapeSize(); i++) {
                op::internal::PtrCastTo<T>(scalarValue.data())[i] = static_cast<T>(
                    op::bfloat16(op::internal::PtrCastTo<uint16_t>(tensor->MutableTensorData().GetAddr())[i],
                                 op::bfloat16::from_bits()));
            }
            tensor->MutableTensorData().SetAddr(scalarValue.data(), nullptr);
            return;
        }
    }

    if (dataType == ge::DataType::DT_COMPLEX64) {
        if constexpr (std::is_same<std::complex<double>, typename std::decay<T>::type>::value) {
            for (int64_t i = 0; i < tensor->GetShapeSize(); i++) {
                (op::internal::PtrCastTo<std::complex<double>>(
                    scalarValue.data()))[i] = (op::internal::PtrCastTo<std::complex<float>>(tensor->MutableTensorData()
                                                                                                .GetAddr()))[i];
            }
        }
        tensor->MutableTensorData().SetAddr(scalarValue.data(), nullptr);
        return;
    }

    if (dataType == ge::DataType::DT_COMPLEX128) {
        if constexpr (std::is_same<std::complex<float>, typename std::decay<T>::type>::value) {
            for (int64_t i = 0; i < tensor->GetShapeSize(); i++) {
                (op::internal::PtrCastTo<std::complex<float>>(
                    scalarValue.data()))[i] = (op::internal::PtrCastTo<std::complex<double>>(tensor->MutableTensorData()
                                                                                                 .GetAddr()))[i];
            }
            tensor->MutableTensorData().SetAddr(scalarValue.data(), nullptr);
        }
        return;
    }

    for (int64_t i = 0; i < tensor->GetShapeSize(); i++) {
        op::internal::PtrCastTo<T>(scalarValue.data())[i] = static_cast<T>(
            op::internal::PtrCastTo<S>(tensor->MutableTensorData().GetAddr())[i]);
    }

    tensor->MutableTensorData().SetAddr(scalarValue.data(), nullptr);
}

static std::map<ge::DataType,
                std::map<ge::DataType, std::function<void(GertTensor*, ge::DataType, std::vector<uint8_t>&)>>>
    g_nnopbaseConvertMap{
        {ge::DataType::DT_INT8,
         {{ge::DataType::DT_INT16, NnopbaseExecutorConvertTensorType<int8_t, int16_t>},
          {ge::DataType::DT_INT32, NnopbaseExecutorConvertTensorType<int8_t, int32_t>},
          {ge::DataType::DT_INT64, NnopbaseExecutorConvertTensorType<int8_t, int64_t>},
          {ge::DataType::DT_UINT8, NnopbaseExecutorConvertTensorType<int8_t, uint8_t>},
          {ge::DataType::DT_UINT16, NnopbaseExecutorConvertTensorType<int8_t, uint16_t>},
          {ge::DataType::DT_UINT32, NnopbaseExecutorConvertTensorType<int8_t, uint32_t>},
          {ge::DataType::DT_UINT64, NnopbaseExecutorConvertTensorType<int8_t, uint64_t>},
          {ge::DataType::DT_FLOAT, NnopbaseExecutorConvertTensorType<int8_t, float>},
          {ge::DataType::DT_FLOAT16, NnopbaseExecutorConvertTensorType<int8_t, uint16_t>},
          {ge::DataType::DT_BF16, NnopbaseExecutorConvertTensorType<int8_t, uint16_t>},
          {ge::DataType::DT_BOOL, NnopbaseExecutorConvertTensorType<int8_t, bool>},
          {ge::DataType::DT_DOUBLE, NnopbaseExecutorConvertTensorType<int8_t, double>}}},
        {ge::DataType::DT_INT16,
         {{ge::DataType::DT_INT8, NnopbaseExecutorConvertTensorType<int16_t, int8_t>},
          {ge::DataType::DT_INT32, NnopbaseExecutorConvertTensorType<int16_t, int32_t>},
          {ge::DataType::DT_INT64, NnopbaseExecutorConvertTensorType<int16_t, int64_t>},
          {ge::DataType::DT_UINT8, NnopbaseExecutorConvertTensorType<int16_t, uint8_t>},
          {ge::DataType::DT_UINT16, NnopbaseExecutorConvertTensorType<int16_t, uint16_t>},
          {ge::DataType::DT_UINT32, NnopbaseExecutorConvertTensorType<int16_t, uint32_t>},
          {ge::DataType::DT_UINT64, NnopbaseExecutorConvertTensorType<int16_t, uint64_t>},
          {ge::DataType::DT_FLOAT, NnopbaseExecutorConvertTensorType<int16_t, float>},
          {ge::DataType::DT_FLOAT16, NnopbaseExecutorConvertTensorType<int16_t, uint16_t>},
          {ge::DataType::DT_BF16, NnopbaseExecutorConvertTensorType<int16_t, uint16_t>},
          {ge::DataType::DT_BOOL, NnopbaseExecutorConvertTensorType<int16_t, bool>},
          {ge::DataType::DT_DOUBLE, NnopbaseExecutorConvertTensorType<int16_t, double>}}},
        {ge::DataType::DT_INT32,
         {{ge::DataType::DT_INT8, NnopbaseExecutorConvertTensorType<int32_t, int8_t>},
          {ge::DataType::DT_INT16, NnopbaseExecutorConvertTensorType<int32_t, int16_t>},
          {ge::DataType::DT_INT64, NnopbaseExecutorConvertTensorType<int32_t, int64_t>},
          {ge::DataType::DT_UINT8, NnopbaseExecutorConvertTensorType<int32_t, uint8_t>},
          {ge::DataType::DT_UINT16, NnopbaseExecutorConvertTensorType<int32_t, uint16_t>},
          {ge::DataType::DT_UINT32, NnopbaseExecutorConvertTensorType<int32_t, uint32_t>},
          {ge::DataType::DT_UINT64, NnopbaseExecutorConvertTensorType<int32_t, uint64_t>},
          {ge::DataType::DT_FLOAT, NnopbaseExecutorConvertTensorType<int32_t, float>},
          {ge::DataType::DT_FLOAT16, NnopbaseExecutorConvertTensorType<int32_t, uint16_t>},
          {ge::DataType::DT_BF16, NnopbaseExecutorConvertTensorType<int32_t, uint16_t>},
          {ge::DataType::DT_BOOL, NnopbaseExecutorConvertTensorType<int32_t, bool>},
          {ge::DataType::DT_DOUBLE, NnopbaseExecutorConvertTensorType<int32_t, double>}}},
        {ge::DataType::DT_INT64,
         {{ge::DataType::DT_INT8, NnopbaseExecutorConvertTensorType<int64_t, int8_t>},
          {ge::DataType::DT_INT16, NnopbaseExecutorConvertTensorType<int64_t, int16_t>},
          {ge::DataType::DT_INT32, NnopbaseExecutorConvertTensorType<int64_t, int32_t>},
          {ge::DataType::DT_UINT8, NnopbaseExecutorConvertTensorType<int64_t, uint8_t>},
          {ge::DataType::DT_UINT16, NnopbaseExecutorConvertTensorType<int64_t, uint16_t>},
          {ge::DataType::DT_UINT32, NnopbaseExecutorConvertTensorType<int64_t, uint32_t>},
          {ge::DataType::DT_UINT64, NnopbaseExecutorConvertTensorType<int64_t, uint64_t>},
          {ge::DataType::DT_FLOAT, NnopbaseExecutorConvertTensorType<int64_t, float>},
          {ge::DataType::DT_FLOAT16, NnopbaseExecutorConvertTensorType<int64_t, uint16_t>},
          {ge::DataType::DT_BF16, NnopbaseExecutorConvertTensorType<int64_t, uint16_t>},
          {ge::DataType::DT_BOOL, NnopbaseExecutorConvertTensorType<int64_t, bool>},
          {ge::DataType::DT_DOUBLE, NnopbaseExecutorConvertTensorType<int64_t, double>}}},
        {ge::DataType::DT_UINT8,
         {{ge::DataType::DT_INT8, NnopbaseExecutorConvertTensorType<uint8_t, int8_t>},
          {ge::DataType::DT_INT16, NnopbaseExecutorConvertTensorType<uint8_t, int16_t>},
          {ge::DataType::DT_INT32, NnopbaseExecutorConvertTensorType<uint8_t, int32_t>},
          {ge::DataType::DT_INT64, NnopbaseExecutorConvertTensorType<uint8_t, int64_t>},
          {ge::DataType::DT_UINT16, NnopbaseExecutorConvertTensorType<uint8_t, uint16_t>},
          {ge::DataType::DT_UINT32, NnopbaseExecutorConvertTensorType<uint8_t, uint32_t>},
          {ge::DataType::DT_UINT64, NnopbaseExecutorConvertTensorType<uint8_t, uint64_t>},
          {ge::DataType::DT_FLOAT, NnopbaseExecutorConvertTensorType<uint8_t, float>},
          {ge::DataType::DT_FLOAT16, NnopbaseExecutorConvertTensorType<uint8_t, uint16_t>},
          {ge::DataType::DT_BF16, NnopbaseExecutorConvertTensorType<uint8_t, uint16_t>},
          {ge::DataType::DT_BOOL, NnopbaseExecutorConvertTensorType<uint8_t, bool>},
          {ge::DataType::DT_DOUBLE, NnopbaseExecutorConvertTensorType<uint8_t, double>}}},
        {ge::DataType::DT_UINT16,
         {{ge::DataType::DT_INT8, NnopbaseExecutorConvertTensorType<uint16_t, int8_t>},
          {ge::DataType::DT_INT16, NnopbaseExecutorConvertTensorType<uint16_t, int32_t>},
          {ge::DataType::DT_INT32, NnopbaseExecutorConvertTensorType<uint16_t, int32_t>},
          {ge::DataType::DT_INT64, NnopbaseExecutorConvertTensorType<uint16_t, int64_t>},
          {ge::DataType::DT_UINT8, NnopbaseExecutorConvertTensorType<uint16_t, uint8_t>},
          {ge::DataType::DT_UINT32, NnopbaseExecutorConvertTensorType<uint16_t, uint32_t>},
          {ge::DataType::DT_UINT64, NnopbaseExecutorConvertTensorType<uint16_t, uint64_t>},
          {ge::DataType::DT_FLOAT, NnopbaseExecutorConvertTensorType<uint16_t, float>},
          {ge::DataType::DT_FLOAT16, NnopbaseExecutorConvertTensorType<uint16_t, uint16_t>},
          {ge::DataType::DT_BF16, NnopbaseExecutorConvertTensorType<uint16_t, uint16_t>},
          {ge::DataType::DT_BOOL, NnopbaseExecutorConvertTensorType<uint16_t, bool>},
          {ge::DataType::DT_DOUBLE, NnopbaseExecutorConvertTensorType<uint16_t, double>}}},
        {ge::DataType::DT_UINT32,
         {{ge::DataType::DT_INT8, NnopbaseExecutorConvertTensorType<uint32_t, int8_t>},
          {ge::DataType::DT_INT16, NnopbaseExecutorConvertTensorType<uint32_t, int16_t>},
          {ge::DataType::DT_INT32, NnopbaseExecutorConvertTensorType<uint32_t, int32_t>},
          {ge::DataType::DT_INT64, NnopbaseExecutorConvertTensorType<uint32_t, int64_t>},
          {ge::DataType::DT_UINT8, NnopbaseExecutorConvertTensorType<uint32_t, uint8_t>},
          {ge::DataType::DT_UINT16, NnopbaseExecutorConvertTensorType<uint32_t, uint16_t>},
          {ge::DataType::DT_UINT64, NnopbaseExecutorConvertTensorType<uint32_t, uint64_t>},
          {ge::DataType::DT_FLOAT, NnopbaseExecutorConvertTensorType<uint32_t, float>},
          {ge::DataType::DT_FLOAT16, NnopbaseExecutorConvertTensorType<uint32_t, uint16_t>},
          {ge::DataType::DT_BF16, NnopbaseExecutorConvertTensorType<uint32_t, uint16_t>},
          {ge::DataType::DT_BOOL, NnopbaseExecutorConvertTensorType<uint32_t, bool>},
          {ge::DataType::DT_DOUBLE, NnopbaseExecutorConvertTensorType<uint32_t, double>}}},
        {ge::DataType::DT_UINT64,
         {{ge::DataType::DT_INT8, NnopbaseExecutorConvertTensorType<uint64_t, int8_t>},
          {ge::DataType::DT_INT16, NnopbaseExecutorConvertTensorType<uint64_t, int16_t>},
          {ge::DataType::DT_INT32, NnopbaseExecutorConvertTensorType<uint64_t, int32_t>},
          {ge::DataType::DT_INT64, NnopbaseExecutorConvertTensorType<uint64_t, int64_t>},
          {ge::DataType::DT_UINT8, NnopbaseExecutorConvertTensorType<uint64_t, uint8_t>},
          {ge::DataType::DT_UINT16, NnopbaseExecutorConvertTensorType<uint64_t, uint16_t>},
          {ge::DataType::DT_UINT32, NnopbaseExecutorConvertTensorType<uint64_t, uint32_t>},
          {ge::DataType::DT_FLOAT, NnopbaseExecutorConvertTensorType<uint64_t, float>},
          {ge::DataType::DT_FLOAT16, NnopbaseExecutorConvertTensorType<uint64_t, uint16_t>},
          {ge::DataType::DT_BF16, NnopbaseExecutorConvertTensorType<uint64_t, uint16_t>},
          {ge::DataType::DT_BOOL, NnopbaseExecutorConvertTensorType<uint64_t, bool>},
          {ge::DataType::DT_DOUBLE, NnopbaseExecutorConvertTensorType<uint64_t, double>}}},
        {ge::DataType::DT_FLOAT,
         {{ge::DataType::DT_INT8, NnopbaseExecutorConvertTensorType<float, int8_t>},
          {ge::DataType::DT_INT16, NnopbaseExecutorConvertTensorType<float, int16_t>},
          {ge::DataType::DT_INT32, NnopbaseExecutorConvertTensorType<float, int32_t>},
          {ge::DataType::DT_INT64, NnopbaseExecutorConvertTensorType<float, int64_t>},
          {ge::DataType::DT_UINT8, NnopbaseExecutorConvertTensorType<float, uint8_t>},
          {ge::DataType::DT_UINT16, NnopbaseExecutorConvertTensorType<float, uint16_t>},
          {ge::DataType::DT_UINT32, NnopbaseExecutorConvertTensorType<float, uint32_t>},
          {ge::DataType::DT_UINT64, NnopbaseExecutorConvertTensorType<float, uint64_t>},
          {ge::DataType::DT_FLOAT16, NnopbaseExecutorConvertTensorType<float, uint16_t>},
          {ge::DataType::DT_BF16, NnopbaseExecutorConvertTensorType<float, uint16_t>},
          {ge::DataType::DT_BOOL, NnopbaseExecutorConvertTensorType<float, bool>},
          {ge::DataType::DT_DOUBLE, NnopbaseExecutorConvertTensorType<uint16_t, double>}}},
        {ge::DataType::DT_FLOAT16,
         {{ge::DataType::DT_INT8, NnopbaseExecutorConvertTensorType<uint16_t, int8_t>},
          {ge::DataType::DT_INT16, NnopbaseExecutorConvertTensorType<uint16_t, int16_t>},
          {ge::DataType::DT_INT32, NnopbaseExecutorConvertTensorType<uint16_t, int32_t>},
          {ge::DataType::DT_INT64, NnopbaseExecutorConvertTensorType<uint16_t, int64_t>},
          {ge::DataType::DT_UINT8, NnopbaseExecutorConvertTensorType<uint16_t, uint8_t>},
          {ge::DataType::DT_UINT16, NnopbaseExecutorConvertTensorType<uint16_t, uint16_t>},
          {ge::DataType::DT_UINT32, NnopbaseExecutorConvertTensorType<uint16_t, uint32_t>},
          {ge::DataType::DT_UINT64, NnopbaseExecutorConvertTensorType<uint16_t, uint64_t>},
          {ge::DataType::DT_FLOAT, NnopbaseExecutorConvertTensorType<uint16_t, float>},
          {ge::DataType::DT_BF16, NnopbaseExecutorConvertTensorType<uint16_t, uint16_t>},
          {ge::DataType::DT_BOOL, NnopbaseExecutorConvertTensorType<uint16_t, bool>},
          {ge::DataType::DT_DOUBLE, NnopbaseExecutorConvertTensorType<uint16_t, double>}}},
        {ge::DataType::DT_BF16,
         {{ge::DataType::DT_INT8, NnopbaseExecutorConvertTensorType<uint16_t, int8_t>},
          {ge::DataType::DT_INT16, NnopbaseExecutorConvertTensorType<uint16_t, int16_t>},
          {ge::DataType::DT_INT32, NnopbaseExecutorConvertTensorType<uint16_t, int32_t>},
          {ge::DataType::DT_INT64, NnopbaseExecutorConvertTensorType<uint16_t, int64_t>},
          {ge::DataType::DT_UINT8, NnopbaseExecutorConvertTensorType<uint16_t, uint8_t>},
          {ge::DataType::DT_UINT16, NnopbaseExecutorConvertTensorType<uint16_t, uint16_t>},
          {ge::DataType::DT_UINT32, NnopbaseExecutorConvertTensorType<uint16_t, uint32_t>},
          {ge::DataType::DT_UINT64, NnopbaseExecutorConvertTensorType<uint16_t, uint64_t>},
          {ge::DataType::DT_FLOAT, NnopbaseExecutorConvertTensorType<uint16_t, float>},
          {ge::DataType::DT_FLOAT16, NnopbaseExecutorConvertTensorType<uint16_t, uint16_t>},
          {ge::DataType::DT_BOOL, NnopbaseExecutorConvertTensorType<uint16_t, bool>},
          {ge::DataType::DT_DOUBLE, NnopbaseExecutorConvertTensorType<uint16_t, double>}}},
        {ge::DataType::DT_BOOL,
         {{ge::DataType::DT_INT8, NnopbaseExecutorConvertTensorType<bool, int8_t>},
          {ge::DataType::DT_INT16, NnopbaseExecutorConvertTensorType<bool, int16_t>},
          {ge::DataType::DT_INT32, NnopbaseExecutorConvertTensorType<bool, int32_t>},
          {ge::DataType::DT_INT64, NnopbaseExecutorConvertTensorType<bool, int64_t>},
          {ge::DataType::DT_UINT8, NnopbaseExecutorConvertTensorType<bool, uint8_t>},
          {ge::DataType::DT_UINT16, NnopbaseExecutorConvertTensorType<bool, uint16_t>},
          {ge::DataType::DT_UINT32, NnopbaseExecutorConvertTensorType<bool, uint32_t>},
          {ge::DataType::DT_UINT64, NnopbaseExecutorConvertTensorType<bool, uint64_t>},
          {ge::DataType::DT_FLOAT, NnopbaseExecutorConvertTensorType<bool, float>},
          {ge::DataType::DT_FLOAT16, NnopbaseExecutorConvertTensorType<bool, uint16_t>},
          {ge::DataType::DT_BF16, NnopbaseExecutorConvertTensorType<bool, uint16_t>},
          {ge::DataType::DT_DOUBLE, NnopbaseExecutorConvertTensorType<bool, double>}}},
        {ge::DataType::DT_DOUBLE,
         {{ge::DataType::DT_INT8, NnopbaseExecutorConvertTensorType<double, int8_t>},
          {ge::DataType::DT_INT16, NnopbaseExecutorConvertTensorType<double, int16_t>},
          {ge::DataType::DT_INT32, NnopbaseExecutorConvertTensorType<double, int32_t>},
          {ge::DataType::DT_INT64, NnopbaseExecutorConvertTensorType<double, int64_t>},
          {ge::DataType::DT_UINT8, NnopbaseExecutorConvertTensorType<double, uint8_t>},
          {ge::DataType::DT_UINT16, NnopbaseExecutorConvertTensorType<double, uint16_t>},
          {ge::DataType::DT_UINT32, NnopbaseExecutorConvertTensorType<double, uint32_t>},
          {ge::DataType::DT_UINT64, NnopbaseExecutorConvertTensorType<double, uint64_t>},
          {ge::DataType::DT_FLOAT, NnopbaseExecutorConvertTensorType<double, float>},
          {ge::DataType::DT_FLOAT16, NnopbaseExecutorConvertTensorType<double, uint16_t>},
          {ge::DataType::DT_BF16, NnopbaseExecutorConvertTensorType<double, uint16_t>},
          {ge::DataType::DT_BOOL, NnopbaseExecutorConvertTensorType<double, bool>}}},
        {ge::DataType::DT_COMPLEX64,
         {{ge::DataType::DT_COMPLEX128, NnopbaseExecutorConvertTensorType<std::complex<float>, std::complex<double>>}}},
        {ge::DataType::DT_COMPLEX128,
         {{ge::DataType::DT_COMPLEX64, NnopbaseExecutorConvertTensorType<std::complex<double>, std::complex<float>>}}}};

bool NnopbaseNeedSetFormat(op::Format format, NnopbaseFormatCheckOption formatCheckOption)
{
    return ((formatCheckOption == kNnopbaseDefault) && nnopbase::FormatsLikeND.count(format) > 0U);
}

aclnnStatus NnopbaseSaveTensor(NnopbaseExecutor* executor, const aclTensor* in_tensor, NnopbaseTensor* tensor)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(in_tensor);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(tensor);
    tensor->isNull = false;

    // 当前op::shape和gert::shape是同一个数据结构，后续这个仓全部使用op::shape
    GertTensor* rt2Tensor = &tensor->rt2Tensor;
    NNOPBASE_ASSERT_NOTNULL_RETVAL(rt2Tensor);
    rt2Tensor->MutableOriginShape() = in_tensor->GetViewShape();
    rt2Tensor->MutableStorageShape() = in_tensor->GetViewShape();
    rt2Tensor->SetDataType(in_tensor->GetDataType());
    auto format = in_tensor->GetStorageFormat();
    const bool needSetFormat = NnopbaseNeedSetFormat(format, executor->formatCheckOption);
    if (needSetFormat) {
        rt2Tensor->SetOriginFormat(ge::FORMAT_ND);
        rt2Tensor->SetStorageFormat(ge::FORMAT_ND);
    } else {
        rt2Tensor->SetOriginFormat(format);
        rt2Tensor->SetStorageFormat(format);
    }
    rt2Tensor->MutableTensorData().SetPlacement(in_tensor->GetPlacement());
    rt2Tensor->SetSize(op::CalcShapeBytes(in_tensor->Numel(), in_tensor->GetDataType()));
    NNOPBASE_ASSERT_TRUE_RETVAL(rt2Tensor->MutableTensorData().SetAddr(in_tensor->GetData(), nullptr) ==
                                ge::GRAPH_SUCCESS);
    const auto& viewStrides = in_tensor->GetViewStrides();
    auto& geStride = rt2Tensor->MutableStride();
    geStride.SetDimNum(viewStrides.size());
    for (size_t i = 0U; i < viewStrides.size(); ++i) {
        geStride.SetStride(i, viewStrides[i]);
    }
    rt2Tensor->SetOffset(in_tensor->GetViewOffset());
    return OK;
}
aclnnStatus NnopbaseExecutorSaveTensor(NnopbaseExecutor* executor, NnopbaseTensors* inputs, const aclTensor* in_tensor,
                                       NnopbaseTensor* tensor)
{
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseSaveTensor(executor, in_tensor, tensor));
    if (tensor->rt2Tensor.GetPlacement() != gert::kOnDeviceHbm) {
        inputs->hostInputNum++;
        inputs->hostInputSize += tensor->rt2Tensor.GetSize();
    }
    return OK;
}

aclnnStatus NnopbaseExecutorConvertScalarTensorType(NnopbaseTensors* tensors, ge::DataType dataType,
                                                    const uint32_t index, const int32_t scalarIndex)
{
    auto scalarTensor = &tensors->extTensors[tensors->paramDescs.instances[scalarIndex].startIndex];
    GertTensor* scalarRt2Tensor = &scalarTensor->rt2Tensor;
    const ge::DataType scalarDataType = scalarRt2Tensor->GetDataType();
    if (dataType != scalarDataType) {
        const auto& funcMap = nnopbase::g_nnopbaseConvertMap.find(dataType);
        CHECK_COND((funcMap != nnopbase::g_nnopbaseConvertMap.end()), ACLNN_ERR_PARAM_INVALID,
                   "Not supported data type[%s].", op::ToString(dataType).GetString());
        const auto& func = funcMap->second.find(scalarDataType);
        CHECK_COND((func != funcMap->second.end()), ACLNN_ERR_PARAM_INVALID, "Not supported data type[%s].",
                   op::ToString(scalarDataType).GetString());

        (func->second)(scalarRt2Tensor, scalarDataType, tensors->paramDescs.instances[index].scalarValue);
        scalarRt2Tensor->SetDataType(dataType);
        scalarRt2Tensor->SetSize(static_cast<size_t>(scalarRt2Tensor->GetShapeSize()) * op::TypeSize(dataType));
    }
    return OK;
}
} // namespace nnopbase

void NnopbaseExecutorClearUnContiguousTensors(NnopbaseTensors* tensors)
{
    tensors->unContiguousTensors.tensors.clear();
    tensors->unContiguousTensors.idxs.clear();
    tensors->unContiguousTensors.workspaceOffsets.clear();
    tensors->unContiguousTensors.refUnContTensors.clear();
    tensors->unContiguousTensors.refContTensors.clear();
    tensors->unContiguousTensors.refIdxs.clear();
}

void NnopbaseGetRealIndex(const NnopbaseParamDesc& paramDescs, size_t* realIndex)
{
    for (size_t i = 0U; i < paramDescs.count; i++) {
        if ((paramDescs.instances[i].isDisable) && (i <= *realIndex)) {
            (*realIndex)++;
        }
    }
}

aclnnStatus NnopbaseExecutorAddTensor(NnopbaseExecutor* executor, const aclTensor* tensor, const uint32_t index,
                                      const bool isInput, const bool ignoreCont)
{
    NnopbaseTensors* tensors = isInput ? &(executor->ownArgs.inputs) : &(executor->ownArgs.outputs);
    tensors->paramDescs.instances[index].isInput = isInput;
    bool isEmpty = true;
    if (tensor != nullptr) {
        isEmpty = tensor->IsEmpty();
        const size_t startIndex = static_cast<size_t>(tensors->paramDescs.instances[index].startIndex);
        if (isInput && (!ignoreCont)) {
            NnopbaseCheckContiguous(tensors, tensor, startIndex, index);
        }
        NNOPBASE_ASSERT_OK_RETVAL(
            nnopbase::NnopbaseExecutorSaveTensor(executor, tensors, tensor, &tensors->extTensors[startIndex]));
        tensors->paramDescs.instances[index].num = 1U;
        const int32_t scalarIndex = tensors->paramDescs.instances[index].scalarIndex;
        if (scalarIndex != -1) {
            // 前面添加的scalar输入要和当前输入的dtype保持一致，需要修改scalar转成的gert tensor的dtype和内存排布
            NNOPBASE_ASSERT_OK_RETVAL(
                nnopbase::NnopbaseExecutorConvertScalarTensorType(tensors, tensor->GetDataType(), index, scalarIndex));
        }
    } else {
        tensors->paramDescs.instances[index].num = 0U; // for null option
    }

    if (isEmpty) {
        tensors->paramDescs.emptyNum++;
    }
    if (isInput) {
        OP_LOGI("Executor add input tensors[%u] successfully.", index);
    } else {
        OP_LOGI("Executor add output tensors[%u] successfully.", index);
    }
    return OK;
}

aclnnStatus NnopbaseExecutorUpdateTensorsIndex(NnopbaseTensors* tensors, const uint32_t index)
{
    CHECK_COND(index >= tensors->expectIndex, ACLNN_ERR_PARAM_INVALID,
               "Failed to add tensor[%u], expect index is [%u].", index, tensors->expectIndex);
    CHECK_COND(index < tensors->paramDescs.count, ACLNN_ERR_PARAM_INVALID,
               "Tensor index [%zu] is greater than or equal to IrTensor num: %d", index, tensors->paramDescs.count);
    uint32_t startIndex = index;
    if (tensors->hasDynamic) {
        uint32_t count = 0;
        for (uint32_t i = tensors->expectIndex; i < index; i++) {
            if (tensors->paramDescs.instances[i].cfgNum <= 1) {
                count++;
            }
        }
        tensors->usedNum += count;
        startIndex = tensors->usedNum;
        tensors->usedNum++;
        tensors->paramDescs.instances[index].startIndex = startIndex;
        tensors->extTensors[startIndex].isRequired = (tensors->paramDescs.instances[index].cfgNum > 0) ? true : false;
        tensors->extTensors[startIndex].isOptional = (tensors->paramDescs.instances[index].cfgNum == 0) ? true : false;
        tensors->extTensors[startIndex].valueDepend = false;
    }
    tensors->expectIndex = index + 1U;
    OP_LOGI("Executor Update tensors index[%u] successfully.", index);
    return OK;
}

static aclnnStatus NnopbaseCheckAndSaveTensor(NnopbaseExecutor* executor, const aclTensorList* tensorList,
                                              const size_t startIndex, const uint32_t index,
                                              const NnopbaseTensorIoOptions& options)
{
    NnopbaseTensors* tensors = options.isInput ? &(executor->ownArgs.inputs) : &(executor->ownArgs.outputs);
    bool isEmpty = true;
    bool flag = false;
    ge::DataType dataType = ge::DataType::DT_UNDEFINED;
    for (uint64_t i = 0U; i < tensorList->Size(); i++) {
        if ((*tensorList)[i] != nullptr) {
            if (options.isInput && (!options.ignoreCont)) {
                NnopbaseCheckContiguous(tensors, (*tensorList)[i], startIndex + i, index);
            }
            isEmpty = isEmpty && ((*tensorList)[i]->IsEmpty());
            NNOPBASE_ASSERT_OK_RETVAL(nnopbase::NnopbaseExecutorSaveTensor(executor, tensors, (*tensorList)[i],
                                                                           &tensors->extTensors[startIndex + i]));
            if (!flag) {
                dataType = (*tensorList)[i]->GetDataType();
                flag = true;
            }
        }
        tensors->extTensors[startIndex + i].isRequired = false;
    }
    const int32_t scalarIndex = tensors->paramDescs.instances[index].scalarIndex;
    if (scalarIndex != -1) {
        // 前面添加的scalar输入要和当前输入的dtype保持一致，需要修改scalar转成的gert tensor的dtype和内存排布
        NNOPBASE_ASSERT_OK_RETVAL(
            nnopbase::NnopbaseExecutorConvertScalarTensorType(tensors, dataType, index, scalarIndex));
    }
    if (isEmpty) {
        tensors->paramDescs.emptyNum++;
    }
    return OK;
}

aclnnStatus NnopbaseExecutorAddDynamicTensors(NnopbaseExecutor* executor, const aclTensorList* tensorList,
                                              const uint32_t index, const bool isInput, const bool ignoreCont)
{
    NnopbaseTensors* tensors = isInput ? &(executor->ownArgs.inputs) : &(executor->ownArgs.outputs);
    if ((tensorList == nullptr) || (tensorList->Size() == 0U)) {
        return OK;
    }
    if (isInput) {
        OP_LOGI("Dynamic input[%u] size is %lu", index, tensorList->Size());
    } else {
        OP_LOGI("Dynamic output[%u] size is %lu", index, tensorList->Size());
    }
    CHECK_COND(index >= tensors->expectIndex, ACLNN_ERR_PARAM_INVALID,
               "Failed to add dynamic tensor[%u], expect index is [%u].", index, tensors->expectIndex);
    CHECK_COND(index < tensors->paramDescs.count, ACLNN_ERR_PARAM_INVALID,
               "Tensor index [%zu] is greater than or equal to IrTensor num: %d", index, tensors->paramDescs.count);
    CHECK_COND(tensorList->Size() <= NNOPBASE_DYNAMIC_PARAM_DEF_NUM, ACLNN_ERR_PARAM_INVALID,
               "Size of tensorList in dynamic input[%zu] is %llu, which exceeds limit: %d", index, tensorList->Size(),
               NNOPBASE_DYNAMIC_PARAM_DEF_NUM);

    uint32_t count = 0U;
    for (uint32_t i = tensors->expectIndex; i < index; i++) {
        if (tensors->paramDescs.instances[i].cfgNum <= 1) {
            count++;
        }
    }
    tensors->paramDescs.instances[index].isInput = isInput;
    tensors->usedNum += count;
    const size_t startIndex = tensors->usedNum;
    tensors->paramDescs.instances[index].startIndex = startIndex;
    tensors->paramDescs.instances[index].num = static_cast<uint32_t>(tensorList->Size());
    NNOPBASE_ASSERT_TRUE_RETVAL(!(ge::AddOverflow(tensors->num, tensorList->Size(), tensors->num)));
    NNOPBASE_ASSERT_TRUE_RETVAL(!(ge::AddOverflow(tensors->usedNum, tensorList->Size(), tensors->usedNum)));
    NNOPBASE_ASSERT_TRUE_RETVAL(!(ge::AddOverflow(tensors->dynamicCnt, tensorList->Size(), tensors->dynamicCnt)));
    if (tensors->num >= tensors->arrayLen) {
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseExecutorExtendIoCaches(tensors));
    }
    const NnopbaseTensorIoOptions options = {isInput, ignoreCont};
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseCheckAndSaveTensor(executor, tensorList, startIndex, index, options));
    /* the first tensor of one dynamic tensor group is set as required, it will be used when generate verbose key. */
    tensors->extTensors[startIndex].isRequired = true;
    tensors->expectIndex = index + 1U;
    OP_LOGI("Executor add dynamic tensors[%u] successfully.", index);
    return OK;
}

aclnnStatus NnopbaseExecutorAddAttr(NnopbaseExecutor* executor, const void* const attrAddr, const size_t attrLen,
                                    const size_t index, const size_t elementSize, const NnopbaseAttrDtype dtype)
{
    NnopbaseAttrs& opAttrs = executor->attrs;
    if (index >= opAttrs.num) {
        OP_LOGE_FOR_INVALID_ARGUMENT_INDEX_OUT_OF_RANGE(std::to_string(index).c_str(), "index",
                                                        std::to_string(opAttrs.num).c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    CHECK_COND(attrAddr != nullptr, ACLNN_ERR_INNER_NULLPTR,
               "Failed to add attr[%zu], please check the value of input attr[%zu].", index, index);
    opAttrs.attrs[index].addr.addr = attrAddr;
    opAttrs.attrs[index].addr.size = attrLen;
    opAttrs.attrs[index].addr.elementSize = elementSize;
    opAttrs.attrs[index].addr.isVector = elementSize == 0U ? false : true;
    opAttrs.attrs[index].dtype = dtype;
    opAttrs.totalSize += attrLen;
    opAttrs.totalDataLen += attrLen;
    if (opAttrs.attrs[index].addr.isVector) {
        opAttrs.totalSize += sizeof(gert::ContinuousVector);
    }

    if ((op::internal::PtrCastTo<NnopbaseExecutor>(executor)->matchArgsV2) &&
        (g_nnopbaseSysCfgParams.enableArgsCache) && !op::internal::GetOpProfilingRecordArgFlag()) {
        NnopbaseUChar* key = op::internal::PtrCastTo<NnopbaseUChar>(executor->ownArgs.inputKey.data()) +
                             executor->ownArgs.keyLen;
        // 第0个属性和输入中间用占位符隔开
        if (index == 0) {
            key = Indv::CacheKeyBuilder::AppendPlaceHolder(&executor->ownArgs, key);
        }
        if (executor->ownArgs.remainKeyLen < attrLen) {
            size_t multiples = attrLen / NNOPBASE_MAX_ARGS_KEY_LEN + 1;
            executor->ownArgs.remainKeyLen += multiples * NNOPBASE_MAX_ARGS_KEY_LEN;
            executor->ownArgs.inputKey.resize(executor->ownArgs.inputKey.size() +
                                              multiples * NNOPBASE_MAX_ARGS_KEY_LEN);
            key = op::internal::PtrCastTo<NnopbaseUChar>(executor->ownArgs.inputKey.data() + executor->ownArgs.keyLen);
        }
        key = NnopbaseAppendBinary(key, attrLen, attrAddr, attrLen);
        executor->ownArgs.keyLen += attrLen;
        executor->ownArgs.remainKeyLen -= attrLen;
    }
    return OK;
}
