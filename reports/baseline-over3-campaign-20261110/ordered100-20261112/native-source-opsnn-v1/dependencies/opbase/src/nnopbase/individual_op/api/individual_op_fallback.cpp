/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "individual_op_api.h"
#include "utils/indv_lib_wrapper.h"
#include "utils/indv_debug_assert.h"
#include "executor/indv_executor.h"

#ifdef __cplusplus
extern "C" {
#endif
constexpr int64_t NNOPBASE_STRIDES_NUM = 2;

// 此接口仅会在第一次调用时调用，性能不敏感
void* NnopbaseGetOpApiFunc(const char* funcName)
{
    std::vector<std::string> basePath;
    NnopbaseGetCustomOpApiPath(basePath);
    NnopbaseGetOppApiPath(basePath);
    return nnopbase::NnopbaseSoLoader::GetInstance().FindFunction(basePath, funcName);
}

aclTensor* NnopbaseConvertTensor(const gert::Tensor* tensor)
{
    OP_CHECK(tensor != nullptr, OP_LOGI("Gert tensor is nullptr."), return nullptr);
    const auto& storageShape = tensor->GetStorageShape();
    op::FVector<int64_t> shape;
    for (size_t i = 0U; i < storageShape.GetDimNum(); ++i) {
        shape.push_back(storageShape.GetDim(i));
    }

    // 计算连续tensor的strides
    op::FVector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - NNOPBASE_STRIDES_NUM; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    return aclCreateTensor(shape.data(), shape.size(), op::ToAclDataType(tensor->GetDataType()), strides.data(), 0,
                           op::ToAclFormat(tensor->GetStorageFormat()), shape.data(), shape.size(),
                           const_cast<void*>(tensor->GetAddr()));
}

aclTensorList* NnopbaseConvertTensorList(std::vector<const gert::Tensor*>& tensorList)
{
    OP_CHECK(tensorList.size() != 0UL, OP_LOGI("TensorList size is 0."), return nullptr);
    op::FVector<aclTensor*> tmp;
    for (size_t i = 0U; i < tensorList.size(); i++) {
        tmp.push_back(NnopbaseConvertTensor(tensorList[i]));
    }

    return aclCreateTensorList(tmp.data(), tmp.size());
}

aclIntArray* NnopbaseCovertIntArray(const gert::Tensor* tensor)
{
    OP_CHECK(tensor != nullptr, OP_LOGI("Gert tensor is nullptr."), return nullptr);
    OP_CHECK(tensor->GetPlacement() == gert::kOnHost,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gert tensor is not host placement."), return nullptr);
    return aclCreateIntArray(static_cast<const int64_t*>(tensor->GetAddr()), tensor->GetStorageShape().GetShapeSize());
}

aclBoolArray* NnopbaseCovertBoolArray(const gert::Tensor* tensor)
{
    OP_CHECK(tensor != nullptr, OP_LOGI("Gert tensor is nullptr."), return nullptr);
    OP_CHECK(tensor->GetPlacement() == gert::kOnHost,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gert tensor is not host placement."), return nullptr);
    return aclCreateBoolArray(static_cast<const bool*>(tensor->GetAddr()), tensor->GetStorageShape().GetShapeSize());
}

aclFloatArray* NnopbaseCovertFloatArray(const gert::Tensor* tensor)
{
    OP_CHECK(tensor != nullptr, OP_LOGI("Gert tensor is nullptr."), return nullptr);
    OP_CHECK(tensor->GetPlacement() == gert::kOnHost,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gert tensor is not host placement."), return nullptr);
    return aclCreateFloatArray(static_cast<const float*>(tensor->GetAddr()), tensor->GetStorageShape().GetShapeSize());
}

aclScalar* NnopbaseConvertScalar(const gert::Tensor* tensor)
{
    OP_CHECK(tensor != nullptr, OP_LOGI("Gert tensor is nullptr."), return nullptr);
    OP_CHECK(tensor->GetPlacement() == gert::kOnHost,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gert tensor is not host placement."), return nullptr);
    return aclCreateScalar(const_cast<void*>(tensor->GetAddr()), op::ToAclDataType(tensor->GetDataType()));
}

aclScalarList* NnopbaseConvertScalarList(const gert::Tensor* tensor)
{
    OP_CHECK(tensor != nullptr, OP_LOGI("Gert tensor is nullptr."), return nullptr);
    OP_CHECK(tensor->GetPlacement() == gert::kOnHost,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gert tensor is not host placement."), return nullptr);
    op::FVector<aclScalar*> tmp;
    aclDataType dataType = op::ToAclDataType(tensor->GetDataType());
    const size_t typeSize = op::TypeSize(tensor->GetDataType());
    void* addr = const_cast<void*>(tensor->GetAddr());
    for (int64_t i = 0; i < tensor->GetStorageShape().GetShapeSize(); i++) {
        aclScalar* scalar = aclCreateScalar(op::internal::PtrCastTo<void>(static_cast<uint8_t*>(addr) + typeSize * i),
                                            dataType);
        tmp.push_back(scalar);
    }
    return aclCreateScalarList(tmp.data(), tmp.size());
}

aclIntArray* NnopbaseCovertIntArrayAttr(const gert::TypedContinuousVector<int64_t>* arr)
{
    OP_CHECK(arr != nullptr, OP_LOGI("int64_t array is nullptr."), return nullptr);
    OP_CHECK(arr->GetSize() != 0U, OP_LOGI("Array size is 0."), return nullptr);
    return aclCreateIntArray(arr->GetData(), arr->GetSize());
}

aclBoolArray* NnopbaseCovertBoolArrayAttr(const gert::TypedContinuousVector<bool>* arr)
{
    OP_CHECK(arr != nullptr, OP_LOGI("bool array is nullptr."), return nullptr);
    OP_CHECK(arr->GetSize() != 0UL, OP_LOGI("Array size is 0."), return nullptr);
    return aclCreateBoolArray(arr->GetData(), arr->GetSize());
}

aclFloatArray* NnopbaseCovertFloatArrayAttr(const gert::TypedContinuousVector<float>* arr)
{
    OP_CHECK(arr != nullptr, OP_LOGI("float array is nullptr."), return nullptr);
    OP_CHECK(arr->GetSize() != 0UL, OP_LOGI("Array size is 0."), return nullptr);
    return aclCreateFloatArray(arr->GetData(), arr->GetSize());
}

void NnopbaseDestroyTensor(const aclTensor* tensor) { aclDestroyTensor(tensor); }

void NnopbaseDestroyTensorList(const aclTensorList* tensorList) { aclDestroyTensorList(tensorList); }

void NnopbaseDestroyScalar(const aclScalar* scalar) { aclDestroyScalar(scalar); }

void NnopbaseDestroyScalarList(const aclScalarList* scalar) { aclDestroyScalarList(scalar); }

void NnopbaseDestroyIntArray(const aclIntArray* array) { aclDestroyIntArray(array); }

void NnopbaseDestroyBoolArray(const aclBoolArray* array) { aclDestroyBoolArray(array); }

void NnopbaseDestroyFloatArray(const aclFloatArray* array) { aclDestroyFloatArray(array); }

#ifdef __cplusplus
}
#endif