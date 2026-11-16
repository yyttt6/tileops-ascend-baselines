/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>

#include "opdev/format_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"

namespace op {

using StrideShapePairs = FVector<std::pair<int64_t, int64_t>, op::MAX_DIM_NUM>;

inline bool IsContiguous(const Shape& shape, const Strides& strides)
{
    int64_t validStride = 1;
    for (int64_t i = static_cast<int64_t>(strides.size()) - 1; i >= 0; --i) {
        if (shape[i] == 1) {
            continue;
        }
        if (validStride != strides[i]) {
            return false;
        }
        validStride *= shape[i];
    }
    return true;
}

bool IsContiguous(const aclTensor* tensor)
{
    CHECK_RET(tensor != nullptr, true);

    if (IsPrivateFormat(tensor->GetStorageFormat())) {
        return true;
    }
    const auto& viewShape = tensor->GetViewShape();
    if (viewShape.GetShapeSize() == 0 || viewShape.GetShapeSize() == 1) {
        return true;
    }
    return IsContiguous(viewShape, tensor->GetViewStrides());
}

bool Validate(const aclTensor* tensor)
{
    auto viewShape = tensor->GetViewShape();
    auto viewStrides = tensor->GetViewStrides();
    auto viewOffset = tensor->GetViewOffset();
    if (viewShape.GetDimNum() != viewStrides.size()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ViewShape and ViewStride mismatch.");
        return false;
    }
    auto storageSize = tensor->GetStorageShape().GetShapeSize();
    int64_t maxViewOffset = viewOffset;
    int64_t minViewOffset = viewOffset;
    for (size_t i = 0; i < viewStrides.size(); i++) {
        maxViewOffset += std::max(static_cast<int64_t>(0), (viewStrides[i] * (viewShape[i] - 1)));
        minViewOffset += std::min(static_cast<int64_t>(0), (viewStrides[i] * (viewShape[i] - 1)));
    }
    if (maxViewOffset + 1 > storageSize || minViewOffset < 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ViewShape overlap.");
        return false;
    }
    return true;
}

inline StrideShapePairs BuildStrideShapePairs(const op::Shape& viewShape, const op::Strides& viewStrides,
                                              bool& mayTranspose, bool& mayBroadcast)
{
    StrideShapePairs strideShapePairs;
    strideShapePairs.reserve(viewStrides.size());
    int64_t lastStride = INT64_MAX;
    for (size_t i = 0; i < viewStrides.size(); i++) {
        if (viewStrides[i] == 0 && viewShape[i] != 1) {
            mayBroadcast = true;
            return strideShapePairs;
        }
        if (viewStrides[i] != 0 && viewShape[i] != 1) {
            strideShapePairs.emplace_back(std::make_pair(viewStrides[i], viewShape[i]));
            if (lastStride < viewStrides[i]) {
                mayTranspose = true;
            }
            lastStride = viewStrides[i];
        }
    }
    return strideShapePairs;
}

inline bool IsContiguous(const StrideShapePairs& strideShapePairs)
{
    int64_t stride = 1;
    for (auto it = strideShapePairs.rbegin(); it != strideShapePairs.rend(); it++) {
        if (it->first != stride) {
            return false;
        }
        stride *= it->second;
    }
    return true;
}

bool CanPickViewAsContiguous(const aclTensor* tensor)
{
    const auto& viewShape = tensor->GetViewShape();
    const auto& viewStrides = tensor->GetViewStrides();
    if (IsContiguous(viewShape, viewStrides)) {
        return true;
    }

    bool mayTranspose = false;
    bool mayBroadcast = false;
    auto strideShapePairs = BuildStrideShapePairs(viewShape, viewStrides, mayTranspose, mayBroadcast);
    if (mayBroadcast) {
        return false;
    }

    if (!mayTranspose) {
        return false;
    }
    std::sort(strideShapePairs.rbegin(), strideShapePairs.rend());
    return IsContiguous(strideShapePairs);
}

bool CanPickViewAsContiguous(std::initializer_list<const aclTensor*> tensorList)
{
    if (tensorList.size() == 0) {
        return true;
    }
    auto firstTensor = *(tensorList.begin());
    for (auto tensor = tensorList.begin() + 1; tensor != tensorList.end(); tensor++) {
        if ((*tensor)->GetViewShape() != firstTensor->GetViewShape() ||
            (*tensor)->GetViewStrides() != firstTensor->GetViewStrides()) {
            return false;
        }
    }
    return CanPickViewAsContiguous(firstTensor);
}
} // namespace op