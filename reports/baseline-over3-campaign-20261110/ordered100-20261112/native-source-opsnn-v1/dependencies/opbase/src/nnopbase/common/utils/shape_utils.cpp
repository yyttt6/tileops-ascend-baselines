/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "opdev/shape_utils.h"
#include "nnopbase_error_msg.h"
#include <sstream>

namespace op {

op::ShapeVector ToShapeVector(const op::Shape& shape)
{
    op::ShapeVector ret;
    ret.resize(shape.GetDimNum());
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        ret[i] = shape[i];
    }

    return ret;
}

ge::AscendString ToString(const op::Shape& shape)
{
    std::ostringstream oss;
    auto v = ToShapeVector(shape);
    oss << "[";
    if (v.size() > 0) {
        for (size_t i = 0; i < v.size() - 1; ++i) {
            oss << v[i] << ", ";
        }
        oss << v[v.size() - 1];
    }
    oss << "]";
    return ge::AscendString((oss.str()).c_str());
}

ge::AscendString ToString(const op::Strides& strides)
{
    std::ostringstream oss;
    oss << "[";
    if (strides.size() > 0) {
        for (size_t i = 0; i < strides.size() - 1; ++i) {
            oss << strides[i] << ", ";
        }
        oss << strides[strides.size() - 1];
    }
    oss << "]";
    return ge::AscendString((oss.str()).c_str());
}

void ToShape(const int64_t* dims, uint64_t dimNum, op::Shape& shape)
{
    shape.SetDimNum(dimNum);
    for (uint64_t i = 0; i < dimNum; i++) {
        shape.SetDim(i, dims[i]);
    }
}

void ToShape(const op::ShapeVector& shapeVector, op::Shape& shape)
{
    ToShape(shapeVector.data(), shapeVector.size(), shape);
}

void ToContiguousStrides(const op::Shape& shape, op::Strides& strides)
{
    strides.assign(shape.GetDimNum(), 1);
    for (int64_t i = static_cast<int64_t>(shape.GetDimNum() - 2); i >= 0; i--) {
        strides[i] = shape.GetDim(i + 1) * strides[i + 1];
    }
}

static bool BroadcastDim(int64_t& dim1, const int64_t dim2)
{
    if (dim1 == dim2) {
        return true;
    }
    /* column is dim1, row is dim2, matrix value is broadcast(dim1, dim2)
  dim   0     1    d2
  0     0     0    E
  1     0     1    d2
  d1    E     d1   E
  */
    if ((dim1 != 1) && (dim2 != 1)) {
        return false;
    }
    dim1 = (dim1 == 1) ? dim2 : dim1;

    return true;
}

bool CheckBroadcastShape(const op::Shape& self, const op::Shape& other)
{
    OP_LOGD("check broadcast %s and %s!", op::ToString(self).GetString(), ToString(other).GetString());
    size_t selfShapeLen = self.GetDimNum();
    size_t otherShapeLen = other.GetDimNum();
    const auto& largerDimShape = selfShapeLen > otherShapeLen ? self : other;
    const auto& smallerDimShape = &largerDimShape == &self ? other : self;
    auto largerDimNum = largerDimShape.GetDimNum();
    auto smallerDimNum = smallerDimShape.GetDimNum();
    auto lenSub = largerDimNum - smallerDimNum;
    for (size_t i = smallerDimNum; i > 0; i--) {
        int64_t dim1 = largerDimShape.GetDim(lenSub + i - 1);
        int64_t dim2 = smallerDimShape.GetDim(i - 1);
        if (!BroadcastDim(dim1, dim2)) {
            return false;
        }
    }

    return true;
}

bool BroadcastInferShape(const op::Shape& self, const op::Shape& other, op::Shape& broadcastShape)
{
    OP_LOGD("start broadcast %s and %s!", op::ToString(self).GetString(), ToString(other).GetString());
    OP_LOGD("check broadcast %s and %s!", op::ToString(self).GetString(), ToString(other).GetString());
    size_t selfShapeLen = self.GetDimNum();
    size_t otherShapeLen = other.GetDimNum();
    const auto& largerDimShape = selfShapeLen > otherShapeLen ? self : other;
    const auto& smallerDimShape = &largerDimShape == &self ? other : self;
    auto largerDimNum = largerDimShape.GetDimNum();
    auto smallerDimNum = smallerDimShape.GetDimNum();
    auto lenSub = largerDimNum - smallerDimNum;
    broadcastShape.SetDimNum(largerDimNum);
    for (size_t i = smallerDimNum; i > 0; i--) {
        int64_t dim1 = largerDimShape.GetDim(lenSub + i - 1);
        int64_t dim2 = smallerDimShape.GetDim(i - 1);
        if (!BroadcastDim(dim1, dim2)) {
            std::string largerDimShapeStr = std::string(op::ToString(largerDimShape).GetString());
            std::string smallerDimShapeStr = std::string(op::ToString(smallerDimShape).GetString());
            std::string dim1Str = std::to_string(dim1);
            std::string reason = "The tensor whose shape is " + largerDimShapeStr + " and the tensor whose shape is " +
                                 smallerDimShapeStr + " do not meet the broadcast condition";
            OP_LOGE_FOR_INVALID_ARGUMENT_TENSOR_INPUT_SHAPE(largerDimShapeStr.c_str(), dim1Str.c_str(), reason.c_str());
            return false;
        }
        broadcastShape.SetDim(lenSub + i - 1, dim1);
    }
    for (size_t i = 0; i < lenSub; i++) {
        broadcastShape.SetDim(i, largerDimShape.GetDim(i));
    }

    return true;
}

} // namespace op