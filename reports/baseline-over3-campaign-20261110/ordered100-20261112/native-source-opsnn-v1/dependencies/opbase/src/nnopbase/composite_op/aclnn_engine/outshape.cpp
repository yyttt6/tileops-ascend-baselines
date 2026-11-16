/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "outshape.h"

namespace op::internal {
static constexpr size_t OP_OUTSHAPE_RANK = 9;
static constexpr size_t OP_OUTSHAPE_COUNT = 2;
aclnnStatus RefreshOutputShape([[maybe_unused]] size_t index, OpArgList& outputShape, OpArgList& outputs)
{
    aclTensor* outputShapeTensor = reinterpret_cast<aclTensor*>(outputShape[0]->pointer);
    void* shapeData = SyncTensorData(outputShapeTensor);
    OP_CHECK(shapeData != nullptr, OP_LOGE(ACLNN_ERR_INNER, "Sync Tensor Data failed."), return ACLNN_ERR_INNER);
    op::DataType dtype = outputShapeTensor->GetDataType();
    size_t dtypeSize = op::TypeSize(dtype);
    OP_LOGI("Output shape tensor dtype: %s, arg count: %zu", op::ToString(dtype).GetString(), outputShape.count);
    if (dtype == op::DataType::DT_FLOAT) {
        for (size_t i = 1; i < outputShape.count; i += OP_OUTSHAPE_COUNT) {
            void* currShape = PtrShift(shapeData, (i / OP_OUTSHAPE_COUNT * OP_OUTSHAPE_RANK * dtypeSize));
            outputs.VisitAt(outputShape[i]->value, [currShape](size_t idx, OpArg& elem) {
                UpdateTensorShape<float>(idx, reinterpret_cast<aclTensor*>(elem->pointer), currShape);
            });
        }
    } else if (dtype == op::DataType::DT_INT32) {
        for (size_t i = 1; i < outputShape.count; i += OP_OUTSHAPE_COUNT) {
            void* currShape = PtrShift(shapeData, (i / OP_OUTSHAPE_COUNT * OP_OUTSHAPE_RANK * dtypeSize));
            outputs.VisitAt(outputShape[i]->value, [currShape](size_t idx, OpArg& elem) {
                UpdateTensorShape<int32_t>(idx, reinterpret_cast<aclTensor*>(elem->pointer), currShape);
            });
        }
    } else if (dtype == op::DataType::DT_INT64) {
        for (size_t i = 1; i < outputShape.count; i += OP_OUTSHAPE_COUNT) {
            void* currShape = PtrShift(shapeData, (i / OP_OUTSHAPE_COUNT * OP_OUTSHAPE_RANK * dtypeSize));
            outputs.VisitAt(outputShape[i]->value, [currShape](size_t idx, OpArg& elem) {
                UpdateTensorShape<int64_t>(idx, reinterpret_cast<aclTensor*>(elem->pointer), currShape);
            });
        }
    } else {
        OP_LOGE(ACLNN_ERR_INNER, "unsupported outshape dtype. %d", dtype);
        BlockPool::Free(shapeData);
        return ACLNN_ERR_INNER;
    }
    BlockPool::Free(shapeData);
    return ACLNN_SUCCESS;
}
} // namespace op::internal