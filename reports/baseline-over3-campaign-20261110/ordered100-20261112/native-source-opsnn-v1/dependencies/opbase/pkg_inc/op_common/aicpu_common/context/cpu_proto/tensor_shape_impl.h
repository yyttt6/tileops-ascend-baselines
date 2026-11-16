/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AICPU_CONTEXT_CPU_PROTO_TENSOR_SHAPE_IMPL_H
#define AICPU_CONTEXT_CPU_PROTO_TENSOR_SHAPE_IMPL_H
#include <functional>
#include <memory>
#include <vector>

#include "cpu_types.h"
#include "proto/cpu_tensor_shape.pb.h"

namespace aicpu {
class TensorShapeImpl {
    friend class CpuKernelUtils;

public:
    TensorShapeImpl(
        aicpuops::TensorShape* shape,
        std::function<void(aicpuops::TensorShape*)> del_func = [](const aicpuops::TensorShape* p) { (void)p; })
        : tensor_shape_(shape, del_func)
    {}

    ~TensorShapeImpl() = default;
    TensorShapeImpl(const TensorShapeImpl&) = delete;
    TensorShapeImpl(TensorShapeImpl&&) = delete;
    TensorShapeImpl& operator=(const TensorShapeImpl&) = delete;
    TensorShapeImpl& operator=(TensorShapeImpl&&) = delete;

    /*
     * set format value to tensor shape.
     * @param format: format value need to set to tensor shape
     */
    void SetFormat(Format format);

    /*
     * get format value of tensor shape.
     * @return Format: format value of tensor shape
     */
    Format GetFormat() const;

    /*
     * get unknown rank value of tensor shape.
     * @return bool: unknown rank value of tensor shape
     */
    bool GetUnknownRank() const;

    /*
     * set unknown rank value to tensor shape.
     * @param unknown_rank: unknown rank value need to set to tensor shape
     */
    void SetUnknownRank(bool unknown_rank);

    /*
     * set dims value to tensor shape.
     * @param dims: dims value need to set to tensor shape
     */
    void SetDimSizes(const std::vector<int64_t>& dims);

    /*
     * get dims value of tensor shape.
     * @return int32_t: dims value of tensor shape
     */
    std::vector<int64_t> GetDimSizes() const;

    /*
     * get dim value of tensor shape index dim.
     * @param index: index dim of tensor shape
     * @return int64_t: dim value of tensor shape index dim
     */
    int64_t GetDimSize(int32_t index) const;

    /*
     * get dims size of tensor shape.
     * @return int32_t: dims size of tensor shape
     */
    int32_t GetDims() const;

    /*
     * get data elements number.
     * @return success->not less than 0, unknown->less than 0
     */
    int64_t NumElements() const;

    /*
     * get tensor shape proto.
     */
    aicpuops::TensorShape* GetProto() const;

private:
    std::shared_ptr<aicpuops::TensorShape> tensor_shape_{nullptr};
};
} // namespace aicpu
#endif // AICPU_CONTEXT_CPU_PROTO_TENSOR_SHAPE_IMPL_H
