/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AICPU_CONTEXT_CPU_PROTO_TENSOR_IMPL_H
#define AICPU_CONTEXT_CPU_PROTO_TENSOR_IMPL_H
#include <functional>
#include <memory>

#include "cpu_tensor_shape.h"
#include "proto/cpu_tensor.pb.h"

namespace aicpu {
class TensorImpl {
    friend class CpuKernelUtils;

public:
    TensorImpl(
        aicpuops::Tensor* tensor,
        std::function<void(aicpuops::Tensor*)> delFunc = [](const aicpuops::Tensor* p) { (void)p; })
        : tensor_(tensor, delFunc)
    {}

    ~TensorImpl() = default;
    TensorImpl(const TensorImpl&) = delete;
    TensorImpl(TensorImpl&&) = delete;
    TensorImpl& operator=(const TensorImpl&) = delete;
    TensorImpl& operator=(TensorImpl&&) = delete;

    /*
     * set tensor shape value to tensor.
     * @param shape: tensor shape value need to set to tensor
     * @return bool: true->success, false->failed
     */
    bool SetTensorShape(const TensorShape* shape);

    /*
     * get tensor shape value of tensor.
     * @return std::shared_ptr<TensorShape>: tensor shape value of tensor
     */
    std::shared_ptr<TensorShape> GetTensorShape() const;

    /*
     * set data type value to tensor.
     * @param type: data type value need to set to tensor
     */
    void SetDataType(DataType type);

    /*
     * get data type value of tensor.
     * @return DataType: data type value of tensor
     */
    DataType GetDataType() const;

    /*
     * set data ptr to tensor.
     * @param addr: tensor data ptr
     */
    void SetData(void* addr);

    /*
     * get data ptr of tensor.
     * @return void *: tensor data ptr
     */
    void* GetData() const;

    /*
     * set data size to tensor.
     * @param size: tensor data size
     */
    void SetDataSize(uint64_t size);

    /*
     * get data size of tensor.
     * @return uint64_t: tensor data size
     */
    uint64_t GetDataSize() const;

    /*
     * get name of tensor.
     * @return std::string: tensor name
     */
    std::string GetName() const;

    /*
     * set name of tensor.
     * @param name: tensor name
     */
    void SetName(const std::string& name);

    /*
     * calculate data size by tensor shape.
     * @return success->not less than 0, failed->less than 0
     */
    int64_t CalcDataSizeByShape() const;

    /*
     * get data elements number.
     * @return success->not less than 0, unknown->less than 0
     */
    int64_t NumElements() const;

    /*
     * get tensor proto.
     */
    aicpuops::Tensor* GetProto() const;

private:
    std::shared_ptr<aicpuops::Tensor> tensor_{nullptr};
};
} // namespace aicpu
#endif // AICPU_CONTEXT_CPU_PROTO_TENSOR_IMPL_H
