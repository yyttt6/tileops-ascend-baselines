/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _AICPU_AICPU_DEVICE_CPU_KERNELS_UTILS_BCAST_H_
#define _AICPU_AICPU_DEVICE_CPU_KERNELS_UTILS_BCAST_H_

#include "cpu_context.h"

namespace aicpu {
// broadcast shape type
// 1. SAME_SHAPE : x and y have the same shape
// 2. X_ONE : x has only one element
// 3. Y_ONE : y has only one element
enum class BcastShapeType {
    SAME_SHAPE = 0,
    X_ONE_ELEMENT = 1,
    Y_ONE_ELEMENT = 2,
    DIFF_SHAPE = 3,
};

struct BCalcInfo {
    BCalcInfo() : input_0(nullptr), input_1(nullptr), output(nullptr) {}
    Tensor* input_0;
    Tensor* input_1;
    Tensor* output;
    std::vector<int64_t> reshape_0;
    std::vector<int64_t> reshape_1;
    std::vector<int64_t> shape_out;
    std::vector<int64_t> bcast_0;
    std::vector<int64_t> bcast_1;
    std::vector<int64_t> x_indexes;
    std::vector<int64_t> y_indexes;
};

class Bcast {
public:
    Bcast() : valid_(true) {};
    Bcast(std::vector<int64_t>& x_shape, std::vector<int64_t>& y_shape);
    ~Bcast() = default;

    uint32_t GenerateBcastInfo(const BCalcInfo& calc_info);
    void GetBcastVec(BCalcInfo& calc_info);
    void BCastIndexes(std::vector<int64_t>& x_indexes, std::vector<int64_t>& y_indexes);
    int64_t GetBroadcastXIndex(int64_t index) const;
    int64_t GetBroadcastYIndex(int64_t index) const;
    bool IsValid() const { return valid_; }
    const std::vector<int64_t>& XReshape() const { return x_reshape_; }
    const std::vector<int64_t>& YReshape() const { return y_reshape_; }
    const std::vector<int64_t>& ResultShape() const { return result_shape_; }
    const std::vector<int64_t>& XBcast() const { return x_bcast_; }
    const std::vector<int64_t>& YBcast() const { return y_bcast_; }

private:
    uint32_t Init(const std::vector<int64_t>& x, const std::vector<int64_t>& y);

    bool valid_;
    std::vector<int64_t> x_reshape_;
    std::vector<int64_t> y_reshape_;
    std::vector<int64_t> shape_out_;
    std::vector<int64_t> x_bcast_;
    std::vector<int64_t> y_bcast_;
    std::vector<int64_t> result_shape_;
    std::vector<int64_t> x_input_strides_;
    std::vector<int64_t> y_input_strides_;
    std::vector<int64_t> x_output_strides_;
    std::vector<int64_t> y_output_strides_;
};
} // namespace aicpu
#endif // _AICPU_AICPU_DEVICE_CPU_KERNELS_UTILS_BCAST_H_
