/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef NODE_DEF_BUILDER_H
#define NODE_DEF_BUILDER_H
#include <string>
#include "cpu_kernel.h"
#include "status.h"
#include "cpu_kernel_register.h"
#include "aicpu_task_struct.h"
#include "device_cpu_kernel.h"

namespace aicpu {
class NodeDefBuilder {
public:
    struct InputOutputNode {
        std::string node;
        aicpu::DataType d_type;
        std::vector<int64_t> dims;
        void* data;
        aicpu::Format format;
    };

    static std::shared_ptr<NodeDef> CreateNodeDef();

    NodeDefBuilder(NodeDef* nodeDef, std::string name, std::string opName);

    NodeDefBuilder& Input(const InputOutputNode& input);

    NodeDefBuilder& Output(const InputOutputNode& output);

    NodeDefBuilder& Attr(std::string name, int32_t value);

    NodeDefBuilder& Attr(std::string name, int64_t value);

    NodeDefBuilder& Attr(std::string name, float value);

    NodeDefBuilder& Attr(std::string name, double value);

    NodeDefBuilder& Attr(std::string name, bool value);

    NodeDefBuilder& Attr(std::string name, aicpu::DataType value);

    NodeDefBuilder& Attr(std::string name, const std::vector<bool>& value);

    NodeDefBuilder& Attr(std::string name, const std::string& value);

    NodeDefBuilder& Attr(std::string name, const std::vector<std::string>& value);

    NodeDefBuilder& Attr(std::string name, const std::vector<int64_t>& value);

    NodeDefBuilder& Attr(std::string name, const std::vector<std::vector<int64_t>>& value);

    NodeDefBuilder& Attr(std::string name, const std::vector<float>& value);

    NodeDefBuilder& Attr(std::string name, const std::vector<aicpu::DataType>& value);

    NodeDefBuilder& Attr(std::string name, const std::vector<int64_t>& dims, std::string type);

    NodeDefBuilder& Attr(std::string name, const std::vector<std::vector<int64_t>>& shape_lists, std::string type);

    NodeDefBuilder& Attr(std::string name, aicpu::Tensor* tensor);

    NodeDefBuilder& Attr(std::string name, std::vector<aicpu::Tensor*>& tensors);

private:
    void BuildNodeFromInputOutputNode(const InputOutputNode& node, bool isInput);

    NodeDef* node_def_;

    std::string name_;

    std::string op_name_;
};
} // namespace aicpu

#endif
