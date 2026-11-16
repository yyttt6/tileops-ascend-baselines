/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "node_def_builder.h"
#include "cpu_kernel_utils.h"

using namespace std;

namespace aicpu {
std::shared_ptr<NodeDef> NodeDefBuilder::CreateNodeDef() { return CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); }

NodeDefBuilder::NodeDefBuilder(NodeDef* nodeDef, std::string name, std::string opName)
{
    node_def_ = nodeDef;
    name_ = name;
    node_def_->SetOpType(opName);
}

void NodeDefBuilder::BuildNodeFromInputOutputNode(const InputOutputNode& node, bool isInput)
{
    std::shared_ptr<Tensor> tensor;
    if (isInput) {
        tensor = node_def_->AddInputs();
    } else {
        tensor = node_def_->AddOutputs();
    }
    aicpu::CpuKernelUtils::SetTensorName(node.node, tensor);
    tensor->SetDataType(node.d_type);
    auto shape = tensor->GetTensorShape();
    shape->SetDimSizes(node.dims);
    shape->SetFormat(node.format);
    int64_t dataSize = 1;
    for (size_t i = 0; i < node.dims.size(); i++) {
        dataSize = dataSize * node.dims[i];
    }
    dataSize = dataSize * GetSizeByDataType(node.d_type);
    if (node.dims.empty()) {
        dataSize = GetSizeByDataType(node.d_type);
    }
    if (node.data == nullptr) {
        dataSize = 0;
    }
    tensor->SetDataSize(static_cast<uint64_t>(dataSize));
    tensor->SetData(node.data);
}

NodeDefBuilder& NodeDefBuilder::Input(const InputOutputNode& input)
{
    BuildNodeFromInputOutputNode(input, true);
    return *this;
}

NodeDefBuilder& NodeDefBuilder::Output(const InputOutputNode& output)
{
    BuildNodeFromInputOutputNode(output, false);
    return *this;
}

NodeDefBuilder& NodeDefBuilder::Attr(std::string name, int32_t value)
{
    auto attr = CpuKernelUtils::CreateAttrValue();
    attr->SetInt(value);
    (void)node_def_->AddAttrs(name, attr.get());
    return *this;
}

NodeDefBuilder& NodeDefBuilder::Attr(std::string name, int64_t value)
{
    auto attr = CpuKernelUtils::CreateAttrValue();
    attr->SetInt(value);
    node_def_->AddAttrs(name, attr.get());
    return *this;
}

NodeDefBuilder& NodeDefBuilder::Attr(std::string name, float value)
{
    auto attr = CpuKernelUtils::CreateAttrValue();
    attr->SetFloat(value);
    node_def_->AddAttrs(name, attr.get());
    return *this;
}

NodeDefBuilder& NodeDefBuilder::Attr(std::string name, double value)
{
    auto attr = CpuKernelUtils::CreateAttrValue();
    attr->SetFloat(static_cast<float>(value));
    node_def_->AddAttrs(name, attr.get());
    return *this;
}

NodeDefBuilder& NodeDefBuilder::Attr(std::string name, bool value)
{
    auto attr = CpuKernelUtils::CreateAttrValue();
    attr->SetBool(value);
    node_def_->AddAttrs(name, attr.get());
    return *this;
}

NodeDefBuilder& NodeDefBuilder::Attr(std::string name, aicpu::DataType value)
{
    auto attr = CpuKernelUtils::CreateAttrValue();
    attr->SetDataType(value);
    node_def_->AddAttrs(name, attr.get());
    return *this;
}

NodeDefBuilder& NodeDefBuilder::Attr(std::string name, const std::vector<bool>& value)
{
    auto attr = CpuKernelUtils::CreateAttrValue();
    attr->SetListBool(value);
    node_def_->AddAttrs(name, attr.get());
    return *this;
}

NodeDefBuilder& NodeDefBuilder::Attr(std::string name, const std::string& value)
{
    auto attr = CpuKernelUtils::CreateAttrValue();
    attr->SetString(value);
    node_def_->AddAttrs(name, attr.get());
    return *this;
}

NodeDefBuilder& NodeDefBuilder::Attr(std::string name, const std::vector<std::string>& value)
{
    auto attr = CpuKernelUtils::CreateAttrValue();
    attr->SetListString(value);
    node_def_->AddAttrs(name, attr.get());
    return *this;
}

NodeDefBuilder& NodeDefBuilder::Attr(std::string name, const std::vector<int64_t>& value)
{
    auto attr = CpuKernelUtils::CreateAttrValue();
    attr->SetListInt(value);
    node_def_->AddAttrs(name, attr.get());
    return *this;
}

NodeDefBuilder& NodeDefBuilder::Attr(std::string name, const std::vector<std::vector<int64_t>>& value)
{
    auto attr = CpuKernelUtils::CreateAttrValue();
    attr->SetListListInt(value);
    node_def_->AddAttrs(name, attr.get());
    return *this;
}

NodeDefBuilder& NodeDefBuilder::Attr(std::string name, const std::vector<float>& value)
{
    auto attr = CpuKernelUtils::CreateAttrValue();
    attr->SetListFloat(value);
    node_def_->AddAttrs(name, attr.get());
    return *this;
}

NodeDefBuilder& NodeDefBuilder::Attr(std::string name, const std::vector<aicpu::DataType>& value)
{
    auto attr = CpuKernelUtils::CreateAttrValue();
    attr->SetListDataType(value);
    node_def_->AddAttrs(name, attr.get());
    return *this;
}

NodeDefBuilder& NodeDefBuilder::Attr(std::string name, const std::vector<int64_t>& dims, std::string type)
{
    if (type == "shape") {
        auto shape = CpuKernelUtils::CreateAttrValue();
        auto value = CpuKernelUtils::CreateTensorShape();
        value->SetDimSizes(dims);
        (void)node_def_->AddAttrs(name, shape.get());
        (void)shape->SetTensorShape(value.get());
    }
    return *this;
}

NodeDefBuilder& NodeDefBuilder::Attr(std::string name, const std::vector<std::vector<int64_t>>& shape_lists,
                                     std::string type)
{
    if (type == "shape_list") {
        auto shapeItems = CpuKernelUtils::CreateAttrValue();
        for (size_t i = 0; i < shape_lists.size(); i++) {
            auto value = shapeItems->AddListTensorShape();
            value->SetDimSizes(shape_lists[i]);
        }
        (void)node_def_->AddAttrs(name, shapeItems.get());
    }
    return *this;
}

NodeDefBuilder& NodeDefBuilder::Attr(std::string name, aicpu::Tensor* tensor)
{
    auto attr = CpuKernelUtils::CreateAttrValue();
    (void)attr->SetTensor(tensor);
    (void)node_def_->AddAttrs(name, attr.get());
    return *this;
}

NodeDefBuilder& NodeDefBuilder::Attr(std::string name, std::vector<aicpu::Tensor*>& tensors)
{
    auto attr = CpuKernelUtils::CreateAttrValue();
    (void)attr->SetListTensor(tensors);
    (void)node_def_->AddAttrs(name, attr.get());
    return *this;
}
} // namespace aicpu
