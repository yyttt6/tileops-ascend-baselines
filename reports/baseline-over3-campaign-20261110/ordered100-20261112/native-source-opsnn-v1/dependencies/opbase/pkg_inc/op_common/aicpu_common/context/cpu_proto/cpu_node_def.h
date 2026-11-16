/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AICPU_CONTEXT_INC_NODE_DEF_H_
#define AICPU_CONTEXT_INC_NODE_DEF_H_
#include <memory>
#include <string>
#include <unordered_map>

#include "cpu_attr_value.h"
#include "cpu_tensor.h"

namespace aicpu {
class NodeDefImpl;
class AICPU_VISIBILITY NodeDef {
    friend class CpuKernelUtils;

public:
    NodeDef() = delete;
    ~NodeDef() = default;

    std::shared_ptr<Tensor> MutableInputs(int32_t index) const;

    std::shared_ptr<Tensor> MutableOutputs(int32_t index) const;

    std::unordered_map<std::string, std::shared_ptr<AttrValue> > Attrs() const;

    bool ParseFromString(const std::string& str);

    bool SerializeToString(std::string& str) const;

    void SetOpType(const std::string& op);

    std::string GetOpType() const;

    std::shared_ptr<Tensor> AddInputs();

    std::shared_ptr<Tensor> AddOutputs();

    bool AddAttrs(const std::string& name, const AttrValue* attr);

    int32_t InputsSize() const;

    int32_t OutputsSize() const;

private:
    explicit NodeDef(NodeDefImpl* impl);

    std::shared_ptr<NodeDefImpl> impl_{nullptr};
};
} // namespace aicpu
#endif // AICPU_CONTEXT_INC_NODE_DEF_H_
