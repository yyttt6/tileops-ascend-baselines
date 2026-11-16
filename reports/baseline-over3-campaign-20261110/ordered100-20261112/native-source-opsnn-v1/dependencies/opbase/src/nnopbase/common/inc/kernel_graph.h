/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_OP_API_COMMON_SRC_MEM_MGR_KERNEL_GRAPH_H
#define OP_API_OP_API_COMMON_SRC_MEM_MGR_KERNEL_GRAPH_H

#include "kernel_node.h"
#include "opdev/object.h"

namespace op::mem {
enum class SortType { USER_CUSTOM, DFS, BFS, DFS_WITH_REARRANGEMENT };

bool cmp(const KernelTensor* a, const KernelTensor* b);

class KernelGraph : public op::Object {
public:
    explicit KernelGraph(int64_t graphId);

    ~KernelGraph();

    int64_t GetGraphId() const;

    const op::FVector<KernelNode*, DEFAULT_NODE_NUM>& GetKernelNodes() const;

    aclnnStatus AddKernelNode(KernelNode* kernelNode);

    aclnnStatus SetKernelNodes(op::FVector<KernelNode*, DEFAULT_NODE_NUM>& kernelNodes);

    aclnnStatus TopologicalSortBFS();

    aclnnStatus TopologicalSortDFS();

    const FVector<KernelNode*, DEFAULT_NODE_NUM>& GetSortedNodes();
    const op::FVector<KernelTensor*, DEFAULT_TENSOR_NUM>& GetSortedKernelTensors(
        SortType sortType = SortType::USER_CUSTOM);

private:
    aclnnStatus SortKernelTensor(SortType sortType);

    aclnnStatus JudgeTensorType() const;

    int64_t CalcIndegrees();

    aclnnStatus CalcLifeTimeAndSize();

    size_t PrintTensors(const op::FVector<KernelTensor*, DEFAULT_TENSOR_NUM>& tensors) const;
    size_t PrintGraph();

    int64_t graphId_;
    op::FVector<KernelNode*, DEFAULT_NODE_NUM> kernelNodes_;
    op::FVector<KernelNode*, DEFAULT_NODE_NUM> sortedNodes_;
    op::FVector<KernelTensor*, DEFAULT_TENSOR_NUM> sortedKernelTensors_;
    op::FVector<int64_t, DEFAULT_STACK_SIZE> indegrees_;
    op::FVector<size_t, DEFAULT_STACK_SIZE> stack_;
};
} // namespace op::mem
#endif // OP_API_OP_API_COMMON_SRC_MEM_MGR_KERNEL_GRAPH_H
