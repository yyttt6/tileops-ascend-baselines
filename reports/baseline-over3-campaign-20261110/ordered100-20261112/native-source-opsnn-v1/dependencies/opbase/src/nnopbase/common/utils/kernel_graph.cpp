/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_graph.h"
#include <queue>
#include <stack>
#include <unordered_set>
#include "opdev/op_def.h"

namespace op::mem {
namespace {
const std::vector<std::string> TENSOR_TYPE_STR = {"input",      "output",      "workspace", "output_ref",
                                                  "user_input", "user_output", "invalid"};
} // namespace
KernelGraph::KernelGraph(int64_t graphId) : graphId_(graphId) {}

KernelGraph::~KernelGraph()
{
    for (auto& node : kernelNodes_) {
        if (node != nullptr) {
            delete node;
            node = nullptr;
        }
    }
    kernelNodes_.clear();
}

int64_t KernelGraph::GetGraphId() const { return graphId_; }

const op::FVector<KernelNode*, DEFAULT_NODE_NUM>& KernelGraph::GetKernelNodes() const { return kernelNodes_; }

aclnnStatus KernelGraph::AddKernelNode(KernelNode* kernelNode)
{
    OP_LOGD("Add kernel nodes %s with id %zu.", op::OpTypeDict::ToString(kernelNode->GetOpType()).GetString(),
            kernelNodes_.size());
    kernelNode->SetOriginalId(static_cast<int64_t>(kernelNodes_.size()));
    kernelNode->SetTopoId(static_cast<int64_t>(kernelNodes_.size()));
    sortedNodes_.emplace_back(kernelNode);
    kernelNodes_.emplace_back(kernelNode);
    return ACLNN_SUCCESS;
}

aclnnStatus KernelGraph::SetKernelNodes(op::FVector<KernelNode*, DEFAULT_NODE_NUM>& kernelNodes)
{
    for (size_t i = 0; i < kernelNodes.size(); i++) {
        sortedNodes_.emplace_back(kernelNodes[i]);
        sortedNodes_[i]->SetOriginalId(static_cast<int64_t>(i));
        sortedNodes_[i]->SetTopoId(static_cast<int64_t>(i));
    }
    kernelNodes_ = std::move(kernelNodes);
    return ACLNN_SUCCESS;
}

aclnnStatus KernelGraph::TopologicalSortBFS()
{
    std::vector<KernelNode*> sortedNodes;
    std::vector<int64_t> indegrees(kernelNodes_.size(), 0);
    for (auto& node : kernelNodes_) {
        for (auto& tensor : node->GetOutputs()) {
            for (auto& peer : tensor->GetPeerTensors()) {
                auto topoId = peer->GetOwnerNode()->GetTopoId();
                if (topoId >= static_cast<int64_t>(indegrees.size())) {
                    OP_LOGE(ACLNN_ERR_INNER, "peer node's topo id %ld is larger thank the in-degree size %zu.", topoId,
                            indegrees.size());
                    return ACLNN_ERR_INNER;
                }
                indegrees[topoId] += 1;
            }
        }
    }

    std::queue<size_t> q;
    for (size_t i = 0; i < indegrees.size(); ++i) {
        if (indegrees[i] == 0) {
            q.push(i);
        }
    }

    while (!q.empty()) {
        size_t nodeIdx = q.front();
        q.pop();
        auto& node = kernelNodes_[nodeIdx];
        sortedNodes.push_back(node);
        for (auto& tensor : node->GetOutputs()) {
            for (auto& peer : tensor->GetPeerTensors()) {
                int peerIdx = peer->GetOwnerNode()->GetTopoId();
                indegrees[peerIdx] -= 1;
                if (indegrees[peerIdx] == 0) {
                    q.push(peerIdx);
                }
            }
        }
    }

    kernelNodes_.assign(sortedNodes.begin(), sortedNodes.end());
    int64_t kernelId = 0;
    for (auto& node : kernelNodes_) {
        node->SetTopoId(kernelId++);
    }
    return ACLNN_SUCCESS;
}

int64_t KernelGraph::CalcIndegrees()
{
    indegrees_.resize(kernelNodes_.size());
    stack_.resize(kernelNodes_.size());
    for (auto& node : kernelNodes_) {
        for (auto& output : node->GetOutputs()) {
            for (auto& peer : output->GetPeerTensors()) {
                auto peerIdx = peer->GetOwnerNode()->GetOriginalId();
                OP_LOGI("node %ld output , peerIdx is %ld.", node->GetOriginalId(), peerIdx);
                if (peerIdx != node->GetOriginalId()) {
                    indegrees_[peerIdx] += 1;
                } else {
                    output->SetRefTensor(peer);
                }
            }
        }
    }

    int64_t stackIdx = -1;
    for (auto i = static_cast<int32_t>(kernelNodes_.size() - 1); i >= 0; --i) {
        if (indegrees_[i] == 0) {
            stack_[++stackIdx] = i;
        }
    }
    ++stackIdx;
    return stackIdx;
}

aclnnStatus KernelGraph::TopologicalSortDFS()
{
    auto stackIdx = CalcIndegrees();
    size_t realIndex = 0;
    while (stackIdx > 0) {
        size_t nodeIdx = stack_[--stackIdx];
        if (nodeIdx >= kernelNodes_.size()) {
            OP_LOGE(ACLNN_ERR_INNER, "stackIdx %ld, nodeIdx %zu is larger than kernel nodes size %zu.", stackIdx,
                    nodeIdx, kernelNodes_.size());
            return ACLNN_ERR_INNER;
        }
        auto& node = kernelNodes_[nodeIdx];
        sortedNodes_[realIndex] = node;
        node->SetTopoId(static_cast<int64_t>(realIndex)); // this node will not be traversed again.
        ++realIndex;
        for (auto& output : node->GetOutputs()) {
            for (auto& peer : output->GetPeerTensors()) {
                int64_t peerIdx = peer->GetOwnerNode()->GetOriginalId();
                // Reference input does not need to traverse. Otherwise there will be a self cycle.
                if (output->GetRefTensor() != nullptr) {
                    continue;
                }
                --indegrees_[peerIdx];
                if (indegrees_[peerIdx] == 0) {
                    stack_[stackIdx] = (peerIdx);
                    ++stackIdx;
                }
            }
        }
    }
    if (realIndex != kernelNodes_.size()) {
        OP_LOGE(ACLNN_ERR_INNER, "Cycle detected in graph %ld. Real index %zu, expected %zu.", graphId_, realIndex,
                kernelNodes_.size());
        return ACLNN_ERR_INNER;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus KernelGraph::JudgeTensorType() const
{
    for (auto node : kernelNodes_) {
        for (auto output : node->GetOutputs()) {
            OP_CHECK_NOTNULL(output);

            auto aclTensor = output->GetAclTensor();
            OP_CHECK_NOTNULL(aclTensor);
            if (aclTensor->IsFromWorkspace()) {
                output->SetType(TensorType::OUTPUT);
            } else {
                output->SetType(TensorType::USER_OUTPUT);
            }
        }
        for (auto wsp : node->GetWorkspace()) {
            OP_CHECK_NOTNULL(wsp);
            wsp->SetType(TensorType::WORKSPACE);
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus KernelGraph::CalcLifeTimeAndSize()
{
    sortedKernelTensors_.clear();
    for (size_t i = 0; i < kernelNodes_.size(); i++) {
        auto& node = sortedNodes_[i];
        auto topoId = node->GetTopoId();
        for (const auto& workspace : node->GetWorkspace()) {
            workspace->SetLifeTimeStart(topoId);
            workspace->SetLifeTimeEnd(topoId);
            workspace->CalcSize();
            sortedKernelTensors_.push_back(workspace);
        }

        for (auto& output : node->GetOutputs()) {
            if (output == nullptr) {
                OP_LOGE(ACLNN_ERR_INNER, "node %ld is nullptr.", node->GetOriginalId());
                return ACLNN_ERR_INNER;
            }
            CHECK_RET(output->GetAclTensor() != nullptr, ACLNN_ERR_INNER);
            if (!output->GetAclTensor()->IsFromWorkspace()) {
                continue;
            }

            output->SetLifeTimeStart(topoId);
            int64_t lifeTimeEnd = topoId;
            for (const auto& peerIn : output->GetPeerTensors()) {
                OP_CHECK_NOTNULL(peerIn);
                auto peerInOwnerNode = peerIn->GetOwnerNode();
                OP_CHECK_NOTNULL(peerInOwnerNode);
                auto peerInId = peerInOwnerNode->GetTopoId();
                if (peerInId > lifeTimeEnd) {
                    lifeTimeEnd = peerInId;
                }
            }
            output->SetLifeTimeEnd(lifeTimeEnd);

            output->CalcSize();
            sortedKernelTensors_.push_back(output);
        }
    }
    return ACLNN_SUCCESS;
}

bool cmp(const KernelTensor* a, const KernelTensor* b) { return a->GetLifeTimeEnd() > b->GetLifeTimeEnd(); }

static std::string GetChildNodesName(const KernelTensor* output)
{
    std::string childName;
    for (auto peer : output->GetPeerTensors()) {
        if (peer == nullptr) {
            OP_LOGD("peer is null.");
            continue;
        }
        if (peer->GetOwnerNode() == nullptr) {
            OP_LOGD("peer node is null.");
            continue;
        }
        childName += "[";
        childName += op::OpTypeDict::ToString(peer->GetOwnerNode()->GetOpType()).GetString();
        childName += ' ';
        childName += std::to_string(peer->GetOwnerNode()->GetOriginalId());
        childName += "],";
    }
    return childName;
}

size_t KernelGraph::PrintGraph()
{
    size_t nodeSize = 0;
    for (auto& node : kernelNodes_) {
        size_t count = 0;
        for (auto& output : node->GetOutputs()) {
            OP_LOGD("node[%s %ld] output total[%zu], current %zu, is from wsp %d, acltensor: %p, child-nodes are: %s.",
                    op::OpTypeDict::ToString(node->GetOpType()).GetString(), node->GetOriginalId(),
                    node->GetOutputs().size(), count, output->GetAclTensor()->IsFromWorkspace(), output->GetAclTensor(),
                    GetChildNodesName(output).c_str());
            ++count;
        }
        count = 0;
        for (auto& input : node->GetInputs()) {
            OP_LOGD("node[%s %ld],input total [%zu] current %zu , is from wsp %d, acltensor: %p, father-nodes are: %s.",
                    op::OpTypeDict::ToString(node->GetOpType()).GetString(), node->GetOriginalId(),
                    node->GetInputs().size(), count, input->GetAclTensor()->IsFromWorkspace(), input->GetAclTensor(),
                    GetChildNodesName(input).c_str());
            ++count;
        }
        ++nodeSize;
    }
    return nodeSize;
}

aclnnStatus KernelGraph::SortKernelTensor(SortType sortType)
{
    if (kernelNodes_.empty()) {
        return ACLNN_SUCCESS;
    }
    OP_LOGD("Kernel graph before sorting with node size %zu looks like above.", PrintGraph());
    aclnnStatus ret = ACLNN_SUCCESS;
    // To use Dfs, check this case:
    // ./build/op_api/test/ut/l2/op_api_l2_ut  --gtest_filter=l2_scatter_add_test.case_11
    if (sortType == SortType::DFS) {
        ret = TopologicalSortDFS();
        OP_LOGD("Kernel graph with node size %zu looks like above.", PrintGraph());
    } else if (sortType == SortType::BFS) {
        ret = TopologicalSortBFS();
    }

    if (ret != ACLNN_SUCCESS) {
        return ret;
    }
    CalcLifeTimeAndSize();
    return ACLNN_SUCCESS;
}

size_t KernelGraph::PrintTensors(const op::FVector<KernelTensor*, DEFAULT_TENSOR_NUM>& tensors) const
{
    JudgeTensorType();
    size_t tensorSize = 0;
    for (auto& kt : tensors) {
        OP_LOGD("Node[%s],position[%ld], type %s, index %zu, [lifetime Start %ld->lifetime End %ld] memory size %ld.",
                op::OpTypeDict::ToString(kt->GetOwnerNode()->GetOpType()).GetString(),
                kt->GetOwnerNode()->GetOriginalId(), TENSOR_TYPE_STR[static_cast<size_t>(kt->GetType())].c_str(),
                kt->GetIndex(), kt->GetLifeTimeStart(), kt->GetLifeTimeEnd(), kt->GetSize());
        ++tensorSize;
    }

    return tensorSize;
}

const op::FVector<KernelTensor*, DEFAULT_TENSOR_NUM>& KernelGraph::GetSortedKernelTensors(SortType sortType)
{
    if (SortKernelTensor(sortType) != ACLNN_SUCCESS) {
        sortedKernelTensors_.clear();
    }

    OP_LOGD("Tensor info for graph %ld with size %zu are above.", graphId_, PrintTensors(sortedKernelTensors_));
    return sortedKernelTensors_;
}

const FVector<KernelNode*, DEFAULT_NODE_NUM>& KernelGraph::GetSortedNodes() { return sortedNodes_; }
} // namespace op::mem
