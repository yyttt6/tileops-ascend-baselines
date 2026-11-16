/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <atomic>
#include "kernel_graph.h"
#include "kernel_node.h"
#include "kernel_tensor.h"
#include "opdev/op_def.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "bridge_graph.h"

using namespace op::mem;

namespace op {
namespace internal {

static std::atomic<uint64_t> GRAPH_ID;

void* CreateGraphImpl()
{
    KernelGraph* graph = new KernelGraph(GRAPH_ID++);
    return static_cast<void*>(graph);
}

void FreeGraphImpl(void* graph)
{
    KernelGraph* g = (KernelGraph*)graph;
    delete g;
}

void FreeExtendTensorImpl(void* extendTensor)
{
    aclTensorExtend* e = (aclTensorExtend*)extendTensor;
    delete e;
}

aclnnStatus AddKernelNodeToGraph(void* kn, void* graph)
{
    OP_LOGI("graph %p, kernel node: %p", graph, kn);
    KernelGraph* g = static_cast<KernelGraph*>(graph);
    g->AddKernelNode(static_cast<KernelNode*>(kn));
    return ACL_SUCCESS;
}

void* BuildKernelNodeImpl(uint32_t opType, FVector<aclTensor*>& aclInputs, FVector<aclTensor*>& aclOutputs,
                          FVector<aclTensor*>& aclWorkspace)
{
    // KernelNode， KernelTensor derived from Object, so new will get memory from pool without nullptr.
    KernelNode* kn = new KernelNode(opType);

    for (aclTensor* tensor : aclInputs) {
        KernelTensor* kt = new KernelTensor(tensor, 0);
        kn->AddInput(kt);
    }

    for (aclTensor* tensor : aclOutputs) {
        KernelTensor* out = new KernelTensor(tensor, 0);
        kn->AddOutput(out);
    }

    for (aclTensor* tensor : aclWorkspace) {
        KernelTensor* kt = new KernelTensor(tensor, 0);
        kn->AddWorkspace(kt);
    }

    const FVector<KernelTensor*, BASIC_NUM>& kts = kn->GetInputs();

    for (KernelTensor* kt : kts) {
        const aclTensor* tensor = kt->GetAclTensor();
        if (tensor == nullptr) {
            continue;
        }

        aclTensorExtend* extendTensor = static_cast<aclTensorExtend*>(tensor->GetExtend());
        if (extendTensor == nullptr) {
            continue;
        }

        KernelTensor* ktp = extendTensor->GetKernelTensor();
        if (ktp == nullptr) {
            continue;
        }
        ktp->AddPeerTensor(kt);
        kt->AddPeerTensor(ktp);
    }

    for (auto& out : kn->GetOutputs()) {
        aclTensorExtend* extend = static_cast<aclTensorExtend*>(out->GetAclTensor()->GetExtend());
        if (extend == nullptr) {
            aclTensorExtend* extendTensor = new aclTensorExtend;
            extendTensor->SetKernelTensor(out);
            out->GetAclTensor()->SetExtend(extendTensor);
        } else {
            // 上一个l0算子的输出作为下一个l0算子的输出
            KernelTensor* outPeerTensor = extend->GetKernelTensor();
            if (outPeerTensor != nullptr) {
                OP_LOGW("the output of the previous op is used as the output of current op!!!");
                outPeerTensor->AddPeerTensor(out);
                out->AddPeerTensor(outPeerTensor);
            }
            /* 这里有一个场景会有奇怪的现象：
             * other_op ---> normal_op ---> inplace_op
             *                   \
             *                    ----------> other_op
             * 如果inplace op的输出有更新输入的操作，即ref类型的输出，
             * 那么在创建normal_op输出的时候这个其storage对应的extend里面的KernelTensor是normal_op输出的
             * KernelTensor， 在创建inplace_op输出的时候这个storage对应的extend里面的KernelTensor会被更新
             * 成inplace_op输出对应的KernelTensor。那么在创建other_op并做AddPeerTensor的时候，会将inplace_op
             * 输出的KernelTensor连接到other_op的输入的KernelTensor上。
             * 目前不确定是否会有这样的场景，如果有的话，似乎把inplace_op作为other_op的一个输入，那么
             * normal_op的输出内存会在inplace_op使用完就有可能被复用，同时inplace_op的输出也会被额外分配一个内存。*/
            extend->SetKernelTensor(out);
        }
    }
    return kn;
}

static void TraitsAclTensor(FVector<aclTensor*>& result, OpArgList& opArgList)
{
    opArgList.VisitByNoReturn([&result]([[maybe_unused]] size_t idx, OpArg& arg) {
        if (arg.type == OpArgType::OPARG_ACLTENSOR) {
            AddToList(result, PtrCastTo<aclTensor>(arg->pointer));
        } else if (arg.type == OpArgType::OPARG_ACLTENSOR_LIST) {
            AddToList(result, PtrCastTo<aclTensorList>(arg->pointer));
        }
    });
}

aclnnStatus BuildGraph(void* graph, uint32_t opType, OpArgList& inputs, OpArgList& outputs, OpArgList& workspace)
{
    FVector<aclTensor*> inputsList;
    FVector<aclTensor*> outputsList;
    FVector<aclTensor*> workspaceList;

    TraitsAclTensor(inputsList, inputs);
    TraitsAclTensor(outputsList, outputs);
    TraitsAclTensor(workspaceList, workspace);

    void* kn = BuildKernelNodeImpl(opType, inputsList, outputsList, workspaceList);
    if (kn != nullptr) {
        AddKernelNodeToGraph(kn, graph);
    }
    return ACL_SUCCESS;
}

aclnnStatus BuildGraph(void* graph, uint32_t opType, OpArgList& inputs, OpArgList& outputs, OpArgList& workspace,
                       OpArgList& outputshape)
{
    FVector<aclTensor*> inputsList;
    FVector<aclTensor*> outputsList;
    FVector<aclTensor*> workspaceList;

    TraitsAclTensor(inputsList, inputs);
    TraitsAclTensor(outputsList, outputs);
    TraitsAclTensor(workspaceList, workspace);
    TraitsAclTensor(workspaceList, outputshape);

    void* kn = BuildKernelNodeImpl(opType, inputsList, outputsList, workspaceList);
    if (kn != nullptr) {
        AddKernelNodeToGraph(kn, graph);
    }
    return ACL_SUCCESS;
}

} // namespace internal
} // namespace op
