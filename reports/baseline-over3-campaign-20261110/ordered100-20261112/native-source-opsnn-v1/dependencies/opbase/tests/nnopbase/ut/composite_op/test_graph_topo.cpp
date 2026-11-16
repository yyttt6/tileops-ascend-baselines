/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include "acl/acl.h"
#include "kernel_graph_utils.h"
#include "memory_allocator.h"
#include "opdev/make_op_executor.h"
#include "thread_local_context.h"
using namespace std;
using namespace std::chrono;
using namespace op::mem;
const std::map<TensorType, string> tensor_type_map{
    {TensorType::INPUT, "input"},           {TensorType::OUTPUT, "output"},
    {TensorType::WORKSPACE, "workspace"},   {TensorType::OUTPUT_REF, "output_ref"},
    {TensorType::USER_INPUT, "user_input"}, {TensorType::USER_OUTPUT, "user_output"}};

struct TensorResult {
    KernelTensor* kt;
    int64_t expected_start;
    int64_t expected_end;
};

void PrintSingleTensor(KernelTensor* kt, int64_t start, int64_t end)
{
    string type;
    auto iter = tensor_type_map.find(kt->GetType());
    if (iter != tensor_type_map.end()) {
        type = iter->second;
    }
    printf("Node %ld, index %zu, type %s [Start %ld -> End %ld].\n", kt->GetOwnerNode()->GetOriginalId(),
           kt->GetIndex(), type.c_str(), start, end);
}

void PrintTensors(const char* name, const op::FVector<TensorResult, DEFAULT_TENSOR_NUM>& tensor_results)
{
    printf("\n----------------------------%s---------------------------\n", name);

    for (auto t_result : tensor_results) {
        PrintSingleTensor(t_result.kt, t_result.expected_start, t_result.expected_end);
    }
}

void PrintTensors(const char* name, const op::FVector<KernelTensor*, DEFAULT_TENSOR_NUM>& tensors)
{
    printf("\n----------------------------%s---------------------------\n", name);
    for (auto tensor : tensors) {
        PrintSingleTensor(tensor, tensor->GetLifeTimeStart(), tensor->GetLifeTimeEnd());
    }
}

KernelNode* AddNode(int64_t original_id, size_t input_size, size_t output_size, size_t workspace_size = 0,
                    bool is_output_from_workspace = true)
{
    static auto ue = CREATE_EXECUTOR();
    KernelNode* node = new KernelNode(0, original_id);

    node->SetTopoId(original_id);
    gert::Shape shape({5, 6, 10, 20});
    for (size_t i = 0; i < input_size; i++) {
        auto* t = ue->AllocTensor(shape, shape, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
        KernelTensor* kt_in = new KernelTensor(t, i);
        kt_in->SetOwnerNode(node);
        node->AddInput(kt_in);
    }

    for (size_t i = 0; i < output_size; i++) {
        auto* t = ue->AllocTensor(shape, shape, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
        KernelTensor* kt_out = new KernelTensor(t, i);
        kt_out->SetOwnerNode(node);
        t->SetFromWorkspace(is_output_from_workspace);

        node->AddOutput(kt_out);
    }

    for (size_t i = 0; i < workspace_size; i++) {
        auto* t = ue->AllocTensor(shape, shape, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
        KernelTensor* kt_wsp = new KernelTensor(t, i);
        kt_wsp->SetOwnerNode(node);
        node->AddWorkspace(kt_wsp);
    }

    return node;
}

int SetSize(KernelNode* node, size_t index, size_t size)
{
    node->GetOutput(index)->SetSize(size);
    for (auto peer : node->GetOutput(index)->GetPeerTensors()) {
        peer->SetSize(size);
    }
    return ACLNN_SUCCESS;
}

uint64_t GetWorkspaceSize(MemoryAllocator& allocator, const op::FVector<KernelTensor*, DEFAULT_TENSOR_NUM>& tensors)
{
    return allocator.Allocate(tensors);
}

class memory_allocator_ut : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::internal::GetThreadLocalContext().cacheHasFull_ = true;
        std::cout << "memory_allocator_ut SetUp" << std::endl;
    }

    static void TearDownTestCase() { std::cout << "memory_allocator_ut TearDown" << std::endl; }
};

void Release(KernelGraph* graph) { delete graph; }

TEST_F(memory_allocator_ut, TestCase1)
{
    // kernel node construction
    KernelNode* a = AddNode(0, 1, 1);
    KernelNode* b = AddNode(1, 1, 1);
    KernelNode* c = AddNode(2, 1, 1);
    KernelNode* d = AddNode(3, 2 /* d has 2 inputs*/, 1);
    KernelNode* e = AddNode(4, 1, 1, 0, false /* e is from workspace; do not allocate memory. */);

    // Set kernel size [992, 2048, 2048, 512, 4096]
    //                  a    b     c     d    e
    SetSize(a, 0, 992);
    SetSize(b, 0, 2048);
    SetSize(c, 0, 2048);
    SetSize(d, 0, 512);
    SetSize(e, 0, 4096);

    // kernel graph construction
    // a--->b--->c--->d--->e
    //       \-------/
    KernelGraphUtils::Link(a, b, 0, 0);
    KernelGraphUtils::Link(b, c, 0, 0);
    KernelGraphUtils::Link(c, d, 0, 0);

    KernelGraphUtils::Link(b, d, 0, 1);
    KernelGraphUtils::Link(d, e, 0, 0);

    KernelGraph* kernel_graph = new KernelGraph(0);

    op::FVector<KernelNode*, DEFAULT_NODE_NUM> kernel_nodes;
    kernel_nodes = {a, b, c, d, e};
    kernel_graph->SetKernelNodes(kernel_nodes);
    // kernel_graph->TopologicalSortDFS();
    //  TODO
    std::vector<KernelTensor*> tensors;
    for (auto node : kernel_graph->GetKernelNodes()) {
        for (auto output : node->GetOutputs()) {
            tensors.push_back(output);
            printf("node %zu : output number %zu, peer_in number %zu, output tensor size %zu\n", node->GetTopoId(),
                   node->GetOutputs().size(), output->GetPeerTensors().size(), output->GetSize());
        }
    }

    auto& sorted_knl_tsr = kernel_graph->GetSortedKernelTensors();
    for (auto& tensor : sorted_knl_tsr) {
        cout << tensor->GetSize() << endl;
    }

    Release(kernel_graph);
}

KernelGraph* CreateGraph1(op::FVector<KernelNode*, DEFAULT_NODE_NUM>& expected)
{
    // kernel node construction
    KernelNode* a = AddNode(3, 1, 1);
    KernelNode* b = AddNode(5, 1, 1);
    KernelNode* c = AddNode(2, 1, 2 /* c has 2 outputs */);
    KernelNode* d = AddNode(1, 2 /* d has 2 inputs */, 1);
    KernelNode* e = AddNode(0, 2 /* e has 2 inputs */, 1);
    KernelNode* f = AddNode(4, 2, 1, 0, false /* f is from workspace; do not allocate memory. */);

    // kernel graph construction

    //            /--------\
    // a--->b--->c--->e--->f
    //  \    \---d---/
    //   -------/
    KernelGraphUtils::Link(a, b, 0, 0);
    KernelGraphUtils::Link(b, c, 0, 0);

    KernelGraphUtils::Link(a, d, 0, 1);
    KernelGraphUtils::Link(b, d, 0, 0);

    KernelGraphUtils::Link(c, e, 0, 0);

    KernelGraphUtils::Link(c, f, 1, 1);
    KernelGraphUtils::Link(d, e, 0, 0);
    KernelGraphUtils::Link(e, f, 0, 0);
    KernelGraph* kernel_graph = new KernelGraph(0);
    op::FVector<KernelNode*, DEFAULT_NODE_NUM> kernel_nodes;
    kernel_nodes = {e, d, c, a, f, b};

    kernel_graph->SetKernelNodes(kernel_nodes);
    expected = {a, b, d, c, e, f};
    // BFS: expected = {a, b, c, d, e, f};
    return kernel_graph;
}

KernelGraph* CreateGraph1WithTensorResult(op::FVector<TensorResult, DEFAULT_TENSOR_NUM>& expected_tensors,
                                          SortType sort_type)
{
    // kernel node construction
    KernelNode* a = AddNode(4, 1, 1);
    KernelNode* b = AddNode(6, 1, 1);
    KernelNode* c = AddNode(3, 1, 2 /* c has 2 outputs */);
    KernelNode* d = AddNode(1, 2 /* d has 2 inputs */, 1);
    KernelNode* e = AddNode(0, 2 /* e has 2 inputs */, 1);
    KernelNode* f = AddNode(5, 1, 1);
    KernelNode* g = AddNode(2, 2, 1, 0, false /* g is from workspace; do not allocate memory. */);

    // kernel graph construction

    //            /-->f-->\
    // a--->b--->c--->e--->g
    //  \    \---d---/
    //   -------/
    // sequence after dfs sort: a, b, c, d, e, f, g
    // sequence after bfs sort:
    KernelGraphUtils::Link(a, b, 0, 0);
    KernelGraphUtils::Link(b, c, 0, 0);
    KernelGraphUtils::Link(a, d, 0, 1);
    KernelGraphUtils::Link(b, d, 0, 0);
    KernelGraphUtils::Link(c, e, 0, 0);
    KernelGraphUtils::Link(c, f, 1, 0);
    KernelGraphUtils::Link(f, g, 0, 0);
    KernelGraphUtils::Link(d, e, 0, 1);
    KernelGraphUtils::Link(e, g, 0, 1);

    KernelGraph* kernel_graph = new KernelGraph(0);
    op::FVector<KernelNode*, DEFAULT_NODE_NUM> kernel_nodes = {e, d, g, c, a, f, b};

    kernel_graph->SetKernelNodes(kernel_nodes);
    if (sort_type == SortType::DFS) {
        expected_tensors = {{a->GetOutput(0), 0, 2}, {b->GetOutput(0), 1, 3}, {d->GetOutput(0), 2, 5},
                            {c->GetOutput(0), 3, 5}, {c->GetOutput(1), 3, 4}, {f->GetOutput(0), 4, 6},
                            {e->GetOutput(0), 5, 6}};
    } else if (sort_type == SortType::DFS_WITH_REARRANGEMENT) {
        expected_tensors = {{a->GetOutput(0), 0, 3}, {b->GetOutput(0), 1, 4}, {c->GetOutput(0), 2, 3},
                            {d->GetOutput(0), 3, 4}, {e->GetOutput(0), 4, 5}, {f->GetOutput(0), 5, 6},
                            {c->GetOutput(1), 2, 5}};
    } else if (sort_type == SortType::BFS) {
        expected_tensors = {{a->GetOutput(0), 0, 3}, {b->GetOutput(0), 1, 3}, {c->GetOutput(0), 2, 5},
                            {c->GetOutput(1), 2, 4}, {d->GetOutput(0), 3, 5}, {f->GetOutput(0), 4, 6},
                            {e->GetOutput(0), 5, 6}};
    }

    return kernel_graph;
}

KernelGraph* CreateGraph2(op::FVector<TensorResult, DEFAULT_TENSOR_NUM>& expected_tensors, SortType sort_type)
{
    // kernel node construction
    KernelNode* a = AddNode(6, 1, 1);
    KernelNode* b = AddNode(3, 1, 2, 1);
    KernelNode* c = AddNode(4, 1, 1);
    KernelNode* d = AddNode(8, 8 /* d has 2 inputs, we set 8 as wrong input */, 1);
    KernelNode* e = AddNode(2, 2 /* e has 2 inputs */, 1);
    KernelNode* f = AddNode(7, 1, 1, 0, false /* f is from workspace; do not allocate memory. */);
    KernelNode* g = AddNode(5, 1, 1);
    KernelNode* h = AddNode(0, 1, 1);
    KernelNode* i = AddNode(1, 1, 1);
    // kernel graph construction
    // b has two outputs and one workspace
    //       /-------------\
    // a--->b--->c--->d--->e--->f
    // g--->h--->i---/
    KernelGraphUtils::Link(a, b, 0, 0);
    KernelGraphUtils::Link(b, c, 0, 0);

    KernelGraphUtils::Link(b, e, 1, 1);
    KernelGraphUtils::Link(b, c, 0, 0);

    KernelGraphUtils::Link(c, d, 0, 0);

    KernelGraphUtils::Link(d, e, 0, 0);
    KernelGraphUtils::Link(e, f, 0, 0);
    KernelGraphUtils::Link(g, h, 0, 0);
    KernelGraphUtils::Link(h, i, 0, 0);
    KernelGraphUtils::Link(i, d, 0, 1);

    KernelGraph* kernel_graph = new KernelGraph(0);
    op::FVector<KernelNode*, DEFAULT_NODE_NUM> kernel_nodes = {h, i, e, b, c, g, a, f, d};

    kernel_graph->SetKernelNodes(kernel_nodes);

    if (sort_type == SortType::DFS) {
        //    expected_tensors = {{a->GetOutput(0), 0, 1}, {b->GetWorkspace(0), 1, 1}, {b->GetOutput(0), 1, 2},
        //                        {b->GetOutput(1), 1, 7}, {c->GetOutput(0), 2, 6}, {g->GetOutput(0), 3, 4},
        //                        {h->GetOutput(0), 4, 5}, {i->GetOutput(0), 5, 6}, {d->GetOutput(0), 6, 7},
        //                        {e->GetOutput(0), 7, 8}};
        expected_tensors = {{g->GetOutput(0), 0, 1}, {h->GetOutput(0), 1, 2},    {i->GetOutput(0), 2, 6},
                            {a->GetOutput(0), 3, 4}, {b->GetWorkspace(0), 4, 4}, {b->GetOutput(0), 4, 5},
                            {b->GetOutput(1), 4, 7}, {c->GetOutput(0), 5, 6},    {d->GetOutput(0), 6, 7},
                            {e->GetOutput(0), 7, 8}};
    } else if (sort_type == SortType::DFS_WITH_REARRANGEMENT) {
    }

    return kernel_graph;
}

KernelGraph* CreateGraphWithCycle()
{
    // kernel node construction
    KernelNode* a = AddNode(0, 1, 1);
    KernelNode* b = AddNode(1, 2, 1);
    KernelNode* c = AddNode(2, 1, 1);
    KernelNode* d = AddNode(3, 1, 1);
    KernelNode* e = AddNode(4, 1, 1);
    KernelNode* f = AddNode(5, 1, 1);
    KernelNode* g = AddNode(6, 1, 1);

    // kernel graph construction
    // b has two outputs and one workspace
    //       /<-----g------\
    // a--->b--->c--->d--->e--->f

    KernelGraphUtils::Link(a, b, 0, 0);
    KernelGraphUtils::Link(b, c, 0, 0);
    KernelGraphUtils::Link(c, d, 0, 0);
    KernelGraphUtils::Link(d, e, 0, 0);
    KernelGraphUtils::Link(e, f, 0, 0);
    KernelGraphUtils::Link(e, g, 0, 0);
    KernelGraphUtils::Link(g, b, 0, 1);

    KernelGraph* kernel_graph = new KernelGraph(0);
    op::FVector<KernelNode*, DEFAULT_NODE_NUM> kernel_nodes = {a, b, c, g, f, e, d};

    kernel_graph->SetKernelNodes(kernel_nodes);

    return kernel_graph;
}

KernelGraph* CreateGraphWithCycle2()
{
    // kernel node construction
    KernelNode* a = AddNode(0, 1, 1);
    KernelNode* b = AddNode(1, 2, 1);
    KernelNode* c = AddNode(2, 1, 1);
    KernelNode* d = AddNode(3, 1, 1);
    KernelNode* e = AddNode(4, 1, 1);
    KernelNode* f = AddNode(5, 1, 1);
    KernelNode* g = AddNode(6, 1, 1);

    // kernel graph construction
    // b has two outputs and one workspace
    //       /<-----g------\
    // a--->b--->c--->d--->e--->f

    KernelGraphUtils::Link(a, b, 0, 0);
    KernelGraphUtils::Link(b, c, 0, 0);
    KernelGraphUtils::Link(c, d, 0, 0);
    KernelGraphUtils::Link(d, e, 0, 0);
    KernelGraphUtils::Link(e, f, 0, 0);
    KernelGraphUtils::Link(e, g, 0, 0);
    KernelGraphUtils::Link(g, b, 0, 1);

    KernelGraph* kernel_graph = new KernelGraph(0);
    op::FVector<KernelNode*, DEFAULT_NODE_NUM> kernel_nodes = {a, b, c, g, f, e, d};

    kernel_graph->SetKernelNodes(kernel_nodes);

    return kernel_graph;
}

bool CheckResult(KernelGraph* graph, op::FVector<TensorResult, DEFAULT_TENSOR_NUM>& expected,
                 bool check_time_cost = false)
{
    printf("############################Check Result############################\n");
    if (check_time_cost) {
        size_t loop_num = 1;
        auto start = steady_clock::now();
        for (auto i = 0; i != loop_num; ++i) {
            (void)graph->GetSortedKernelTensors();
        }
        auto stop = steady_clock::now();
        auto duration = duration_cast<nanoseconds>(stop - start);
        cout << "time cost: " << duration.count() / loop_num << "(ns)" << endl;
    }

    op::FVector<KernelTensor*, DEFAULT_TENSOR_NUM> real = graph->GetSortedKernelTensors(SortType::DFS);
    for (auto node : graph->GetKernelNodes()) {
        printf("[PrintNode] node: %ld, topo id %ld, ", node->GetOriginalId(), node->GetTopoId());
    }

    PrintTensors("expected", expected);
    PrintTensors("real", real);

    if (expected.size() != real.size()) {
        printf("Expected size %zu is not equal to sorted tensor size %zu.\n", expected.size(), real.size());
        printf("\nCompare Failed.\n");
        return false;
    }

    for (size_t i = 0; i < expected.size(); ++i) {
        if (expected[i].kt != real[i]) {
            printf("\nCompare tensor ptr %zu Failed.\n", i);
            return false;
        }

        if (expected[i].expected_end != real[i]->GetLifeTimeEnd()) {
            printf("\nCompare end of node %zu tensor %zu Failed. [expected %ld], [real %ld]\n",
                   expected[i].kt->GetOwnerNode()->GetOriginalId(), real[i]->GetIndex(), expected[i].expected_end,
                   real[i]->GetLifeTimeEnd());
            return false;
        }

        if (expected[i].expected_start != real[i]->GetLifeTimeStart()) {
            printf("\nCompare start of node %zu tensor %zu Failed. [expected %ld], [real %ld]\n",
                   expected[i].kt->GetOwnerNode()->GetOriginalId(), real[i]->GetIndex(), expected[i].expected_start,
                   real[i]->GetLifeTimeStart());
            return false;
        }
    }
    return true;
}

TEST_F(memory_allocator_ut, TestTopoSortBFS)
{
    return;
    op::FVector<KernelNode*, DEFAULT_NODE_NUM> expected;
    KernelGraph* graph = CreateGraph1(expected);
    graph->TopologicalSortBFS();
    if (expected != graph->GetKernelNodes()) {
        for (auto node : graph->GetKernelNodes()) {
            printf("%ld, ", node->GetOriginalId());
        }
    }
    EXPECT_EQ(expected, graph->GetKernelNodes());
    Release(graph);
}

TEST_F(memory_allocator_ut, TestTopoSortDFS)
{
    op::FVector<KernelNode*, DEFAULT_NODE_NUM> expected;
    KernelGraph* graph = CreateGraph1(expected);
    graph->TopologicalSortDFS();

    for (size_t i = 0; i < graph->GetKernelNodes().size(); i++) {
        EXPECT_EQ(expected[i], graph->sortedNodes_[i]);
    }
    Release(graph);
}

TEST_F(memory_allocator_ut, TestGetSortedTensors1)
{
    op::FVector<TensorResult, DEFAULT_TENSOR_NUM> expected;
    KernelGraph* graph = CreateGraph1WithTensorResult(expected, SortType::DFS);
    EXPECT_TRUE(CheckResult(graph, expected));
    Release(graph);
}

TEST_F(memory_allocator_ut, TestGetSortedTensors2)
{
    op::FVector<TensorResult, DEFAULT_TENSOR_NUM> expected;
    KernelGraph* graph = CreateGraph2(expected, SortType::DFS);
    EXPECT_TRUE(CheckResult(graph, expected, true));
    Release(graph);
}

TEST_F(memory_allocator_ut, TestGetSortedTensorsWithCycle)
{
    op::FVector<TensorResult, DEFAULT_TENSOR_NUM> expected;
    KernelGraph* graph = CreateGraphWithCycle();
    aclnnStatus ret = graph->TopologicalSortDFS();
    EXPECT_EQ(ret, ACLNN_ERR_INNER);
    Release(graph);
}
// Create a graph with n nodes and m edges.
void GenerateRandomGraph(int n, int m, std::vector<KernelNode*>& nodes, std::vector<KernelTensor*>& tensors)
{
    std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
    std::uniform_int_distribution<int> dist(0, n - 1);

    for (int i = 0; i < n; ++i) {
        nodes.push_back(new KernelNode(0, i));
    }

    for (int i = 0; i < m; ++i) {
        tensors.push_back(new KernelTensor(nullptr, i));
        int src = dist(rng);
        int dst = dist(rng);
        while (src == dst || nodes[src]->GetOutputs().empty() || nodes[dst]->GetOutputs().empty()) {
            src = dist(rng);
            dst = dist(rng);
        }
        nodes[src]->AddOutput(tensors.back());
        nodes[dst]->AddInput(tensors.back());
    }
}

TEST_F(memory_allocator_ut, TestTopoSortDFSLarge)
{
    return;
    // 创建随机图
    std::vector<KernelNode*> nodes;
    std::vector<KernelTensor*> tensors;
    GenerateRandomGraph(10, 50, nodes, tensors);
    auto graph = new KernelGraph(0);
    for (auto* node : nodes) {
        graph->AddKernelNode(node);
    }

    // 计时并排序
    auto start = std::chrono::steady_clock::now();
    int ret = graph->TopologicalSortDFS();
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Time elapsed: " << elapsed_seconds.count() << " s" << std::endl;
    EXPECT_EQ(ret, ACLNN_SUCCESS);

    // 验证排序结果
    auto kernel_nodes = graph->GetKernelNodes();
    for (KernelNode* node : graph->GetKernelNodes()) {
        for (KernelTensor* input : node->GetInputs()) {
            for (KernelTensor* peer : input->GetPeerTensors()) {
                KernelNode* peer_node = peer->GetOwnerNode();
                auto m = std::find(kernel_nodes.begin(), kernel_nodes.end(), peer_node);
                ASSERT_FALSE(m == kernel_nodes.end());
            }
        }
    }

    // 释放资源
    Release(graph);
}
