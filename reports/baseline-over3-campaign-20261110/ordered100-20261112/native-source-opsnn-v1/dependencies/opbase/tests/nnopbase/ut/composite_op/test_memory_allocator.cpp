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
#include <initializer_list>
#include <memory>

#include "acl/acl.h"
#include "kernel_graph_utils.h"
#include "memory_allocator.h"
#include "opdev/fast_vector.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"

using namespace op;
using namespace op::mem;

class MemoryAllocatorUt : public testing::Test {
protected:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}
};

namespace {
inline int64_t Align(int64_t n)
{
    return (n + EXTRA_SIZE_IN_BYTE + ALIGN_SIZE_IN_BYTE - 1) & (~(ALIGN_SIZE_IN_BYTE - 1));
}

int64_t CalWorkspaceSize(std::initializer_list<int64_t> args)
{
    int64_t size = 0;
    for (auto arg : args) {
        size += Align(arg);
    }
    return size;
}

void PrepareKernelTensor(KernelTensor* kernelTensor, int64_t lifeTimeStart, int64_t lifeTimeEnd)
{
    kernelTensor->CalcSize();
    kernelTensor->SetLifeTimeStart(lifeTimeStart);
    kernelTensor->SetLifeTimeEnd(lifeTimeEnd);
}
} // namespace

TEST_F(MemoryAllocatorUt, MaxAllocatorTestCase0)
{
    op::FVector<KernelTensor*, DEFAULT_TENSOR_NUM> kernelTensors;
    auto aclTensor0 = std::make_unique<aclTensor>(op::Shape{4, 5}, op::DataType::DT_FLOAT, op::Format::FORMAT_ND,
                                                  nullptr);
    auto kernelTensor0 = std::make_unique<KernelTensor>(aclTensor0.get(), 0);
    PrepareKernelTensor(kernelTensor0.get(), 0, 2);
    kernelTensors.push_back(kernelTensor0.get());

    auto aclTensor1 = std::make_unique<aclTensor>(op::Shape{4, 5}, op::DataType::DT_FLOAT, op::Format::FORMAT_ND,
                                                  nullptr);
    auto kernelTensor1 = std::make_unique<KernelTensor>(aclTensor1.get(), 0);
    PrepareKernelTensor(kernelTensor1.get(), 1, 2);
    kernelTensors.push_back(kernelTensor1.get());

    auto allocator = op::mem::MaxAllocator();
    int64_t workspace_size = allocator.Allocate(kernelTensors);
    EXPECT_EQ(workspace_size, CalWorkspaceSize({kernelTensor0->GetSize(), kernelTensor1->GetSize()}));
}

/* two kernel tensors has same aclTensor pointer */
TEST_F(MemoryAllocatorUt, MaxAllocatorTestInplaceL0)
{
    op::FVector<KernelTensor*, DEFAULT_TENSOR_NUM> kernelTensors;
    auto aclTensor0 = std::make_unique<aclTensor>(op::Shape{4, 5}, op::DataType::DT_FLOAT, op::Format::FORMAT_ND,
                                                  nullptr);
    auto kernelTensor0 = std::make_unique<KernelTensor>(aclTensor0.get(), 0);
    PrepareKernelTensor(kernelTensor0.get(), 0, 2);
    kernelTensors.push_back(kernelTensor0.get());

    auto kernelTensor1 = std::make_unique<KernelTensor>(aclTensor0.get(), 0);
    PrepareKernelTensor(kernelTensor1.get(), 1, 2);
    kernelTensors.push_back(kernelTensor1.get());

    auto allocator = op::mem::MaxAllocator();
    int64_t workspace_size = allocator.Allocate(kernelTensors);
    EXPECT_EQ(workspace_size, CalWorkspaceSize({kernelTensor0->GetSize()}));
    EXPECT_EQ(0, kernelTensor0->GetAclTensor()->GetWorkspaceOffset());
    EXPECT_EQ(0, kernelTensor0->GetAclTensor()->GetWorkspaceOffset());
}

TEST_F(MemoryAllocatorUt, MaxAllocatorTestCase1)
{
    op::FVector<KernelTensor*, DEFAULT_TENSOR_NUM> kernelTensors;
    auto aclTensor0 = std::make_unique<aclTensor>(op::Shape{16}, op::DataType::DT_FLOAT, op::Format::FORMAT_ND,
                                                  nullptr);
    auto kernelTensor0 = std::make_unique<KernelTensor>(aclTensor0.get(), 0);
    PrepareKernelTensor(kernelTensor0.get(), 0, 0);
    kernelTensors.push_back(kernelTensor0.get());

    auto aclTensor1 = std::make_unique<aclTensor>(op::Shape{4, 5}, op::DataType::DT_FLOAT, op::Format::FORMAT_ND,
                                                  nullptr);
    auto kernelTensor1 = std::make_unique<KernelTensor>(aclTensor1.get(), 0);
    PrepareKernelTensor(kernelTensor1.get(), 0, 2);
    kernelTensors.push_back(kernelTensor1.get());

    auto aclTensor2 = std::make_unique<aclTensor>(op::Shape{4, 5}, op::DataType::DT_FLOAT, op::Format::FORMAT_ND,
                                                  nullptr);
    auto kernelTensor2 = std::make_unique<KernelTensor>(aclTensor2.get(), 0);
    PrepareKernelTensor(kernelTensor2.get(), 1, 2);
    kernelTensors.push_back(kernelTensor2.get());

    auto allocator = op::mem::MaxAllocator();
    int64_t workspace_size = allocator.Allocate(kernelTensors);
    EXPECT_EQ(workspace_size, CalWorkspaceSize({kernelTensor1->GetSize(), kernelTensor2->GetSize()}));
}

TEST_F(MemoryAllocatorUt, MaxAllocatorTestCase2)
{
    op::FVector<KernelTensor*, DEFAULT_TENSOR_NUM> kernelTensors;
    auto aclTensor0 = std::make_unique<aclTensor>(op::Shape{1024}, op::DataType::DT_UINT8, op::Format::FORMAT_ND,
                                                  nullptr);
    auto kernelTensor0 = std::make_unique<KernelTensor>(aclTensor0.get(), 0);
    PrepareKernelTensor(kernelTensor0.get(), 0, 0);
    kernelTensors.push_back(kernelTensor0.get());

    auto aclTensor1 = std::make_unique<aclTensor>(op::Shape{4, 5}, op::DataType::DT_FLOAT, op::Format::FORMAT_ND,
                                                  nullptr);
    auto kernelTensor1 = std::make_unique<KernelTensor>(aclTensor1.get(), 0);
    PrepareKernelTensor(kernelTensor1.get(), 0, 2);
    kernelTensors.push_back(kernelTensor1.get());

    auto aclTensor2 = std::make_unique<aclTensor>(op::Shape{4, 5}, op::DataType::DT_FLOAT, op::Format::FORMAT_ND,
                                                  nullptr);
    auto kernelTensor2 = std::make_unique<KernelTensor>(aclTensor2.get(), 0);
    PrepareKernelTensor(kernelTensor2.get(), 1, 2);
    kernelTensors.push_back(kernelTensor2.get());

    auto allocator = op::mem::MaxAllocator();
    int64_t workspace_size = allocator.Allocate(kernelTensors);
    EXPECT_EQ(workspace_size, CalWorkspaceSize({kernelTensor0->GetSize(), kernelTensor2->GetSize()}));
}

TEST_F(MemoryAllocatorUt, MaxAllocatorTestCase3)
{
    op::FVector<KernelTensor*, DEFAULT_TENSOR_NUM> kernelTensors;

    auto allocator = op::mem::MaxAllocator();
    int64_t workspace_size = allocator.Allocate(kernelTensors);
    EXPECT_EQ(workspace_size, 0);
}

TEST_F(MemoryAllocatorUt, MaxAllocatorTestCase4)
{
    op::FVector<KernelTensor*, DEFAULT_TENSOR_NUM> kernelTensors;

    auto aclTensor0 = std::make_unique<aclTensor>(op::Shape{18}, op::DataType::DT_UINT8, op::Format::FORMAT_ND,
                                                  nullptr);
    auto kernelTensor0 = std::make_unique<KernelTensor>(aclTensor0.get(), 0);
    PrepareKernelTensor(kernelTensor0.get(), 0, 0);
    kernelTensors.push_back(kernelTensor0.get());

    auto allocator = op::mem::MaxAllocator();
    int64_t workspace_size = allocator.Allocate(kernelTensors);
    EXPECT_EQ(workspace_size, CalWorkspaceSize({kernelTensor0->GetSize()}));
}

TEST_F(MemoryAllocatorUt, MaxAllocatorTestCase5)
{
    op::FVector<KernelTensor*, DEFAULT_TENSOR_NUM> kernelTensors;

    auto aclTensor0 = std::make_unique<aclTensor>(op::Shape{481}, op::DataType::DT_UINT8, op::Format::FORMAT_ND,
                                                  nullptr);
    auto kernelTensor0 = std::make_unique<KernelTensor>(aclTensor0.get(), 0);
    PrepareKernelTensor(kernelTensor0.get(), 0, 0);
    kernelTensors.push_back(kernelTensor0.get());

    auto allocator = op::mem::MaxAllocator();
    int64_t workspace_size = allocator.Allocate(kernelTensors);
    EXPECT_EQ(workspace_size, CalWorkspaceSize({kernelTensor0->GetSize()}));
}

TEST_F(MemoryAllocatorUt, LinearAllocatorTestCase0)
{
    op::FVector<KernelTensor*, DEFAULT_TENSOR_NUM> kernelTensors;
    auto aclTensor0 = std::make_unique<aclTensor>(op::Shape{4, 5}, op::DataType::DT_FLOAT, op::Format::FORMAT_ND,
                                                  nullptr);
    auto kernelTensor0 = std::make_unique<KernelTensor>(aclTensor0.get(), 0);
    PrepareKernelTensor(kernelTensor0.get(), 0, 2);
    kernelTensors.push_back(kernelTensor0.get());

    auto aclTensor1 = std::make_unique<aclTensor>(op::Shape{4, 5}, op::DataType::DT_FLOAT, op::Format::FORMAT_ND,
                                                  nullptr);
    auto kernelTensor1 = std::make_unique<KernelTensor>(aclTensor1.get(), 0);
    PrepareKernelTensor(kernelTensor1.get(), 1, 2);
    kernelTensors.push_back(kernelTensor1.get());

    auto allocator = op::mem::LinearAllocator();
    int64_t workspace_size = allocator.Allocate(kernelTensors);
    EXPECT_EQ(workspace_size, CalWorkspaceSize({kernelTensor0->GetSize(), kernelTensor1->GetSize()}));
}
