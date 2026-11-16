/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * You may refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file test_dynamic_capacity.cpp
 * @brief Unit tests for dynamic capacity expansion in context holders
 *
 * This file contains test cases for the dynamic capacity expansion feature:
 * - MAX_OP_ARG_NUM dynamic expansion (KernelContextHolder, InferShapeContextHolder, TilingCtxHolder)
 * - ATTR_CAPACITY dynamic expansion (KernelContextHolder)
 *
 * The expansion follows std::vector-like growth strategy (2x growth factor)
 * and uses malloc + memcpy instead of realloc.
 */

#include "gtest/gtest.h"
#include <memory>
#include <stdlib.h>
#include <vector>

#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnn/aclnn_base.h"
#include "kernel_context_holder.h"
#include "infershape_context_holder.h"
#include "tilingctx_builder.h"
#include "op_ctx_def.h"
#include "opdev/op_def.h"
#include "opdev/make_op_executor.h"
#include "thread_local_context.h"

using namespace op::internal;

class DynamicCapacityUT : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        setenv("ASCEND_OPP_PATH", OP_API_COMMON_UT_SRC_DIR, 1);
        GetThreadLocalContext().cacheHasFull_ = true;
    }

    static void TearDownTestCase() {}

    void SetUp() override {}
    void TearDown() override {}
};

// ============================================================================
// Test: KernelContextHolder dynamic expansion for opInArg_
// ============================================================================

TEST_F(DynamicCapacityUT, KernelContextHolder_EnsureArgCapacity_NoExpansionNeeded)
{
    KernelContextHolder holder;

    // Initial capacity is MAX_OP_ARG_NUM (1024)
    size_t initialCapacity = holder.opInArgCapacity_;
    EXPECT_EQ(initialCapacity, MAX_OP_ARG_NUM);

    // Request less than initial capacity, no expansion should occur
    aclnnStatus ret = holder.EnsureArgCapacity(512);
    EXPECT_EQ(ret, ACLNN_SUCCESS);
    EXPECT_EQ(holder.opInArgCapacity_, initialCapacity);
}

TEST_F(DynamicCapacityUT, KernelContextHolder_EnsureArgCapacity_SingleExpansion)
{
    KernelContextHolder holder;

    size_t initialCapacity = holder.opInArgCapacity_;
    EXPECT_EQ(initialCapacity, MAX_OP_ARG_NUM); // Initial is 1024

    // Request more than initial capacity, should trigger 2x expansion
    aclnnStatus ret = holder.EnsureArgCapacity(MAX_OP_ARG_NUM + 1);
    EXPECT_EQ(ret, ACLNN_SUCCESS);
    EXPECT_EQ(holder.opInArgCapacity_, MAX_OP_ARG_NUM * 2); // 2048
}

TEST_F(DynamicCapacityUT, KernelContextHolder_EnsureArgCapacity_MultipleExpansion)
{
    KernelContextHolder holder;

    size_t initialCapacity = holder.opInArgCapacity_;
    EXPECT_EQ(initialCapacity, MAX_OP_ARG_NUM); // Initial is 1024

    // Request 10x capacity, should trigger multiple expansions
    aclnnStatus ret = holder.EnsureArgCapacity(MAX_OP_ARG_NUM * 10);
    EXPECT_EQ(ret, ACLNN_SUCCESS);
    // Capacity should be at least 10x, using 2x growth
    EXPECT_GE(holder.opInArgCapacity_, MAX_OP_ARG_NUM * 10);
}

TEST_F(DynamicCapacityUT, KernelContextHolder_EnsureArgCapacity_PowerOfTwoGrowth)
{
    KernelContextHolder holder;

    // Test that capacity follows 2x growth pattern from initial MAX_OP_ARG_NUM
    size_t expectedCapacity = MAX_OP_ARG_NUM;
    for (int i = 0; i < 5; i++) {
        expectedCapacity *= 2;
        aclnnStatus ret = holder.EnsureArgCapacity(expectedCapacity);
        EXPECT_EQ(ret, ACLNN_SUCCESS);
        EXPECT_EQ(holder.opInArgCapacity_, expectedCapacity);
    }
}

// ============================================================================
// Test: KernelContextHolder dynamic expansion for computeNodeInfo_
// ============================================================================

TEST_F(DynamicCapacityUT, KernelContextHolder_EnsureComputeNodeInfoCapacity_NoExpansionNeeded)
{
    KernelContextHolder holder;

    size_t initialCapacity = holder.computeNodeInfoCapacity_;
    EXPECT_EQ(initialCapacity, MAX_OP_ARG_NUM); // Initial is 1024

    aclnnStatus ret = holder.EnsureComputeNodeInfoCapacity(512);
    EXPECT_EQ(ret, ACLNN_SUCCESS);
    EXPECT_EQ(holder.computeNodeInfoCapacity_, initialCapacity);
}

TEST_F(DynamicCapacityUT, KernelContextHolder_EnsureComputeNodeInfoCapacity_SingleExpansion)
{
    KernelContextHolder holder;

    size_t initialCapacity = holder.computeNodeInfoCapacity_;
    EXPECT_EQ(initialCapacity, MAX_OP_ARG_NUM); // Initial is 1024

    aclnnStatus ret = holder.EnsureComputeNodeInfoCapacity(MAX_OP_ARG_NUM + 1);
    EXPECT_EQ(ret, ACLNN_SUCCESS);
    EXPECT_EQ(holder.computeNodeInfoCapacity_, MAX_OP_ARG_NUM * 2); // 2048
}

TEST_F(DynamicCapacityUT, KernelContextHolder_EnsureComputeNodeInfoCapacity_WithAttrs)
{
    KernelContextHolder holder;

    // First expand computeNodeInfo capacity
    size_t initialCapacity = holder.computeNodeInfoCapacity_;
    aclnnStatus ret = holder.EnsureComputeNodeInfoCapacity(MAX_OP_ARG_NUM + 10);
    EXPECT_EQ(ret, ACLNN_SUCCESS);

    // Verify internal pointers are still valid
    EXPECT_NE(holder.anchorInfo_, nullptr);
    EXPECT_NE(holder.computeNodeInfo_, nullptr);
}

// ============================================================================
// Test: KernelContextHolder dynamic expansion for attrCapacity_
// ============================================================================

TEST_F(DynamicCapacityUT, KernelContextHolder_EnsureAttrCapacity_NoExpansionNeeded)
{
    KernelContextHolder holder;

    size_t initialCapacity = holder.attrCapacity_;
    EXPECT_EQ(initialCapacity, ATTR_CAPACITY); // 32KB initial

    // Request less than initial capacity
    aclnnStatus ret = holder.EnsureAttrCapacity(16 * 1024);
    EXPECT_EQ(ret, ACLNN_SUCCESS);
    EXPECT_EQ(holder.attrCapacity_, initialCapacity);
}

TEST_F(DynamicCapacityUT, KernelContextHolder_EnsureAttrCapacity_SingleExpansion)
{
    KernelContextHolder holder;

    // Need to initialize attrDef first by calling UpdateAttrDefOffset
    // Simulate the state during attribute processing
    holder.irInputNum_ = 10;
    holder.inputNum_ = 10;
    holder.outputNum_ = 0;
    holder.attrNum_ = 0;
    holder.UpdateCompileDescOffset(10);
    holder.UpdateAttrDefOffset(10, 0);

    size_t initialCapacity = holder.attrCapacity_;
    EXPECT_EQ(initialCapacity, ATTR_CAPACITY); // 32KB initial

    // Request more than initial capacity (32KB + 1)
    aclnnStatus ret = holder.EnsureAttrCapacity(ATTR_CAPACITY + 1);
    EXPECT_EQ(ret, ACLNN_SUCCESS);
    EXPECT_EQ(holder.attrCapacity_, ATTR_CAPACITY * 2); // 64KB
}

TEST_F(DynamicCapacityUT, KernelContextHolder_EnsureAttrCapacity_MultipleExpansion)
{
    KernelContextHolder holder;

    // Initialize attrDef first
    holder.irInputNum_ = 10;
    holder.inputNum_ = 10;
    holder.outputNum_ = 0;
    holder.attrNum_ = 0;
    holder.UpdateCompileDescOffset(10);
    holder.UpdateAttrDefOffset(10, 0);

    size_t initialCapacity = holder.attrCapacity_;
    EXPECT_EQ(initialCapacity, ATTR_CAPACITY); // 32KB initial

    // Request 4x capacity
    aclnnStatus ret = holder.EnsureAttrCapacity(ATTR_CAPACITY * 4);
    EXPECT_EQ(ret, ACLNN_SUCCESS);
    EXPECT_EQ(holder.attrCapacity_, ATTR_CAPACITY * 4); // 128KB
}

TEST_F(DynamicCapacityUT, KernelContextHolder_EnsureAttrCapacity_LargeExpansion)
{
    KernelContextHolder holder;

    // Initialize attrDef first
    holder.irInputNum_ = 10;
    holder.inputNum_ = 10;
    holder.outputNum_ = 0;
    holder.attrNum_ = 0;
    holder.UpdateCompileDescOffset(10);
    holder.UpdateAttrDefOffset(10, 0);

    // Request a large size (512KB)
    aclnnStatus ret = holder.EnsureAttrCapacity(512 * 1024);
    EXPECT_EQ(ret, ACLNN_SUCCESS);
    EXPECT_GE(holder.attrCapacity_, 512 * 1024);

    // Verify pointers are still valid after expansion
    EXPECT_NE(holder.attrDataStart_, nullptr);
    EXPECT_NE(holder.attrDef_, nullptr);
    EXPECT_NE(holder.computeNodeInfo_, nullptr);
}

// ============================================================================
// Test: InferShapeContextHolder dynamic expansion
// ============================================================================

TEST_F(DynamicCapacityUT, InferShapeContextHolder_BuildAndExpand)
{
    InferShapeContextHolder holder;

    size_t initialCapacity = holder.inferShapeCtxCapacity_;
    EXPECT_EQ(initialCapacity, MAX_OP_ARG_NUM); // Initial is 1024

    // Request expansion
    aclnnStatus ret = const_cast<InferShapeContextHolder&>(holder).EnsureContextCapacity(MAX_OP_ARG_NUM + 10);
    EXPECT_EQ(ret, ACLNN_SUCCESS);
    EXPECT_EQ(holder.inferShapeCtxCapacity_, MAX_OP_ARG_NUM * 2); // 2048
}

TEST_F(DynamicCapacityUT, InferShapeContextHolder_MultipleExpansions)
{
    InferShapeContextHolder holder;

    size_t expectedCapacity = MAX_OP_ARG_NUM; // Start from 1024
    for (int i = 0; i < 4; i++) {
        expectedCapacity *= 2;
        aclnnStatus ret = const_cast<InferShapeContextHolder&>(holder).EnsureContextCapacity(expectedCapacity);
        EXPECT_EQ(ret, ACLNN_SUCCESS);
        EXPECT_EQ(holder.inferShapeCtxCapacity_, expectedCapacity);
    }
}

// ============================================================================
// Test: TilingCtxHolder dynamic expansion
// ============================================================================

TEST_F(DynamicCapacityUT, TilingCtxHolder_EnsureTilingCtxCapacity_NoExpansionNeeded)
{
    TilingCtxHolder holder;

    size_t initialCapacity = holder.tilingCtxCapacity_;
    EXPECT_EQ(initialCapacity, MAX_OP_ARG_NUM); // Initial is 1024

    aclnnStatus ret = holder.EnsureTilingCtxCapacity(512);
    EXPECT_EQ(ret, ACLNN_SUCCESS);
    EXPECT_EQ(holder.tilingCtxCapacity_, initialCapacity);
}

TEST_F(DynamicCapacityUT, TilingCtxHolder_EnsureTilingCtxCapacity_SingleExpansion)
{
    TilingCtxHolder holder;

    size_t initialCapacity = holder.tilingCtxCapacity_;
    EXPECT_EQ(initialCapacity, MAX_OP_ARG_NUM); // Initial is 1024

    aclnnStatus ret = holder.EnsureTilingCtxCapacity(MAX_OP_ARG_NUM + 1);
    EXPECT_EQ(ret, ACLNN_SUCCESS);
    EXPECT_EQ(holder.tilingCtxCapacity_, MAX_OP_ARG_NUM * 2); // 2048
}

TEST_F(DynamicCapacityUT, TilingCtxHolder_EnsureTilingCtxCapacity_MultipleExpansion)
{
    TilingCtxHolder holder;

    size_t initialCapacity = holder.tilingCtxCapacity_;
    EXPECT_EQ(initialCapacity, MAX_OP_ARG_NUM); // Initial is 1024

    // Request 5x capacity
    aclnnStatus ret = holder.EnsureTilingCtxCapacity(MAX_OP_ARG_NUM * 5);
    EXPECT_EQ(ret, ACLNN_SUCCESS);
    EXPECT_GE(holder.tilingCtxCapacity_, MAX_OP_ARG_NUM * 5);
}

// ============================================================================
// Test: Integration test with many tensors
// ============================================================================

TEST_F(DynamicCapacityUT, KernelContextHolder_ManyTensors_ExpansionTriggered)
{
    KernelContextHolder holder;

    // Create many tensors to trigger expansion
    // Initial capacity is MAX_OP_ARG_NUM (1024), creating 1500 tensors should trigger expansion
    std::vector<std::unique_ptr<aclTensor>> tensors;

    for (int i = 0; i < 1500; i++) {
        op::Shape shape{2, 2};
        tensors.push_back(std::make_unique<aclTensor>(shape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND, nullptr));
    }

    // Initialize state properly
    holder.irInputNum_ = 1500;
    holder.irOutputNum_ = 0;
    holder.UpdateCompileDescOffset(1500);

    // Add all tensors as inputs
    for (int i = 0; i < 1500; i++) {
        aclnnStatus ret = holder.UpdateInputArg(i, tensors[i].get());
        EXPECT_EQ(ret, ACLNN_SUCCESS);
    }

    // Verify capacity was expanded
    EXPECT_GE(holder.opInArgCapacity_, 1500);
    EXPECT_GE(holder.computeNodeInfoCapacity_, 1500);
    EXPECT_EQ(holder.inputNum_, 1500);
}

TEST_F(DynamicCapacityUT, KernelContextHolder_InputOutput_ExpansionTriggered)
{
    KernelContextHolder holder;

    // Create tensors - need more than MAX_OP_ARG_NUM to trigger expansion
    std::vector<std::unique_ptr<aclTensor>> inputTensors;
    std::vector<std::unique_ptr<aclTensor>> outputTensors;

    for (int i = 0; i < 600; i++) {
        op::Shape shape{2, 2};
        inputTensors.push_back(
            std::make_unique<aclTensor>(shape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND, nullptr));
        outputTensors.push_back(
            std::make_unique<aclTensor>(shape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND, nullptr));
    }

    // Initialize state properly
    holder.irInputNum_ = 600;
    holder.irOutputNum_ = 600;
    holder.UpdateCompileDescOffset(600);

    // Add inputs
    for (int i = 0; i < 600; i++) {
        aclnnStatus ret = holder.UpdateInputArg(i, inputTensors[i].get());
        EXPECT_EQ(ret, ACLNN_SUCCESS);
    }

    // Add outputs
    for (int i = 0; i < 600; i++) {
        aclnnStatus ret = holder.UpdateOutputArg(i, outputTensors[i].get());
        EXPECT_EQ(ret, ACLNN_SUCCESS);
    }

    // Total args = 1200, should trigger expansion from initial 1024
    EXPECT_GE(holder.opInArgCapacity_, 1200);
    EXPECT_GE(holder.computeNodeInfoCapacity_, 1200);
    EXPECT_EQ(holder.inputNum_, 600);
    EXPECT_EQ(holder.outputNum_, 600);
}

// ============================================================================
// Test: Edge cases
// ============================================================================

TEST_F(DynamicCapacityUT, KernelContextHolder_EnsureArgCapacity_ExactlyAtBoundary)
{
    KernelContextHolder holder;

    size_t initialCapacity = holder.opInArgCapacity_;

    // Request exactly the current capacity, no expansion needed
    aclnnStatus ret = holder.EnsureArgCapacity(initialCapacity);
    EXPECT_EQ(ret, ACLNN_SUCCESS);
    EXPECT_EQ(holder.opInArgCapacity_, initialCapacity);
}

TEST_F(DynamicCapacityUT, KernelContextHolder_EnsureArgCapacity_OneOverBoundary)
{
    KernelContextHolder holder;

    size_t initialCapacity = holder.opInArgCapacity_;

    // Request one more than current capacity, should expand
    aclnnStatus ret = holder.EnsureArgCapacity(initialCapacity + 1);
    EXPECT_EQ(ret, ACLNN_SUCCESS);
    EXPECT_EQ(holder.opInArgCapacity_, initialCapacity * 2);
}

TEST_F(DynamicCapacityUT, DynamicCapacity_InitialCapacityValues)
{
    // Verify initial capacity values match expected constants
    KernelContextHolder kernelHolder;
    EXPECT_EQ(kernelHolder.opInArgCapacity_, MAX_OP_ARG_NUM);         // 1024
    EXPECT_EQ(kernelHolder.computeNodeInfoCapacity_, MAX_OP_ARG_NUM); // 1024
    EXPECT_EQ(kernelHolder.attrCapacity_, ATTR_CAPACITY);             // 32KB

    InferShapeContextHolder inferShapeHolder;
    EXPECT_EQ(inferShapeHolder.inferShapeCtxCapacity_, MAX_OP_ARG_NUM); // 1024

    TilingCtxHolder tilingHolder;
    EXPECT_EQ(tilingHolder.tilingCtxCapacity_, MAX_OP_ARG_NUM); // 1024
}

// ============================================================================
// Test: Memory allocation failure handling (limited test)
// ============================================================================

TEST_F(DynamicCapacityUT, DynamicCapacity_SmallRequest)
{
    // Test with small requests to ensure basic functionality
    KernelContextHolder holder;

    // Sequential small expansions (still within MAX_OP_ARG_NUM, no expansion triggered)
    for (int i = 1; i <= 5; i++) {
        size_t request = 200 * i; // 200, 400, 600, 800, 1000 (all < 1024)
        aclnnStatus ret = holder.EnsureArgCapacity(request);
        EXPECT_EQ(ret, ACLNN_SUCCESS);
        // No expansion should occur as all requests are within MAX_OP_ARG_NUM
        EXPECT_EQ(holder.opInArgCapacity_, MAX_OP_ARG_NUM); // Still 1024
    }
}

TEST_F(DynamicCapacityUT, DynamicCapacity_AttrCapacitySmallExpansion)
{
    KernelContextHolder holder;

    // Test small incremental expansions for attr capacity
    size_t currentCap = holder.attrCapacity_;
    for (int i = 1; i <= 3; i++) {
        size_t request = currentCap + (16 * 1024); // Add 16KB each time
        aclnnStatus ret = holder.EnsureAttrCapacity(request);
        EXPECT_EQ(ret, ACLNN_SUCCESS);
        EXPECT_GE(holder.attrCapacity_, request);
        currentCap = holder.attrCapacity_;
    }
}
