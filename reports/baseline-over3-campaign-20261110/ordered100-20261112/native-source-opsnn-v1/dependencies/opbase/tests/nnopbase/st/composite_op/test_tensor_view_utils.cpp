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

#include "opdev/tensor_view_utils.h"
#include "opdev/op_executor.h"

class TensorViewUtilsTest : public ::testing::Test {
public:
    TensorViewUtilsTest() : executor(nullptr) {}
    aclTensor* CreateAclTensor(std::vector<int64_t> view_shape, std::vector<int64_t> stride, int64_t offset,
                               std::vector<int64_t> storage_shape)
    {
        return aclCreateTensor(view_shape.data(), view_shape.size(), ACL_FLOAT, stride.data(), offset, ACL_FORMAT_ND,
                               storage_shape.data(), storage_shape.size(), nullptr);
    }

public:
    UniqueExecutor executor;
};

TEST_F(TensorViewUtilsTest, IsContiguous)
{
    auto tensor = CreateAclTensor({4, 5, 6, 7}, {210, 42, 7, 1}, 0, {4, 5, 6, 7});
    EXPECT_TRUE(IsContiguous(tensor));
    aclDestroyTensor(tensor);

    tensor = CreateAclTensor({3, 5, 6, 7}, {210, 42, 7, 1}, 210, {4, 5, 6, 7});
    EXPECT_TRUE(IsContiguous(tensor));
    aclDestroyTensor(tensor);

    tensor = CreateAclTensor({1, 1, 1, 1}, {210, 42, 7, 1}, 0, {4, 5, 6, 7});
    EXPECT_TRUE(IsContiguous(tensor));
    aclDestroyTensor(tensor);

    tensor = CreateAclTensor({4, 0, 6, 7}, {210, 42, 7, 1}, 0, {4, 5, 6, 7});
    EXPECT_EQ(IsContiguous(tensor), true);
    aclDestroyTensor(tensor);

    tensor = CreateAclTensor({4, 2, 6, 7}, {210, 42, 7, 1}, 0, {4, 5, 6, 7});
    EXPECT_EQ(IsContiguous(tensor), false);
    aclDestroyTensor(tensor);

    tensor = CreateAclTensor({4, 2, 6, 7}, {210, 84, 7, 1}, 0, {4, 5, 6, 7});
    EXPECT_EQ(IsContiguous(tensor), false);
    aclDestroyTensor(tensor);

    tensor = CreateAclTensor({2, 3, 32, 16}, {3072, 512, 16, 1}, 0, {4, 3, 32, 16});
    EXPECT_EQ(IsContiguous(tensor), false);
    aclDestroyTensor(tensor);
}

TEST_F(TensorViewUtilsTest, Validate)
{
    // auto tensor = CreateAclTensor({4, 5, 6, 7}, {210, 42, 7, 1}, 0, {4, 5, 6, 7});
    //    tensor->SetViewStrides(op::Strides({42, 7, 1}));
    //    EXPECT_FALSE(op::Validate(tensor));

    auto tensor = CreateAclTensor({4, 5, 6, 7}, {210, 42, 7, 1}, 0, {4, 5, 6, 7});
    EXPECT_TRUE(op::Validate(tensor));
    aclDestroyTensor(tensor);

    tensor = CreateAclTensor({4, 5, 6, 7}, {211, 42, 7, 1}, 0, {4, 5, 6, 7});
    EXPECT_FALSE(op::Validate(tensor));
    aclDestroyTensor(tensor);
}

TEST_F(TensorViewUtilsTest, CanPickViewAsContiguous)
{
    auto tensor = CreateAclTensor({4, 5, 6, 7}, {42, 0, 0, 1}, 0, {4, 1, 6, 7});
    EXPECT_FALSE(op::CanPickViewAsContiguous({tensor, tensor, tensor}));
    aclDestroyTensor(tensor);

    tensor = CreateAclTensor({4, 6, 5, 7}, {42, 7, 0, 1}, 0, {4, 1, 6, 7});
    EXPECT_FALSE(op::CanPickViewAsContiguous({tensor, tensor, tensor}));

    auto tensor2 = CreateAclTensor({3, 5, 5, 7}, {42, 0, 0, 1}, 0, {4, 1, 6, 7});
    EXPECT_FALSE(op::CanPickViewAsContiguous({tensor, tensor2, tensor}));
    aclDestroyTensor(tensor);
    aclDestroyTensor(tensor2);

    tensor = CreateAclTensor({4, 5, 6, 7}, {210, 42, 7, 1}, 0, {4, 5, 6, 7});
    EXPECT_TRUE(op::CanPickViewAsContiguous({tensor, tensor, tensor}));
    aclDestroyTensor(tensor);

    tensor = CreateAclTensor({5, 4, 6, 7}, {42, 210, 7, 1}, 0, {4, 5, 6, 7});
    EXPECT_TRUE(op::CanPickViewAsContiguous({tensor, tensor, tensor}));
    aclDestroyTensor(tensor);

    tensor = CreateAclTensor({4, 5, 6, 7}, {42, 0, 7, 1}, 0, {4, 1, 6, 7});
    EXPECT_FALSE(op::CanPickViewAsContiguous({tensor, tensor, tensor}));
    aclDestroyTensor(tensor);
}