/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "opdev/shape_utils.h"
#include "gtest/gtest.h"
#include "shape_inference.h"
#include "op_ctx_def.h"

class TestShapeUtils : public testing::Test {};

TEST_F(TestShapeUtils, TestCheckBroadcastShape)
{
    op::Shape shape1({2, 2});
    op::Shape shape2({2});
    op::Shape shape3({2, 1});
    op::Shape shape4({2, 1});
    op::Shape shape5({2, 5});
    op::Shape shape6({2, 2, 5});
    op::Shape shape7({2, 1, 5});

    EXPECT_TRUE(op::CheckBroadcastShape(shape1, shape2));
    EXPECT_TRUE(op::CheckBroadcastShape(shape2, shape1));
    EXPECT_TRUE(op::CheckBroadcastShape(shape2, shape3));
    EXPECT_FALSE(op::CheckBroadcastShape(shape1, shape5));
    EXPECT_TRUE(op::CheckBroadcastShape(shape3, shape4));
    EXPECT_TRUE(op::CheckBroadcastShape(shape6, shape7));
    EXPECT_FALSE(op::CheckBroadcastShape(shape1, shape6));
}

inline void ClearShape(op::Shape& shape)
{
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        shape[i] = -3;
    }
}

TEST_F(TestShapeUtils, TestUpdateOutputShape)
{
    op::Shape selfShape{33, 15, 64};
    auto self = std::make_unique<aclTensor>(selfShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND, nullptr);

    auto input = OP_INPUT(self.get());
    auto ctx = op::MakeOpArgContext(input);

    auto ret = op::internal::UpdateOutputShape(*ctx->GetOpArg(op::OpArgDef::OP_INPUT_ARG));
    EXPECT_EQ(ret, ACLNN_SUCCESS);
}

TEST_F(TestShapeUtils, TestBroadcastInferShape)
{
    op::Shape shape1({2, 2});
    op::Shape shape2({2});
    op::Shape shape3({2, 1});
    op::Shape shape4({2, 1});
    op::Shape shape5({2, 5});
    op::Shape shape6({2, 2, 5});
    op::Shape shape7({2, 1, 5});
    op::Shape shape8({2, 2, 3, 2});
    op::Shape shape9({2, 3, 2});
    op::Shape outShape;

    ClearShape(outShape);
    EXPECT_TRUE(op::BroadcastInferShape(shape2, shape1, outShape));
    EXPECT_EQ(op::ToString(outShape), op::ToString(op::Shape({2, 2})));
    ClearShape(outShape);
    EXPECT_TRUE(op::BroadcastInferShape(shape1, shape2, outShape));
    EXPECT_EQ(op::ToString(outShape), op::ToString(op::Shape({2, 2})));
    ClearShape(outShape);
    EXPECT_TRUE(op::BroadcastInferShape(shape2, shape3, outShape));
    EXPECT_EQ(op::ToString(outShape), op::ToString(op::Shape({2, 2})));
    ClearShape(outShape);
    EXPECT_FALSE(op::BroadcastInferShape(shape1, shape5, outShape));
    ClearShape(outShape);
    EXPECT_TRUE(op::BroadcastInferShape(shape3, shape4, outShape));
    EXPECT_EQ(op::ToString(outShape), op::ToString(op::Shape({2, 1})));
    ClearShape(outShape);
    EXPECT_TRUE(op::BroadcastInferShape(shape6, shape7, outShape));
    EXPECT_EQ(op::ToString(outShape), op::ToString(op::Shape({2, 2, 5})));
    ClearShape(outShape);
    EXPECT_FALSE(op::BroadcastInferShape(shape1, shape6, outShape));
    ClearShape(outShape);
    EXPECT_TRUE(op::BroadcastInferShape(shape2, shape8, outShape));
    EXPECT_EQ(op::ToString(outShape), op::ToString(op::Shape({2, 2, 3, 2})));
    ClearShape(outShape);
    EXPECT_TRUE(op::BroadcastInferShape(shape8, shape2, outShape));
    EXPECT_EQ(op::ToString(outShape), op::ToString(op::Shape({2, 2, 3, 2})));
    ClearShape(outShape);
    EXPECT_TRUE(op::BroadcastInferShape(shape2, shape9, outShape));
    EXPECT_EQ(op::ToString(outShape), op::ToString(op::Shape({2, 3, 2})));
    ClearShape(outShape);
    EXPECT_TRUE(op::BroadcastInferShape(shape9, shape2, outShape));
    EXPECT_EQ(op::ToString(outShape), op::ToString(op::Shape({2, 3, 2})));

    ClearShape(outShape);
    EXPECT_TRUE(op::BroadcastInferShape(shape8, shape9, outShape));
    EXPECT_EQ(op::ToString(outShape), op::ToString(op::Shape({2, 2, 3, 2})));
    ClearShape(outShape);
    EXPECT_TRUE(op::BroadcastInferShape(shape9, shape8, outShape));
    EXPECT_EQ(op::ToString(outShape), op::ToString(op::Shape({2, 2, 3, 2})));
}