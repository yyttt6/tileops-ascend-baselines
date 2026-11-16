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
#include <array>
#include <memory>
#include "opdev/fast_vector.h"
#include "acl/acl.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"

OP_TYPE_REGISTER(Add);
OP_TYPE_REGISTER(TransData)
OP_TYPE_REGISTER(Relu);
OP_TYPE_REGISTER(ReduceSum);
OP_TYPE_REGISTER(SortDemo);

using graphStatus = uint32_t;

class InferShapeUt : public testing::Test {
protected:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}
};

// TEST_F(InferShapeUt, InferShapeTestCase0)
// {
//     const auto infer_shape_func = [](gert::InferShapeContext *context) -> graphStatus {
//         auto input_shape = context->GetInputShape(1);
//         auto output = context->GetOutputShape(0);
//         *output = *input_shape;
//         return ACLNN_SUCCESS;
//     };
//     IMPL_OP(Add).InferShape(infer_shape_func);
//     auto space_registry = std::make_shared<gert::OpImplSpaceRegistry>();
//     auto registry_holder = std::make_shared<gert::OpImplRegistryHolder>();
//     gert::OpImplKernelRegistry::OpImplFunctions op_impl_func;
//     op_impl_func.infer_shape = infer_shape_func;
//     registry_holder->AddTypesToImpl("Add", op_impl_func);
//     space_registry->AddRegistry(registry_holder);
//     gert::DefaultOpImplSpaceRegistry::GetInstance().SetDefaultSpaceRegistry(space_registry);
//
//     op::Shape selfShape{33, 15, 1, 48};
//     op::Shape otherShape{33, 15, 14, 48};
//
//     auto self = std::make_unique<aclTensor>(selfShape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND, nullptr);
//     auto other = std::make_unique<aclTensor>(otherShape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND, nullptr);
//     auto out = std::make_unique<aclTensor>(op::DataType::DT_FLOAT, op::Format::FORMAT_ND, op::Format::FORMAT_ND);
//     auto ret = INFER_SHAPE(Add, OP_INPUT(self.get(), other.get()), OP_OUTPUT(out.get()));
//     EXPECT_EQ(ret, ACL_SUCCESS);
//     EXPECT_EQ(out->GetOriginalShape(), otherShape);
//     EXPECT_EQ(out->GetStorageShape(), otherShape);
//     EXPECT_EQ(out->GetViewShape(), otherShape);
// }
//
// TEST_F(InferShapeUt, InferShapeTestCase1)
// {
//     const auto infer_shape_func = [](gert::InferShapeContext *context) -> graphStatus {
//         auto input_shape = context->GetInputShape(0);
//         auto output = context->GetOutputShape(0);
//         *output = *input_shape;
//         return ACLNN_SUCCESS;
//     };
//     IMPL_OP(TransData).InferShape(infer_shape_func);
//
//     auto space_registry = std::make_shared<gert::OpImplSpaceRegistry>();
//     auto registry_holder = std::make_shared<gert::OpImplRegistryHolder>();
//     gert::OpImplKernelRegistry::OpImplFunctions op_impl_func;
//     op_impl_func.infer_shape = infer_shape_func;
//     registry_holder->AddTypesToImpl("TransData", op_impl_func);
//     space_registry->AddRegistry(registry_holder);
//     gert::DefaultOpImplSpaceRegistry::GetInstance().SetDefaultSpaceRegistry(space_registry);
//
//     op::Shape xShape{33, 15, 14, 48};
//
//     auto x = std::make_unique<aclTensor>(xShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_NCHW, nullptr);
//     auto out = std::make_unique<aclTensor>(op::DataType::DT_FLOAT16, op::Format::FORMAT_NC1HWC0,
//                                            op::Format::FORMAT_NCHW);
//     auto ret = INFER_SHAPE(TransData, OP_INPUT(x.get()), OP_OUTPUT(out.get()));
//     EXPECT_EQ(ret, ACL_SUCCESS);
//     EXPECT_EQ(out->GetOriginalShape(), xShape);
//     EXPECT_EQ(out->GetStorageShape(), op::Shape({33, 1, 14, 48, 16}));
//     EXPECT_EQ(out->GetViewShape(), xShape);
// }
//
// TEST_F(InferShapeUt, InferShapeTestCase2)
// {
//     const auto infer_shape_func = [](gert::InferShapeContext *context) -> graphStatus {
//         auto input_shape = context->GetInputShape(0);
//         auto output = context->GetOutputShape(0);
//         *output = *input_shape;
//         return ACLNN_SUCCESS;
//     };
//     IMPL_OP(TransData).InferShape(infer_shape_func);
//
//     auto space_registry = std::make_shared<gert::OpImplSpaceRegistry>();
//     auto registry_holder = std::make_shared<gert::OpImplRegistryHolder>();
//     gert::OpImplKernelRegistry::OpImplFunctions op_impl_func;
//     op_impl_func.infer_shape = infer_shape_func;
//     registry_holder->AddTypesToImpl("TransData", op_impl_func);
//     space_registry->AddRegistry(registry_holder);
//     gert::DefaultOpImplSpaceRegistry::GetInstance().SetDefaultSpaceRegistry(space_registry);
//
//     op::Shape xStorageShape{33, 1, 14, 48, 16};
//     op::Shape xOriginalShape{33, 16, 14, 48};
//
//     auto x = std::make_unique<aclTensor>(xStorageShape,
//                                          xOriginalShape,
//                                          op::DataType::DT_FLOAT16,
//                                          op::Format::FORMAT_NC1HWC0,
//                                          op::Format::FORMAT_NCHW,
//                                          nullptr);
//     auto out = std::make_unique<aclTensor>(op::DataType::DT_FLOAT16, op::Format::FORMAT_NCHW,
//     op::Format::FORMAT_NCHW); auto ret = INFER_SHAPE(TransData, OP_INPUT(x.get()), OP_OUTPUT(out.get()));
//     EXPECT_EQ(ret, ACL_SUCCESS);
//     EXPECT_EQ(out->GetOriginalShape(), xOriginalShape);
//     EXPECT_EQ(out->GetStorageShape(), xOriginalShape);
//     EXPECT_EQ(out->GetViewShape(), xOriginalShape);
// }
//
// TEST_F(InferShapeUt, InferShapeTestCase3)
// {
//     const auto infer_shape_func = [](gert::InferShapeContext *context) -> graphStatus {
//         auto input_shape = context->GetInputShape(0);
//         auto output = context->GetOutputShape(0);
//         *output = *input_shape;
//         return ACLNN_SUCCESS;
//     };
//     IMPL_OP(TransData).InferShape(infer_shape_func);
//
//     auto space_registry = std::make_shared<gert::OpImplSpaceRegistry>();
//     auto registry_holder = std::make_shared<gert::OpImplRegistryHolder>();
//     gert::OpImplKernelRegistry::OpImplFunctions op_impl_func;
//     op_impl_func.infer_shape = infer_shape_func;
//     registry_holder->AddTypesToImpl("TransData", op_impl_func);
//     space_registry->AddRegistry(registry_holder);
//     gert::DefaultOpImplSpaceRegistry::GetInstance().SetDefaultSpaceRegistry(space_registry);
//
//     op::Shape selfShape{33, 15, 1, 48};
//
//     auto self = std::make_unique<aclTensor>(selfShape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND, nullptr);
//     auto out = std::make_unique<aclTensor>(op::DataType::DT_FLOAT, op::Format::FORMAT_ND, op::Format::FORMAT_ND);
//     auto ret = INFER_SHAPE(Relu, OP_INPUT(self.get()), OP_OUTPUT(out.get()));
//     EXPECT_EQ(ret, ACL_SUCCESS);
//     EXPECT_EQ(out->GetOriginalShape(), selfShape);
//     EXPECT_EQ(out->GetStorageShape(), selfShape);
//     EXPECT_EQ(out->GetViewShape(), selfShape);
// }
//
// TEST_F(InferShapeUt, InferShapeTestCase4)
// {
//     const auto infer_shape_func = [](gert::InferShapeContext *context) -> graphStatus {
//         auto output = context->GetOutputShape(0);
//         *output = op::Shape{15, 14, 48};
//         return ACLNN_SUCCESS;
//     };
//     IMPL_OP(ReduceSum).InferShape(infer_shape_func);
//
//     auto space_registry = std::make_shared<gert::OpImplSpaceRegistry>();
//     auto registry_holder = std::make_shared<gert::OpImplRegistryHolder>();
//     gert::OpImplKernelRegistry::OpImplFunctions op_impl_func;
//     op_impl_func.infer_shape = infer_shape_func;
//     registry_holder->AddTypesToImpl("ReduceSum", op_impl_func);
//     space_registry->AddRegistry(registry_holder);
//     gert::DefaultOpImplSpaceRegistry::GetInstance().SetDefaultSpaceRegistry(space_registry);
//
//     op::Shape xShape{33, 15, 14, 48};
//     op::Shape expectShape{15, 14, 48};
//     auto x = std::make_unique<aclTensor>(xShape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND, nullptr);
//     std::array<int64_t, 1> axes{0};
//     aclIntArray rAxesArray{axes.data(), axes.size()};
//     auto rAxesTensor = std::make_unique<aclTensor>(&rAxesArray, op::DataType::DT_INT64);
//     auto out = std::make_unique<aclTensor>(op::DataType::DT_FLOAT, op::Format::FORMAT_ND, op::Format::FORMAT_ND);
//     auto ret = INFER_SHAPE(ReduceSum, OP_INPUT(x.get(), rAxesTensor.get()), OP_OUTPUT(out.get()), OP_ATTR(false));
//     EXPECT_EQ(ret, ACL_SUCCESS);
//     EXPECT_EQ(out->GetOriginalShape(), expectShape);
//     EXPECT_EQ(out->GetStorageShape(), expectShape);
//     EXPECT_EQ(out->GetViewShape(), expectShape);
// }
//
// TEST_F(InferShapeUt, InferShapeTestCase5)
// {
//     const auto infer_shape_func = [](gert::InferShapeContext *context) -> graphStatus {
//         auto output = context->GetOutputShape(0);
//         *output = op::Shape{1, 15, 14, 48};
//         return ACLNN_SUCCESS;
//     };
//     IMPL_OP(ReduceSum).InferShape(infer_shape_func);
//
//     auto space_registry = std::make_shared<gert::OpImplSpaceRegistry>();
//     auto registry_holder = std::make_shared<gert::OpImplRegistryHolder>();
//     gert::OpImplKernelRegistry::OpImplFunctions op_impl_func;
//     op_impl_func.infer_shape = infer_shape_func;
//     registry_holder->AddTypesToImpl("ReduceSum", op_impl_func);
//     space_registry->AddRegistry(registry_holder);
//     gert::DefaultOpImplSpaceRegistry::GetInstance().SetDefaultSpaceRegistry(space_registry);
//
//     op::Shape xShape{33, 15, 14, 48};
//     op::Shape expectShape{1, 15, 14, 48};
//     auto x = std::make_unique<aclTensor>(xShape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND, nullptr);
//     std::array<int64_t, 1> axes{0};
//     aclIntArray rAxesArray{axes.data(), axes.size()};
//     auto rAxesTensor = std::make_unique<aclTensor>(&rAxesArray, op::DataType::DT_INT64);
//     auto out = std::make_unique<aclTensor>(op::DataType::DT_FLOAT, op::Format::FORMAT_ND, op::Format::FORMAT_ND);
//     auto ret = INFER_SHAPE(ReduceSum, OP_INPUT(x.get(), rAxesTensor.get()), OP_OUTPUT(out.get()), OP_ATTR(true));
//     EXPECT_EQ(ret, ACL_SUCCESS);
//     EXPECT_EQ(out->GetOriginalShape(), expectShape);
//     EXPECT_EQ(out->GetStorageShape(), expectShape);
//     EXPECT_EQ(out->GetViewShape(), expectShape);
// }

// TEST_F(InferShapeUt, InferShapeTestCase6)
//{
//     const auto infer_shape_func = [](gert::InferShapeContext *context) -> graphStatus {
//         auto input = context->GetInputShape(0);
//         auto output0 = context->GetOutputShape(0);
//         auto output1 = context->GetOutputShape(1);
//         *output0 = *input;
//         *output1 = *input;
//         return ACLNN_SUCCESS;
//     };
//
//     auto space_registry = std::make_shared<gert::OpImplSpaceRegistry>();
//     auto registry_holder = std::make_shared<gert::OpImplRegistryHolder>();
//     gert::OpImplKernelRegistry::OpImplFunctions op_impl_func;
//     op_impl_func.infer_shape = infer_shape_func;
//     registry_holder->AddTypesToImpl("SortDemo", op_impl_func);
//     space_registry->AddRegistry(registry_holder);
//     gert::DefaultOpImplSpaceRegistry::GetInstance().SetDefaultSpaceRegistry(space_registry);
//
//     op::Shape selfShape{1, 1, 1, 1, 2, 4};
//     auto self = std::make_unique<aclTensor>(selfShape, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND, nullptr);
//
//     auto values = std::make_unique<aclTensor>(op::DataType::DT_FLOAT16, op::Format::FORMAT_ND,
//     op::Format::FORMAT_ND); auto indices = std::make_unique<aclTensor>(op::DataType::DT_INT32, op::Format::FORMAT_ND,
//     op::Format::FORMAT_ND);
//
//     op::FVector<aclTensor *> tensors{values.get(), indices.get()};
//     auto outputs = std::make_unique<aclTensorList>(tensors.data(), tensors.size());
//
//     auto ret = INFER_SHAPE(SortDemo, OP_INPUT(self.get()), OP_OUTPUT(outputs.get()), OP_ATTR(-1, false));
//     EXPECT_EQ(ret, ACL_SUCCESS);
//     EXPECT_EQ(values->GetOriginalShape(), selfShape);
//     EXPECT_EQ(values->GetStorageShape(), selfShape);
//     EXPECT_EQ(values->GetViewShape(), selfShape);
//     EXPECT_EQ(indices->GetOriginalShape(), selfShape);
//     EXPECT_EQ(indices->GetStorageShape(), selfShape);
//     EXPECT_EQ(indices->GetViewShape(), selfShape);
// }
