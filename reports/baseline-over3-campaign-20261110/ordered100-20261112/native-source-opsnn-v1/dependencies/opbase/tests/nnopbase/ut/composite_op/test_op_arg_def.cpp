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
#include <iostream>
#include <memory>
#include <stdlib.h>

#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnn/aclnn_base.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_errno.h"
#include "register/op_impl_registry.h"
#include "rts_arg.h"
#include "opdev/op_arg_def.h"
#include "bridge_pool.h"

using namespace op::internal;

class OpArgDefUt : public testing::Test {
protected:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}
};

TEST_F(OpArgDefUt, OpArgType_aclTensor)
{
    // OPARG_ACLTENSOR
    std::vector<op::DataType> dtypes{op::DataType::DT_INT8,   op::DataType::DT_UINT8,  op::DataType::DT_INT16,
                                     op::DataType::DT_INT32,  op::DataType::DT_INT64,  op::DataType::DT_UINT16,
                                     op::DataType::DT_UINT32, op::DataType::DT_UINT64, op::DataType::DT_FLOAT16,
                                     op::DataType::DT_DOUBLE, op::DataType::DT_FLOAT,  op::DataType::DT_BF16,
                                     op::DataType::DT_BOOL};
    op::Shape tShape{1, 2, 3};
    for (auto dtype : dtypes) {
        aclTensor t1(tShape, dtype, ge::FORMAT_ND, nullptr);
        aclTensor t2(tShape, dtype, ge::FORMAT_ND, nullptr);

        auto input_arg = OP_INPUT(&t1);
        auto output_arg = OP_OUTPUT(&t2);

        auto ctx = op::MakeOpArgContext(input_arg, output_arg);

        EXPECT_EQ(ctx->ContainsOpArgType(op::OpArgDef::OP_INPUT_ARG), true);
        EXPECT_EQ(ctx->ContainsOpArgType(op::OpArgDef::OP_OUTPUT_ARG), true);

        op::OpArgList input = *ctx->GetOpArg(op::OpArgDef::OP_INPUT_ARG);
        op::OpArgList output = *ctx->GetOpArg(op::OpArgDef::OP_OUTPUT_ARG);

        EXPECT_EQ(input.count, 1);
        EXPECT_EQ(output.count, 1);
        EXPECT_EQ(input.args[0].type, op::OpArgType::OPARG_ACLTENSOR);
        EXPECT_EQ(output.args[0].type, op::OpArgType::OPARG_ACLTENSOR);
        op::DestroyOpArgContext(ctx);
    }
}

TEST_F(OpArgDefUt, OpArgType_aclTensorList)
{
    // aclTensorList
    op::Shape tShape{1, 2, 3};
    aclTensor t3(tShape, op::DataType::DT_FLOAT16, ge::FORMAT_ND, nullptr);
    aclTensor t4(tShape, op::DataType::DT_FLOAT16, ge::FORMAT_ND, nullptr);
    aclTensor t5(tShape, op::DataType::DT_FLOAT16, ge::FORMAT_ND, nullptr);
    aclTensor t6(tShape, op::DataType::DT_FLOAT16, ge::FORMAT_ND, nullptr);
    aclTensor* list[] = {&t3, &t4};
    auto tensorList = aclCreateTensorList(list, 2);

    auto input_arg1 = OP_INPUT(tensorList, &t5);
    auto output_arg1 = OP_OUTPUT(&t6);

    auto ctx1 = op::MakeOpArgContext(input_arg1, output_arg1);

    EXPECT_EQ(ctx1->ContainsOpArgType(op::OpArgDef::OP_INPUT_ARG), true);
    EXPECT_EQ(ctx1->ContainsOpArgType(op::OpArgDef::OP_OUTPUT_ARG), true);
    EXPECT_EQ(ctx1->ContainsOpArgType(op::OpArgDef::OP_ATTR_ARG), false);
    EXPECT_EQ(ctx1->ContainsOpArgType(op::OpArgDef::OP_WORKSPACE_ARG), false);
    EXPECT_EQ(ctx1->ContainsOpArgType(op::OpArgDef::OP_OUTSHAPE_ARG), false);
    EXPECT_EQ(ctx1->ContainsOpArgType(op::OpArgDef::OP_OPTION_ARG), false);
    EXPECT_EQ(ctx1->ContainsOpArgType(op::OpArgDef::OP_MEMSET_ARG), false);
    EXPECT_EQ(ctx1->ContainsOpArgType(op::OpArgDef::OP_ARG_DEF_BUTT), false);

    op::OpArgList input1 = *ctx1->GetOpArg(op::OpArgDef::OP_INPUT_ARG);
    op::OpArgList output1 = *ctx1->GetOpArg(op::OpArgDef::OP_OUTPUT_ARG);

    EXPECT_EQ(input1.count, 2);
    EXPECT_EQ(output1.count, 1);
    EXPECT_EQ(input1.args[0].type, op::OpArgType::OPARG_ACLTENSOR_LIST);
    EXPECT_EQ(input1.args[1].type, op::OpArgType::OPARG_ACLTENSOR);
    EXPECT_EQ(output1.args[0].type, op::OpArgType::OPARG_ACLTENSOR);
    op::DestroyOpArgContext(ctx1);
    delete tensorList;

    // aclTensorList1 contains nullptr
    aclTensor* list2[] = {&t3, &t4, nullptr};
    aclTensorList* tensorList2 = aclCreateTensorList(list2, 3);
    aclTensor outShapeTensor(tShape, op::DataType::DT_FLOAT16, ge::FORMAT_ND, nullptr);

    auto input_arg2 = OP_INPUT(tensorList2);
    auto output_arg2 = OP_OUTPUT(&t5);

    auto ctx2 = op::MakeOpArgContext(input_arg2, output_arg2, OP_OUTSHAPE({&outShapeTensor, 0}));

    EXPECT_EQ(ctx2->ContainsOpArgType(op::OpArgDef::OP_INPUT_ARG), true);
    EXPECT_EQ(ctx2->ContainsOpArgType(op::OpArgDef::OP_OUTPUT_ARG), true);
    EXPECT_EQ(ctx2->ContainsOpArgType(op::OpArgDef::OP_OUTSHAPE_ARG), true);
    EXPECT_EQ(ctx2->ContainsOpArgType(op::OpArgDef::OP_ATTR_ARG), false);
    EXPECT_EQ(ctx2->ContainsOpArgType(op::OpArgDef::OP_WORKSPACE_ARG), false);
    EXPECT_EQ(ctx2->ContainsOpArgType(op::OpArgDef::OP_OPTION_ARG), false);
    EXPECT_EQ(ctx2->ContainsOpArgType(op::OpArgDef::OP_MEMSET_ARG), false);
    EXPECT_EQ(ctx2->ContainsOpArgType(op::OpArgDef::OP_ARG_DEF_BUTT), false);

    op::OpArgList input2 = *ctx2->GetOpArg(op::OpArgDef::OP_INPUT_ARG);
    op::OpArgList output2 = *ctx2->GetOpArg(op::OpArgDef::OP_OUTPUT_ARG);

    EXPECT_EQ(input2.count, 1);
    EXPECT_EQ(output2.count, 1);
    EXPECT_EQ(input2.args[0].type, op::OpArgType::OPARG_ACLTENSOR_LIST);
    EXPECT_EQ(output2.args[0].type, op::OpArgType::OPARG_ACLTENSOR);
    op::DestroyOpArgContext(ctx2);
    delete tensorList2;
}

TEST_F(OpArgDefUt, OpArgType_aclScalar)
{
    // aclScalar
    float fpValue = 3.2;
    aclScalar scalar(&fpValue, op::DataType::DT_FLOAT);

    op::Shape tShape{1, 2, 3};
    aclTensor t3(tShape, op::DataType::DT_FLOAT16, ge::FORMAT_ND, nullptr);
    aclTensor t4(tShape, op::DataType::DT_FLOAT16, ge::FORMAT_ND, nullptr);
    aclTensor t5(tShape, op::DataType::DT_FLOAT16, ge::FORMAT_ND, nullptr);
    aclTensor t6(tShape, op::DataType::DT_FLOAT16, ge::FORMAT_ND, nullptr);

    auto input_arg1 = OP_INPUT(&t3, &t4, &scalar);
    auto output_arg1 = OP_OUTPUT(&t5);
    auto attr_arg1 = OP_ATTR(&t6);

    auto ctx1 = op::MakeOpArgContext(input_arg1, output_arg1, attr_arg1);

    EXPECT_EQ(ctx1->ContainsOpArgType(op::OpArgDef::OP_INPUT_ARG), true);
    EXPECT_EQ(ctx1->ContainsOpArgType(op::OpArgDef::OP_OUTPUT_ARG), true);
    EXPECT_EQ(ctx1->ContainsOpArgType(op::OpArgDef::OP_ATTR_ARG), true);

    op::OpArgList input1 = *ctx1->GetOpArg(op::OpArgDef::OP_INPUT_ARG);
    op::OpArgList output1 = *ctx1->GetOpArg(op::OpArgDef::OP_OUTPUT_ARG);
    op::OpArgList attr1 = *ctx1->GetOpArg(op::OpArgDef::OP_ATTR_ARG);

    EXPECT_EQ(input1.count, 3);
    EXPECT_EQ(output1.count, 1);
    EXPECT_EQ(attr1.count, 1);
    EXPECT_EQ(input1.args[2].type, op::OpArgType::OPARG_ACLSCALAR);
    EXPECT_EQ(output1.args[0].type, op::OpArgType::OPARG_ACLTENSOR);
    EXPECT_EQ(attr1.args[0].type, op::OpArgType::OPARG_ACLTENSOR);
    op::DestroyOpArgContext(ctx1);
}

TEST_F(OpArgDefUt, OpArgType_string)
{
    op::Shape tShape{1, 2, 3};
    aclTensor t3(tShape, op::DataType::DT_FLOAT16, ge::FORMAT_ND, nullptr);
    aclTensor t4(tShape, op::DataType::DT_FLOAT16, ge::FORMAT_ND, nullptr);
    aclTensor t5(tShape, op::DataType::DT_FLOAT16, ge::FORMAT_ND, nullptr);
    aclTensor t6(tShape, op::DataType::DT_FLOAT16, ge::FORMAT_ND, nullptr);

    std::string name = "hello";

    auto input_arg1 = OP_INPUT(&t3, &t4);
    auto output_arg1 = OP_OUTPUT(&t5);
    auto attr_arg1 = OP_ATTR(&t6, name);

    auto ctx1 = op::MakeOpArgContext(input_arg1, output_arg1, attr_arg1);

    EXPECT_EQ(ctx1->ContainsOpArgType(op::OpArgDef::OP_INPUT_ARG), true);
    EXPECT_EQ(ctx1->ContainsOpArgType(op::OpArgDef::OP_OUTPUT_ARG), true);
    EXPECT_EQ(ctx1->ContainsOpArgType(op::OpArgDef::OP_ATTR_ARG), true);

    op::OpArgList input1 = *ctx1->GetOpArg(op::OpArgDef::OP_INPUT_ARG);
    op::OpArgList output1 = *ctx1->GetOpArg(op::OpArgDef::OP_OUTPUT_ARG);
    op::OpArgList attr1 = *ctx1->GetOpArg(op::OpArgDef::OP_ATTR_ARG);

    EXPECT_EQ(input1.count, 2);
    EXPECT_EQ(output1.count, 1);
    EXPECT_EQ(attr1.count, 2);
    EXPECT_EQ(attr1.args[0].type, op::OpArgType::OPARG_ACLTENSOR);
    EXPECT_EQ(attr1.args[1].type, op::OpArgType::OPARG_STRING);
    op::DestroyOpArgContext(ctx1);
}

TEST_F(OpArgDefUt, OpArgType_arithmetic)
{
    op::Shape tShape{1, 2, 3};
    aclTensor t1(tShape, op::DataType::DT_FLOAT16, ge::FORMAT_ND, nullptr);
    aclTensor t2(tShape, op::DataType::DT_FLOAT16, ge::FORMAT_ND, nullptr);
    aclTensor t3(tShape, op::DataType::DT_FLOAT16, ge::FORMAT_ND, nullptr);
    aclTensor t4(tShape, op::DataType::DT_FLOAT16, ge::FORMAT_ND, nullptr);

    bool name = true;
    int32_t dim = 0;
    uint32_t dim1 = 0;
    float weight = 0.5;
    double weight1 = 0.5;

    auto input_arg1 = OP_INPUT(&t1, &t2);
    auto output_arg1 = OP_OUTPUT(&t3);
    auto attr_arg1 = OP_ATTR(&t4, name, dim, dim1, weight, weight1);

    auto ctx1 = op::MakeOpArgContext(input_arg1, output_arg1, attr_arg1);

    EXPECT_EQ(ctx1->ContainsOpArgType(op::OpArgDef::OP_INPUT_ARG), true);
    EXPECT_EQ(ctx1->ContainsOpArgType(op::OpArgDef::OP_OUTPUT_ARG), true);
    EXPECT_EQ(ctx1->ContainsOpArgType(op::OpArgDef::OP_ATTR_ARG), true);

    op::OpArgList input1 = *ctx1->GetOpArg(op::OpArgDef::OP_INPUT_ARG);
    op::OpArgList output1 = *ctx1->GetOpArg(op::OpArgDef::OP_OUTPUT_ARG);
    op::OpArgList attr1 = *ctx1->GetOpArg(op::OpArgDef::OP_ATTR_ARG);

    EXPECT_EQ(input1.count, 2);
    EXPECT_EQ(output1.count, 1);
    EXPECT_EQ(attr1.count, 6);
    EXPECT_EQ(attr1.args[0].type, op::OpArgType::OPARG_ACLTENSOR);
    EXPECT_EQ(attr1.args[1].type, op::OpArgType::OPARG_BOOL);
    EXPECT_EQ(attr1.args[2].type, op::OpArgType::OPARG_INT);
    EXPECT_EQ(attr1[3].type, op::OpArgType::OPARG_UINT);
    EXPECT_EQ(attr1[4].type, op::OpArgType::OPARG_FLOAT);
    EXPECT_EQ(attr1[5].type, op::OpArgType::OPARG_DOUBLE);
    op::DestroyOpArgContext(ctx1);
}

TEST_F(OpArgDefUt, OpArgType_DATATYPE_and_list)
{
    op::Shape tShape{1, 2, 3};
    aclTensor t1(tShape, op::DataType::DT_FLOAT16, ge::FORMAT_ND, nullptr);
    aclTensor t2(tShape, op::DataType::DT_FLOAT16, ge::FORMAT_ND, nullptr);
    aclTensor t3(tShape, op::DataType::DT_FLOAT16, ge::FORMAT_ND, nullptr);

    op::DataType outDdataType = op::DataType::DT_FLOAT16;
    bool value = true;
    aclBoolArray name(&value, 1);
    int64_t valueDim[] = {0, 1, 2};
    aclIntArray dim(valueDim, 3);
    float valueFloat = 0.5;
    aclFloatArray weight(&valueFloat, 1);

    auto input_arg1 = OP_INPUT(&t1, &t2);
    auto output_arg1 = OP_OUTPUT(&t3);
    auto attr_arg1 = OP_ATTR(outDdataType, &dim, &weight, &name);

    auto ctx1 = op::MakeOpArgContext(input_arg1, output_arg1, attr_arg1);

    EXPECT_EQ(ctx1->ContainsOpArgType(op::OpArgDef::OP_INPUT_ARG), true);
    EXPECT_EQ(ctx1->ContainsOpArgType(op::OpArgDef::OP_OUTPUT_ARG), true);
    EXPECT_EQ(ctx1->ContainsOpArgType(op::OpArgDef::OP_ATTR_ARG), true);

    op::OpArgList input1 = *ctx1->GetOpArg(op::OpArgDef::OP_INPUT_ARG);
    op::OpArgList output1 = *ctx1->GetOpArg(op::OpArgDef::OP_OUTPUT_ARG);
    op::OpArgList attr1 = *ctx1->GetOpArg(op::OpArgDef::OP_ATTR_ARG);

    EXPECT_EQ(input1.count, 2);
    EXPECT_EQ(output1.count, 1);
    EXPECT_EQ(attr1.count, 4);
    EXPECT_EQ(attr1.args[0].type, op::OpArgType::OPARG_DATATYPE);
    EXPECT_EQ(attr1.args[1].type, op::OpArgType::OPARG_INT_LIST);
    EXPECT_EQ(attr1.args[2].type, op::OpArgType::OPARG_FLOAT_LIST);
    EXPECT_EQ(attr1.args[3].type, op::OpArgType::OPARG_BOOL_LIST);
    op::DestroyOpArgContext(ctx1);
}

TEST_F(OpArgDefUt, OpArgType_other)
{
    op::Shape tShape{1, 2, 3};
    aclTensor t1(tShape, op::DataType::DT_FLOAT16, ge::FORMAT_ND, nullptr);
    aclTensor t2(tShape, op::DataType::DT_FLOAT16, ge::FORMAT_ND, nullptr);
    aclTensor t3(tShape, op::DataType::DT_FLOAT16, ge::FORMAT_ND, nullptr);

    op::OpImplMode mode = op::OpImplMode::IMPL_MODE_HIGH_PRECISION;

    std::vector<std::tuple<void*, const aclTensor*>> tensor;
    tensor.emplace_back(t1.GetData(), &t1);
    auto workspace = OP_WORKSPACE(std::move(tensor));

    auto ctx1 = op::MakeOpArgContext(workspace, OP_OPTION(mode));

    EXPECT_EQ(ctx1->ContainsOpArgType(op::OpArgDef::OP_OPTION_ARG), true);
    EXPECT_EQ(ctx1->ContainsOpArgType(op::OpArgDef::OP_WORKSPACE_ARG), true);

    op::OpArgList option = *ctx1->GetOpArg(op::OpArgDef::OP_OPTION_ARG);
    op::OpArgList workspaceArg = *ctx1->GetOpArg(op::OpArgDef::OP_WORKSPACE_ARG);

    EXPECT_EQ(option.count, 1);
    EXPECT_EQ(workspaceArg.count, 1);
    EXPECT_EQ(option.args[0].type, op::OpArgType::OPARG_IMPLMODE);
    EXPECT_EQ(workspaceArg.args[0].type, op::OpArgType::OPARG_MEMSET_WORKSPACE);
    op::DestroyOpArgContext(ctx1);
}

TEST_F(OpArgDefUt, OpArgValue_String)
{
    std::string str = "clkncskzn";
    op::OpArgValue opArgValue(str);
    EXPECT_NE(opArgValue.data.pointer, nullptr);
    op::OpArgValue opArgValue1(&str);
    EXPECT_NE(opArgValue1.data.pointer, nullptr);
    const std::string* str1 = nullptr;
    op::OpArgValue opArgValue2(str1);
    EXPECT_EQ(opArgValue2.data.pointer, nullptr);

    const char* cstr = "clkncskzn";
    op::OpArgValue opArgValue5(cstr);
    EXPECT_NE(opArgValue5.data.pointer, nullptr);
    const char* cstr1 = nullptr;
    op::OpArgValue opArgValue6(cstr1);
    EXPECT_EQ(opArgValue6.data.pointer, nullptr);
}

TEST_F(OpArgDefUt, DeAllocated)
{
    uint8_t* addr = static_cast<uint8_t*>(op::internal::Allocate(10));
    op::DeAllocated(addr);
}

TEST_F(OpArgDefUt, TestAllType)
{
    op::Shape tShape{1, 2, 3};
    int64_t shape[] = {1, 2, 3};
    aclTensor t1(tShape, op::DataType::DT_FLOAT16, ge::FORMAT_ND, nullptr);
    aclTensor t2(tShape, op::DataType::DT_FLOAT16, ge::FORMAT_ND, nullptr);
    aclTensor* nullTensor = nullptr;
    aclTensorList* nullTensorList = nullptr;
    aclTensor* list2[] = {
        aclCreateTensor(shape, 3, aclDataType::ACL_FLOAT, nullptr, 0, aclFormat::ACL_FORMAT_ND, shape, 3, nullptr),
        aclCreateTensor(shape, 3, aclDataType::ACL_FLOAT, nullptr, 0, aclFormat::ACL_FORMAT_ND, shape, 3, nullptr),
        aclCreateTensor(shape, 3, aclDataType::ACL_FLOAT, nullptr, 0, aclFormat::ACL_FORMAT_ND, shape, 3, nullptr)};
    aclTensorList* tensorList = aclCreateTensorList(list2, 3);

    bool boolAttr = false;
    op::DataType dtypeAttr = op::DataType::DT_FLOAT16;
    float scalar_value = 5;
    aclScalar* scalarAttr = aclCreateScalar(&scalar_value, aclDataType::ACL_FLOAT);
    int64_t array[] = {3, 4, 5};
    aclIntArray* intArrayAttr = aclCreateIntArray(array, sizeof(array) / sizeof(array[0]));
    std::string stringAttr = "123";
    std::string s = "456";
    std::string* stringPAttr = &s;
    char* charPAttr = "789";
    double doubleAttr = 3.14159;
    float floatAttr = 3.14;
    int32_t intAttr = -1;
    uint32_t uintAttr = 1;
    int64_t int64Attr = -2;
    uint64_t uint64Attr = 2;

    op::OpImplMode implMode = op::OpImplMode::IMPL_MODE_HIGH_PRECISION;
    aclTensor outShapeTensor(tShape, op::DataType::DT_FLOAT16, ge::FORMAT_ND, nullptr);

    auto input_arg = OP_INPUT(&t1, nullTensor, nullTensorList, tensorList);
    auto output_arg = OP_OUTPUT(&t2);
    auto attr_arg = OP_ATTR(boolAttr, dtypeAttr, scalarAttr, intArrayAttr, stringAttr, stringPAttr, charPAttr,
                            doubleAttr, floatAttr, intAttr, uintAttr, int64Attr, uint64Attr);
    auto option_arg = OP_OPTION(implMode);
    auto outshape_arg = OP_OUTSHAPE({&outShapeTensor, 0});

    auto ctx = op::MakeOpArgContext(input_arg, output_arg, attr_arg, outshape_arg);

    EXPECT_EQ(ctx->ContainsOpArgType(op::OpArgDef::OP_INPUT_ARG), true);
    EXPECT_EQ(ctx->ContainsOpArgType(op::OpArgDef::OP_OUTPUT_ARG), true);
    EXPECT_EQ(ctx->ContainsOpArgType(op::OpArgDef::OP_ATTR_ARG), true);
    EXPECT_EQ(ctx->ContainsOpArgType(op::OpArgDef::OP_OUTSHAPE_ARG), true);

    op::OpArgList& input = *ctx->GetOpArg(op::OpArgDef::OP_INPUT_ARG);
    op::OpArgList& output = *ctx->GetOpArg(op::OpArgDef::OP_OUTPUT_ARG);
    op::OpArgList& attr = *ctx->GetOpArg(op::OpArgDef::OP_ATTR_ARG);
    op::OpArgList& outshape = *ctx->GetOpArg(op::OpArgDef::OP_OUTSHAPE_ARG);

    EXPECT_EQ(input.count, 4);
    EXPECT_EQ(output.count, 1);
    EXPECT_EQ(attr.count, 13);
    EXPECT_EQ(outshape.count, 2);

    EXPECT_EQ(input.args[0].type, op::OpArgType::OPARG_ACLTENSOR);
    EXPECT_EQ(input.args[0]->pointer, &t1);
    EXPECT_EQ(input.args[1].type, op::OpArgType::OPARG_ACLTENSOR);
    EXPECT_EQ(input.args[1]->pointer, nullptr);
    EXPECT_EQ(input.args[2].type, op::OpArgType::OPARG_ACLTENSOR_LIST);
    EXPECT_EQ(input.args[2]->pointer, nullptr);
    EXPECT_EQ(input.args[3].type, op::OpArgType::OPARG_ACLTENSOR_LIST);
    EXPECT_EQ(input.args[3]->pointer, tensorList);
    EXPECT_EQ(output.args[0].type, op::OpArgType::OPARG_ACLTENSOR);
    EXPECT_EQ(output.args[0]->pointer, &t2);

    EXPECT_EQ(attr.args[0].type, op::OpArgType::OPARG_BOOL);
    EXPECT_EQ(attr.args[0]->value, static_cast<uint64_t>(false));
    EXPECT_EQ(attr.args[1].type, op::OpArgType::OPARG_DATATYPE);
    EXPECT_EQ(attr.args[1]->value, static_cast<uint64_t>(dtypeAttr));
    EXPECT_EQ(attr.args[2].type, op::OpArgType::OPARG_ACLSCALAR);
    EXPECT_EQ(attr.args[2]->pointer, scalarAttr);
    EXPECT_EQ(attr.args[3].type, op::OpArgType::OPARG_INT_LIST);
    EXPECT_EQ(attr.args[3]->pointer, intArrayAttr);
    EXPECT_EQ(attr.args[4].type, op::OpArgType::OPARG_STRING);
    EXPECT_EQ(0, strcmp(static_cast<char*>(attr.args[4]->pointer), "123"));
    EXPECT_EQ(attr.args[5].type, op::OpArgType::OPARG_STRING);
    EXPECT_EQ(0, strcmp(static_cast<char*>(attr.args[5]->pointer), "456"));
    EXPECT_EQ(attr.args[6].type, op::OpArgType::OPARG_STRING);
    EXPECT_EQ(0, strcmp(static_cast<char*>(attr.args[6]->pointer), "789"));
    EXPECT_EQ(attr.args[7].type, op::OpArgType::OPARG_DOUBLE);
    EXPECT_LE(attr.args[7]->dvalue - 3.14159, std::numeric_limits<double>::epsilon());
    EXPECT_EQ(attr.args[8].type, op::OpArgType::OPARG_FLOAT);
    EXPECT_LE(attr.args[8]->fvalue - 3.14, std::numeric_limits<float>::epsilon());
    EXPECT_EQ(attr.args[9].type, op::OpArgType::OPARG_INT);
    EXPECT_EQ(static_cast<int64_t>(attr.args[9]->ivalue), -1);
    EXPECT_EQ(attr.args[10].type, op::OpArgType::OPARG_UINT);
    EXPECT_EQ(static_cast<uint64_t>(attr.args[10]->value), 1);
    EXPECT_EQ(attr.args[11].type, op::OpArgType::OPARG_INT);
    EXPECT_EQ(attr.args[11]->ivalue, -2);
    EXPECT_EQ(attr.args[12].type, op::OpArgType::OPARG_INT);
    EXPECT_EQ(attr.args[12]->value, 2);

    op::DestroyOpArgContext(ctx);
    aclDestroyTensorList(tensorList);
    aclDestroyScalar(scalarAttr);
    aclDestroyIntArray(intArrayAttr);
}