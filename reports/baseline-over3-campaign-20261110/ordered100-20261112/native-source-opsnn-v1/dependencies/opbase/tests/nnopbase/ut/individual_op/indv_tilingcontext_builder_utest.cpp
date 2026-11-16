/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <cstdint>
#include <initializer_list>

#include <gtest/gtest.h>

#include "executor/indv_tilingcontext_builder.h"

namespace {
constexpr uint32_t kTilingContextPadding = 32U;

void SetParamInstance(NnopbaseParamInstance& instance, uint32_t startIndex, uint32_t num, bool isDynamic, bool isInput)
{
    instance.startIndex = startIndex;
    instance.num = num;
    instance.cfgNum = isDynamic ? 2U : 1U;
    instance.isDynamic = isDynamic;
    instance.isInput = isInput;
}

void SetTensor(NnopbaseTensor& tensor, void* addr, std::initializer_list<int64_t> shape, ge::DataType dtype,
               ge::Format format)
{
    tensor.isNull = false;
    GertShape gertShape(shape);
    tensor.rt2Tensor.MutableOriginShape() = gertShape;
    tensor.rt2Tensor.MutableStorageShape() = gertShape;
    tensor.rt2Tensor.SetDataType(dtype);
    tensor.rt2Tensor.SetOriginFormat(format);
    tensor.rt2Tensor.SetStorageFormat(format);
    tensor.rt2Tensor.MutableTensorData().SetPlacement(gert::kOnDeviceHbm);
    tensor.rt2Tensor.SetSize(sizeof(uint64_t));
    ASSERT_EQ(tensor.rt2Tensor.MutableTensorData().SetAddr(addr, nullptr), ge::GRAPH_SUCCESS);
}

size_t ExpectedKernelRunContextLen(uint32_t valueNum)
{
    const size_t offset = valueNum * sizeof(void*) + sizeof(NnopbaseKernelRunContext) + kTilingContextPadding;
    return offset + valueNum * sizeof(NnopbaseAsyncAnyValue);
}
} // namespace

class NnopbaseTilingContextBuilderTest : public testing::Test {
protected:
    void SetUp() override
    {
        executor_.args = &executor_.ownArgs;
        executor_.opType = opType_.data();
    }

    void TearDown() override
    {
        if (executor_.tiling.contextExt.context != nullptr || executor_.tiling.contextExt.nodeExt.buf != nullptr) {
            NnopbaseTilingContextDeInit(&executor_);
        }
    }

    std::array<char, 32U> opType_ = {"TilingBusinessOp"};
    NnopbaseExecutor executor_{};
};

TEST_F(NnopbaseTilingContextBuilderTest, ContextInitAllocatesDynamicSlotsForExecutorTilingValues)
{
    executor_.ownArgs.inputs.nonDynamicCnt = 2U;
    executor_.ownArgs.inputs.hasDynamic = true;
    executor_.ownArgs.inputs.dynamicNum = 1U;
    executor_.ownArgs.outputs.nonDynamicCnt = 1U;

    ASSERT_EQ(NnopbaseTilingContextInit(&executor_), OK);

    const uint32_t expectedValues = executor_.ownArgs.inputs.nonDynamicCnt + executor_.ownArgs.outputs.nonDynamicCnt +
                                    static_cast<uint32_t>(kInputsAppendEnd) +
                                    static_cast<uint32_t>(gert::TilingContext::kOutputNum) +
                                    NNOPBASE_DYNAMIC_PARAM_DEF_NUM;
    EXPECT_EQ(executor_.tiling.contextExt.contextLen, ExpectedKernelRunContextLen(expectedValues));
    EXPECT_FALSE(executor_.tiling.contextExt.hasPrepared);
    ASSERT_NE(executor_.tiling.contextExt.context, nullptr);
    EXPECT_EQ(executor_.tiling.contextExt.context->compute_node_info, executor_.tiling.contextExt.nodeExt.node);
}

TEST_F(NnopbaseTilingContextBuilderTest, BuildOpInputsAndOutputsBindRuntimeTensorsAndCompileTensorDescs)
{
    auto& args = executor_.ownArgs;
    args.inputs.paramDescs.count = 3U;
    args.inputs.paramDescs.instances.resize(3U);
    SetParamInstance(args.inputs.paramDescs.instances[0], 0U, 1U, false, true);
    SetParamInstance(args.inputs.paramDescs.instances[1], 1U, 1U, false, true);
    SetParamInstance(args.inputs.paramDescs.instances[2], 2U, 1U, false, true);
    args.inputs.num = 3U;
    args.inputs.nonDynamicCnt = 3U;
    args.inputs.extTensors.resize(3U);

    args.outputs.paramDescs.count = 2U;
    args.outputs.paramDescs.instances.resize(2U);
    SetParamInstance(args.outputs.paramDescs.instances[0], 0U, 1U, false, false);
    SetParamInstance(args.outputs.paramDescs.instances[1], 1U, 1U, false, false);
    args.outputs.num = 2U;
    args.outputs.nonDynamicCnt = 2U;
    args.outputs.extTensors.resize(2U);

    uint64_t input0 = 0x11U;
    uint64_t input2 = 0x22U;
    uint64_t output0 = 0x33U;
    SetTensor(args.inputs.extTensors[0], &input0, {2, 2}, ge::DT_FLOAT16, ge::FORMAT_ND);
    args.inputs.extTensors[1].isNull = true;
    SetTensor(args.inputs.extTensors[2], &input2, {4}, ge::DT_INT32, ge::FORMAT_ND);
    SetTensor(args.outputs.extTensors[0], &output0, {2, 2}, ge::DT_FLOAT, ge::FORMAT_ND);
    args.outputs.extTensors[1].isNull = true;

    ASSERT_EQ(NnopbaseTilingContextInit(&executor_), OK);
    ASSERT_EQ(NnopbaseTilingContextUpdtPrepare(&executor_), OK);
    NnopbaseTilingBuildOpInputs(&executor_);
    NnopbaseTilingBuildOpOutputs(&executor_);

    auto* context = executor_.tiling.contextExt.context;
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(context->input_size, args.inputs.num + args.outputs.num + static_cast<uint32_t>(kInputsAppendEnd));
    EXPECT_EQ(context->output_size, gert::TilingContext::kOutputNum);
    EXPECT_EQ(context->compute_node_info, executor_.tiling.contextExt.nodeExt.node);
    EXPECT_EQ(context->values[0]->data.pointer, &args.inputs.extTensors[0].rt2Tensor);
    EXPECT_EQ(context->values[1]->data.pointer, nullptr);
    EXPECT_EQ(context->values[2]->data.pointer, &args.inputs.extTensors[2].rt2Tensor);
    EXPECT_EQ(context->values[3]->data.pointer, &args.outputs.extTensors[0].rt2Tensor);
    EXPECT_EQ(context->values[4]->data.pointer, nullptr);

    auto* inputTd = executor_.tiling.contextExt.nodeExt.inputTdStart;
    ASSERT_NE(inputTd, nullptr);
    EXPECT_EQ(inputTd[0].GetDataType(), ge::DT_FLOAT16);
    EXPECT_EQ(inputTd[0].GetStorageFormat(), ge::FORMAT_ND);
    EXPECT_EQ(inputTd[2].GetDataType(), ge::DT_INT32);
    EXPECT_EQ(inputTd[2].GetStorageFormat(), ge::FORMAT_ND);

    auto* outputTd = executor_.tiling.contextExt.nodeExt.outputTdStart;
    ASSERT_NE(outputTd, nullptr);
    EXPECT_EQ(outputTd[0].GetDataType(), ge::DT_FLOAT);
    EXPECT_EQ(outputTd[0].GetStorageFormat(), ge::FORMAT_ND);
}
