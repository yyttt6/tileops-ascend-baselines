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
#include <cstring>

#include <gtest/gtest.h>

#include "executor/indv_compute_node_info.h"
#include "executor/indv_executor.h"
#include "exe_graph/runtime/continuous_vector.h"

namespace {
void SetParamInstance(NnopbaseParamInstance& instance, uint32_t startIndex, uint32_t num, bool isDynamic, bool isInput)
{
    instance.startIndex = startIndex;
    instance.num = num;
    instance.cfgNum = isDynamic ? 2U : 1U;
    instance.isDynamic = isDynamic;
    instance.isInput = isInput;
}
} // namespace

class NnopbaseComputeNodeInfoTest : public testing::Test {
protected:
    void SetUp() override
    {
        executor_.args = &args_;
        executor_.opType = opType_.data();
        ASSERT_EQ(NnopbaseComputeNodeInfoInit(&executor_.tiling.contextExt.nodeExt), OK);
    }

    void TearDown() override { NnopbaseComputeNodeInfoDeInit(&executor_.tiling.contextExt.nodeExt); }

    std::array<char, 32U> opType_ = {"BusinessExecutorOp"};
    NnopbaseExecutor executor_{};
    NnopbaseExecutorArgs args_{};
};

TEST_F(NnopbaseComputeNodeInfoTest, InfoUpdtLaysOutDynamicAndOptionalIrInstancesForExecutorNode)
{
    args_.inputs.paramDescs.count = 3U;
    args_.inputs.paramDescs.instances.resize(3U);
    SetParamInstance(args_.inputs.paramDescs.instances[0], 0U, 1U, false, true);
    SetParamInstance(args_.inputs.paramDescs.instances[1], 1U, 1U, false, true);
    SetParamInstance(args_.inputs.paramDescs.instances[2], 2U, 2U, true, true);
    args_.inputs.num = 4U;
    args_.inputs.hasDynamic = true;

    args_.outputs.paramDescs.count = 2U;
    args_.outputs.paramDescs.instances.resize(2U);
    SetParamInstance(args_.outputs.paramDescs.instances[0], 0U, 1U, false, false);
    SetParamInstance(args_.outputs.paramDescs.instances[1], 1U, 3U, true, false);
    args_.outputs.num = 4U;
    args_.outputs.hasDynamic = true;

    auto& nodeExt = executor_.tiling.contextExt.nodeExt;
    ASSERT_EQ(NnopbaseComputeNodeInfoUpdt(&executor_), OK);

    ASSERT_EQ(nodeExt.node->nodeType, opType_.data());
    ASSERT_EQ(nodeExt.node->nodeName, opType_.data());
    EXPECT_EQ(nodeExt.node->irInputsNum, 3U);
    EXPECT_EQ(nodeExt.node->irOutputsNum, 2U);
    EXPECT_EQ(nodeExt.node->inputsNum, 4U);
    EXPECT_EQ(nodeExt.node->outputsNum, 4U);
    EXPECT_EQ(nodeExt.inputTdStart, reinterpret_cast<NnopbaseCompileTimeTensorDesc*>(nodeExt.instStart + 3U));
    EXPECT_EQ(nodeExt.outputTdStart, nodeExt.inputTdStart + 4U);
    EXPECT_EQ(nodeExt.attrStart, reinterpret_cast<NnopbaseRuntimeAttrsDef*>(nodeExt.outputTdStart + 4U));
    EXPECT_EQ(nodeExt.outputInstStart,
              reinterpret_cast<NnopbaseAnchorInstanceInfo*>(reinterpret_cast<NnopbaseUChar*>(nodeExt.attrStart) +
                                                            nodeExt.node->attrSize));

    EXPECT_EQ(nodeExt.instStart[0].GetInstanceStart(), 0U);
    EXPECT_EQ(nodeExt.instStart[0].GetInstanceNum(), 1U);
    EXPECT_EQ(nodeExt.instStart[1].GetInstanceStart(), 1U);
    EXPECT_EQ(nodeExt.instStart[1].GetInstanceNum(), 1U);
    EXPECT_EQ(nodeExt.instStart[2].GetInstanceStart(), 2U);
    EXPECT_EQ(nodeExt.instStart[2].GetInstanceNum(), 2U);
    EXPECT_EQ(nodeExt.outputInstStart[0].GetInstanceStart(), 0U);
    EXPECT_EQ(nodeExt.outputInstStart[0].GetInstanceNum(), 1U);
    EXPECT_EQ(nodeExt.outputInstStart[1].GetInstanceStart(), 1U);
    EXPECT_EQ(nodeExt.outputInstStart[1].GetInstanceNum(), 3U);
}

TEST_F(NnopbaseComputeNodeInfoTest, AttrsUpdtCopiesScalarAndVectorAttrsIntoRuntimeBuffer)
{
    args_.inputs.paramDescs.count = 1U;
    args_.inputs.paramDescs.instances.resize(1U);
    SetParamInstance(args_.inputs.paramDescs.instances[0], 0U, 1U, false, true);
    args_.inputs.num = 1U;

    args_.outputs.paramDescs.count = 1U;
    args_.outputs.paramDescs.instances.resize(1U);
    SetParamInstance(args_.outputs.paramDescs.instances[0], 0U, 1U, false, false);
    args_.outputs.num = 1U;

    int64_t scalarValue = 42;
    std::array<int32_t, 3U> vectorValue = {7, 8, 9};
    std::array<NnopbaseAttr, 2U> attrStorage{};
    attrStorage[0].addr = {&scalarValue, sizeof(scalarValue), false, false, sizeof(scalarValue)};
    attrStorage[0].dtype = kNnopbaseInt;
    attrStorage[1].addr = {vectorValue.data(), vectorValue.size() * sizeof(int32_t), false, true, sizeof(int32_t)};
    attrStorage[1].dtype = kNnopbaseInt;
    executor_.attrs.attrs = attrStorage.data();
    executor_.attrs.num = attrStorage.size();
    executor_.attrs.totalSize = sizeof(scalarValue) + sizeof(gert::ContinuousVector) +
                                vectorValue.size() * sizeof(int32_t);

    ASSERT_EQ(NnopbaseComputeNodeInfoUpdt(&executor_), OK);
    ASSERT_EQ(NnopbaseComputeNodeAttrsUpdt(&executor_.tiling.contextExt.nodeExt, &executor_.attrs), OK);

    auto* attrDef = executor_.tiling.contextExt.nodeExt.attrStart;
    ASSERT_NE(attrDef, nullptr);
    EXPECT_EQ(attrDef->attr_num, attrStorage.size());
    auto* attrBase = reinterpret_cast<NnopbaseUChar*>(attrDef);
    EXPECT_EQ(*reinterpret_cast<int64_t*>(attrBase + attrDef->offset[0]), scalarValue);

    auto* vector = reinterpret_cast<gert::ContinuousVector*>(attrBase + attrDef->offset[1]);
    ASSERT_EQ(vector->GetSize(), vectorValue.size());
    ASSERT_EQ(vector->GetCapacity(), vectorValue.size());
    EXPECT_EQ(std::memcmp(vector->GetData(), vectorValue.data(), vectorValue.size() * sizeof(int32_t)), 0);
}

TEST_F(NnopbaseComputeNodeInfoTest, InfoUpdtReallocatesNodeBufferWhenRuntimeAttrsExceedDefaultStorage)
{
    args_.inputs.paramDescs.count = 1U;
    args_.inputs.paramDescs.instances.resize(1U);
    SetParamInstance(args_.inputs.paramDescs.instances[0], 0U, 1U, false, true);
    args_.inputs.num = 1U;

    args_.outputs.paramDescs.count = 1U;
    args_.outputs.paramDescs.instances.resize(1U);
    SetParamInstance(args_.outputs.paramDescs.instances[0], 0U, 1U, false, false);
    args_.outputs.num = 1U;

    std::array<NnopbaseAttr, 1U> attrs{};
    uint8_t attrValue = 1U;
    attrs[0].addr = {&attrValue, sizeof(attrValue), false, false, sizeof(attrValue)};
    executor_.attrs.attrs = attrs.data();
    executor_.attrs.num = attrs.size();
    executor_.attrs.totalSize = NNOPBASE_COMPUTE_NODE_BUF_LEN + 1U;

    auto* oldBuf = executor_.tiling.contextExt.nodeExt.buf;
    ASSERT_EQ(NnopbaseComputeNodeInfoUpdt(&executor_), OK);

    EXPECT_NE(executor_.tiling.contextExt.nodeExt.buf, oldBuf);
    EXPECT_GT(executor_.tiling.contextExt.nodeExt.bufLen,
              sizeof(NnopbaseComputeNodeInfo) + NNOPBASE_COMPUTE_NODE_BUF_LEN);
    EXPECT_EQ(executor_.tiling.contextExt.nodeExt.node->nodeType, opType_.data());
}
