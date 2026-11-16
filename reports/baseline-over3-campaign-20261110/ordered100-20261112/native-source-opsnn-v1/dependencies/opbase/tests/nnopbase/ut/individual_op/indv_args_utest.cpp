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
#include <initializer_list>

#include <gtest/gtest.h>

#include "executor/indv_args.h"
#include "executor/indv_bininfo.h"
#include "executor/indv_executor.h"

namespace {
void SetParamInstance(NnopbaseParamInstance& instance, uint32_t startIndex, uint32_t num, bool isDynamic, bool isInput)
{
    instance.startIndex = startIndex;
    instance.num = num;
    instance.cfgNum = isDynamic ? 2U : 1U;
    instance.isDynamic = isDynamic;
    instance.isInput = isInput;
}

void SetTensor(NnopbaseTensor& tensor, void* addr, gert::TensorPlacement placement,
               std::initializer_list<int64_t> shape, ge::DataType dtype = ge::DT_FLOAT)
{
    tensor.isNull = false;
    GertShape gertShape(shape);
    tensor.rt2Tensor.MutableOriginShape() = gertShape;
    tensor.rt2Tensor.MutableStorageShape() = gertShape;
    tensor.rt2Tensor.SetDataType(dtype);
    tensor.rt2Tensor.SetOriginFormat(ge::FORMAT_ND);
    tensor.rt2Tensor.SetStorageFormat(ge::FORMAT_ND);
    tensor.rt2Tensor.MutableTensorData().SetPlacement(placement);
    tensor.rt2Tensor.SetSize(sizeof(uint64_t));
    ASSERT_EQ(tensor.rt2Tensor.MutableTensorData().SetAddr(addr, nullptr), ge::GRAPH_SUCCESS);
}
} // namespace

class NnopbaseIndvArgsTest : public testing::Test {
protected:
    void SetUp() override
    {
        executor_.args = &args_;
        executor_.argsExt.args = launchBuf_.data();
        args_.binInfo = &binInfo_;
    }

    NnopbaseExecutor executor_{};
    NnopbaseExecutorArgs args_{};
    NnopbaseBinInfo binInfo_{};
    std::array<NnopbaseUChar, 2048U> launchBuf_{};
};

TEST_F(NnopbaseIndvArgsTest, PrepareInputsParamsExtEncodesDeviceHostDynamicAndOptionalInputs)
{
    args_.inputs.paramDescs.count = 4U;
    args_.inputs.paramDescs.instances.resize(4U);
    SetParamInstance(args_.inputs.paramDescs.instances[0], 0U, 1U, false, true);
    SetParamInstance(args_.inputs.paramDescs.instances[1], 1U, 1U, false, true);
    SetParamInstance(args_.inputs.paramDescs.instances[2], 2U, 1U, false, true);
    SetParamInstance(args_.inputs.paramDescs.instances[3], 3U, 2U, true, true);
    args_.inputs.extTensors.resize(5U);
    args_.inputs.num = 5U;

    uint64_t deviceData = 0x1111U;
    uint64_t hostData = 0x2222U;
    uint64_t dynData0 = 0x3333U;
    uint64_t dynData1 = 0x4444U;
    SetTensor(args_.inputs.extTensors[0], &deviceData, gert::kOnDeviceHbm, {4});
    args_.inputs.extTensors[1].isNull = true;
    SetTensor(args_.inputs.extTensors[2], &hostData, gert::kOnHost, {1}, ge::DT_UINT64);
    SetTensor(args_.inputs.extTensors[3], &dynData0, gert::kOnDeviceHbm, {2, 3});
    SetTensor(args_.inputs.extTensors[4], &dynData1, gert::kOnDeviceHbm, {5});

    auto* params = reinterpret_cast<void**>(launchBuf_.data());
    auto* hostInfo = reinterpret_cast<aclrtPlaceHolderInfo*>(launchBuf_.data() + 512U);
    auto* hostDataStart = launchBuf_.data() + 768U;
    NnopbaseHcclCommParamDesc hcclDesc{};
    NnopbaseExecutorArgsAddr argsAddr{hostDataStart, hostInfo, launchBuf_.data() + 1600U, nullptr, &hcclDesc};

    void** ret = NnopbaseExecutorPrepareInputsParamsExt(&executor_, params, &argsAddr);

    ASSERT_EQ(ret, params + 4U);
    EXPECT_EQ(params[0], &deviceData);
    EXPECT_EQ(params[1], nullptr);
    EXPECT_EQ(params[2], hostDataStart);
    EXPECT_EQ(*reinterpret_cast<uint64_t*>(params[2]), hostData);
    EXPECT_EQ(hcclDesc.isDyn, 1ULL << 3U);

    auto* dynamicBase = reinterpret_cast<NnopbaseUChar*>(params[3]);
    ASSERT_NE(dynamicBase, nullptr);
    const auto dynamicDataOffset = *reinterpret_cast<uint64_t*>(dynamicBase);
    EXPECT_EQ(dynamicDataOffset, 48U);
    EXPECT_EQ(*reinterpret_cast<void**>(dynamicBase + dynamicDataOffset), &dynData0);
    EXPECT_EQ(*reinterpret_cast<void**>(dynamicBase + dynamicDataOffset + sizeof(void*)), &dynData1);

    const auto base = reinterpret_cast<uintptr_t>(launchBuf_.data());
    EXPECT_EQ(hostInfo[0].addrOffset, reinterpret_cast<uintptr_t>(&params[2]) - base);
    EXPECT_EQ(hostInfo[0].dataOffset, reinterpret_cast<uintptr_t>(hostDataStart) - base);
    EXPECT_EQ(hostInfo[1].addrOffset, reinterpret_cast<uintptr_t>(&params[3]) - base);
    EXPECT_EQ(hostInfo[1].dataOffset, reinterpret_cast<uintptr_t>(dynamicBase) - base);
    EXPECT_GT(args_.inputs.extTensors[3].argsOffset, 0U);
    EXPECT_GT(args_.inputs.extTensors[4].argsOffset, args_.inputs.extTensors[3].argsOffset);
}

TEST_F(NnopbaseIndvArgsTest, PrepareNullTensorsReservesLaunchSlotOnlyForDynamicShapeBin)
{
    std::array<void*, 2U> params = {reinterpret_cast<void*>(0x1234), reinterpret_cast<void*>(0x5678)};
    size_t tensorIndex = 0U;

    binInfo_.isStaticShape = false;
    void** ret = NnopbaseExecutorPrepareNullTensors(&executor_, params.data(), &tensorIndex);
    EXPECT_EQ(ret, params.data() + 1U);
    EXPECT_EQ(tensorIndex, 1U);
    EXPECT_EQ(params[0], nullptr);

    params = {reinterpret_cast<void*>(0x1234), reinterpret_cast<void*>(0x5678)};
    tensorIndex = 0U;
    binInfo_.isStaticShape = true;
    ret = NnopbaseExecutorPrepareNullTensors(&executor_, params.data(), &tensorIndex);
    EXPECT_EQ(ret, params.data());
    EXPECT_EQ(tensorIndex, 1U);
    EXPECT_EQ(params[0], reinterpret_cast<void*>(0x1234));
}

TEST_F(NnopbaseIndvArgsTest, GetIrIndexMapsFlattenedDynamicTensorIndexBackToIrInput)
{
    NnopbaseParamDesc desc{};
    desc.count = 3U;
    desc.instances.resize(3U);
    SetParamInstance(desc.instances[0], 0U, 1U, false, true);
    SetParamInstance(desc.instances[1], 1U, 3U, true, true);
    SetParamInstance(desc.instances[2], 4U, 1U, false, true);

    size_t irIndex = 0U;
    size_t relativeIndex = 0U;
    NnopbaseGetIrIndex(desc, 3U, irIndex, relativeIndex);

    EXPECT_EQ(irIndex, 1U);
    EXPECT_EQ(relativeIndex, 2U);
}
