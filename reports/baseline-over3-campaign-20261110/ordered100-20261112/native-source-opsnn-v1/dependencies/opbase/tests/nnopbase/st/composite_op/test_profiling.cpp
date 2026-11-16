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

#include "acl/acl.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "op_dfx_internal.h"
#include "depends/profiler/profiler_stub.h"
#include "thread_local_context.h"

using namespace op;

static bool profiling_called = false;
class ProfilingUtProfiler : public ProfilerStub {
public:
    int32_t MsprofReportApi(uint32_t agingFlag, const MsprofApi* api)
    {
        profiling_called = true;
        return 0;
    }
};
static ProfilingUtProfiler prof;

uint64_t MsprofGetHashId(const char* hashInfo, size_t length) { return 100; }

namespace op {
namespace internal {
extern OpProfilingSwitch opProfilingSwitch;
int32_t ProfilingCallBack(uint32_t type, VOID_PTR data, uint32_t len);
} // namespace internal
} // namespace op

class ProfilingUt : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::internal::opProfilingSwitch.reportFlag = true;
        op::internal::opProfilingSwitch.kernelLaunchFlag = true;
        op::internal::opProfilingSwitch.kernelLaunchFlag = true;
        ProfilerStub::GetInstance()->Install(&prof);
    }

    static void TearDownTestCase()
    {
        op::internal::opProfilingSwitch.reportFlag = false;
        op::internal::opProfilingSwitch.kernelLaunchFlag = false;
        op::internal::opProfilingSwitch.kernelLaunchFlag = false;
        ProfilerStub::GetInstance()->UnInstall();
    }
};

TEST_F(ProfilingUt, l2_phase_one_api_profiling)
{
    profiling_called = false;
    /*
    op::internal::GetThreadLocalContext().logInfo_.l2ApiName = "aclnnAdd";
    op::internal::GetThreadLocalContext().logInfo_.l2SequenceCounter = op::internal::OpGetLogSequence();
    thread_local uint32_t aclnnAddGetWorkspaceSize_Profiling_Phase_1_Id_L2_DFX_REGISTER =
        op::internal::CollectProfilingStr("aclnnAddGetWorkspaceSize");
    */
    aclTensor* inTensor = 0;
    aclTensor* outTensor = 0;
    {
        op::OpDfxGuard opDfxGuard(__FILE__, __LINE__, op::LevelTwo, __func__, "DFX_IN(in)", "DFX_OUT(out)",
                                  std::make_tuple(inTensor), std::make_tuple(outTensor));
    }
    EXPECT_TRUE(profiling_called);
}

TEST_F(ProfilingUt, l2_phase_two_api_profiling)
{
    profiling_called = false;
    {
        op::OpDfxGuard opDfxGuard(__FILE__, __LINE__, op::LevelTwo, __func__);
    }
    EXPECT_TRUE(profiling_called);
}

TEST_F(ProfilingUt, l0_api_profiling)
{
    profiling_called = false;
    aclTensor* inTensor = 0;
    {
        op::OpDfxGuard opDfxGuard(100001, __FILE__, __LINE__, op::LevelZero, __func__, "aa", std::make_tuple(inTensor));
    }
    EXPECT_TRUE(profiling_called);
}

TEST_F(ProfilingUt, kernel_launch_profiling)
{
    MsprofCommandHandle handle;

    EXPECT_EQ(op::internal::ProfilingCallBack(PROF_CTRL_INVALID, nullptr, 0), -1);
    EXPECT_EQ(op::internal::ProfilingCallBack(PROF_CTRL_SWITCH, &handle, 0), -1);

    op::internal::ProfilingCallBack(PROF_CTRL_SWITCH, &handle, sizeof(MsprofCommandHandle));
    op::internal::CollectProfilingStr("aaaaa");

    profiling_called = false;
    op::internal::opProfilingSwitch.reportFlag = true;
    op::internal::opProfilingSwitch.kernelLaunchFlag = true;
    op::internal::opProfilingSwitch.kernelLaunchFlag = true;

    vector<int64_t> shapeA = {2, 1, 32, 16};
    aclDataType dtype1 = aclDataType::ACL_FLOAT16;
    int64_t multiStride1 = 2;
    auto storageShapeA = shapeA;
    void* deviceDataA = nullptr;
    vector<int64_t> stridesA = {2, 1, 32, 16};

    const aclTensor* tensor = aclCreateTensor(shapeA.data(), shapeA.size(), dtype1, stridesA.data(), 0,
                                              aclFormat::ACL_FORMAT_ND, storageShapeA.data(), storageShapeA.size(),
                                              deviceDataA);
    ;
    auto ctx = op::MakeOpArgContext(OP_INPUT(tensor, tensor, tensor, tensor, tensor, tensor, tensor),
                                    OP_OUTPUT(tensor, tensor, tensor, tensor, tensor, tensor, tensor));
    {
        op::internal::ReportAdditionInfo(*ctx->GetOpArg(op::OpArgDef::OP_INPUT_ARG),
                                         *ctx->GetOpArg(op::OpArgDef::OP_OUTPUT_ARG), MSPROF_GE_TASK_TYPE_AI_CORE,
                                         100001);
        op::OpDfxGuard opDfxGuard(100001, op::DfxProfilingKernelLaunch);
        op::internal::TaskInfo info;
        info.type = MSPROF_GE_TASK_TYPE_AI_CORE;
        info.ration = 0;
        op::internal::ReportAdditionInfo(info, 100, 100001);
    }
    op::DestroyOpArgContext(ctx);
    aclDestroyTensor(tensor);
    EXPECT_TRUE(profiling_called);
}

TEST_F(ProfilingUt, gen_summary_itemid)
{
    auto id = op::internal::GenSummaryItemId("a", "b", "c");
    EXPECT_EQ(id, 100U);
}

TEST_F(ProfilingUt, GetTimeStampFlag)
{
    MsprofCommandHandle handle;
    handle.type = PROF_COMMANDHANDLE_TYPE_START;
    handle.profSwitch = 0x0000100000000ULL;
    EXPECT_EQ(op::internal::ProfilingCallBack(PROF_CTRL_SWITCH, &handle, sizeof(MsprofCommandHandle)), 0);
    EXPECT_EQ(op::internal::opProfilingSwitch.timeStampFlag, true);

    handle.type = PROF_COMMANDHANDLE_TYPE_STOP;
    EXPECT_EQ(op::internal::ProfilingCallBack(PROF_CTRL_SWITCH, &handle, sizeof(MsprofCommandHandle)), 0);
    EXPECT_EQ(op::internal::opProfilingSwitch.timeStampFlag, false);
}
