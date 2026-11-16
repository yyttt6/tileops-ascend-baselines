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
    aclIntArray* intArrNull = nullptr;
    int64_t intValues[] = {3, 4, 5};
    aclIntArray* intArr = aclCreateIntArray(intValues, sizeof(intValues) / sizeof(intValues[0]));

    aclFloatArray* floatArrNull = nullptr;
    float floatValues[] = {3.3f, 4.4f, 5.5f};
    aclFloatArray* floatArr = aclCreateFloatArray(floatValues, sizeof(floatValues) / sizeof(floatValues[0]));

    aclBoolArray* boolArrNull = nullptr;
    bool boolValues[] = {true, false, true};
    aclBoolArray* boolArr = aclCreateBoolArray(boolValues, sizeof(boolValues) / sizeof(boolValues[0]));

    aclTensorList* tensorListNull = nullptr;
    op::Shape tShape{1, 2, 3};
    aclTensor t1(tShape, op::DataType::DT_INT32, ge::FORMAT_ND, nullptr);
    aclTensor t2(tShape, op::DataType::DT_INT32, ge::FORMAT_ND, nullptr);
    aclTensor* tensorNull = nullptr;
    aclTensor* tensorPtr = &t1;
    aclTensor* list1[] = {&t1, &t2, nullptr};
    aclTensorList tensorList(list1, 3);
    const aclTensorList* tensorListPtr = &tensorList;

    aclScalarList* scalarListNull = nullptr;
    int32_t intVal = 8;
    aclScalar s1(intVal);
    aclScalar s2(intVal);
    aclScalar* scalarNull = nullptr;
    aclScalar* scalarPtr = &s1;
    aclScalar* list2[] = {&s1, &s2, nullptr};
    aclScalarList scalarList(list2, 3);
    const aclScalarList* scalarListPtr = &scalarList;

    op::DataType dt = op::DataType::DT_FLOAT16;
    op::Format fm = op::Format::FORMAT_FRACTAL_NZ;
    int64_t number = 100;

    profiling_called = false;
    {
        op::OpDfxGuard opDfxGuard(
            __FILE__, __LINE__, op::LevelTwo, __func__,
            "DFX_IN(intArrNull, intArr, floatArrNull, floatArr, boolArrNull, boolArr, scalarNull, scalarPtr, "
            "tensorNull, tensorPtr, dt, fm, number)",
            "DFX_OUT(tensorListPtr, scalarListPtr)",
            std::make_tuple(intArrNull, intArr, floatArrNull, floatArr, boolArrNull, boolArr, scalarNull, scalarPtr,
                            tensorNull, tensorPtr, dt, fm, number),
            std::make_tuple(tensorListPtr, scalarListPtr));
    }
    EXPECT_TRUE(profiling_called);

    aclDestroyIntArray(intArr);
    aclDestroyFloatArray(floatArr);
    aclDestroyBoolArray(boolArr);
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
    op::internal::opProfilingSwitch.additionInfoFlag = true;
    op::internal::opProfilingSwitch.level2ProfilingFlag = true;

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
    aclDestroyTensor(tensor);
    op::DestroyOpArgContext(ctx);
    EXPECT_TRUE(profiling_called);

    MsprofCommandHandle handleStop;
    handleStop.type = PROF_COMMANDHANDLE_TYPE_STOP;
    EXPECT_EQ(op::internal::ProfilingCallBack(PROF_CTRL_SWITCH, &handleStop, sizeof(MsprofCommandHandle)), 0);
    EXPECT_EQ(op::internal::opProfilingSwitch.reportFlag, false);
    EXPECT_EQ(op::internal::opProfilingSwitch.kernelLaunchFlag, false);
    EXPECT_EQ(op::internal::opProfilingSwitch.additionInfoFlag, false);
    EXPECT_EQ(op::internal::opProfilingSwitch.level2ProfilingFlag, false);
    handleStop.type = PROF_COMMANDHANDLE_TYPE_INIT;
    EXPECT_EQ(op::internal::ProfilingCallBack(PROF_CTRL_SWITCH, &handleStop, sizeof(MsprofCommandHandle)), 0);
    EXPECT_EQ(op::internal::opProfilingSwitch.reportFlag, false);
    EXPECT_EQ(op::internal::opProfilingSwitch.kernelLaunchFlag, false);
    EXPECT_EQ(op::internal::opProfilingSwitch.additionInfoFlag, false);
    EXPECT_EQ(op::internal::opProfilingSwitch.level2ProfilingFlag, false);
}

TEST_F(ProfilingUt, gen_summary_itemid)
{
    auto id = op::internal::GenSummaryItemId("a", "b", "c");
    EXPECT_EQ(id, 100U);
}

TEST_F(ProfilingUt, report_context_id)
{
    auto id = op::internal::GenSummaryItemId("a", "b", "c");
    op::internal::ReportNodeContextIdInfo(id);
}

TEST_F(ProfilingUt, AddTensorToThreadLocalCtx)
{
    op::Shape tShape{1, 2, 3};
    auto self = std::make_unique<aclTensor>(tShape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND, nullptr);
    auto out = std::make_unique<aclTensor>(tShape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND, nullptr);
    aclTensor* wsArr[] = {self.get(), out.get()};
    const aclTensorList* wsList = aclCreateTensorList(wsArr, 2);
    AddInputTensorToThreadLocalCtx(self.get());
    AddOutputTensorToThreadLocalCtx(out.get());
    AddInputTensorToThreadLocalCtx(wsList);
    AddOutputTensorToThreadLocalCtx(wsList);
    aclTensorList* wsListUnconst = const_cast<aclTensorList*>(wsList);
    AddInputTensorToThreadLocalCtx(wsListUnconst);
    AddOutputTensorToThreadLocalCtx(wsListUnconst);
    delete wsList;
}
TEST_F(ProfilingUt, StringToVec)
{
    op::OpDfxGuard opDfxGuard(__FILE__, __LINE__, op::LevelTwo, __func__);

    std::vector<std::string> v;
    opDfxGuard.StringToVec("aclnnAffineGrid", v);
    opDfxGuard.StringToVecWithBrackets("DFX_IN(aa, bb, cc)", v);
}

TEST_F(ProfilingUt, GetLogApiInfo)
{
    op::internal::GetThreadLocalContext().logInfo_.l2ApiName = "TestStaticAdd";
    op::internal::GetThreadLocalContext().logInfo_.l0Name = "TestStaticAdd";
    std::cout << "GetLogApiInfo() " << op::internal::GetLogApiInfo() << std::endl;
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

    handle.type = PROF_COMMANDHANDLE_TYPE_START;
    handle.profSwitch = 0x00001ULL;
    EXPECT_EQ(op::internal::ProfilingCallBack(PROF_CTRL_SWITCH, &handle, sizeof(MsprofCommandHandle)), 0);
    EXPECT_EQ(op::internal::opProfilingSwitch.timeStampFlag, false);
}
