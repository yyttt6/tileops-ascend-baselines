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
#include "aclnn/acl_meta.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "op_cache_internal.h"
#include "depends/profiler/profiler_stub.h"
#include "thread_local_context.h"

using namespace op;
using namespace op::internal;

static bool profiling_called = false;

class ProfilingCacheUtProfiler : public ProfilerStub {
public:
    int32_t MsprofReportApi(uint32_t agingFlag, const MsprofApi* api)
    {
        profiling_called = true;
        return 0;
    }
};
static ProfilingCacheUtProfiler prof;

namespace op {
namespace internal {
void CacheTensorInfo(const FVector<const aclTensor*>& inTensors, const FVector<const aclTensor*>& outTensors);
void CacheDfxInfo(uint32_t blockDim, const ProfilingInfoId& id, const TaskInfo& taskInfo, bool isMemSet);
aclnnStatus DoReportAdditionInfo(void* infoLists, const TaskInfo& taskInfo,
                                 const op::internal::ProfilingInfoId& profilingInfoId);
} // namespace internal
} // namespace op

class ProfilingCacheUt : public testing::Test {
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

TEST_F(ProfilingCacheUt, test_storeage)
{
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
    int64_t* storageDims = nullptr;
    uint64_t storageDimsNum = 0;
    EXPECT_EQ(aclGetStorageShape(tensor, &storageDims, &storageDimsNum), OK);
    EXPECT_EQ(aclGetStorageShape(nullptr, &storageDims, &storageDimsNum), ACLNN_ERR_PARAM_NULLPTR);

    // free resource
    delete[] storageDims;
    aclDestroyTensor(tensor);
}

TEST_F(ProfilingCacheUt, test_cache_tensor_has_op_cache)
{
    vector<int64_t> shapeA = {2, 1, 32, 16};
    aclDataType dtype1 = aclDataType::ACL_FLOAT16;
    int64_t multiStride1 = 2;
    auto storageShapeA = shapeA;
    void* deviceDataA = nullptr;
    vector<int64_t> stridesA = {2, 1, 32, 16};

    const aclTensor* tensorIn = aclCreateTensor(shapeA.data(), shapeA.size(), dtype1, stridesA.data(), 0,
                                                aclFormat::ACL_FORMAT_ND, storageShapeA.data(), storageShapeA.size(),
                                                deviceDataA);
    ;
    const aclTensor* tensorOut = aclCreateTensor(shapeA.data(), shapeA.size(), dtype1, stridesA.data(), 0,
                                                 aclFormat::ACL_FORMAT_ND, storageShapeA.data(), storageShapeA.size(),
                                                 deviceDataA);
    ;
    FVector<const aclTensor*> in;
    FVector<const aclTensor*> out;

    in.push_back(tensorIn);
    out.push_back(tensorOut);
    op::internal::ProfilingInfoId id;

    // test cache exist
    auto opExecCache = new OpExecCache();
    opExecCache->SetCacheBuf(GetCacheBuf());
    GetOpCacheContext().SetOpCache(opExecCache);
    CacheTensorInfo(in, out);
    TaskInfo info;
    info.type = MSPROF_GE_TASK_TYPE_AI_CORE;
    info.ration = 0;
    CacheDfxInfo(32, id, info, false);

    // release resource
    aclDestroyTensor(tensorIn);
    aclDestroyTensor(tensorOut);
    delete opExecCache;
}

TEST_F(ProfilingCacheUt, test_cache_tensor_restore_and_report)
{
    vector<int64_t> shapeA = {2, 1, 32, 16};
    aclDataType dtype1 = aclDataType::ACL_FLOAT16;
    int64_t multiStride1 = 2;
    auto storageShapeA = shapeA;
    void* deviceDataA = nullptr;
    vector<int64_t> stridesA = {2, 1, 32, 16};

    const aclTensor* tensorIn = aclCreateTensor(shapeA.data(), shapeA.size(), dtype1, stridesA.data(), 0,
                                                aclFormat::ACL_FORMAT_ND, storageShapeA.data(), storageShapeA.size(),
                                                deviceDataA);
    ;
    const aclTensor* tensorOut = aclCreateTensor(shapeA.data(), shapeA.size(), dtype1, stridesA.data(), 0,
                                                 aclFormat::ACL_FORMAT_ND, storageShapeA.data(), storageShapeA.size(),
                                                 deviceDataA);
    ;
    FVector<const aclTensor*> in;
    FVector<const aclTensor*> out;

    in.push_back(tensorIn);
    out.push_back(tensorOut);
    op::internal::ProfilingInfoId id;

    // test cache exist
    auto opExecCache = new OpExecCache();
    opExecCache->SetCacheBuf(GetCacheBuf());
    GetOpCacheContext().SetOpCache(opExecCache);
    CacheTensorInfo(in, out);
    TaskInfo info;
    info.type = MSPROF_GE_TASK_TYPE_AI_CORE;
    info.ration = 0;
    CacheDfxInfo(32, id, info, false);

    // test restore/report
    EXPECT_EQ(DoReportAdditionInfo(opExecCache->GetCacheTensorInfo(0), info, id), OK);

    // release resource
    aclDestroyTensor(tensorIn);
    aclDestroyTensor(tensorOut);
    delete opExecCache;
}
