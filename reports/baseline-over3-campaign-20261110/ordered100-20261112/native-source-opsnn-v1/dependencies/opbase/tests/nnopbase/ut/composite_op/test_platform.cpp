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
#include <vector>
#include <random>
#include <thread>
#include "opdev/common_types.h"
#include "acl/acl.h"
#include "opdev/platform.h"
typedef std::function<void(void)> Functional;
using namespace std;
using namespace op;
class OpPlatformTest : public testing::Test {
    void SetUp() {}
};

TEST_F(OpPlatformTest, TestGetSocVersion)
{
    // auto ret = aclrtSetDevice(0);
    // ASSERT_TRUE(ret == ACL_SUCCESS);
    // aclrtContext context;
    // ret = aclrtCreateContext(&context, 0);
    // ASSERT_TRUE(ret == ACL_SUCCESS);
    // ret = aclrtSetCurrentContext(context);
    // ASSERT_TRUE(ret == ACL_SUCCESS);
    auto& platformInfo = op::GetCurrentPlatformInfo();
    SocVersion socVersion = platformInfo.GetSocVersion();
    EXPECT_EQ(socVersion, op::SocVersion::ASCEND910);
    auto socLongVersion = platformInfo.GetSocLongVersion();
    EXPECT_EQ(socLongVersion, "Ascend910B2");
    EXPECT_EQ(ToString(static_cast<SocVersion>(100)), ge::AscendString("UnknownSocVersion"));
}

TEST_F(OpPlatformTest, TestGetFftsPlusMode)
{
    auto& platformInfo = op::GetCurrentPlatformInfo();
    auto fftsModel = platformInfo.GetFftsPlusMode();
    EXPECT_EQ(fftsModel, true);
}

TEST_F(OpPlatformTest, TestMMadInstAbility)
{
    // auto ret = aclrtSetDevice(0);
    // ASSERT_TRUE(ret == ACL_SUCCESS);
    // aclrtContext context;
    // ret = aclrtCreateContext(&context, 0);
    // ASSERT_TRUE(ret == ACL_SUCCESS);
    // ret = aclrtSetCurrentContext(context);
    // ASSERT_TRUE(ret == ACL_SUCCESS);

    auto& platformInfo = op::GetCurrentPlatformInfo();
    auto checkRet = platformInfo.CheckSupport(op::SocSpec::INST_MMAD, op::SocSpecAbility::INST_MMAD_F322F32);
    EXPECT_EQ(checkRet, false);
    checkRet = platformInfo.CheckSupport(op::SocSpec::INST_MMAD, op::SocSpecAbility::INST_MMAD_F162F16);
    EXPECT_EQ(checkRet, true);
}

TEST_F(OpPlatformTest, TestBlockSize)
{
    // auto ret = aclrtSetDevice(0);
    // ASSERT_TRUE(ret == ACL_SUCCESS);
    // aclrtContext context;
    // ret = aclrtCreateContext(&context, 0);
    // ASSERT_TRUE(ret == ACL_SUCCESS);
    // ret = aclrtSetCurrentContext(context);
    // ASSERT_TRUE(ret == ACL_SUCCESS);

    auto& platformInfo = op::GetCurrentPlatformInfo();
    auto blockSize = platformInfo.GetBlockSize();
    EXPECT_EQ(blockSize, 32);
}

TEST_F(OpPlatformTest, TestToString)
{
    auto socVersion = ToString(SocVersion::ASCEND910);
    EXPECT_EQ(socVersion, "Ascend910");
    socVersion = ToString(SocVersion::RESERVED_VERSION);
    EXPECT_EQ(socVersion, "UnknownSocVersion");
}

TEST_F(OpPlatformTest, TestGetDeviceId)
{
    auto& platformInfo = op::GetCurrentPlatformInfo();
    EXPECT_EQ(platformInfo.GetDeviceId(), 0);
}

static void TestGetCubeCoreNum()
{
    auto& platformInfo = op::GetCurrentPlatformInfo();
    auto currentCubeCoreNum = platformInfo.GetCubeCoreNum();
    EXPECT_EQ(currentCubeCoreNum, 64);
}

static void TestGetVectorCoreNum()
{
    auto& platformInfo = op::GetCurrentPlatformInfo();
    auto currentVectorCoreNum = platformInfo.GetVectorCoreNum();
    EXPECT_EQ(currentVectorCoreNum, 32);
}

TEST_F(OpPlatformTest, TestGetCoreNumMutiThreadTest)
{
    vector<Functional> funVec = {TestGetCubeCoreNum, TestGetVectorCoreNum};
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, funVec.size() - 1);
    const uint64_t threadCount = funVec.size() * 100;
    vector<std::thread> threadVec;
    for (int i = 0; i < threadCount; i++) {
        threadVec.emplace_back(std::thread(funVec[distrib(gen)]));
    }
    for (int i = 0; i < threadCount; i++) {
        threadVec[i].join();
    }
}
