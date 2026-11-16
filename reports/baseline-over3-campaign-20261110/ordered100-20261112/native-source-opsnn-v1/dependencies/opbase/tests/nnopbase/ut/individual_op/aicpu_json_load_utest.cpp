/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include <fstream>
#include <sys/stat.h>
#include <unistd.h>

#define private public
#include "aicpu_json_load_manager.h"
#undef private

#include "depends/mmpa/mmpa_stub.h"

using op::internal::JsonLoadManger;

namespace {
const std::string kCustAicpuJsonRelPath = "/op_impl/cpu/config/cust_aicpu_kernel.json";

void MakeDir(const std::string& dir) { system(("mkdir -p " + dir).c_str()); }

void WriteFile(const std::string& path, const std::string& content)
{
    MakeDir(path.substr(0, path.rfind('/')));
    std::ofstream ofs(path);
    ofs << content;
    ofs.close();
}

const std::string kSampleCustJson = R"({
    "CustomAicpuTestOp":{
        "input0":{"name":"x","type":"DT_FLOAT"},
        "output0":{"name":"y","type":"DT_FLOAT"},
        "opInfo":{
            "computeCost":"100",
            "engine":"DNN_VM_AICPU",
            "flagAsync":"False",
            "flagPartial":"False",
            "formatAgnostic":"False",
            "functionName":"RunCpuKernel",
            "kernelSo":"libcust_test_kernels.so",
            "opKernelLib":"CUSTAICPUKernel",
            "opsFlag":"OPS_FLAG_OPEN",
            "userDefined":"True",
            "workspaceSize":"100"
        }
    }
})";
} // namespace

class AicpuJsonLoadTest : public testing::Test {
protected:
    std::string tmpDir = "";

    void SetUp() override
    {
        tmpDir = "/tmp/opbase_aicpu_ut_" + std::to_string(getpid());
        MakeDir(tmpDir);
    }

    void TearDown() override
    {
        unsetenv("ASCEND_CUSTOM_OPP_PATH");
        unsetenv("ASCEND_OPP_PATH");
        system(("rm -rf " + tmpDir).c_str());
        JsonLoadManger::hasAicpuLoadBin_ = false;
        JsonLoadManger::aicpuBinHandle_ = nullptr;
        JsonLoadManger::hasTfLoadBin_ = false;
        JsonLoadManger::tfBinHandle_ = nullptr;
        JsonLoadManger::isSupportNewLaunch_ = true;
        JsonLoadManger::socVersion_ = "";
        JsonLoadManger::aicpuCustLoadFlag_ = false;
        JsonLoadManger::customOpsInfos_.clear();
        JsonLoadManger::custOpJsonInfo_.clear();
        JsonLoadManger::custRegisterInfos_.clear();
        JsonLoadManger::customBinhandleInfos_.clear();
        JsonLoadManger::bufferCache_.clear();
    }

    void CreateVendorWithCustJson(const std::string& vendorName)
    {
        const std::string vendorDir = tmpDir + "/vendors/" + vendorName;
        WriteFile(vendorDir + kCustAicpuJsonRelPath, kSampleCustJson);
    }

    void CreateVendorWithoutCustJson(const std::string& vendorName) { MakeDir(tmpDir + "/vendors/" + vendorName); }

    void WriteConfigIni(const std::string& content) { WriteFile(tmpDir + "/vendors/config.ini", content); }
};

TEST_F(AicpuJsonLoadTest, custom_opp_path_set_use_directly)
{
    const std::string customDir = tmpDir + "/custom_dir";
    WriteFile(customDir + kCustAicpuJsonRelPath, kSampleCustJson);
    setenv("ASCEND_CUSTOM_OPP_PATH", customDir.c_str(), 1);

    EXPECT_EQ(JsonLoadManger::CustJsonLoadAndParse(), ACLNN_SUCCESS);

    std::string kernelSo = "";
    std::string functionName = "";
    EXPECT_TRUE(JsonLoadManger::FindAndGetInCustomRegistry("CustomAicpuTestOp", kernelSo, functionName));
    EXPECT_EQ(kernelSo, "libcust_test_kernels.so");
    EXPECT_EQ(functionName, "RunCpuKernel");
}

TEST_F(AicpuJsonLoadTest, custom_opp_path_empty_fallback_to_vendors)
{
    CreateVendorWithCustJson("custom_math");
    WriteConfigIni("load_priority=custom_math");
    setenv("ASCEND_OPP_PATH", tmpDir.c_str(), 1);
    setenv("ASCEND_CUSTOM_OPP_PATH", "", 1);

    EXPECT_EQ(JsonLoadManger::CustJsonLoadAndParse(), ACLNN_SUCCESS);

    std::string kernelSo = "";
    std::string functionName = "";
    EXPECT_TRUE(JsonLoadManger::FindAndGetInCustomRegistry("CustomAicpuTestOp", kernelSo, functionName));
}

TEST_F(AicpuJsonLoadTest, custom_opp_path_not_set_fallback_to_vendors)
{
    CreateVendorWithCustJson("custom_math");
    WriteConfigIni("load_priority=custom_math");
    setenv("ASCEND_OPP_PATH", tmpDir.c_str(), 1);
    unsetenv("ASCEND_CUSTOM_OPP_PATH");

    EXPECT_EQ(JsonLoadManger::CustJsonLoadAndParse(), ACLNN_SUCCESS);

    std::string kernelSo = "";
    std::string functionName = "";
    EXPECT_TRUE(JsonLoadManger::FindAndGetInCustomRegistry("CustomAicpuTestOp", kernelSo, functionName));
}

TEST_F(AicpuJsonLoadTest, both_envs_set_custom_opp_path_priority)
{
    const std::string customDir = tmpDir + "/explicit_custom";
    WriteFile(customDir + kCustAicpuJsonRelPath, kSampleCustJson);
    setenv("ASCEND_CUSTOM_OPP_PATH", customDir.c_str(), 1);

    CreateVendorWithCustJson("custom_math");
    WriteConfigIni("load_priority=custom_math");
    setenv("ASCEND_OPP_PATH", tmpDir.c_str(), 1);

    EXPECT_EQ(JsonLoadManger::CustJsonLoadAndParse(), ACLNN_SUCCESS);

    std::string kernelSo = "";
    std::string functionName = "";
    EXPECT_TRUE(JsonLoadManger::FindAndGetInCustomRegistry("CustomAicpuTestOp", kernelSo, functionName));
}

TEST_F(AicpuJsonLoadTest, no_custom_json_anywhere_returns_success_empty_registry)
{
    setenv("ASCEND_OPP_PATH", tmpDir.c_str(), 1);
    unsetenv("ASCEND_CUSTOM_OPP_PATH");

    EXPECT_EQ(JsonLoadManger::CustJsonLoadAndParse(), ACLNN_SUCCESS);

    std::string kernelSo = "";
    std::string functionName = "";
    EXPECT_FALSE(JsonLoadManger::FindAndGetInCustomRegistry("CustomAicpuTestOp", kernelSo, functionName));
}

TEST_F(AicpuJsonLoadTest, vendor_without_cust_json_skipped)
{
    CreateVendorWithCustJson("custom_math");
    CreateVendorWithoutCustJson("custom_nn");
    WriteConfigIni("load_priority=custom_nn,custom_math");
    setenv("ASCEND_OPP_PATH", tmpDir.c_str(), 1);
    unsetenv("ASCEND_CUSTOM_OPP_PATH");

    EXPECT_EQ(JsonLoadManger::CustJsonLoadAndParse(), ACLNN_SUCCESS);

    std::string kernelSo = "";
    std::string functionName = "";
    EXPECT_TRUE(JsonLoadManger::FindAndGetInCustomRegistry("CustomAicpuTestOp", kernelSo, functionName));
}

TEST_F(AicpuJsonLoadTest, config_ini_priority_format_without_load_prefix)
{
    CreateVendorWithCustJson("custom_math");
    WriteConfigIni("priority=custom_math");
    setenv("ASCEND_OPP_PATH", tmpDir.c_str(), 1);
    unsetenv("ASCEND_CUSTOM_OPP_PATH");

    EXPECT_EQ(JsonLoadManger::CustJsonLoadAndParse(), ACLNN_SUCCESS);

    std::string kernelSo = "";
    std::string functionName = "";
    EXPECT_TRUE(JsonLoadManger::FindAndGetInCustomRegistry("CustomAicpuTestOp", kernelSo, functionName));
}

TEST_F(AicpuJsonLoadTest, both_envs_no_valid_custom_json_returns_success)
{
    setenv("ASCEND_CUSTOM_OPP_PATH", "", 1);
    setenv("ASCEND_OPP_PATH", "/nonexistent_path", 1);

    EXPECT_EQ(JsonLoadManger::CustJsonLoadAndParse(), ACLNN_SUCCESS);

    std::string kernelSo = "";
    std::string functionName = "";
    EXPECT_FALSE(JsonLoadManger::FindAndGetInCustomRegistry("CustomAicpuTestOp", kernelSo, functionName));
}
