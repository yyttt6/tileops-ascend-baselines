/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include "executor/indv_bininfo.h"
#include "executor/indv_executor.h"
#include <fstream>
#include "utils/file_faker.h"
#include "utils/indv_soc.h"
#include "runtime/runtime/kernel.h"
#include "mockcpp/mockcpp.hpp"

namespace {
void SetSocVersion(const std::string& version)
{
    auto& soc = nnopbase::IndvSoc::GetInstance();
    soc.socVersion = version;
    soc.isInit = true;
}
} // namespace

class NnopbaseKernelLaunchUt : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}
};

TEST_F(NnopbaseKernelLaunchUt, test)
{
    NnopbaseSetStubFiles(OP_API_COMMON_UT_SRC_DIR);
    NnopbaseBinInfo* p = new NnopbaseBinInfo;
    auto& binInfo = *p;
    binInfo.binHandle = (void*)0x0002;
    binInfo.bin = new unsigned char[10];
    binInfo.binLen = 1;

    binInfo.coreType = kMix;
    binInfo.hasReg = false;
    binInfo.isStaticShape = false;

    nlohmann::json binJson = nlohmann::json::parse(R"(
        {
            "coreType": "VectorCore",
            "kernelList": [
                {
                    "kernelName": "AddCustom_1e04ee05ab491cc5ae9c3d5c9ee8950b_1"
                }
            ],
            "filePath": "ascend910b/add_custom/AddCustom_402e355eb717124771cfc7dbebfe946c.json",
            "supportInfo": {
                "simplifiedKey": [
                    "AddCustom/d=0,p=0/3,2/3,2/3,2"
                ]
            }
        }  
        )");
    std::ofstream file("test.json");
    file << binJson.dump(4);
    file.close();

    binInfo.binPath = "test.o";
    static NnopbaseBinCollector* gBinCollector = nullptr;
    if (gBinCollector == nullptr) {
        gBinCollector = new NnopbaseBinCollector;
        ASSERT_NE(gBinCollector, nullptr);
        auto ret = NnopbaseCollectorInit(gBinCollector);
        ASSERT_EQ(ret, OK);
        ret = NnopbaseCollectorWork(gBinCollector);
        ASSERT_EQ(ret, OK);
    }
    NnopbaseExecutor* executor = new NnopbaseExecutor;
    NnopbaseExecutorSetCollector(executor, gBinCollector);
    executor->collector->useCoreTypeMagic = true;
    executor->collector->oppPath = "/usr/local/Ascend/latest/opp";
    executor->opType = "AddCustom";
    std::string socVersion = "ascend910b";
    auto oriSocVersion = nnopbase::IndvSoc::GetInstance().GetCurSocVersion();
    struct SocVersionGuard {
        std::string originalVersion;
        ~SocVersionGuard() { SetSocVersion(originalVersion); }
    } socGuard{oriSocVersion};
    SetSocVersion(socVersion);
    EXPECT_EQ(NnopbaseKernelRegister(executor, &binInfo), OK);
    EXPECT_EQ(nnopbase::GetMagicFormBin(true, &binInfo), "ACL_RT_BINARY_MAGIC_ELF_AICORE");
    EXPECT_EQ(binInfo.hasReg, true);
    EXPECT_EQ(NnopbaseKernelRegister(executor, &binInfo), OK);

    binInfo.coreType = kAicore;
    binInfo.hasReg = false;
    EXPECT_EQ(NnopbaseKernelRegister(executor, &binInfo), OK);
    EXPECT_EQ(nnopbase::GetMagicFormBin(true, &binInfo), "ACL_RT_BINARY_MAGIC_ELF_CUBE_CORE");

    binInfo.coreType = kVectorcore;
    binInfo.hasReg = false;
    EXPECT_EQ(NnopbaseKernelRegister(executor, &binInfo), OK);
    EXPECT_EQ(nnopbase::GetMagicFormBin(true, &binInfo), "ACL_RT_BINARY_MAGIC_ELF_VECTOR_CORE");

    binInfo.coreType = kCoreTypeEnd;
    binInfo.hasReg = false;
    EXPECT_EQ(NnopbaseKernelRegister(executor, &binInfo), OK);
    NnopbaseBinInfoDestroy(&p);
    delete executor;
    NnopbaseUnsetEnvAndClearFolder();
}
