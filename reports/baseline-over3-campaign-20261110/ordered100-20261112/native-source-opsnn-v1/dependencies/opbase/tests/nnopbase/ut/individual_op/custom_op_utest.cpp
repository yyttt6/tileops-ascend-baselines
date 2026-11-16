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
#include "executor/indv_executor.h"
#include "executor/indv_args_pool.h"
#include "individual_op_api.h"
#include "individual_op_internal.h"
#include "op_cache_internal.h"
#include "executor/indv_collector.h"
#include "utils/file_faker.h"
#include "depends/dump/dump_stub.h"
#include "depends/mmpa/mmpa_stub.h"
#include "depends/op/op_stub.h"
#include "depends/op/aclnn_bninference_d_kernel_stub.h"
#include "opdev/op_executor.h"
#include "utils/indv_lib_wrapper.h"
#include "depends/runtime/runtime_stub.h"

class NnopbaseOpPathTest : public testing::Test {
protected:
    void SetUp() { setenv("ASCEND_C", "1", 1); }
    void TearDown() { unsetenv("ASCEND_C"); }
};

string GetModelPath()
{
    mmDlInfo dl_info;
    if ((mmDladdr(reinterpret_cast<void*>(&GetModelPath), &dl_info) != EN_OK) || (dl_info.dli_fname == nullptr)) {
        return std::string();
    }

    if (strlen(dl_info.dli_fname) >= MMPA_MAX_PATH) {
        return std::string();
    }

    char_t path[MMPA_MAX_PATH] = {};
    if (mmRealPath(dl_info.dli_fname, &path[0], MMPA_MAX_PATH) != EN_OK) {
        return std::string();
    }

    string so_path = path;
    so_path = so_path.substr(0U, so_path.rfind('/') + 1U);
    return so_path;
}

TEST_F(NnopbaseOpPathTest, test_custom_opp_path_no_specified)
{
    unsetenv("ASCEND_CUSTOM_OPP_PATH");
    std::vector<std::pair<std::string, gert::OppImplVersionTag>> base_path;
    NnopbaseGetCustomOppPath(base_path);
    EXPECT_EQ(base_path.size(), 0);
}

TEST_F(NnopbaseOpPathTest, test_custom_opp_path_specified_with_empty)
{
    setenv("ASCEND_CUSTOM_OPP_PATH", "", 1);
    std::vector<std::pair<std::string, gert::OppImplVersionTag>> base_path;
    NnopbaseGetCustomOppPath(base_path);
    unsetenv("ASCEND_CUSTOM_OPP_PATH");
    EXPECT_EQ(base_path.size(), 0);
}

TEST_F(NnopbaseOpPathTest, test_custom_opp_path_specified)
{
    string custom_path = GetModelPath();
    custom_path = custom_path.substr(0, custom_path.rfind('/'));
    custom_path = custom_path.substr(0, custom_path.rfind('/'));
    custom_path = custom_path.substr(0, custom_path.rfind('/') + 1);

    custom_path.append("Nnopbase_custom_test");
    std::cout << "temp path is:" << custom_path << std::endl;
    system(("mkdir -p " + custom_path).c_str());

    custom_path.append(":");
    custom_path.append("/Nnopbase_test/"); // invalid path
    setenv("ASCEND_CUSTOM_OPP_PATH", custom_path.c_str(), 1);

    std::vector<std::pair<std::string, gert::OppImplVersionTag>> base_path;
    NnopbaseGetCustomOppPath(base_path);
    unsetenv("ASCEND_CUSTOM_OPP_PATH");
    EXPECT_EQ(base_path.size(), 1);
}

TEST_F(NnopbaseOpPathTest, test_custom_opp_path_with_default)
{
    std::string opp_path = GetModelPath();
    opp_path = opp_path.substr(0, opp_path.rfind('/'));
    opp_path = opp_path.substr(0, opp_path.rfind('/'));
    opp_path = opp_path.substr(0, opp_path.rfind('/') + 1);

    std::cout << "opp path is:" << opp_path << std::endl;
    setenv("ASCEND_OPP_PATH", opp_path.c_str(), 1);

    std::string path_vendors = opp_path + "vendors";
    std::string path_config = path_vendors + "/config.ini";
    system(("mkdir -p " + path_vendors).c_str());
    system(("echo 'load_priority=mdc' > " + path_config).c_str());

    NnopbaseBinCollector* bin_collector = new NnopbaseBinCollector;
    std::vector<std::pair<std::string, gert::OppImplVersionTag>> base_path;
    int32_t builtInStartIndex = -1;
    NnopbaseGetOppPath(bin_collector, base_path, builtInStartIndex);
    unsetenv("ASCEND_OPP_PATH");
    EXPECT_EQ(base_path.size(), 2);
    EXPECT_EQ(builtInStartIndex, base_path.size() - 1);
    delete bin_collector;
}

TEST_F(NnopbaseOpPathTest, test_read_config_file_format_error)
{
    std::string opp_path = GetModelPath();
    opp_path = opp_path.substr(0, opp_path.rfind('/'));
    opp_path = opp_path.substr(0, opp_path.rfind('/'));
    opp_path = opp_path.substr(0, opp_path.rfind('/') + 1);

    std::cout << "opp path is:" << opp_path << std::endl;
    setenv("ASCEND_OPP_PATH", opp_path.c_str(), 1);

    std::string path_vendors = opp_path + "vendors";
    std::string path_config = path_vendors + "/config.ini";
    system(("mkdir -p " + path_vendors).c_str());
    system(("echo 'load_prioritymdc' > " + path_config).c_str());

    std::vector<std::string> subPath;
    bool ret = NnopbaseReadConfigFile(path_config, subPath);
    unsetenv("ASCEND_OPP_PATH");
    EXPECT_EQ(ret, false);
}

TEST_F(NnopbaseOpPathTest, test_model_path_no_exist)
{
    std::string opp_path = "/usr/local/latest/opp";
    setenv("ASCEND_OPP_PATH", opp_path.c_str(), 1);
    std::cout << "opp path is:" << opp_path << std::endl;
    std::string os_type;
    std::string cpu_type;
    NnopbaseGetCurEnvPackageOsAndCpuType(os_type, cpu_type);
    std::cout << "cpu_type" << cpu_type << std::endl;
    std::cout << "os_type" << os_type << std::endl;
    EXPECT_EQ(os_type, "");
    EXPECT_EQ(os_type, "");
}