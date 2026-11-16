/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <stdio.h>

#include "nlohmann/json.hpp"
#include "register/op_binary_resource_manager.h"
#include "executor/indv_executor.h"

#define ASSERT_TRUE(exp)                                                                \
    do {                                                                                \
        if (!(exp)) {                                                                   \
            std::cout << __FUNCTION__ << " Assert " << #exp << " failed!" << std::endl; \
            return 0;                                                                   \
        }                                                                               \
    } while (false)

#define ASSERT_SUCC(exp) ASSERT_TRUE((exp) == 1)

namespace {
std::string OPP_PATH;

int32_t CreateStubCustConfig()
{
    FILE* file = fopen((OPP_PATH + "/vendors/config.ini").c_str(), "w");
    ASSERT_TRUE(file != nullptr);
    fprintf(file, "load_priority=cust\n");
    fclose(file);
    return 1;
}

int32_t CreateStubSceneFile()
{
    FILE* file = fopen((OPP_PATH + "/scene.info").c_str(), "w");
    ASSERT_TRUE(file != nullptr);
    fprintf(file, "os=linux\n");
    fprintf(file, "os_version=\n");
    fprintf(file, "arch=x86_64\n");
    fclose(file);
    return 1;
}

void CreateOppDir(const std::string& dirPrefix)
{
    mkdir(dirPrefix.c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
    mkdir((dirPrefix + "/op_impl").c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
    mkdir((dirPrefix + "/op_impl/ai_core").c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
    mkdir((dirPrefix + "/op_impl/ai_core/tbe").c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
    mkdir((dirPrefix + "/op_impl/ai_core/tbe/kernel").c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
    mkdir((dirPrefix + "/op_impl/ai_core/tbe/kernel/ascend910b").c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
    mkdir((dirPrefix + "/op_impl/ai_core/tbe/kernel/ascend910").c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
    mkdir((dirPrefix + "/op_impl/ai_core/tbe/kernel/ascend310p").c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
    mkdir((dirPrefix + "/op_impl/ai_core/tbe/kernel/config").c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
    mkdir((dirPrefix + "/op_impl/ai_core/tbe/kernel/ascend950").c_str(), S_IRUSR | S_IWUSR | S_IXUSR);

    mkdir((dirPrefix + "/op_impl/ai_core/tbe/kernel/config/ascend910b").c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
    mkdir((dirPrefix + "/op_impl/ai_core/tbe/kernel/config/ascend910").c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
    mkdir((dirPrefix + "/op_impl/ai_core/tbe/kernel/config/ascend310p").c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
    mkdir((dirPrefix + "/op_impl/ai_core/tbe/kernel/config/ascend950").c_str(), S_IRUSR | S_IWUSR | S_IXUSR);

    // for tiling
    mkdir((dirPrefix + "/op_impl/ai_core/tbe/op_tiling").c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
    mkdir((dirPrefix + "/op_impl/ai_core/tbe/op_tiling/lib").c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
    mkdir((dirPrefix + "/op_impl/ai_core/tbe/op_tiling/lib/linux").c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
    mkdir((dirPrefix + "/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64").c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
    mkdir((dirPrefix + "/op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64").c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
    // for tiling 2.0
    mkdir((dirPrefix + "/op_impl/ai_core/tbe/op_host").c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
    mkdir((dirPrefix + "/op_impl/ai_core/tbe/op_host/lib").c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
    mkdir((dirPrefix + "/op_impl/ai_core/tbe/op_host/lib/linux").c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
    mkdir((dirPrefix + "/op_impl/ai_core/tbe/op_host/lib/linux/aarch64").c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
    mkdir((dirPrefix + "/op_impl/ai_core/tbe/op_host/lib/linux/aarch64/emptyDir").c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
    mkdir((dirPrefix + "/op_impl/ai_core/tbe/op_host/lib/linux/x86_64").c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
    mkdir((dirPrefix + "/op_impl/ai_core/tbe/op_host/lib/linux/x86_64/emptyDir").c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
}

void CreateStubFolders(const std::string& opp_path, const bool isOppLastest = false)
{
    if (opp_path.empty()) {
        OPP_PATH = "./usr/local/Ascend/latest/opp";
        mkdir("./usr", S_IRUSR | S_IWUSR | S_IXUSR);
        mkdir("./usr/local", S_IRUSR | S_IWUSR | S_IXUSR);
        mkdir("./usr/local/Ascend", S_IRUSR | S_IWUSR | S_IXUSR);
        mkdir("./usr/local/Ascend/latest", S_IRUSR | S_IWUSR | S_IXUSR);
        mkdir("./usr/local/Ascend/latest/opp", S_IRUSR | S_IWUSR | S_IXUSR);
        if (isOppLastest) {
            setenv("ASCEND_HOME_PATH", "./usr/local/Ascend/latest", 1);
            mkdir("./usr/local/Ascend/latest/opp_latest", S_IRUSR | S_IWUSR | S_IXUSR);
            OPP_PATH = "./usr/local/Ascend/latest/opp_latest";
        }
    } else {
        OPP_PATH = opp_path;
        // 如果设置isOppLastest且出入opp_path, 将opp_path当做是ASCEND_HOME_PATH的路径
        if (isOppLastest) {
            setenv("ASCEND_HOME_PATH", opp_path.c_str(), 1);
            OPP_PATH = opp_path + "/opp_latest";
        }
    }

    std::cout << "ASCEND_OPP_PATH is" << OPP_PATH << std::endl;
    setenv("ASCEND_OPP_PATH", OPP_PATH.c_str(), 1);
    std::string cust_opp_path = OPP_PATH + "/user_define";
    setenv("ASCEND_CUSTOM_OPP_PATH", cust_opp_path.c_str(), 1);

    // build-in
    CreateOppDir(OPP_PATH + "/built-in");

    // 自定义case1:vendor 默认路径下 cust
    mkdir((OPP_PATH + "/vendors/").c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
    CreateOppDir(OPP_PATH + "/vendors/cust");

    // 自定义case2:构造用户指定任意目录 ../opp/user_define
    CreateOppDir(OPP_PATH + "/user_define");

    // staticKernel 路径
    mkdir((OPP_PATH + "/static_kernel").c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
    mkdir((OPP_PATH + "/static_kernel/ai_core").c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
    mkdir((OPP_PATH + "/static_kernel/ai_core/static_kernel_202307261051").c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
    mkdir((OPP_PATH + "/static_kernel/ai_core/static_kernel_202307261051/Flash").c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
    mkdir((OPP_PATH + "/static_kernel/ai_core/static_kernel_202307261051/TestStatic").c_str(),
          S_IRUSR | S_IWUSR | S_IXUSR);
    mkdir((OPP_PATH + "/static_kernel/ai_core/static_kernel_202307261051/TestStaticAdd").c_str(),
          S_IRUSR | S_IWUSR | S_IXUSR);
    mkdir((OPP_PATH + "/static_kernel/ai_core/static_kernel_202307261051/Conv2D").c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
    mkdir((OPP_PATH + "/static_kernel/ai_core/static_kernel_202307261051/TransData").c_str(),
          S_IRUSR | S_IWUSR | S_IXUSR);
    mkdir((OPP_PATH + "/static_kernel/ai_core/config").c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
    mkdir((OPP_PATH + "/static_kernel/ai_core/config/ascend910b").c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
    mkdir((OPP_PATH + "/static_kernel/ai_core/config/ascend910").c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
    mkdir((OPP_PATH + "/static_kernel/ai_core/config/ascend950").c_str(), S_IRUSR | S_IWUSR | S_IXUSR);

    // debug 路径
    mkdir((OPP_PATH + "/debug_kernel").c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
    mkdir((OPP_PATH + "/debug_kernel/config").c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
    mkdir((OPP_PATH + "/debug_kernel/config/ascend910b").c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
    mkdir((OPP_PATH + "/debug_kernel/config/ascend910").c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
    mkdir((OPP_PATH + "/debug_kernel/config/ascend950").c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
    mkdir((OPP_PATH + "/debug_kernel/ascend910b").c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
    mkdir((OPP_PATH + "/debug_kernel/ascend910").c_str(), S_IRUSR | S_IWUSR | S_IXUSR);

    // conf 路径
    mkdir((OPP_PATH + "/conf").c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
    mkdir((OPP_PATH + "/conf/error_manager").c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
    mkdir((OPP_PATH + "/conf/op_api").c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
}

int32_t NnopbaseCreateStubOpInfoConfig(const std::string& filePath, const std::string& soc)
{
    FILE* file = fopen(filePath.c_str(), "w");
    ASSERT_TRUE(file != nullptr);
    const std::string binPathPrefix = std::string("                \"binPath\": \"") + soc;
    const std::string binPath = std::string("                \"binPath\": \"");
    fprintf(file, "{\n");

    // bninference_d_kernel
    fprintf(file, "    \"bninference_d_kernel\": {\n");
    fprintf(file, "        \"dynamicRankSupport\": true,\n");
    fprintf(file, "        \"simplifiedKeyMode\": 0,\n");
    fprintf(file, "        \"binaryList\": [\n");
    fprintf(file, "            {\n");
    fprintf(file, "                \"coreType\": 0,\n");
    fprintf(file, "                \"simplifiedKey\": [\"bninference_d_kernel/d=0,p=0/1,30/1,30/1,30/1,30\",\n");
    fprintf(file, "                                    \"bninference_d_kernel/d=0,p=0/0,2/0,2/0,2/0,2\"],\n");
    fprintf(file, (binPathPrefix + "/kernel_bninference_d_dync_main.o\"\n").c_str());
    fprintf(file, "            }\n");
    fprintf(file, "        ]\n");
    fprintf(file, "    },\n");

    fprintf(file, "    \"NativeSparseAttention\": {\n");
    fprintf(file, "        \"dynamicRankSupport\": true,\n");
    fprintf(file, "        \"simplifiedKeyMode\": 0,\n");
    fprintf(file, "        \"binaryList\": [\n");
    fprintf(file, "            {\n");
    fprintf(file, "                \"coreType\": 0,\n");
    fprintf(file, "                \"simplifiedKey\": [\"NativeSparseAttention/d=0,p=0/0,2/0,2\",\n");
    fprintf(file, "                                    \"NativeSparseAttention/d=0,p=0/0,2/0,2\"],\n");
    fprintf(file, (binPathPrefix + "/NativeSparseAttention_fatbin.o\"\n").c_str());
    fprintf(file, "            }\n");
    fprintf(file, "        ]\n");
    fprintf(file, "    },\n");

    // NonFiniteCheck
    fprintf(file, "    \"NonFiniteCheck\": {\n");
    fprintf(file, "        \"dynamicRankSupport\": true,\n");
    fprintf(file, "        \"simplifiedKeyMode\": 0,\n");
    fprintf(file, "        \"binaryList\": [\n");
    fprintf(file, "            {\n");
    fprintf(file, "                \"coreType\": 0,\n");
    fprintf(file, "                \"simplifiedKey\": [\"NonFiniteCheck/d=0,p=0/0,2/0,2\",\n");
    fprintf(file, "                                    \"NonFiniteCheck/d=0,p=0/1,2/1,2\",\n");
    fprintf(file, "                                    \"NonFiniteCheck/d=0,p=0/27,2/27,2\"],\n");
    fprintf(file, (binPathPrefix + "/kernel_NonFiniteCheck.o\"\n").c_str());
    fprintf(file, "            }\n");
    fprintf(file, "        ]\n");
    fprintf(file, "    },\n");

    // AddTik2
    fprintf(file, "    \"AddTik2\": {\n");
    fprintf(file, "        \"dynamicRankSupport\": true,\n");
    fprintf(file, "        \"simplifiedKeyMode\": 0,\n");
    fprintf(file, "        \"binaryList\": [\n");
    fprintf(file, "            {\n");
    fprintf(file, "                \"coreType\": 2,\n");
    fprintf(file, "                \"simplifiedKey\": [\"AddTik2/d=0,p=0/0,2/0,2/0,2\"],\n");
    fprintf(file, (binPathPrefix + "/add_tik2_entry_ascend910.o\"\n").c_str());
    fprintf(file, "            }\n");
    fprintf(file, "        ]\n");
    fprintf(file, "    },\n");

    // AutomicClean
    fprintf(file, "    \"AutomicClean\": {\n");
    fprintf(file, "        \"dynamicRankSupport\": true,\n");
    fprintf(file, "        \"simplifiedKeyMode\": 0,\n");
    fprintf(file, "        \"binaryList\": [\n");
    fprintf(file, "            {\n");
    fprintf(file, "                \"coreType\": 0,\n");
    fprintf(file, "                \"simplifiedKey\": [\"AutomicClean/d=0,p=0/0,2/0,2/0,2\",\n");
    fprintf(file, "                                    \"AutomicClean/d=0,p=0/1,2/1,2/1,2\"],\n");
    fprintf(file, (binPathPrefix + "/kernel_output_automic_clean.o\"\n").c_str());
    fprintf(file, "            }\n");
    fprintf(file, "        ]\n");
    fprintf(file, "    },\n");

    // TestDavidCustom david only
    fprintf(file, "    \"TestDavidCustom\": {\n");
    fprintf(file, "        \"dynamicRankSupport\": true,\n");
    fprintf(file, "        \"simplifiedKeyMode\": 0,\n");
    fprintf(file, "        \"binaryList\": [\n");
    fprintf(file, "            {\n");
    fprintf(file, "                \"coreType\": 0,\n");
    fprintf(file, "                \"simplifiedKey\": [\"TestDavidCustom/d=0,p=0/0,2/0,2/0,2\"],\n");
    fprintf(file, (binPath + "ascend950" + "/TestDavidCustom.o\"\n").c_str());
    fprintf(file, "            }\n");
    fprintf(file, "        ]\n");
    fprintf(file, "    },\n");

    // memsetV2 david only
    fprintf(file, "    \"MemSetV2\": {\n");
    fprintf(file, "        \"dynamicRankSupport\": true,\n");
    fprintf(file, "        \"simplifiedKeyMode\": 2,\n");
    fprintf(file, "        \"params\": {\n");
    fprintf(file, "            \"inputs\": [\n");
    fprintf(file, "                [\n");
    fprintf(file, "                    {\n");
    fprintf(file, "                        \"name\": \"x\",\n");
    fprintf(file, "                        \"index\": 0,\n");
    fprintf(file, "                        \"paramType\": \"dynamic\",\n");
    fprintf(file, "                        \"dtypeMode\": \"bit\",\n");
    fprintf(file, "                        \"formatMode\": \"static_nd_agnostic\"\n");
    fprintf(file, "                    }\n");
    fprintf(file, "                ]\n");
    fprintf(file, "            ],\n");
    fprintf(file, "            \"outputs\": [\n");
    fprintf(file, "                [\n");
    fprintf(file, "                    {\n");
    fprintf(file, "                        \"name\": \"x\",\n");
    fprintf(file, "                        \"index\": 0,\n");
    fprintf(file, "                        \"paramType\": \"dynamic\",\n");
    fprintf(file, "                        \"dtypeMode\": \"bit\",\n");
    fprintf(file, "                        \"formatMode\": \"static_nd_agnostic\"\n");
    fprintf(file, "                    }\n");
    fprintf(file, "                ]\n");
    fprintf(file, "            ],\n");
    fprintf(file, "            \"attrs\": [\n");
    fprintf(file, "                {\n");
    fprintf(file, "                    \"name\": \"values_int\"\n");
    fprintf(file, "                },\n");
    fprintf(file, "                {\n");
    fprintf(file, "                    \"name\": \"values_float\"\n");
    fprintf(file, "                }\n");
    fprintf(file, "            ]\n");
    fprintf(file, "        },\n");
    fprintf(file, "        \"binaryList\": [\n");
    fprintf(file, "            {\n");
    fprintf(file, "                \"coreType\": 2,\n");
    fprintf(file, "                \"simplifiedKey\": [\n");
    fprintf(file, "                    \"MemSetV2/d=0,p=0/diy,99\",\n");
    fprintf(file, "                    \"MemSetV2/d=1,p=0/diy,99\"\n");
    fprintf(file, "                ],\n");
    fprintf(file, "                \"binPath\": \"ascend950/MemSetV2.o\",\n");
    fprintf(file, "                \"jsonPath\": \"ascend950/MemSetV2.json\"\n");
    fprintf(file, "            }\n");
    fprintf(file, "        ]\n");
    fprintf(file, "    },\n");

    // test_register
    fprintf(file, "    \"test_register\": {\n");
    fprintf(file, "        \"dynamicRankSupport\": true,\n");
    fprintf(file, "        \"simplifiedKeyMode\": 0,\n");
    fprintf(file, "        \"binaryList\": [\n");
    fprintf(file, "            {\n");
    fprintf(file, "                \"coreType\": 2,\n");
    fprintf(file, "                \"simplifiedKey\": [\"test_register/d=0,p=0\"],\n");
    fprintf(file, (binPathPrefix + "/test_register_ascend910.o\"\n").c_str());
    fprintf(file, "            }\n");
    fprintf(file, "        ]\n");
    fprintf(file, "    },\n");

    // noSimplifiedKey
    fprintf(file, "    \"noSimplifiedKey\": {\n");
    fprintf(file, "        \"dynamicRankSupport\": true,\n");
    fprintf(file, "        \"simplifiedKeyMode\": 0,\n");
    fprintf(file, "        \"binaryList\": [\n");
    fprintf(file, "            {\n");
    fprintf(file, "                \"coreType\": 2,\n");
    fprintf(file, (binPathPrefix + "/test_register_ascend910.o\"\n").c_str());
    fprintf(file, "            }\n");
    fprintf(file, "        ]\n");
    fprintf(file, "    },\n");

    // 1971_for_mix_normal
    fprintf(file, "    \"1971_for_mix_normal\": {\n");
    fprintf(file, "        \"dynamicRankSupport\": true,\n");
    fprintf(file, "        \"simplifiedKeyMode\": 0,\n");
    fprintf(file, "        \"binaryList\": [\n");
    fprintf(file, "            {\n");
    fprintf(file, "                \"coreType\": 0,\n");
    fprintf(file, "                \"multiKernelType\": 1,\n");
    fprintf(file, "                \"simplifiedKey\": [\"1971_for_mix_normal/d=0,p=0/1,30/1,30/1,30/1,30\",\n");
    fprintf(file, "                                    \"1971_for_mix_normal/d=0,p=0/0,2/0,2/0,2/0,2\"],\n");
    fprintf(file, (binPathPrefix + "/1971_for_mix_normal.o\"\n").c_str());
    fprintf(file, "            },\n");
    fprintf(file, "            {\n");
    fprintf(file, "                \"coreType\": 0,\n");
    fprintf(file, "                \"simplifiedKey\": [\"1971_for_mix_normal/d=0,p=0\"],\n");
    fprintf(file, (binPathPrefix + "/1971_for_mix_normal.o\"\n").c_str());
    fprintf(file, "            }\n");
    fprintf(file, "        ]\n");
    fprintf(file, "    },\n");

    // 310p_for_mix_aic_normal
    fprintf(file, "    \"310p_for_mix_aic_normal\": {\n");
    fprintf(file, "        \"dynamicRankSupport\": true,\n");
    fprintf(file, "        \"simplifiedKeyMode\": 0,\n");
    fprintf(file, "        \"binaryList\": [\n");
    fprintf(file, "            {\n");
    fprintf(file, "                \"coreType\": 3,\n");
    fprintf(file, "                \"multiKernelType\": 1,\n");
    fprintf(file, "                \"simplifiedKey\": [\"310p_for_mix_aic_normal/d=0,p=0/1,30/1,30/1,30/1,30\",\n");
    fprintf(file, "                                    \"310p_for_mix_aic_normal/d=0,p=0/0,2/0,2/0,2/0,2\"],\n");
    fprintf(file, (binPathPrefix + "/310p_for_mix_aic_normal.o\"\n").c_str());
    fprintf(file, "            },\n");
    fprintf(file, "            {\n");
    fprintf(file, "                \"coreType\": 3,\n");
    fprintf(file, "                \"simplifiedKey\": [\"310p_for_mix_aic_normal/d=0,p=0\"],\n");
    fprintf(file, (binPathPrefix + "/310p_for_mix_aic_normal.o\"\n").c_str());
    fprintf(file, "            }\n");
    fprintf(file, "        ]\n");
    fprintf(file, "    },\n");

    // 310p_for_mix_aiv_normal
    fprintf(file, "    \"310p_for_mix_aiv_normal\": {\n");
    fprintf(file, "        \"dynamicRankSupport\": true,\n");
    fprintf(file, "        \"simplifiedKeyMode\": 0,\n");
    fprintf(file, "        \"binaryList\": [\n");
    fprintf(file, "            {\n");
    fprintf(file, "                \"coreType\": 4,\n");
    fprintf(file, "                \"multiKernelType\": 1,\n");
    fprintf(file, "                \"simplifiedKey\": [\"310p_for_mix_aiv_normal/d=0,p=0/1,30/1,30/1,30/1,30\",\n");
    fprintf(file, "                                    \"310p_for_mix_aiv_normal/d=0,p=0/0,2/0,2/0,2/0,2\"],\n");
    fprintf(file, (binPathPrefix + "/310p_for_mix_aiv_normal.o\"\n").c_str());
    fprintf(file, "            },\n");
    fprintf(file, "            {\n");
    fprintf(file, "                \"coreType\": 3,\n");
    fprintf(file, "                \"simplifiedKey\": [\"310p_for_mix_aiv_normal/d=0,p=0\"],\n");
    fprintf(file, (binPathPrefix + "/310p_for_mix_aiv_normal.o\"\n").c_str());
    fprintf(file, "            }\n");
    fprintf(file, "        ]\n");
    fprintf(file, "    },\n");

    // test_profiling_mix_aic
    fprintf(file, "    \"test_profiling_mix_aic\": {\n");
    fprintf(file, "        \"dynamicRankSupport\": true,\n");
    fprintf(file, "        \"simplifiedKeyMode\": 0,\n");
    fprintf(file, "        \"binaryList\": [\n");
    fprintf(file, "            {\n");
    fprintf(file, "                \"coreType\": 0,\n");
    fprintf(file, "                \"simplifiedKey\": [\"test_profiling_mix_aic/d=0,p=0/1,30/1,30/1,30/1,30\",\n");
    fprintf(file, "                                    \"test_profiling_mix_aic/d=0,p=0/0,2/0,2/0,2/0,2\"],\n");
    fprintf(file, (binPathPrefix + "/test_profiling_mix_aic.o\"\n").c_str());
    fprintf(file, "            }\n");
    fprintf(file, "        ]\n");
    fprintf(file, "    },\n");

    // test_profiling_mix_aiv
    fprintf(file, "    \"test_profiling_mix_aiv\": {\n");
    fprintf(file, "        \"dynamicRankSupport\": true,\n");
    fprintf(file, "        \"simplifiedKeyMode\": 0,\n");
    fprintf(file, "        \"binaryList\": [\n");
    fprintf(file, "            {\n");
    fprintf(file, "                \"coreType\": 0,\n");
    fprintf(file, "                \"simplifiedKey\": [\"test_profiling_mix_aiv/d=0,p=0/1,30/1,30/1,30/1,30\",\n");
    fprintf(file, "                                    \"test_profiling_mix_aiv/d=0,p=0/0,2/0,2/0,2/0,2\"],\n");
    fprintf(file, (binPathPrefix + "/test_profiling_mix_aiv.o\"\n").c_str());
    fprintf(file, "            }\n");
    fprintf(file, "        ]\n");
    fprintf(file, "    },\n");

    // 1971_for_mix_unnormal
    fprintf(file, "    \"1971_for_mix_unnormal\": {\n");
    fprintf(file, "        \"dynamicRankSupport\": true,\n");
    fprintf(file, "        \"simplifiedKeyMode\": 0,\n");
    fprintf(file, "        \"binaryList\": [\n");
    fprintf(file, "            {\n");
    fprintf(file, "                \"coreType\": 0,\n");
    fprintf(file, "                \"multiKernelType\": 1,\n");
    fprintf(file, "                \"simplifiedKey\": [\"1971_for_mix_unnormal/d=0,p=0/1,30/1,30/1,30/1,30\",\n");
    fprintf(file, "                                    \"1971_for_mix_unnormal/d=0,p=0/0,2/0,2/0,2/0,2\"],\n");
    fprintf(file, (binPathPrefix + "/1971_for_mix_unnormal.o\"\n").c_str());
    fprintf(file, "            },\n");
    fprintf(file, "            {\n");
    fprintf(file, "                \"coreType\": 0,\n");
    fprintf(file, "                \"multiKernelType\": 1,\n");
    fprintf(file, "                \"simplifiedKey\": [\"1971_for_mix_unnormal2/d=0,p=0/1,2/1,2/1,2/1,2\"],\n");
    fprintf(file, (binPathPrefix + "/1971_for_mix_unnormal2.o\"\n").c_str());
    fprintf(file, "            }\n");
    fprintf(file, "        ]\n");
    fprintf(file, "    },\n");

    // StubCustomOp（customizedKey模式，用于customizedKey动态bin查找测试）
    fprintf(file, "    \"StubCustomOp\": {\n");
    fprintf(file, "        \"dynamicRankSupport\": true,\n");
    fprintf(file, "        \"simplifiedKeyMode\": 2,\n");
    fprintf(file, "        \"binaryList\": [\n");
    fprintf(file, "            {\n");
    fprintf(file, "                \"coreType\": 0,\n");
    fprintf(file, "                \"simplifiedKey\": [\"StubCustomOp/d=0,p=0/diy,99\"],\n");
    fprintf(file, (binPath + "ascend910/kernel_bninference_d_dync_main.o\"\n").c_str());
    fprintf(file, "            }\n");
    fprintf(file, "        ]\n");
    fprintf(file, "    },\n");

    // StubCustomOpNoGen（customizedKey模式但未注册genSimplifiedKey，用于加载期gen指针为空场景）
    fprintf(file, "    \"StubCustomOpNoGen\": {\n");
    fprintf(file, "        \"dynamicRankSupport\": true,\n");
    fprintf(file, "        \"simplifiedKeyMode\": 2,\n");
    fprintf(file, "        \"binaryList\": [\n");
    fprintf(file, "            {\n");
    fprintf(file, "                \"coreType\": 0,\n");
    fprintf(file, "                \"simplifiedKey\": [\"StubCustomOpNoGen/d=0,p=0/diy,99\"],\n");
    fprintf(file, (binPath + "ascend910/kernel_bninference_d_dync_main.o\"\n").c_str());
    fprintf(file, "            }\n");
    fprintf(file, "        ]\n");
    fprintf(file, "    },\n");

    // StubMixKeyOp（内置包声明 mode=2）。同名算子在自定义包中声明 mode=0，见
    // CreateCustomStubOpInfoConfig。自定义包先加载、内置包后加载，故 binTbl 内两种索引共存，
    // 且 regInfo->customizedSimplifiedKey 被内置包覆写为 true。
    // 此处后缀 "diy,zzz" 与桩 gen 产出的 "diy,99" 不一致，保证 customizedKey 必然未命中。
    fprintf(file, "    \"StubMixKeyOp\": {\n");
    fprintf(file, "        \"dynamicRankSupport\": true,\n");
    fprintf(file, "        \"simplifiedKeyMode\": 2,\n");
    fprintf(file, "        \"binaryList\": [\n");
    fprintf(file, "            {\n");
    fprintf(file, "                \"coreType\": 0,\n");
    fprintf(file, "                \"simplifiedKey\": [\"StubMixKeyOp/d=0,p=0/diy,zzz\"],\n");
    fprintf(file, (binPath + "ascend910/kernel_bninference_d_dync_main.o\"\n").c_str());
    fprintf(file, "            }\n");
    fprintf(file, "        ]\n");
    fprintf(file, "    }\n");
    fprintf(file, "}\n");

    fclose(file);
    return 1;
}

int32_t CreateStaticStubOpInfoConfig(const std::string& filePath)
{
    FILE* file = fopen(filePath.c_str(), "w");
    ASSERT_TRUE(file != nullptr);
    fprintf(file, "{\n");

    // Conv2D
    fprintf(file, "    \"TestStaticAdd\": {\n");
    fprintf(file, "        \"dynamicRankSupport\": true,\n");
    fprintf(file, "        \"simplifiedKeyMode\": 0,\n");
    fprintf(file, "        \"staticList\": [\n");
    fprintf(file, "            {\n");
    fprintf(file, "                \"binDesc\": {\n");
    fprintf(file, "                    \"blockDim\": 16,\n");
    fprintf(file, "                    \"kernelName\": \"TestStaticAdd_1e04ee05ab491cc5ae9c3d5c9ee8950b\"\n");
    fprintf(file, "                },\n");
    fprintf(file, "                \"coreType\": 1,\n");
    fprintf(file, "                \"binPath\": "
                  "\"static_kernel_202307261051/TestStaticAdd/TestStaticAdd_high_performance_0.o\",\n");
    fprintf(file, "                \"simplifiedKey\": "
                  "\"TestStaticAdd/d=0,p=1/0,0,(200,200,3,3)/0,0,(200,200,3,3)/0,0,(200,200,3,3)/\"\n");
    fprintf(file, "            }\n");
    fprintf(file, "        ]\n");
    fprintf(file, "    },\n");

    // Conv2D
    fprintf(file, "    \"Conv2D\": {\n");
    fprintf(file, "        \"dynamicRankSupport\": true,\n");
    fprintf(file, "        \"simplifiedKeyMode\": 0,\n");
    fprintf(file, "        \"staticList\": [\n");
    fprintf(file, "            {\n");
    fprintf(file, "                \"binDesc\": {\n");
    fprintf(file, "                    \"blockDim\": 24,\n");
    fprintf(file, "                    \"kernelName\": \"Conv2D_1e04ee05ab491cc5ae9c3d5c9ee8950b\"\n");
    fprintf(file, "                },\n");
    fprintf(file, "                \"coreType\": 1,\n");
    fprintf(file, "                \"binPath\": \"static_kernel_202307261051/Conv2D/Conv2D_high_performance_0.o\",\n");
    fprintf(file, "                \"simplifiedKey\": "
                  "\"Conv2D/d=0,p=2/1,3,(20,4,60,60,10)/1,4,(24,4,10,10)/_/_/1,3,(20,4,60,60,10)/[1,1,1,1]/[1,1,1,1]/"
                  "[1,1,1,1]/1/5748434e/0\"\n");
    fprintf(file, "            }\n");
    fprintf(file, "        ]\n");
    fprintf(file, "    },\n");

    // TestStatic
    fprintf(file, "    \"TestStatic\": {\n");
    fprintf(file, "        \"dynamicRankSupport\": true,\n");
    fprintf(file, "        \"simplifiedKeyMode\": 0,\n");
    fprintf(file, "        \"staticList\": [\n");
    fprintf(file, "            {\n");
    fprintf(file, "                \"binDesc\": {\n");
    fprintf(file, "                    \"blockDim\": 16,\n");
    fprintf(file, "                    \"kernelName\": \"TestStatic_1e04ee05ab491cc5ae9c3d5c9ee8950b\",\n");
    fprintf(file, "                    \"workspace\": [100,101,102]\n");
    fprintf(file, "                },\n");
    fprintf(file, "                \"coreType\": 0,\n");
    fprintf(
        file,
        "                \"binPath\": \"static_kernel_202307261051/TestStatic/TestStatic_high_performance_0.o\",\n");
    fprintf(file, "                \"simplifiedKey\": \"xxxxx\"\n");
    fprintf(file, "            }\n");
    fprintf(file, "        ]\n");
    fprintf(file, "    },\n");

    // Flash
    fprintf(file, "    \"Flash\": {\n");
    fprintf(file, "        \"dynamicRankSupport\": true,\n");
    fprintf(file, "        \"simplifiedKeyMode\": 0,\n");
    fprintf(file, "        \"staticList\": [\n");
    fprintf(file, "            {\n");
    fprintf(file, "                \"binDesc\": {\n");
    fprintf(file, "                    \"blockDim\": 8,\n");
    fprintf(file, "                    \"kernelName\": \"FlashCustom_1e04ee05ab491cc5ae9c3d5c9ee8950b\",\n");
    fprintf(file, "                    \"workspace\": [100,101,102]\n");
    fprintf(file, "                },\n");
    fprintf(file, "                \"coreType\": 0,\n");
    fprintf(file, "                \"binPath\": \"static_kernel_202307261051/Flash/flash.o\",\n");
    fprintf(file, "                \"simplifiedKey\": "
                  "[\"Flash/d=0,p=0/0,2,(1,1,1,1,1)/0,2,(1,1,1,1,1)/0,2,(1,1,1,1,1)/0,2,(1,1,1,1,1)\",\n");
    fprintf(
        file,
        "                                    \"Flash/d=0,p=0/0,2,(1,1,1,1,1)/_/0,2,(1,1,1,1,1)/0,2,(1,1,1,1,1)\", \n");
    fprintf(
        file,
        "                                    \"Flash/d=0,p=0/3,2,(5),[1b,1b,1b,1b,9bb0]/3,2,(5)/3,2,(5)/3,2,(5)\",\n");
    fprintf(file, "                                    "
                  "\"Flash/d=0,p=0/0,2,(1,1,1,1,1)/0,2,(1,1,1,1,1)/0,2,(1,1,1,1,1)/0,2,(1,1,1,1,1)/[1,0,1]/[3,28,5]/"
                  "65636261/[3f4ccccd,3f4ccccd,3f4ccccd]\",\n");
    fprintf(file, "                                    "
                  "\"Flash/d=0,p=0/0,2,(1,1,1,1,1),[3f4ccccd]/0,2,(1,1,1,1,1)/0,2,(1,1,1,1,1)/0,2,(1,1,1,1,1)/[1,0,1]/"
                  "[3,28,5]/65636261/[3f4ccccd,3f4ccccd,3f4ccccd]\",\n");
    fprintf(file, "                                    \"Flash/d=0,p=0/0,2,(5)/0,2,(5)/0,2,(5),(5),(5)/0,2,(5)\",\n");
    fprintf(
        file,
        "                                    \"Flash/d=0,p=0/0,2,(5)/0,2,(5)/0,2,(5)/0,2,(5)/5a5f4c415443415246\",\n");
    fprintf(file, "                                    "
                  "\"Flash/d=0,p=0/0,2,(1,1,1,1,1)/0,2,(1,1,1,1,1)/0,2,(1,1,1,1,1)/0,2,(1,1,1,1,1)/3f4ccccd\",\n");
    fprintf(file,
            "                                    "
            "\"Flash/d=0,p=0/0,2,(1,1,1,1,1)/0,2,(1,1,1,1,1)/0,2,(1,1,1,1,1)/0,2,(1,1,1,1,1)/[3f4ccccd,3f4ccccd]\",\n");
    fprintf(
        file,
        "                                    "
        "\"Flash/d=0,p=0/0,2,(1,1,1,1,1)/0,2,(1,1,1,1,1),(1,1,1,1,1)/0,2,(1,1,1,1,1)/0,2,(1,1,1,1,1),(1,1,1,1,1)\",\n");
    fprintf(file, "                                    "
                  "\"Flash/d=0,p=0/0,2,(1,1,1,1,1)/0,2,(1,1,1,1,1),(1,1,1,1,1)/0,2,(1,1,1,1,1)/_/"
                  "0,2,(1,1,1,1,1),(1,1,1,1,1)\",\n");
    fprintf(file, "                                    "
                  "\"Flash/d=0,p=0/0,2,(1,1,1,1,1)/0,2,(1,1,1,1,1),(1,1,1,1,1)/0,2,(1,1,1,1,1)/_/"
                  "0,2,(1,1,1,1,1),(1,1,1,1,1)/6061626364656667/1234\",\n");
    fprintf(file,
            "                                    "
            "\"Flash/p=2,d=1/0,2,(8,20)/_/0,2,(2)/3,2,(2710),[0,7b,ffffffff,4,5]/"
            "0,2,(7b,7b),[3f4ccccd,42f60000,bf800000,40800000,40a00000]/1e,2e,(4);1,7,(5,1d)/0,2,(6,6)/[1,2,3]/7b/"
            "61626365/1/3f4ccccd/[3f4ccccd,3f666666,3f800000]/[[1,2,3,4],[7b,3,2,1]]\"]\n");
    fprintf(file, "            }\n");
    fprintf(file, "        ]\n");
    fprintf(file, "    }\n");
    fprintf(file, "}\n");
    fclose(file);
    return 1;
}

int32_t CreateCustomStubOpInfoConfig(const std::string& filePath, const std::string& soc)
{
    FILE* file = fopen(filePath.c_str(), "w");
    ASSERT_TRUE(file != nullptr);
    const std::string binPathPrefix = std::string("                \"binPath\": \"") + soc;
    fprintf(file, "{\n");

    // custom_op
    fprintf(file, "    \"custom_op\": {\n");
    fprintf(file, "        \"dynamicRankSupport\": true,\n");
    fprintf(file, "        \"simplifiedKeyMode\": 0,\n");
    fprintf(file, "        \"binaryList\": [\n");
    fprintf(file, "            {\n");
    fprintf(file, "                \"coreType\": 2,\n");
    fprintf(file, "                \"simplifiedKey\": [\"custom_op/d=0,p=0/1,30/1,30/1,30/1,30\",\n");
    fprintf(file, "                                    \"custom_op/d=0,p=0/0,2/0,2/0,2/0,2\"],\n");
    fprintf(file, (binPathPrefix + "/kernel_custom_op_dync_main.o\"\n").c_str());
    fprintf(file, "            },\n");
    fprintf(file, "            {\n");
    fprintf(file, "                \"simplifiedKey\": [\"custom_op/d=0,p=0\"],\n");
    fprintf(file, (binPathPrefix + "/kernel_custom_op_main_entry.o\"\n").c_str());
    fprintf(file, "            }\n");
    fprintf(file, "        ]\n");
    fprintf(file, "    },\n");

    // custom_op1
    fprintf(file, "    \"custom_op1\": {\n");
    fprintf(file, "        \"dynamicRankSupport\": true,\n");
    fprintf(file, "        \"simplifiedKeyMode\": 0,\n");
    fprintf(file, "        \"binaryList\": [\n");
    fprintf(file, "            {\n");
    fprintf(file, "                \"coreType\": 0,\n");
    fprintf(file, "                \"simplifiedKey\": [\"custom_op1/d=0,p=0/1,30/1,30/1,30/1,30\",\n");
    fprintf(file, "                                    \"custom_op1/d=0,p=0/0,2/0,2/0,2/0,2\"],\n");
    fprintf(file, (binPathPrefix + "/kernel_custom_op1_dync_main.o\"\n").c_str());
    ;
    fprintf(file, "            }\n");
    fprintf(file, "        ]\n");
    fprintf(file, "    },\n");

    // StubMixKeyOp（自定义包声明 mode=0，与内置包的 mode=2 同名）。此处的动态键索引先入 binTbl，
    // 供 customizedKey 未命中后回落命中，详见内置包侧注释。
    fprintf(file, "    \"StubMixKeyOp\": {\n");
    fprintf(file, "        \"dynamicRankSupport\": true,\n");
    fprintf(file, "        \"simplifiedKeyMode\": 0,\n");
    fprintf(file, "        \"binaryList\": [\n");
    fprintf(file, "            {\n");
    fprintf(file, "                \"coreType\": 0,\n");
    fprintf(file, "                \"simplifiedKey\": [\"StubMixKeyOp/d=0,p=0/0,2/0,2/0,2/0,2\"],\n");
    fprintf(file, (binPathPrefix + "/kernel_custom_op1_dync_main.o\"\n").c_str());
    fprintf(file, "            }\n");
    fprintf(file, "        ]\n");
    fprintf(file, "    }\n");
    fprintf(file, "}\n");
    fclose(file);
    return 1;
}

int32_t CreateUserDefineStubOpInfoConfig(const std::string& filePath, const std::string& soc)
{
    FILE* file = fopen(filePath.c_str(), "w");
    ASSERT_TRUE(file != nullptr);
    const std::string binPathPrefix = std::string("                \"binPath\": \"") + soc;

    // custom_op2
    fprintf(file, "{\n");
    fprintf(file, "    \"custom_op2\": {\n");
    fprintf(file, "        \"dynamicRankSupport\": true,\n");
    fprintf(file, "        \"simplifiedKeyMode\": 0,\n");
    fprintf(file, "        \"binaryList\": [\n");
    fprintf(file, "            {\n");
    fprintf(file, "                \"coreType\": 0,\n");
    fprintf(file, "                \"simplifiedKey\": [\"custom_op2/d=0,p=0/1,30/1,30/1,30/1,30\",\n");
    fprintf(file, "                                    \"custom_op2/d=0,p=0/0,2/0,2/0,2/0,2\"],\n");
    fprintf(file, (binPathPrefix + "/kernel_custom_op2_dync_main.o\"\n").c_str());
    fprintf(file, "            }\n");
    fprintf(file, "        ]\n");
    fprintf(file, "    }\n");
    fprintf(file, "}\n");
    fclose(file);
    return 1;
}

int32_t NnopbaseCreateStubBinJsonFor1971ForMixNormal(const std::string& filePath, const std::string& soc)
{
    FILE* file = fopen(filePath.c_str(), "w");
    ASSERT_TRUE(file != nullptr);
    const std::string binPathPrefix = std::string("                \"binPath\": \"") + soc;

    fprintf(file, "{\n");
    fprintf(file, "    \"binFileName\": \"1971_for_mix_normal\",\n");
    fprintf(file, "    \"binFileSuffix\": \".o\",\n");
    fprintf(file, "    \"coreType\": \"MIX\",\n");
    fprintf(file, "    \"debugOptions\": \"printf,timestamp\",\n");
    fprintf(file, "    \"debugBufSize\": 78643200,\n");
    fprintf(file, "    \"kernelList\":[\n");
    fprintf(file, "        {\n");
    fprintf(file, "            \"kernelName\": \"1971_for_mix_normal_1\",\n");
    fprintf(file, "            \"tilingKey\": 6,\n");
    fprintf(file, "            \"kernelType\": \"MIX_AIC\",\n");
    fprintf(file, "            \"crossCoreSync\": 1,\n");
    fprintf(file, "            \"taskRation\": \"1:0\"\n");
    fprintf(file, "        },\n");
    fprintf(file, "        {\n");
    fprintf(file, "            \"kernelName\": \"1971_for_mix_normal_1\",\n");
    fprintf(file, "            \"tilingKey\": 1,\n");
    fprintf(file, "            \"kernelType\": \"MIX_AIC\",\n");
    fprintf(file, "            \"crossCoreSync\": 1,\n");
    fprintf(file, "            \"taskRation\": \"0:1\"\n");
    fprintf(file, "        },\n");
    fprintf(file, "        {\n");
    fprintf(file, "            \"kernelName\": \"1971_for_mix_normal_1\",\n");
    fprintf(file, "            \"tilingKey\": 2,\n");
    fprintf(file, "            \"kernelType\": \"MIX_AIC\",\n");
    fprintf(file, "            \"crossCoreSync\": 0,\n");
    fprintf(file, "            \"taskRation\": \"1:0\"\n");
    fprintf(file, "        },\n");
    fprintf(file, "        {\n");
    fprintf(file, "            \"kernelName\": \"1971_for_mix_normal_1\",\n");
    fprintf(file, "            \"tilingKey\": 3,\n");
    fprintf(file, "            \"kernelType\": \"MIX_AIC\",\n");
    fprintf(file, "            \"crossCoreSync\": 0,\n");
    fprintf(file, "            \"taskRation\": \"0:1\"\n");
    fprintf(file, "        },\n");
    fprintf(file, "        {\n");
    fprintf(file, "            \"kernelName\": \"1971_for_mix_normal_1\",\n");
    fprintf(file, "            \"tilingKey\": 4,\n"); // tilingKey is 4 taskRation is 1:1
    fprintf(file, "            \"kernelType\": \"MIX_AIC\",\n");
    fprintf(file, "            \"crossCoreSync\": 1,\n");
    fprintf(file, "            \"taskRation\": \"1:1\"\n");
    fprintf(file, "        },\n");
    fprintf(file, "        {\n");
    fprintf(file, "            \"kernelName\": \"1971_for_mix_normal_1\",\n");
    fprintf(file, "            \"tilingKey\": 5,\n"); // tilingKey is 5 taskRation is 1:2
    fprintf(file, "            \"kernelType\": \"MIX_AIC\",\n");
    fprintf(file, "            \"crossCoreSync\": 1,\n");
    fprintf(file, "            \"taskRation\": \"1:2\"\n");
    fprintf(file, "        }\n");
    fprintf(file, "    ]\n");
    fprintf(file, "}\n");
    fclose(file);
    return 1;
}

int32_t NnopbaseCreateStubBinJsonFor1971ForMixUnNormal(const std::string& filePath,
                                                       const std::string& soc) // 缺少字段 crossCoreSync
{
    FILE* file = fopen(filePath.c_str(), "w");
    ASSERT_TRUE(file != nullptr);
    const std::string binPathPrefix = std::string("                \"binPath\": \"") + soc;

    fprintf(file, "{\n");
    fprintf(file, "    \"binFileName\": \"1971_for_mix_unnormal\",\n");
    fprintf(file, "    \"binFileSuffix\": \".o\",\n");
    fprintf(file, "    \"coreType\": \"MIX\",\n");
    fprintf(file, "    \"kernelList\":[\n");
    fprintf(file, "        {\n");
    fprintf(file, "            \"kernelName\": \"1971_for_mix_unnormal_1\",\n");
    fprintf(file, "            \"tilingKey\": 1,\n");
    fprintf(file, "            \"kernelType\": \"MIX_AIC\",\n");
    fprintf(file, "            \"taskRation\": \"1:2\"\n");
    fprintf(file, "        }\n");
    fprintf(file, "    ]\n");
    fprintf(file, "}\n");
    fclose(file);
    return 1;
}

int32_t NnopbaseCreateStubBinJsonFor1971ForMixUnNormal2(const std::string& filePath,
                                                        const std::string& soc) // 缺少字段 tilingKey
{
    FILE* file = fopen(filePath.c_str(), "w");
    ASSERT_TRUE(file != nullptr);
    const std::string binPathPrefix = std::string("                \"binPath\": \"") + soc;

    fprintf(file, "{\n");
    fprintf(file, "    \"binFileName\": \"1971_for_mix_unnormal2\",\n");
    fprintf(file, "    \"binFileSuffix\": \".o\",\n");
    fprintf(file, "    \"coreType\": \"MIX\",\n");
    fprintf(file, "    \"kernelList\":[\n");
    fprintf(file, "        {\n");
    fprintf(file, "            \"kernelName\": \"1971_for_mix_unormal2_1\",\n");
    fprintf(file, "            \"kernelType\": \"MIX_AIC\",\n");
    fprintf(file, "            \"taskRation\": \"1:2\"\n");
    fprintf(file, "        }\n");
    fprintf(file, "    ]\n");
    fprintf(file, "}\n");
    fclose(file);
    return 1;
}

int32_t NnopbaseCreateStubBinJsonFor310pForMixAicNormal(const std::string& filePath, const std::string& soc)
{
    FILE* file = fopen(filePath.c_str(), "w");
    ASSERT_TRUE(file != nullptr);
    const std::string binPathPrefix = std::string("                \"binPath\": \"") + soc;

    fprintf(file, "{\n");
    fprintf(file, "    \"binFileName\": \"310p_for_mix_normal\",\n");
    fprintf(file, "    \"binFileSuffix\": \".o\",\n");
    fprintf(file, "    \"coreType\": \"MIX_AICORE\",\n");
    fprintf(file, "    \"kernelList\":[\n");
    fprintf(file, "        {\n");
    fprintf(file, "            \"kernelName\": \"310p_for_mix_normal_1\",\n");
    fprintf(file, "            \"tilingKey\": 0,\n");
    fprintf(file, "            \"kernelType\": \"MIX_AICORE\",\n");
    fprintf(file, "            \"taskRation\": \"1:0\"\n");
    fprintf(file, "        },\n");
    fprintf(file, "        {\n");
    fprintf(file, "            \"kernelName\": \"310p_for_mix_normal_1\",\n");
    fprintf(file, "            \"tilingKey\": 1,\n"); // tilingKey is 1 taskRation is 1:1
    fprintf(file, "            \"kernelType\": \"MIX_AICORE\",\n");
    fprintf(file, "            \"taskRation\": \"1:1\"\n");
    fprintf(file, "        },\n");
    fprintf(file, "        {\n");
    fprintf(file, "            \"kernelName\": \"310p_for_mix_normal_1\",\n");
    fprintf(file, "            \"tilingKey\": 2,\n"); // tilingKey is 2 taskRation is 1:2
    fprintf(file, "            \"kernelType\": \"MIX_AICORE\",\n");
    fprintf(file, "            \"taskRation\": \"1:2\"\n");
    fprintf(file, "        }\n");
    fprintf(file, "    ]\n");
    fprintf(file, "}\n");
    fclose(file);
    return 1;
}

int32_t NnopbaseCreateStubBinJsonFor310pForMixAivNormal(const std::string& filePath, const std::string& soc)
{
    FILE* file = fopen(filePath.c_str(), "w");
    ASSERT_TRUE(file != nullptr);
    const std::string binPathPrefix = std::string("                \"binPath\": \"") + soc;

    fprintf(file, "{\n");
    fprintf(file, "    \"binFileName\": \"310p_for_mix_normal\",\n");
    fprintf(file, "    \"binFileSuffix\": \".o\",\n");
    fprintf(file, "    \"coreType\": \"MIX_AIV\",\n");
    fprintf(file, "    \"kernelList\":[\n");
    fprintf(file, "        {\n");
    fprintf(file, "            \"kernelName\": \"310p_for_mix_normal_1\",\n");
    fprintf(file, "            \"tilingKey\": 0,\n");
    fprintf(file, "            \"kernelType\": \"MIX_VECTOR_CORE\",\n");
    fprintf(file, "            \"taskRation\": \"1:0\"\n");
    fprintf(file, "        },\n");
    fprintf(file, "        {\n");
    fprintf(file, "            \"kernelName\": \"310p_for_mix_normal_1\",\n");
    fprintf(file, "            \"tilingKey\": 1,\n"); // tilingKey is 1 taskRation is 1:1
    fprintf(file, "            \"kernelType\": \"MIX_VECTOR_CORE\",\n");
    fprintf(file, "            \"taskRation\": \"1:1\"\n");
    fprintf(file, "        },\n");
    fprintf(file, "        {\n");
    fprintf(file, "            \"kernelName\": \"310p_for_mix_normal_1\",\n");
    fprintf(file, "            \"tilingKey\": 2,\n"); // tilingKey is 2 taskRation is 1:2
    fprintf(file, "            \"kernelType\": \"MIX_VECTOR_CORE\",\n");
    fprintf(file, "            \"taskRation\": \"1:2\"\n");
    fprintf(file, "        }\n");
    fprintf(file, "    ]\n");
    fprintf(file, "}\n");
    fclose(file);
    return 1;
}

int32_t NnopbaseCreateStubBinJsonForProfilingMixAic(const std::string& filePath, const std::string& soc)
{
    FILE* file = fopen(filePath.c_str(), "w");
    ASSERT_TRUE(file != nullptr);
    const std::string binPathPrefix = std::string("                \"binPath\": \"") + soc;
    fprintf(file, "{\n");
    fprintf(file, "    \"binFileName\": \"test_profiling_mix_aic\",\n");
    fprintf(file, "    \"binFileSuffix\": \".o\",\n");
    fprintf(file, "    \"coreType\": \"MIX\",\n");
    fprintf(file, "    \"taskRation\": \"tilingKey\",\n");
    fprintf(file, "    \"kernelList\":[\n");
    fprintf(file, "        {\n");
    fprintf(file, "            \"kernelName\": \"test_profiling_mix_aic\"\n");
    fprintf(file, "        }\n");
    fprintf(file, "    ]\n");
    fprintf(file, "}\n");
    fclose(file);
    return 1;
}

int32_t NnopbaseCreateStubBinJsonForProfilingMixAiv(const std::string& filePath, const std::string& soc)
{
    FILE* file = fopen(filePath.c_str(), "w");
    ASSERT_TRUE(file != nullptr);
    const std::string binPathPrefix = std::string("                \"binPath\": \"") + soc;
    fprintf(file, "{\n");
    fprintf(file, "    \"binFileName\": \"test_profiling_mix_aic\",\n");
    fprintf(file, "    \"binFileSuffix\": \".o\",\n");
    fprintf(file, "    \"coreType\": \"MIX\",\n");
    fprintf(file, "    \"taskRation\": \"0:1\",\n");
    fprintf(file, "    \"kernelList\":[\n");
    fprintf(file, "        {\n");
    fprintf(file, "            \"kernelName\": \"test_profiling_mix_aic\"\n");
    fprintf(file, "        }\n");
    fprintf(file, "    ]\n");
    fprintf(file, "}\n");
    fclose(file);
    return 1;
}

int32_t NnopbaseCreateStubBinOpInfoConfigForAdd(const std::string& filePath, const std::string& soc)
{
    FILE* file = fopen(filePath.c_str(), "w");
    ASSERT_TRUE(file != nullptr);
    const std::string binPathPrefix = std::string("                \"binPath\": \"") + soc;

    fprintf(file, "{\n");
    fprintf(file, "    \"binFileName\": \"add_tik2_entry_ascend910\",\n");
    fprintf(file, "    \"binFileSuffix\": \".o\",\n");
    fprintf(file, "    \"coreType\": \"MIX\"\n");
    fprintf(file, "}\n");
    fclose(file);
    return 1;
}

int32_t CreateDebugStubOpInfoConfig(const std::string& filePath, const std::string& soc)
{
    FILE* file = fopen(filePath.c_str(), "w");
    ASSERT_TRUE(file != nullptr);
    const std::string binPathPrefix = std::string("                \"binPath\": \"") + soc;
    fprintf(file, "{\n");

    // conv2d no .o
    fprintf(file, "    \"Conv2D\": {\n");
    fprintf(file, "        \"dynamicRankSupport\": true,\n");
    fprintf(file, "        \"simplifiedKeyMode\": 0,\n");
    fprintf(file, "        \"staticList\": [\n");
    fprintf(file, "            {\n");
    fprintf(file, "                \"binDesc\": {\n");
    fprintf(file, "                    \"blockDim\": 24,\n");
    fprintf(file, "                    \"kernelName\": \"Conv2D_1e04ee05ab491cc5ae9c3d5c9ee8950b\"\n");
    fprintf(file, "                },\n");
    fprintf(file, "                \"coreType\": 1,\n");
    fprintf(file, "                \"simplifiedKey\": "
                  "\"Conv2D/d=0,p=2/1,3,(20,4,60,60,10)/1,4,(24,4,10,10)/_/_/1,3,(20,4,60,60,10)/[1,1,1,1]/[1,1,1,1]/"
                  "[1,1,1,1]/1/5748434e/0\",\n");
    fprintf(file, (binPathPrefix + "/Conv2D_high_performance_0.o\"\n").c_str());
    fprintf(file, "            }\n");
    fprintf(file, "        ]\n");
    fprintf(file, "    },\n");

    // StaticDebug
    fprintf(file, "    \"StaticDebug\": {\n");
    fprintf(file, "        \"dynamicRankSupport\": true,\n");
    fprintf(file, "        \"simplifiedKeyMode\": 0,\n");
    fprintf(file, "        \"staticList\": [\n");
    fprintf(file, "            {\n");
    fprintf(file, "                \"binDesc\": {\n");
    fprintf(file, "                    \"blockDim\": 24,\n");
    fprintf(file, "                    \"kernelName\": \"StaticDebug\"\n");
    fprintf(file, "                },\n");
    fprintf(file, "                \"coreType\": 1,\n");
    fprintf(file,
            "                \"simplifiedKey\": "
            "\"StaticDebug/d=0,p=0/1,3,(20,4,60,60,10)/1,4,(24,4,10,10)/_/_/1,3,(20,4,60,60,10)/[1,1,1,1]/[1,1,1,1]/"
            "[1,1,1,1]/1/5748434e/0\",\n");
    fprintf(file, (binPathPrefix + "/StaticDebug_high_performance_0.o\"\n").c_str());
    fprintf(file, "            }\n");
    fprintf(file, "        ]\n");
    fprintf(file, "    },\n");

    // custom_op3
    fprintf(file, "    \"custom_op3\": {\n");
    fprintf(file, "        \"dynamicRankSupport\": true,\n");
    fprintf(file, "        \"simplifiedKeyMode\": 0,\n");
    fprintf(file, "        \"binaryList\": [\n");
    fprintf(file, "            {\n");
    fprintf(file, "                \"coreType\": 0,\n");
    fprintf(file, "                \"simplifiedKey\": [\"custom_op3/d=0,p=0/1,30/1,30/1,30/1,30\",\n");
    fprintf(file, "                                    \"custom_op3/d=0,p=0/0,2/0,2/0,2/0,2\"],\n");
    fprintf(file, (binPathPrefix + "/kernel_custom_op3_dync_main.o\"\n").c_str());
    ;
    fprintf(file, "            }\n");
    fprintf(file, "        ]\n");
    fprintf(file, "    }\n");
    fprintf(file, "}\n");
    fclose(file);
    return 1;
}

int32_t CreateBinFile(const std::string& filePath)
{
    FILE* file = fopen(filePath.c_str(), "w");
    ASSERT_TRUE(file != nullptr);
    fprintf(file, "000\n");
    fclose(file);
    return 1;
}

int32_t CreateStubTilingSo()
{
    // build-in
    std::string buildInPrefix = OPP_PATH + "/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/";
    ASSERT_SUCC(CreateBinFile(buildInPrefix + "aarch64/libopmaster_rt2.0.so"));
    ASSERT_SUCC(CreateBinFile(buildInPrefix + "x86_64/libopmaster_rt2.0.so"));

    std::string buildInNewPrefix = OPP_PATH + "/built-in/op_impl/ai_core/tbe/op_host/lib/linux/";
    ASSERT_SUCC(CreateBinFile(buildInNewPrefix + "aarch64/libophost_legacy.so"));
    ASSERT_SUCC(CreateBinFile(buildInNewPrefix + "aarch64/libophost_cv.so"));
    ASSERT_SUCC(CreateBinFile(buildInNewPrefix + "aarch64/libophost_fake.bin"));
    ASSERT_SUCC(CreateBinFile(buildInNewPrefix + "x86_64/libophost_legacy.so"));
    ASSERT_SUCC(CreateBinFile(buildInNewPrefix + "x86_64/libophost_cv.so"));
    ASSERT_SUCC(CreateBinFile(buildInNewPrefix + "x86_64/libophost_fake.bin"));
    // cust
    std::string custPrefix = OPP_PATH + "/vendors/cust/op_impl/ai_core/tbe/op_tiling/lib/linux/";
    ASSERT_SUCC(CreateBinFile(custPrefix + "aarch64/libcust_opmaster_rt2.0.so"));
    ASSERT_SUCC(CreateBinFile(custPrefix + "x86_64/libcust_opmaster_rt2.0.so"));

    // userDefine
    std::string userDefinePrefix = OPP_PATH + "/user_define/op_impl/ai_core/tbe/op_tiling/lib/linux/";
    ASSERT_SUCC(CreateBinFile(userDefinePrefix + "aarch64/libcust_opmaster_rt2.0.so"));
    ASSERT_SUCC(CreateBinFile(userDefinePrefix + "x86_64/libcust_opmaster_rt2.0.so"));
    return 1;
}

int32_t NnopbaseCreateStubBinJsonForSparseAttention(const std::string& filePath, const std::string& soc)
{
    FILE* file = fopen(filePath.c_str(), "w");
    ASSERT_TRUE(file != nullptr);

    fprintf(file, "{\n");
    fprintf(file, "    \"binFileName\": \"NativeSparseAttention_fatbin\",\n");
    fprintf(file, "    \"binFileSuffix\": \".o\",\n");
    fprintf(file, "    \"blockDim\": -1,\n");
    fprintf(file, "    \"compileInfo\": {},\n");
    fprintf(file, "    \"coreType\": \"MIX\",\n");
    fprintf(file, "    \"dynamicParamMode\": \"folded_with_desc\",\n");
    fprintf(file, "    \"kernelList\": [\n");
    fprintf(file, "        {\n");
    fprintf(file, "            \"blockDim\": 24,\n");
    fprintf(file, "            \"configKey\": 0,\n");
    fprintf(file, "            \"kernelName\": \"ast_main_0\",\n");
    fprintf(file, "            \"workspaceSize\": 6291456\n");
    fprintf(file, "        },\n");
    fprintf(file, "        {\n");
    fprintf(file, "            \"blockDim\": 24,\n");
    fprintf(file, "            \"configKey\": 1,\n");
    fprintf(file, "            \"kernelName\": \"ast_main_0\",\n");
    fprintf(file, "            \"workspaceSize\": 6291456\n");
    fprintf(file, "        },\n");
    fprintf(file, "        {\n");
    fprintf(file, "            \"blockDim\": 24,\n");
    fprintf(file, "            \"configKey\": 2,\n");
    fprintf(file, "            \"kernelName\": \"ast_main_0\",\n");
    fprintf(file, "            \"workspaceSize\": 6291456\n");
    fprintf(file, "        }\n");
    fprintf(file, "    ],\n");
    fprintf(file, "    \"kernelName\": \"ast_main_0\",\n");
    fprintf(file, "    \"magic\": \"RT_DEV_BINARY_MAGIC_ELF\",\n");
    fprintf(file, "    \"workspace\": {\n");
    fprintf(file, "        \"num\": 1,\n");
    fprintf(file, "        \"size\": [\n");
    fprintf(file, "            6291456\n");
    fprintf(file, "        ],\n");
    fprintf(file, "        \"type\": [\n");
    fprintf(file, "            0\n");
    fprintf(file, "        ]\n");
    fprintf(file, "    },\n");
    fprintf(file, "    \"supportInfo\": {\n");
    fprintf(file, "        \"int64Mode\": false,\n");
    fprintf(file, "        \"simplifiedKeyMode\": 0,\n");
    fprintf(file, "        \"simplifiedKey\": [\n");
    fprintf(file, "            \"NativeSparseAttention/d=0,p=0/0,2/0,2\",\n");
    fprintf(file, "            \"NativeSparseAttention/d=1,p=0/0,2/0,2\"\n");
    fprintf(file, "        ],\n");
    fprintf(file, "        \"dynamicParamMode\": \"folded_with_desc\",\n");
    fprintf(file, "        \"staticKey\": \"795b8d601a472a550b6babd80423cd56aadaf02b495186c06f5022c39928ebba\",\n");
    fprintf(file, "        \"inputs\": [\n");
    fprintf(file, "            [\n");
    fprintf(file, "                {\n");
    fprintf(file, "                    \"name\": \"x\",\n");
    fprintf(file, "                    \"index\": 0,\n");
    fprintf(file, "                    \"dtype\": \"float32\",\n");
    fprintf(file, "                    \"format\": \"ND\",\n");
    fprintf(file, "                    \"paramType\": \"dynamic\",\n");
    fprintf(file, "                    \"shape\": [\n");
    fprintf(file, "                        -2\n");
    fprintf(file, "                    ],\n");
    fprintf(file, "                    \"format_match_mode\": \"FormatAgnostic\"\n");
    fprintf(file, "                }\n");
    fprintf(file, "            ]\n");
    fprintf(file, "        ],\n");
    fprintf(file, "        \"outputs\": [\n");
    fprintf(file, "            [\n");
    fprintf(file, "                {\n");
    fprintf(file, "                    \"name\": \"y\",\n");
    fprintf(file, "                    \"index\": 0,\n");
    fprintf(file, "                    \"dtype\": \"float32\",\n");
    fprintf(file, "                    \"format\": \"ND\",\n");
    fprintf(file, "                    \"paramType\": \"dynamic\",\n");
    fprintf(file, "                    \"shape\": [\n");
    fprintf(file, "                        -2\n");
    fprintf(file, "                    ],\n");
    fprintf(file, "                    \"format_match_mode\": \"FormatAgnostic\"\n");
    fprintf(file, "                }\n");
    fprintf(file, "            ]\n");
    fprintf(file, "        ],\n");
    fprintf(file, "        \"opMode\": \"dynamic\",\n");
    fprintf(file, "        \"deterministic\": \"ignore\"\n");
    fprintf(file, "    }\n");
    fprintf(file, "}\n");

    fclose(file);
    return 1;
}

int32_t NnopbaseCreateStubBinJsonFor1971(const std::string& filePath, const std::string& soc)
{
    FILE* file = fopen(filePath.c_str(), "w");
    ASSERT_TRUE(file != nullptr);
    const std::string binPathPrefix = std::string("                \"binPath\": \"") + soc;

    fprintf(file, "{\n");
    fprintf(file, "    \"binFileName\": \"kernel_bninference_d_dync_main\",\n");
    fprintf(file, "    \"binFileSuffix\": \".o\",\n");
    fprintf(file, "    \"coreType\": \"MIX\",\n");
    fprintf(file, "    \"opParaSize\": 60,\n");
    fprintf(file, "    \"supportInfo\": {\n");
    fprintf(file, "        \"op_debug_config\": \"oom,ccec\"\n");
    fprintf(file, "    },\n");
    fprintf(file, "    \"parameters\": [\n");
    fprintf(file, "        {\n");
    fprintf(file, "            \"init_value\": 0.0\n");
    fprintf(file, "        }\n");
    fprintf(file, "    ],\n");
    fprintf(file, "    \"kernelList\":[\n");
    fprintf(file, "        {\n");
    fprintf(file, "            \"kernelName\": \"kernel_bninference_d_dync_main\"\n");
    fprintf(file, "        }\n");
    fprintf(file, "    ]\n");
    fprintf(file, "}\n");
    fclose(file);
    return 1;
}

int32_t NnopbaseCreateStubBinJsonFor1980(const std::string& filePath, const std::string& soc)
{
    FILE* file = fopen(filePath.c_str(), "w");
    ASSERT_TRUE(file != nullptr);
    const std::string binPathPrefix = std::string("                \"binPath\": \"") + soc;

    fprintf(file, "{\n");
    fprintf(file, "    \"binFileName\": \"kernel_bninference_d_dync_main\",\n");
    fprintf(file, "    \"binFileSuffix\": \".o\",\n");
    fprintf(file, "    \"coreType\": \"MIX\",\n");
    fprintf(file, "    \"opParaSize\": 6900,\n");
    fprintf(file, "    \"supportInfo\": {\n");
    fprintf(file, "        \"op_debug_config\": \"oom,ccec\"\n");
    fprintf(file, "    },\n");
    fprintf(file, "    \"kernelList\":[\n");
    fprintf(file, "        {\n");
    fprintf(file, "            \"kernelName\": \"kernel_bninference_d_dync_main\"\n");
    fprintf(file, "        }\n");
    fprintf(file, "    ]\n");
    fprintf(file, "}\n");
    fclose(file);
    return 1;
}

int32_t NnopbaseCreateStubBinJsonForAutomicClean(const std::string& filePath, const std::string& soc)
{
    FILE* file = fopen(filePath.c_str(), "w");
    ASSERT_TRUE(file != nullptr);
    const std::string binPathPrefix = std::string("                \"binPath\": \"") + soc;

    fprintf(file, "{\n");
    fprintf(file, "    \"binFileName\": \"kernel_output_automic_clean\",\n");
    fprintf(file, "    \"binFileSuffix\": \".o\",\n");
    fprintf(file, "    \"coreType\": \"MIX\",\n");
    fprintf(file, "    \"opParaSize\": 60,\n");
    fprintf(file, "    \"parameters\": [\n");
    fprintf(file, "        null,\n");
    fprintf(file, "        null,\n");
    fprintf(file, "        {\n");
    fprintf(file, "            \"dtype\": \"float16\",\n");
    fprintf(file, "            \"init_value\": 0.0\n");
    fprintf(file, "        },\n");
    fprintf(file, "        {\n");
    fprintf(file, "            \"dtype\": \"float32\",\n");
    fprintf(file, "            \"init_value\": 0.0\n");
    fprintf(file, "        },\n");
    fprintf(file, "        {\n");
    fprintf(file, "            \"dtype\": \"int64\",\n");
    fprintf(file, "            \"init_value\": 0\n");
    fprintf(file, "        },\n");
    fprintf(file, "        {\n");
    fprintf(file, "            \"dtype\": \"uint64\",\n");
    fprintf(file, "            \"init_value\": 0\n");
    fprintf(file, "        },\n");
    fprintf(file, "        {\n");
    fprintf(file, "            \"dtype\": \"int32\",\n");
    fprintf(file, "            \"init_value\": 0\n");
    fprintf(file, "        },\n");
    fprintf(file, "        {\n");
    fprintf(file, "            \"dtype\": \"uint32\",\n");
    fprintf(file, "            \"init_value\": 0\n");
    fprintf(file, "        },\n");
    fprintf(file, "        {\n");
    fprintf(file, "            \"dtype\": \"int16\",\n");
    fprintf(file, "            \"init_value\": 0\n");
    fprintf(file, "        },\n");
    fprintf(file, "        {\n");
    fprintf(file, "            \"dtype\": \"uint16\",\n");
    fprintf(file, "            \"init_value\": 0\n");
    fprintf(file, "        },\n");
    fprintf(file, "        {\n");
    fprintf(file, "            \"dtype\": \"int8\",\n");
    fprintf(file, "            \"init_value\": 0\n");
    fprintf(file, "        },\n");
    fprintf(file, "        {\n");
    fprintf(file, "            \"dtype\": \"uint8\",\n");
    fprintf(file, "            \"init_value\": 0\n");
    fprintf(file, "        },\n");
    fprintf(file, "        {\n");
    fprintf(file, "            \"dtype\": \"uint1\",\n");
    fprintf(file, "            \"init_value\": 0\n");
    fprintf(file, "        },\n");
    fprintf(file, "        null\n");
    fprintf(file, "    ],\n");
    fprintf(file, "    \"kernelList\":[\n");
    fprintf(file, "        {\n");
    fprintf(file, "            \"kernelName\": \"kernel_output_automic_clean\"\n");
    fprintf(file, "        }\n");
    fprintf(file, "    ]\n");
    fprintf(file, "}\n");
    fclose(file);
    return 1;
}

int32_t NnopbaseCreateStubBinJsonForTestDavidCustom(const std::string& filePath, const std::string& soc)
{
    FILE* file = fopen(filePath.c_str(), "w");
    ASSERT_TRUE(file != nullptr);
    const std::string binPathPrefix = std::string("                \"binPath\": \"") + soc;

    fprintf(file, "{\n");
    fprintf(file, "    \"binFileName\": \"TestDavidCustom\",\n");
    fprintf(file, "    \"binFileSuffix\": \".o\",\n");
    fprintf(file, "    \"coreType\": \"MIX\",\n");
    fprintf(file, "    \"opParaSize\": 60,\n");
    fprintf(file, "    \"parameters\": [\n");
    fprintf(file, "        null,\n");
    fprintf(file, "        null,\n");
    fprintf(file, "        {\n");
    fprintf(file, "            \"dtype\": \"float16\",\n");
    fprintf(file, "            \"init_value\": 0.0\n");
    fprintf(file, "        },\n");
    fprintf(file, "        {\n");
    fprintf(file, "            \"dtype\": \"float32\",\n");
    fprintf(file, "            \"init_value\": 1.0\n");
    fprintf(file, "        },\n");
    fprintf(file, "        {\n");
    fprintf(file, "            \"dtype\": \"float32\",\n");
    fprintf(file, "            \"init_value\": 1.0\n");
    fprintf(file, "        },\n");
    fprintf(file, "        null\n");
    fprintf(file, "    ],\n");
    fprintf(file, "    \"kernelList\":[\n");
    fprintf(file, "        {\n");
    fprintf(file, "            \"kernelName\": \"TestDavidCustom\"\n");
    fprintf(file, "        }\n");
    fprintf(file, "    ]\n");
    fprintf(file, "}\n");
    fclose(file);
    return 1;
}

int32_t NnopbaseCreateStubBinJsonForMemSetV2(const std::string& filePath, const std::string& soc)
{
    FILE* file = fopen(filePath.c_str(), "w");
    ASSERT_TRUE(file != nullptr);
    const std::string binPathPrefix = std::string("                \"binPath\": \"") + soc;

    fprintf(file, "{\n");
    fprintf(file, "    \"binFileName\": \"MemSetV2\",\n");
    fprintf(file, "    \"binFileSuffix\": \".o\",\n");
    fprintf(file, "    \"blockDim\": -1,\n");
    fprintf(file, "    \"coreType\": \"VectorCore\",\n");
    fprintf(file, "    \"core_type\": \"AIV\",\n");
    fprintf(file, "    \"intercoreSync\": 0,\n");
    fprintf(file, "    \"kernelName\": \"MemSetV2\",\n");
    fprintf(file, "    \"magic\": \"RT_DEV_BINARY_MAGIC_ELF_AIVEC\",\n");
    fprintf(file, "    \"memoryStamping\": [],\n");
    fprintf(file, "    \"opParaSize\": 11304,\n");
    fprintf(file, "    \"parameters\": [\n");
    fprintf(file, "        null,\n");
    fprintf(file, "        null,\n");
    fprintf(file, "        null\n");
    fprintf(file, "    ],\n");
    fprintf(file, "    \"workspace\": {\n");
    fprintf(file, "        \"num\": 1,\n");
    fprintf(file, "        \"size\": [\n");
    fprintf(file, "            -1\n");
    fprintf(file, "        ],\n");
    fprintf(file, "        \"type\": [\n");
    fprintf(file, "            0\n");
    fprintf(file, "        ]\n");
    fprintf(file, "    },\n");
    fprintf(file, "    \"kernelList\":[\n");
    fprintf(file, "        {\n");
    fprintf(file, "            \"kernelName\": \"TestDavidCustom\"\n");
    fprintf(file, "        }\n");
    fprintf(file, "    ],\n");
    fprintf(file, "    \"supportSuperKernel\": 1,\n");
    fprintf(file, "    \"compileInfo\": {}\n");
    fprintf(file, "}\n");
    fclose(file);
    return 1;
}

int32_t CreateStubKernelFiles()
{
    // build-in
    std::string buildInPrefix = OPP_PATH + "/built-in/op_impl/ai_core/tbe/kernel/";
    ASSERT_SUCC(CreateBinFile(buildInPrefix + "ascend910b/kernel_bninference_d_dync_main.o"));
    ASSERT_SUCC(CreateBinFile(buildInPrefix + "ascend910b/kernel_NonFiniteCheck.o"));
    ASSERT_SUCC(CreateBinFile(buildInPrefix + "ascend910/kernel_NonFiniteCheck.o"));
    ASSERT_SUCC(CreateBinFile(buildInPrefix + "ascend910b/add_tik2_entry_ascend910.o"));
    ASSERT_SUCC(CreateBinFile(buildInPrefix + "ascend910/kernel_bninference_d_dync_main.o"));
    ASSERT_SUCC(CreateBinFile(buildInPrefix + "ascend910/add_tik2_entry_ascend910.o"));

    ASSERT_SUCC(CreateBinFile(buildInPrefix + "ascend910b/kernel_bninference_d_dync_main.json"));
    ASSERT_SUCC(CreateBinFile(buildInPrefix + "ascend910b/kernel_NonFiniteCheck.json"));
    ASSERT_SUCC(CreateBinFile(buildInPrefix + "ascend910/kernel_NonFiniteCheck.json"));
    ASSERT_SUCC(CreateBinFile(buildInPrefix + "ascend910b/add_tik2_entry_ascend910.json"));
    ASSERT_SUCC(CreateBinFile(buildInPrefix + "ascend910/kernel_bninference_d_dync_main.json"));
    ASSERT_SUCC(CreateBinFile(buildInPrefix + "ascend910/add_tik2_entry_ascend910.json"));

    ASSERT_SUCC(
        NnopbaseCreateStubOpInfoConfig(buildInPrefix + "config/ascend910b/binary_info_config.json", "ascend910b"));
    ASSERT_SUCC(
        NnopbaseCreateStubOpInfoConfig(buildInPrefix + "config/ascend910/binary_info_config.json", "ascend910"));
    ASSERT_SUCC(
        NnopbaseCreateStubOpInfoConfig(buildInPrefix + "config/ascend950/binary_info_config.json", "ascend950"));
    ASSERT_SUCC(NnopbaseCreateStubBinJsonFor1971(buildInPrefix + "ascend910b/kernel_bninference_d_dync_main.json",
                                                 "ascend910b"));
    ASSERT_SUCC(
        NnopbaseCreateStubBinJsonFor1980(buildInPrefix + "ascend910/kernel_bninference_d_dync_main.json", "ascend910"));

    ASSERT_SUCC(CreateBinFile(buildInPrefix + "ascend910b/NativeSparseAttention_fatbin.o"));
    ASSERT_SUCC(CreateBinFile(buildInPrefix + "ascend910b/NativeSparseAttention_fatbin.json"));
    ASSERT_SUCC(NnopbaseCreateStubBinJsonForSparseAttention(
        buildInPrefix + "ascend910b/NativeSparseAttention_fatbin.json", "ascend910b"));

    // AutomicClean 1980
    ASSERT_SUCC(CreateBinFile(buildInPrefix + "ascend910/kernel_output_automic_clean.o"));
    ASSERT_SUCC(CreateBinFile(buildInPrefix + "ascend910/kernel_output_automic_clean.json"));
    ASSERT_SUCC(NnopbaseCreateStubBinJsonForAutomicClean(buildInPrefix + "ascend910/kernel_output_automic_clean.json",
                                                         "ascend910"));

    // AutomicClean 1971
    ASSERT_SUCC(CreateBinFile(buildInPrefix + "ascend910b/kernel_output_automic_clean.o"));
    ASSERT_SUCC(CreateBinFile(buildInPrefix + "ascend910b/kernel_output_automic_clean.json"));
    ASSERT_SUCC(NnopbaseCreateStubBinJsonForAutomicClean(buildInPrefix + "ascend910b/kernel_output_automic_clean.json",
                                                         "ascend910b"));

    // MemSetV2 david only
    ASSERT_SUCC(CreateBinFile(buildInPrefix + "ascend950/MemSetV2.o"));
    ASSERT_SUCC(CreateBinFile(buildInPrefix + "ascend950/MemSetV2.json"));
    ASSERT_SUCC(NnopbaseCreateStubBinJsonForMemSetV2(buildInPrefix + "ascend950/MemSetV2.json", "ascend950"));

    // TestDavidCustom david only
    ASSERT_SUCC(CreateBinFile(buildInPrefix + "ascend950/TestDavidCustom.o"));
    ASSERT_SUCC(CreateBinFile(buildInPrefix + "ascend950/TestDavidCustom.json"));
    ASSERT_SUCC(
        NnopbaseCreateStubBinJsonForTestDavidCustom(buildInPrefix + "ascend950/TestDavidCustom.json", "ascend950"));

    // test_register
    ASSERT_SUCC(CreateBinFile(buildInPrefix + "ascend910/test_registe_ascend910.o"));
    ASSERT_SUCC(CreateBinFile(buildInPrefix + "ascend910/test_registe_ascend910.json"));

    // 1971_for_mix_normal 多个.o创建同一个json文件
    ASSERT_SUCC(CreateBinFile(buildInPrefix + "ascend910/1971_for_mix_normal.o"));
    ASSERT_SUCC(CreateBinFile(buildInPrefix + "ascend910/1971_for_mix_normal.json"));
    ASSERT_SUCC(NnopbaseCreateStubBinJsonFor1971ForMixNormal(buildInPrefix + "ascend910/1971_for_mix_normal.json",
                                                             "ascend910"));

    // 1971_for_mix_normal 多个.o创建同一个json文件
    ASSERT_SUCC(CreateBinFile(buildInPrefix + "ascend910b/1971_for_mix_normal.o"));
    ASSERT_SUCC(CreateBinFile(buildInPrefix + "ascend910b/1971_for_mix_normal.json"));
    ASSERT_SUCC(NnopbaseCreateStubBinJsonFor1971ForMixNormal(buildInPrefix + "ascend910b/1971_for_mix_normal.json",
                                                             "ascend910b"));

    // 1971_for_mix_unnormal 多个.o创建同一个json文件
    ASSERT_SUCC(CreateBinFile(buildInPrefix + "ascend910/1971_for_mix_unnormal.o"));
    ASSERT_SUCC(CreateBinFile(buildInPrefix + "ascend910/1971_for_mix_unnormal.json"));
    ASSERT_SUCC(NnopbaseCreateStubBinJsonFor1971ForMixUnNormal(buildInPrefix + "ascend910/1971_for_mix_unnormal.json",
                                                               "ascend910"));

    // 1971_for_mix_unnormal 多个.o创建同一个json文件
    ASSERT_SUCC(CreateBinFile(buildInPrefix + "ascend910b/1971_for_mix_unnormal.o"));
    ASSERT_SUCC(CreateBinFile(buildInPrefix + "ascend910b/1971_for_mix_unnormal.json"));
    ASSERT_SUCC(NnopbaseCreateStubBinJsonFor1971ForMixUnNormal(buildInPrefix + "ascend910b/1971_for_mix_unnormal.json",
                                                               "ascend910b"));

    // 1971_for_mix_unnormal2 多个.o创建同一个json文件
    ASSERT_SUCC(CreateBinFile(buildInPrefix + "ascend910/1971_for_mix_unnormal2.o"));
    ASSERT_SUCC(CreateBinFile(buildInPrefix + "ascend910/1971_for_mix_unnormal2.json"));
    ASSERT_SUCC(NnopbaseCreateStubBinJsonFor1971ForMixUnNormal2(buildInPrefix + "ascend910/1971_for_mix_unnormal2.json",
                                                                "ascend910"));

    // 310p_for_mix_aic_normal 多个.o创建同一个json文件
    ASSERT_SUCC(CreateBinFile(buildInPrefix + "ascend910/310p_for_mix_aic_normal.o"));
    ASSERT_SUCC(CreateBinFile(buildInPrefix + "ascend910/310p_for_mix_aic_normal.json"));
    ASSERT_SUCC(NnopbaseCreateStubBinJsonFor310pForMixAicNormal(
        buildInPrefix + "ascend910/310p_for_mix_aic_normal.json", "ascend910"));

    // 310p_for_mix_aic_normal 多个.o创建同一个json文件
    ASSERT_SUCC(CreateBinFile(buildInPrefix + "ascend910b/310p_for_mix_aic_normal.o"));
    ASSERT_SUCC(CreateBinFile(buildInPrefix + "ascend910b/310p_for_mix_aic_normal.json"));
    ASSERT_SUCC(NnopbaseCreateStubBinJsonFor310pForMixAicNormal(
        buildInPrefix + "ascend910b/310p_for_mix_aic_normal.json", "ascend910b"));

    // 310p_for_mix_aiv_normal 多个.o创建同一个json文件
    ASSERT_SUCC(CreateBinFile(buildInPrefix + "ascend910/310p_for_mix_aiv_normal.o"));
    ASSERT_SUCC(CreateBinFile(buildInPrefix + "ascend910/310p_for_mix_aiv_normal.json"));
    ASSERT_SUCC(NnopbaseCreateStubBinJsonFor310pForMixAivNormal(
        buildInPrefix + "ascend910/310p_for_mix_aiv_normal.json", "ascend910"));

    // 310p_for_mix_aiv_normal 多个.o创建同一个json文件
    ASSERT_SUCC(CreateBinFile(buildInPrefix + "ascend910b/310p_for_mix_aiv_normal.o"));
    ASSERT_SUCC(CreateBinFile(buildInPrefix + "ascend910b/310p_for_mix_aiv_normal.json"));
    ASSERT_SUCC(NnopbaseCreateStubBinJsonFor310pForMixAivNormal(
        buildInPrefix + "ascend910b/310p_for_mix_aiv_normal.json", "ascend910b"));

    // test_profiling_mix_aic
    ASSERT_SUCC(CreateBinFile(buildInPrefix + "ascend910/test_profiling_mix_aic.o"));
    ASSERT_SUCC(CreateBinFile(buildInPrefix + "ascend910/test_profiling_mix_aic.json"));
    ASSERT_SUCC(NnopbaseCreateStubBinJsonForProfilingMixAic(buildInPrefix + "ascend910/test_profiling_mix_aic.json",
                                                            "ascend910"));

    // test_profiling_mix_aic
    ASSERT_SUCC(CreateBinFile(buildInPrefix + "ascend910b/test_profiling_mix_aic.o"));
    ASSERT_SUCC(CreateBinFile(buildInPrefix + "ascend910b/test_profiling_mix_aic.json"));
    ASSERT_SUCC(NnopbaseCreateStubBinJsonForProfilingMixAic(buildInPrefix + "ascend910b/test_profiling_mix_aic.json",
                                                            "ascend910b"));

    // test_profiling_mix_aiv
    ASSERT_SUCC(CreateBinFile(buildInPrefix + "ascend910/test_profiling_mix_aiv.o"));
    ASSERT_SUCC(CreateBinFile(buildInPrefix + "ascend910/test_profiling_mix_aiv.json"));
    ASSERT_SUCC(NnopbaseCreateStubBinJsonForProfilingMixAiv(buildInPrefix + "ascend910/test_profiling_mix_aiv.json",
                                                            "ascend910"));

    // test_profiling_mix_aiv
    ASSERT_SUCC(CreateBinFile(buildInPrefix + "ascend910b/test_profiling_mix_aiv.o"));
    ASSERT_SUCC(CreateBinFile(buildInPrefix + "ascend910b/test_profiling_mix_aiv.json"));
    ASSERT_SUCC(NnopbaseCreateStubBinJsonForProfilingMixAiv(buildInPrefix + "ascend910b/test_profiling_mix_aiv.json",
                                                            "ascend910b"));

    // cust
    std::string custPrefix = OPP_PATH + "/vendors/cust/op_impl/ai_core/tbe/kernel/";
    ASSERT_SUCC(CreateBinFile(custPrefix + "ascend910b/kernel_custom_op_dync_main.o"));
    ASSERT_SUCC(CreateBinFile(custPrefix + "ascend910b/kernel_custom_op_main_entry.o"));
    ASSERT_SUCC(CreateBinFile(custPrefix + "ascend910b/kernel_custom_op1_dync_main.o"));
    ASSERT_SUCC(CreateBinFile(custPrefix + "ascend910b/kernel_custom_op1_main_entry.o"));
    ASSERT_SUCC(CreateBinFile(custPrefix + "ascend910/kernel_custom_op_dync_main.o"));
    ASSERT_SUCC(CreateBinFile(custPrefix + "ascend910/kernel_custom_op_main_entry.o"));
    ASSERT_SUCC(CreateBinFile(custPrefix + "ascend910/kernel_custom_op1_dync_main.o"));
    ASSERT_SUCC(CreateBinFile(custPrefix + "ascend910/kernel_custom_op1_main_entry.o"));

    ASSERT_SUCC(CreateBinFile(custPrefix + "ascend910b/kernel_custom_op_dync_main.json"));
    ASSERT_SUCC(CreateBinFile(custPrefix + "ascend910b/kernel_custom_op_main_entry.json"));
    ASSERT_SUCC(CreateBinFile(custPrefix + "ascend910b/kernel_custom_op1_dync_main.json"));
    ASSERT_SUCC(CreateBinFile(custPrefix + "ascend910b/kernel_custom_op1_main_entry.json"));
    ASSERT_SUCC(CreateBinFile(custPrefix + "ascend910/kernel_custom_op_dync_main.json"));
    ASSERT_SUCC(CreateBinFile(custPrefix + "ascend910/kernel_custom_op_main_entry.json"));
    ASSERT_SUCC(CreateBinFile(custPrefix + "ascend910/kernel_custom_op1_dync_main.json"));
    ASSERT_SUCC(CreateBinFile(custPrefix + "ascend910/kernel_custom_op1_main_entry.json"));

    ASSERT_SUCC(CreateCustomStubOpInfoConfig(custPrefix + "config/ascend910b/binary_info_config.json", "ascend910b"));
    ASSERT_SUCC(CreateCustomStubOpInfoConfig(custPrefix + "config/ascend910/binary_info_config.json", "ascend910"));
    ASSERT_SUCC(CreateCustomStubOpInfoConfig(custPrefix + "config/ascend950/binary_info_config.json", "ascend950"));

    // userDefine
    std::string userDefinePrefix = OPP_PATH + "/user_define/op_impl/ai_core/tbe/kernel/";
    ASSERT_SUCC(CreateBinFile(userDefinePrefix + "ascend910b/kernel_custom_op2_dync_main.o"));
    ASSERT_SUCC(CreateBinFile(userDefinePrefix + "ascend910b/kernel_custom_op2_main_entry.o"));
    ASSERT_SUCC(CreateBinFile(userDefinePrefix + "ascend910/kernel_custom_op2_dync_main.o"));
    ASSERT_SUCC(CreateBinFile(userDefinePrefix + "ascend910/kernel_custom_op2_main_entry.o"));

    ASSERT_SUCC(CreateBinFile(userDefinePrefix + "ascend910b/kernel_custom_op2_dync_main.json"));
    ASSERT_SUCC(CreateBinFile(userDefinePrefix + "ascend910b/kernel_custom_op2_main_entry.json"));
    ASSERT_SUCC(CreateBinFile(userDefinePrefix + "ascend910/kernel_custom_op2_dync_main.json"));
    ASSERT_SUCC(CreateBinFile(userDefinePrefix + "ascend910/kernel_custom_op2_main_entry.json"));

    ASSERT_SUCC(
        CreateUserDefineStubOpInfoConfig(userDefinePrefix + "config/ascend910b/binary_info_config.json", "ascend910b"));
    ASSERT_SUCC(
        CreateUserDefineStubOpInfoConfig(userDefinePrefix + "config/ascend910/binary_info_config.json", "ascend910"));

    // static
    std::string staticPrefix = OPP_PATH + "/static_kernel/ai_core/";
    ASSERT_SUCC(CreateBinFile(staticPrefix + "static_kernel_202307261051/Flash/flash.o"));
    ASSERT_SUCC(CreateBinFile(staticPrefix + "static_kernel_202307261051/Flash/flash.json"));
    ASSERT_SUCC(CreateStaticStubOpInfoConfig(staticPrefix + "config/ascend910b/binary_info_config.json"));
    ASSERT_SUCC(CreateStaticStubOpInfoConfig(staticPrefix + "config/ascend910/binary_info_config.json"));

    // debug
    std::string debugPrefix = OPP_PATH + "/debug_kernel/";
    ASSERT_SUCC(CreateBinFile(debugPrefix + "ascend910b/StaticDebug_high_performance_0.o"));
    ASSERT_SUCC(CreateBinFile(debugPrefix + "ascend910b/kernel_custom_op3_dync_main.o"));
    ASSERT_SUCC(CreateBinFile(debugPrefix + "ascend910/StaticDebug_high_performance_0.o"));
    ASSERT_SUCC(CreateBinFile(debugPrefix + "ascend910/kernel_custom_op3_dync_main.o"));

    ASSERT_SUCC(CreateBinFile(debugPrefix + "ascend910b/StaticDebug_high_performance_0.json"));
    ASSERT_SUCC(CreateBinFile(debugPrefix + "ascend910b/kernel_custom_op3_dync_main.json"));
    ASSERT_SUCC(CreateBinFile(debugPrefix + "ascend910/StaticDebug_high_performance_0.json"));
    ASSERT_SUCC(CreateBinFile(debugPrefix + "ascend910/kernel_custom_op3_dync_main.json"));

    ASSERT_SUCC(CreateDebugStubOpInfoConfig(debugPrefix + "config/ascend910b/binary_info_config.json", "ascend910b"));
    ASSERT_SUCC(CreateDebugStubOpInfoConfig(debugPrefix + "config/ascend910/binary_info_config.json", "ascend910"));
    ASSERT_SUCC(CreateDebugStubOpInfoConfig(debugPrefix + "config/ascend950/binary_info_config.json", "ascend950"));
    return 1;
}

} // namespace

int32_t NnopbaseSetStaticBinOp()
{
    const std::string opType = "AddCustom";
    nlohmann::json binDesc = nlohmann::json::parse(R"(
        {
            "binList": [
                {
                "simplifiedKey": [
                    "AddCustom/d=0,p=0/3,2/3,2/3,2",
                    "AddCustom/d=1,p=0/3,2/3,2/3,2"
                ],
                "binInfo": {
                    "jsonFilePath": "ascend910b/add_custom/AddCustom_402e355eb717124771cfc7dbebfe946c.json"
                }
                },
                {
                "simplifiedKey": [
                    "AddCustom/d=0,p=0/1,2/1,2/1,2",
                    "AddCustom/d=1,p=0/1,2/1,2/1,2"
                ],
                "binInfo": {
                    "jsonFilePath": "ascend910b/add_custom/AddCustom_1e04ee05ab491cc5ae9c3d5c9ee8950b.json"
                }
                },
                {
                "simplifiedKey": [
                    "AddCustom/d=0,p=0/0,2/0,2/0,2",
                    "AddCustom/d=1,p=0/0,2/0,2/0,2"
                ],
                "binInfo": {
                    "jsonFilePath": "ascend910b/add_custom/AddCustom_ccd748392d99d04b8205210970fde2b9.json"
                }
                },
                {
                "simplifiedKey": [
                    "AddCustom/d=0,p=0/4,2/4,2/4,2",
                    "AddCustom/d=1,p=0/4,2/4,2/4,2"
                ],
                "binInfo": {
                    "jsonFilePath": "ascend910b/add_custom/AddCustom_ccd748392d99d04b8205210970fde2b94.json"
                }
                },
                {
                "simplifiedKey": [
                    "AddCustom/d=0,p=0/5,2/5,2/5,2",
                    "AddCustom/d=1,p=0/5,2/5,2/5,2"
                ],
                "binInfo": {
                    "jsonFilePath": "ascend910b/add_custom/AddCustom_ccd748392d99d04b8205210970fde2b95.txt"
                }
                }
            ]
        }
        )");
    std::string json_str = binDesc.dump();
    size_t size = json_str.size();
    const uint8_t* ptr1 = (uint8_t*)(&json_str[0]);
    const uint8_t* ptr2 = ptr1 + size;

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
    std::string binJsonStr = binJson.dump();
    size = binJsonStr.size();
    const uint8_t* ptr3 = (uint8_t*)(&binJsonStr[0]);
    const uint8_t* ptr4 = ptr3 + size;

    std::string bin = "000";
    size = bin.size();
    const uint8_t* ptr5 = (uint8_t*)(&bin[0]);
    const uint8_t* ptr6 = ptr5 + size;

    // 添加包含taskRation的场景 + buildType的场景
    nlohmann::json binJson2 = nlohmann::json::parse(R"(
        {
            "coreType": "VectorCore",
            "debugOptions": "assert",
            "debugBufSize": 76800,
            "kernelList": [
                {
                    "tilingKey": 1,
                    "kernelType": "VectorCore",
                    "crossCoreSync": 1,
                    "taskRation": "0:1",
                    "kernelName": "AddCustom_f70b557fe2733f00b58895b9092a4784_1"
                }
            ],
            "filePath": "ascend910b/add_custom/AddCustom_1e04ee05ab491cc5ae9c3d5c9ee8950b.json",
            "supportInfo": {
                "simplifiedKey": [
                    "AddCustom/d=0,p=0/1,2/1,2/1,2",
                    "AddCustom/d=1,p=0/1,2/1,2/1,2"
                ]
            }
        }  
        )");
    std::string binJson2Str = binJson2.dump();
    size = binJson2Str.size();
    const uint8_t* ptr7 = (uint8_t*)(&binJson2Str[0]);
    const uint8_t* ptr8 = ptr7 + size;

    nlohmann::json binJson3 = nlohmann::json::parse(R"(
        {
            "coreType": "VectorCore",
            "kernelList": [
                {
                    "tilingKey": 1,
                    "kernelType": "MIX_AIC",
                    "kernelName": "AddCustom_f70b557fe2733f00b58895b9092a4784_1"
                }
            ],
            "filePath": "ascend910b/add_custom/AddCustom_ccd748392d99d04b8205210970fde2b9.json",
            "supportInfo": {
                "simplifiedKey": [
                    "AddCustom/d=0,p=0/0,2/0,2/0,2",
                    "AddCustom/d=1,p=0/0,2/0,2/0,2"
                ]
            }
        }  
        )");
    std::string binJson3Str = binJson3.dump();
    size = binJson3Str.size();
    const uint8_t* ptr9 = (uint8_t*)(&binJson3Str[0]);
    const uint8_t* ptr10 = ptr9 + size;
    // 没有coreType场景
    nlohmann::json binJson4 = nlohmann::json::parse(R"(
        {
            "kernelList": [
                {
                    "kernelName": "AddCustom_f70b557fe2733f00b58895b9092a4784_1"
                }
            ],
            "filePath": "ascend910b/add_custom/AddCustom_ccd748392d99d04b8205210970fde2b94.json",
            "supportInfo": {
                "simplifiedKey": [
                    "AddCustom/d=0,p=0/4,2/4,2/4,2",
                    "AddCustom/d=1,p=0/4,2/4,2/4,2"
                ]
            }
        }  
        )");
    std::string binJson4Str = binJson4.dump();
    size = binJson4Str.size();
    const uint8_t* ptr11 = (uint8_t*)(&binJson4Str[0]);
    const uint8_t* ptr12 = ptr11 + size;

    // jsonPath不以json结尾场景
    nlohmann::json binJson5 = nlohmann::json::parse(R"(
        {
            "coreType": "VectorCore",
            "kernelList": [
                {
                    "kernelName": "AddCustom_1e04ee05ab491cc5ae9c3d5c9ee8950b_1"
                }
            ],
            "filePath": "ascend910b/add_custom/AddCustom_ccd748392d99d04b8205210970fde2b95.txt",
            "supportInfo": {
                "simplifiedKey": [
                    "AddCustom/d=0,p=0/5,2/5,2/5,2"
                ]
            }
        }  
        )");
    std::string binJson5Str = binJson5.dump();
    size = binJson5Str.size();
    const uint8_t* ptr13 = (uint8_t*)(&binJson5Str[0]);
    const uint8_t* ptr14 = ptr13 + size;

    const std::vector<std::tuple<const uint8_t*, const uint8_t*>> opBinary = {
        std::make_tuple(ptr1, ptr2),   std::make_tuple(ptr3, ptr4),   std::make_tuple(ptr5, ptr6),
        std::make_tuple(ptr7, ptr8),   std::make_tuple(ptr5, ptr6),   std::make_tuple(ptr9, ptr10),
        std::make_tuple(ptr5, ptr6),   std::make_tuple(ptr11, ptr12), std::make_tuple(ptr5, ptr6),
        std::make_tuple(ptr13, ptr14), std::make_tuple(ptr5, ptr6)};
    nnopbase::OpBinaryResourceManager::GetInstance().AddBinary(opType.c_str(), opBinary);

    return 1;
}

int32_t CreateStubConf()
{
    // op_debug_config.json
    const std::string opDebugConfigPath = OPP_PATH + "/conf/op_api/op_debug_config.json";
    FILE* opDebugConfigFile = fopen(opDebugConfigPath.c_str(), "w");
    ASSERT_TRUE(opDebugConfigFile != nullptr);
    nlohmann::json json1 = nlohmann::json::parse(R"(
        {
            "op_debug_config": {
                "enable_debug_kernel": "on"
            }
        } 
    )");
    fprintf(opDebugConfigFile, json1.dump().c_str());

    fclose(opDebugConfigFile);

    // error_code.json
    const std::string errorCodePath = OPP_PATH + "/conf/error_manager/error_code.json";
    FILE* errorCodeFile = fopen(errorCodePath.c_str(), "w");
    ASSERT_TRUE(errorCodeFile != nullptr);
    nlohmann::json json2 = nlohmann::json::parse(R"(
        {
        }  
    )");
    fprintf(errorCodeFile, json2.dump().c_str());

    fclose(errorCodeFile);
    return 1;
}

int32_t NnopbaseSetStubFiles(const std::string& opp_path, const bool isOppLastest)
{
    CreateStubFolders(opp_path, isOppLastest);
    ASSERT_SUCC(CreateStubCustConfig());
    ASSERT_SUCC(CreateStubKernelFiles());
    ASSERT_SUCC(CreateStubSceneFile());
    ASSERT_SUCC(CreateStubTilingSo());
    ASSERT_SUCC(NnopbaseSetStaticBinOp());
    ASSERT_SUCC(CreateStubConf());
    return 1;
}

int32_t NnopbaseSetStubNoTilingSoFiles()
{
    CreateStubFolders("");
    return 1;
}

int32_t NnopbaseSetStubNoConfigFiles()
{
    CreateStubFolders("");
    ASSERT_SUCC(CreateStubTilingSo());
    ASSERT_SUCC(CreateStubSceneFile());
    return 1;
}

void NnopbaseUnsetEnvAndClearFolder()
{
    unsetenv("ASCEND_OPP_PATH");
    unsetenv("ASCEND_CUSTOM_OPP_PATH");
    unsetenv("ASCEND_HOME_PATH");
    std::string oppPath = "./usr";
    system(("rm -rf " + oppPath).c_str());
}