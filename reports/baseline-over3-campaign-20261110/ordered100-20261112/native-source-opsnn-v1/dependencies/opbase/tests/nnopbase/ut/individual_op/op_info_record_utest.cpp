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
#include <fstream>
#include <cstdio>
#include "acl/acl_op.h"
#include "register/op_impl_kernel_registry.h"
#include "graph/types.h"
#include "graph/operator_factory.h"
#include "graph/utils/type_utils.h"
#include "register/op_impl_registry.h"
#include "opdev/common_types.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "op_run_context.h"
#include "op_info_serialize.h"
#include "ini_parse.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "lib_path.h"
#include "nlohmann/json.hpp"

namespace Adx {
int32_t AdumpSaveToFile(const char* data, size_t dataLen, const char* filename, SaveType type) { return 0; }
} // namespace Adx

class OpInfoRecordUtest : public testing::Test {
protected:
    virtual void SetUp()
    {
        aclnnOpInfoRecord::Path confDir = aclnnOpInfoRecord::LibPath::Instance().GetInstallParentPath().Append("conf");
        ASSERT_TRUE(confDir.CreateDirectory(true));
        std::ifstream src(NNOPBASE_UT_DEPENDS_DIR "/conf/dump_tool_config.ini", std::ios::binary);
        ASSERT_TRUE(src.is_open());
        aclnnOpInfoRecord::Path confFile = confDir.Concat("dump_tool_config.ini");
        std::ofstream dst(confFile.GetString(), std::ios::binary);
        ASSERT_TRUE(dst.is_open());
        dst << src.rdbuf();
    }

    virtual void TearDown()
    {
        gert::DefaultOpImplSpaceRegistryV2::GetInstance().ClearSpaceRegistry();
        aclnnOpInfoRecord::Path confFile = aclnnOpInfoRecord::LibPath::Instance().GetInstallParentPath().Concat(
            "conf/dump_tool_config.ini");
        (void)std::remove(confFile.GetCString());
    }
};

TEST_F(OpInfoRecordUtest, Utest_OpInfoEmptyBinInfo)
{
    aclnnOpInfoRecord::OpCompilerOption opt("", 0);
    aclnnOpInfoRecord::OpKernelInfo kernelInfo("", 0);
    EXPECT_EQ(kernelInfo.isMc2, false);
    gert::TilingContext ctx;
    EXPECT_EQ(aclnnOpInfoRecord::OpInfoSerialize(&ctx, opt, &kernelInfo), -1);
    EXPECT_EQ(aclnnOpInfoRecord::OpInfoDump(), 0);
}

TEST_F(OpInfoRecordUtest, Utest_OpInfoSerialize)
{
    aclnnOpInfoRecord::OpCompilerOption opt("", 0);
    aclnnOpInfoRecord::OpKernelInfo kernelInfo(OP_API_COMMON_UT_SRC_DIR
                                               "/built-in/op_impl/ai_core/tbe/kernel/ascend910/axpy/"
                                               "Axpy_233851a3505389e43928a8bba133a74d_high_performance.json",
                                               0);
    op::Shape shape{33, 15, 1, 48};
    auto self = std::make_unique<aclTensor>(shape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND, nullptr);
    auto other = std::make_unique<aclTensor>(shape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND, nullptr);
    auto out = std::make_unique<aclTensor>(shape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND, nullptr);
    float alpha = 13.37;
    auto input = OP_INPUT(self.get(), other.get());
    auto output = OP_OUTPUT(out.get());
    auto attr = OP_ATTR(alpha);
    auto ctx = op::MakeOpArgContext(input, output, attr);
    auto spaceRegistry = std::make_shared<gert::OpImplSpaceRegistryV2>();
    gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(spaceRegistry);
    uint32_t opType = op::GenOpTypeId("Axpy");
    gert::TilingContext* tilingCtx = op::internal::OpRunContextMgr::opRunCtx_.UpdateTilingCtx(
        opType, *ctx->GetOpArg(op::OpArgDef::OP_INPUT_ARG), *ctx->GetOpArg(op::OpArgDef::OP_OUTPUT_ARG),
        *ctx->GetOpArg(op::OpArgDef::OP_ATTR_ARG));
    ASSERT_NE(tilingCtx, nullptr);

    EXPECT_EQ(aclnnOpInfoRecord::OpInfoSerialize(tilingCtx, opt, &kernelInfo), 0);
    EXPECT_EQ(aclnnOpInfoRecord::OpInfoDump(), 0);
    op::DestroyOpArgContext(ctx);
}

TEST_F(OpInfoRecordUtest, Utest_OpInfoSerialize_without_supportInfo)
{
    aclnnOpInfoRecord::OpCompilerOption opt("", 0);
    aclnnOpInfoRecord::OpKernelInfo kernelInfo(
        OP_API_COMMON_UT_SRC_DIR "/built-in/op_impl/ai_core/tbe/kernel/config/ascend910/add_n.json", 0);
    gert::TilingContext ctx;
    EXPECT_EQ(aclnnOpInfoRecord::OpInfoSerialize(&ctx, opt, &kernelInfo), -1);
    EXPECT_EQ(aclnnOpInfoRecord::OpInfoDump(), 0);
}

TEST_F(OpInfoRecordUtest, Utest_OpInfoPathEqual)
{
    std::string dependPath = NNOPBASE_UT_DEPENDS_DIR "/";
    aclnnOpInfoRecord::Path pathA = aclnnOpInfoRecord::Path(dependPath);
    aclnnOpInfoRecord::Path pathB = aclnnOpInfoRecord::Path(dependPath);
    std::string configPath = OP_API_COMMON_UT_SRC_DIR "/built-in/op_impl/ai_core/tbe/kernel/config/";
    aclnnOpInfoRecord::Path pathC = aclnnOpInfoRecord::Path(configPath);
    EXPECT_EQ(pathA == pathB, true);
    EXPECT_EQ(pathA == pathC, false);
}
