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
#include "mockcpp/mockcpp.hpp"
#include "acl/acl_op.h"
#include "register/op_impl_kernel_registry.h"
#include "graph/types.h"
#include "graph/operator_factory.h"
#include "graph/utils/type_utils.h"
#include "register/op_impl_registry.h"
#include "op_info_serialize.h"
#include "ini_parse.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "lib_path.h"
#include "nlohmann/json.hpp"
class OpInfoRecordStest : public testing::Test {
protected:
    virtual void SetUp() {}
    virtual void TearDown() { GlobalMockObject::verify(); }
};

TEST_F(OpInfoRecordStest, Stest_OpInfoEmptyBinInfo)
{
    std::string path = NNOPBASE_UT_DEPENDS_DIR "/";
    MOCKER(mmAccess2).stubs().will(returnValue(EN_OK));
    MOCKER_CPP(&aclnnOpInfoRecord::LibPath::GetInstallParentPath)
        .stubs()
        .will(returnValue(aclnnOpInfoRecord::Path(path)));
    aclnnOpInfoRecord::OpCompilerOption opt("", 0);
    aclnnOpInfoRecord::OpKernelInfo kernelInfo("", 0);
    gert::TilingContext ctx;
    EXPECT_EQ(aclnnOpInfoRecord::OpInfoSerialize(&ctx, opt, &kernelInfo), -1);
    EXPECT_EQ(aclnnOpInfoRecord::OpInfoDump(), 0);
}

TEST_F(OpInfoRecordStest, Stest_OpInfoSerializeFailed)
{
    std::string path = NNOPBASE_UT_DEPENDS_DIR "/";
    MOCKER(mmAccess2).stubs().will(returnValue(EN_OK));
    MOCKER_CPP(&aclnnOpInfoRecord::LibPath::GetInstallParentPath)
        .stubs()
        .will(returnValue(aclnnOpInfoRecord::Path(path)));
    aclnnOpInfoRecord::OpCompilerOption opt("", 0);
    aclnnOpInfoRecord::OpKernelInfo kernelInfo(
        OP_API_COMMON_UT_SRC_DIR "/built-in/op_impl/ai_core/tbe/kernel/config/ascend910/add_n.json", 0);
    gert::TilingContext ctx;
    EXPECT_EQ(aclnnOpInfoRecord::OpInfoSerialize(&ctx, opt, &kernelInfo), -1);
    EXPECT_EQ(aclnnOpInfoRecord::OpInfoDump(), 0);
}

TEST_F(OpInfoRecordStest, Stest_OpInfoPathEqual)
{
    std::string dependPath = NNOPBASE_UT_DEPENDS_DIR "/";
    aclnnOpInfoRecord::Path pathA = aclnnOpInfoRecord::Path(dependPath);
    aclnnOpInfoRecord::Path pathB = aclnnOpInfoRecord::Path(dependPath);
    std::string configPath = OP_API_COMMON_UT_SRC_DIR "/built-in/op_impl/ai_core/tbe/kernel/config/";
    aclnnOpInfoRecord::Path pathC = aclnnOpInfoRecord::Path(configPath);
    EXPECT_EQ(pathA == pathB, true);
    EXPECT_EQ(pathA == pathC, false);
}