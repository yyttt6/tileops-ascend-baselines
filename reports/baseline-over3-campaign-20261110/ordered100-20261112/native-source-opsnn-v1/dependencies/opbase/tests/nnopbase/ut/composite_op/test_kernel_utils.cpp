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
#include "kernel_utils.h"
#include "opdev/op_errno.h"

class KernelUtilsUT : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        unsetenv("ASCEND_HOME_PATH");
        unsetenv("ASCEND_OPP_PATH");
    }

    static void TearDownTestCase()
    {
        unsetenv("ASCEND_OPP_PATH");
        unsetenv("ASCEND_HOME_PATH");
    }
};

TEST_F(KernelUtilsUT, TestGetOppKernelPath)
{
    KernelUtilsUT::SetUpTestCase();

    std::string oppKernelPath;
    aclnnStatus ret = op::internal::GetOppKernelPath(oppKernelPath);
    EXPECT_EQ(ret, ACLNN_ERR_INNER_OPP_PATH_NOT_FOUND);
    EXPECT_EQ(oppKernelPath.length(), 0);

    setenv("ASCEND_OPP_PATH", "", 1);
    ret = op::internal::GetOppKernelPath(oppKernelPath);
    EXPECT_EQ(ret, ACLNN_ERR_INNER_OPP_PATH_NOT_FOUND);
    EXPECT_EQ(oppKernelPath.length(), 0);

    KernelUtilsUT::TearDownTestCase();
}
