/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "opdev/format_utils.h"
#include "gtest/gtest.h"

class TestFormatUtils : public testing::Test {};

TEST_F(TestFormatUtils, FormatToString) { op::ToString(static_cast<op::Format>(1000000)); }

TEST_F(TestFormatUtils, ToFormat)
{
    EXPECT_EQ(op::ToFormat("NC1HWC0"), op::Format::FORMAT_NC1HWC0);
    EXPECT_EQ(op::ToFormat("NC1HWC0XX"), op::Format::FORMAT_RESERVED);
}
