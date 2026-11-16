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
#include "gtest/gtest.h"
#include "opdev/op_errno.h"
#include "opdev/op_log.h"
#include "base/err_mgr.h"

#define REPORT_ERROR_MESSAGE(code, ...)                                \
    do {                                                               \
        ReportErrorMessage(std::to_string(code).c_str(), __VA_ARGS__); \
    } while (0)
#define REPORT_ERROR_MESSAGE_UT(code, ...) REPORT_ERROR_MESSAGE(code, __VA_ARGS__)

class ErrorManagerUt : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        int32_t eRes = error_message::ErrMgrInit(error_message::ErrorMessageMode::INTERNAL_MODE);
        OP_LOGI("Init%d", eRes);
        auto errMsg = error_message::GetErrMgrErrorMessage(); // clear error message container
        OP_LOGI("clear error massage, cached error msg:\n%s", errMsg.get());
    }

    static void TearDownTestCase() {}
};

TEST_F(ErrorManagerUt, ErrorManagerTestCase0)
{
    auto errMsg = error_message::GetErrMgrErrorMessage();
    EXPECT_EQ(std::string(errMsg.get()), "");
}

TEST_F(ErrorManagerUt, ErrorManagerTestCase1)
{
    std::string errorCode = "EZ9999";
    std::string errorMsg = "ErrorManagerTestCase1";
    REPORT_ERROR_MESSAGE_UT(ACLNN_ERR_INNER_TILING_ERROR, errorMsg.c_str());
    auto errMsg = error_message::GetErrMgrErrorMessage();
    OP_LOGI("error msg:\n%s", errMsg.get());
    EXPECT_NE(errMsg, nullptr);
    EXPECT_TRUE(std::string(errMsg.get()).find(errorCode) != std::string::npos);
    EXPECT_TRUE(std::string(errMsg.get()).find(errorMsg) != std::string::npos);
}

TEST_F(ErrorManagerUt, ErrorManagerTestCase2)
{
    std::string errorCode = "EZ1001";
    std::string errorMsg = "ErrorManagerTestCase2";
    int32_t errorNo = ACLNN_ERR_PARAM_INVALID;
    REPORT_ERROR_MESSAGE_UT(errorNo, errorMsg.c_str());
    auto errMsg = error_message::GetErrMgrErrorMessage();
    OP_LOGI("error msg:\n%s", errMsg.get());
    EXPECT_NE(errMsg, nullptr);
    EXPECT_TRUE(std::string(errMsg.get()).find(errorCode) != std::string::npos);
    EXPECT_TRUE(std::string(errMsg.get()).find(errorMsg) != std::string::npos);
}

TEST_F(ErrorManagerUt, ErrorManagerTestCas3)
{
    std::string errorCode = "EZ9903";
    std::string errorMsg = "ErrorManagerTestCase3";
    REPORT_ERROR_MESSAGE_UT(361001, "ErrorManagerTestCase%d", 3);
    auto errMsg = error_message::GetErrMgrErrorMessage();
    OP_LOGI("error msg:\n%s", errMsg.get());
    EXPECT_NE(errMsg, nullptr);
    EXPECT_TRUE(std::string(errMsg.get()).find(errorCode) != std::string::npos);
    EXPECT_TRUE(std::string(errMsg.get()).find(errorMsg) != std::string::npos);
}

// TEST_F(ErrorManagerUt, ErrorManagerTestCas4)
// {
//     std::stringstream expect;
//     expect << "EZ9903: ErrorManagerTestCase1" << std::endl;
//     expect << "        TraceBack (most recent call last):" << std::endl;
//     expect << "        ErrorManagerTestCase0" << std::endl;
//     expect << "        ErrorManagerTestCase1" << std::endl;

//     REPORT_ERROR_MESSAGE_UT(ACLNN_ERR_INNER_INFERSHAPE_ERROR, "ErrorManagerTestCase%d", 0);
//     REPORT_ERROR_MESSAGE_UT(ACLNN_ERR_RUNTIME_ERROR, "ErrorManagerTestCase%d", 1);
//     auto errMsg = ErrorManager::GetInstance().GetErrorMessage();
//     EXPECT_EQ(errMsg, expect.str());
// }

// TEST_F(ErrorManagerUt, ErrorManagerTestCas5)
// {
//     std::stringstream expect;
//     expect << "EZ1001: ErrorManagerTestCase0" << std::endl;
//     expect << "        TraceBack (most recent call last):" << std::endl;
//     expect << "        ErrorManagerTestCase1" << std::endl;

//     REPORT_ERROR_MESSAGE_UT(ACLNN_ERR_PARAM_INVALID, "ErrorManagerTestCase%d", 0);
//     REPORT_ERROR_MESSAGE_UT(ACLNN_ERR_INNER, "ErrorManagerTestCase%d", 1);
//     auto errMsg = ErrorManager::GetInstance().GetErrorMessage();
//     EXPECT_EQ(errMsg, expect.str());
// }

// TEST_F(ErrorManagerUt, ErrorManagerTestCas6)
// {
//     std::stringstream expect;
//     expect << "EZ1001: ErrorManagerTestCase0: xx" << std::endl;

//     REPORT_ERROR_MESSAGE_UT(ACLNN_ERR_PARAM_INVALID, "ErrorManagerTestCase%d: %s", 0, "xx");
//     auto errMsg = ErrorManager::GetInstance().GetErrorMessage();
//     EXPECT_EQ(errMsg, expect.str());
// }

// TEST_F(ErrorManagerUt, ErrorManagerTestCas7)
// {
//     std::stringstream expect;
//     expect << "EZ9999: Inner Error!" << std::endl;
//     expect << "EZ9999  ErrorManagerTestCase0: xx" << std::endl;
//     expect << "        TraceBack (most recent call last):" << std::endl;

//     REPORT_ERROR_MESSAGE_UT(XXXX, "ErrorManagerTestCase%d: %s", 0, "xx");
//     auto errMsg = ErrorManager::GetInstance().GetErrorMessage();
//     EXPECT_EQ(errMsg, expect.str());
// }

// TEST_F(ErrorManagerUt, ErrorManagerTestCas8)
// {
//     std::stringstream msg;
//     for (size_t i = 0; i != LIMIT_PER_MESSAGE + 100U; ++i) {
//         msg << "x";
//     }

//     std::stringstream expect;
//     expect << "EZ1001: ";
//     for (size_t i = 0; i != LIMIT_PER_MESSAGE - 1U; ++i) {
//         expect << "x";
//     }
//     expect << std::endl;

//     REPORT_ERROR_MESSAGE_UT(ACLNN_ERR_PARAM_INVALID, msg.str().c_str());
//     auto errMsg = ErrorManager::GetInstance().GetErrorMessage();
//     EXPECT_EQ(errMsg, expect.str());
// }
