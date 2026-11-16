/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <fstream>
#include <memory>
#include <string>
#include <cstdint>
#include <filesystem>
#include "gtest/gtest.h"
#include "file_utils.h"

class FileReaderTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        std::filesystem::path tempFile = std::filesystem::temp_directory_path() / "testfile.bin";
        std::ofstream ofs(tempFile, std::ios::out | std::ios::binary);
        ofs.write("testdata", 8);
        ofs.close();
        testFilePath = tempFile;
    }

    void TearDown() override
    {
        if (!testFilePath.empty()) {
            std::filesystem::remove(testFilePath);
        }
    }

    std::filesystem::path testFilePath;
};

TEST_F(FileReaderTest, ReadExistingFile)
{
    uint32_t dataLen = 0;
    auto data = op::GetBinFromFile(testFilePath.string(), dataLen);

    EXPECT_NE(data, nullptr);
    EXPECT_EQ(dataLen, 8u);
    EXPECT_EQ(std::string(data.get(), dataLen), "testdata");
}

TEST_F(FileReaderTest, ReadNonexistentFile)
{
    uint32_t dataLen = 0;
    auto data = op::GetBinFromFile("nonexistentfile.bin", dataLen);

    EXPECT_EQ(data, nullptr);
    EXPECT_EQ(dataLen, 0u);
}

TEST_F(FileReaderTest, ReadFailedFile)
{
    std::filesystem::path tempFile = std::filesystem::temp_directory_path() / "badfile.bin";
    std::ofstream ofs(tempFile, std::ios::out | std::ios::binary);
    ofs.close();

    std::ifstream ifs(tempFile, std::ios::in | std::ios::binary);
    ifs.setstate(std::ios::badbit);

    uint32_t dataLen = 0;
    auto data = op::GetBinFromFile(tempFile.string(), dataLen);

    EXPECT_NE(data, nullptr);

    std::filesystem::remove(tempFile);
}