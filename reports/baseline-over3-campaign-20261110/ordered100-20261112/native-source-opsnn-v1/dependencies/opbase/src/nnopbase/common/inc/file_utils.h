/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_OP_API_COMMON_SRC_UTILS_FILE_UTILS_H
#define OP_API_OP_API_COMMON_SRC_UTILS_FILE_UTILS_H

#include <cstdint>
#include <string>
#include <memory>
#include <vector>

namespace op {

bool IsDir(const std::string& path);
std::unique_ptr<char[]> GetBinFromFile(const std::string& path, uint32_t& dataLen);
void GetFilesWithSuffix(const std::string& path, const std::string& suffix, std::vector<std::string>& files);
std::string RealPath(const std::string& path);

} // namespace op

#endif // OP_API_OP_API_COMMON_SRC_UTILS_FILE_UTILS_H
