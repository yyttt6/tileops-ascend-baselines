/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "utils.h"
#include <algorithm>
#include <stdexcept>
#include <cstdint>

namespace aclnnOpInfoRecord {
std::string Utils::TrimLeft(const std::string& s)
{
    std::string rs = s;
    (void)rs.erase(rs.begin(), std::find_if(rs.begin(), rs.end(), [](uint8_t ch) { return !std::isspace(ch); }));
    return rs;
}

std::string Utils::TrimRight(const std::string& s)
{
    std::string rs = s;
    (void)rs.erase(std::find_if(rs.rbegin(), rs.rend(), [](uint8_t ch) { return !std::isspace(ch); }).base(), rs.end());
    return rs;
}

std::string Utils::Trim(const std::string& s)
{
    std::string tmp = TrimLeft(s);
    return TrimRight(tmp);
}
} // namespace aclnnOpInfoRecord