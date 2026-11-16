/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <vector>
#include "string_utils.h"

using namespace std;

namespace op {

vector<string> SplitWith(const string& str, const char pattern)
{
    vector<string> strs;
    if (str.empty()) {
        return strs;
    }

    string strAndPattern = str + pattern;
    size_t pos = strAndPattern.find(pattern);
    while (pos != string::npos) {
        string subStr = strAndPattern.substr(0, pos);
        if (!subStr.empty()) {
            strs.push_back(subStr);
        }
        strAndPattern = strAndPattern.substr(pos + 1u);
        pos = strAndPattern.find(pattern);
    }
    return strs;
}

void TrimWith(string& result, const char delims)
{
    if (result.length() <= 0) {
        return;
    }
    string::size_type index = result.find_last_not_of(delims);
    if (index != string::npos) {
        (void)result.erase(++index);
    }

    index = result.find_first_not_of(delims);
    if (index != string::npos) {
        (void)result.erase(0, index);
    }

    if (result[0] == delims && result[result.length() - 1] == delims) {
        result = "";
    }
    return;
}

} // namespace op