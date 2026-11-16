/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file bin_to_json.h
 */
#ifndef __OP_INFO_RECORD_BIN_TO_JSON_H__
#define __OP_INFO_RECORD_BIN_TO_JSON_H__
#include <functional>
#include <map>
#include <nlohmann/json.hpp>

#include "graph/types.h"

namespace aclnnOpInfoRecord {
extern const std::map<ge::DataType, std::function<nlohmann::json(const void*, size_t)>> BIN_TO_JSON;
} // namespace aclnnOpInfoRecord
#endif // __OP_INFO_RECORD_BIN_TO_JSON_H__