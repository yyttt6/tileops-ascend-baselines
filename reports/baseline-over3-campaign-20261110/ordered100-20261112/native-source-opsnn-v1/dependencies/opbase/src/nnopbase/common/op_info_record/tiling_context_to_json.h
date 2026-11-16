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
 * \file tiling_context_to_json.h
 */
#ifndef __OP_INFO_RECORD_TILING_CONTEXT_TO_JSON_H__
#define __OP_INFO_RECORD_TILING_CONTEXT_TO_JSON_H__
#include <nlohmann/json.hpp>

#include "exe_graph/runtime/tiling_context.h"
#include "graph/operator.h"

namespace aclnnOpInfoRecord {
nlohmann::json TilingContextToJson(const gert::TilingContext* ctx,
                                   const std::map<std::string, std::string>& iniConfigMap,
                                   const nlohmann::json& supportInfoJsonConfig, bool isMc2);
}
#endif // __OP_INFO_RECORD_TILING_CONTEXT_TO_JSON_H__