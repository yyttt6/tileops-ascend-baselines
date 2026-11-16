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
 * \file infershape_broadcast_util.h
 * \brief
 */
#ifndef OP_COMMON_OP_HOST_INFERSHAPE_BROADCAST_UTIL_H
#define OP_COMMON_OP_HOST_INFERSHAPE_BROADCAST_UTIL_H

#include <stddef.h>
#include "exe_graph/runtime/infer_shape_context.h"
#include "util/opbase_export.h"

namespace Ops {
namespace Base {

// Do infershape for OP which is single-input single-output and in-shape equal out-shape.
OPBASE_API ge::graphStatus InferShape4Broadcast(gert::InferShapeContext* context);

// Do infershape for OP which is multi-input single-output.
OPBASE_API ge::graphStatus InferShape4Broadcast(gert::InferShapeContext* context, size_t inputNum);

OPBASE_API bool BroadcastShape(const gert::Shape* in1Shape, const gert::Shape* in2Shape, gert::Shape* outShape);

OPBASE_API bool BroadcastShape(const std::vector<const gert::Shape*>& inShapes, gert::Shape* outShape);
} // namespace Base
} // namespace Ops

#endif // OP_COMMON_OP_HOST_INFERSHAPE_BROADCAST_UTIL_H