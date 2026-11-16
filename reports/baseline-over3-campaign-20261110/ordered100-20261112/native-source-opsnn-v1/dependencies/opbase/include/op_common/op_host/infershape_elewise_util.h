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
 * \file infershape_elewise_util.h
 * \brief
 */
#ifndef OP_COMMON_OP_HOST_INFERSHAPE_ELEWISE_UTIL_H
#define OP_COMMON_OP_HOST_INFERSHAPE_ELEWISE_UTIL_H

#include "exe_graph/runtime/infer_shape_context.h"
#include "util/opbase_export.h"

namespace Ops {
namespace Base {

// Do infershape for OP which is single-input single-output and in-shape equal out-shape.
OPBASE_API ge::graphStatus InferShape4Elewise(gert::InferShapeContext* context);
} // namespace Base
} // namespace Ops

#endif // OP_COMMON_OP_HOST_INFERSHAPE_ELEWISE_UTIL_H
