/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INDV_COMMON_TYPES_H_
#define INDV_COMMON_TYPES_H_

#include "exe_graph/runtime/tensor.h"
#include "exe_graph/runtime/shape.h"
#include "exe_graph/runtime/tensor_data.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/kernel_run_context.h"
#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/compute_node_info.h"
#include "graph/types.h"
#include "register/op_impl_registry.h"
#include "register/op_impl_kernel_registry.h"
#include "opdev/common_types.h"
#include "aclnn/acl_meta.h"

using GertTensor = gert::TensorV2;
using GertShape = gert::Shape;
using GeFormat = ge::Format;
using NnopbaseAnchorInstanceInfo = gert::AnchorInstanceInfo;
using NnopbaseChar = char;
using NnopbaseUChar = unsigned char;

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    size_t attr_num;
    uint8_t reserved_[40];
    size_t offset[0];
} NnopbaseRuntimeAttrsDef;

typedef AsyncAnyValue NnopbaseAsyncAnyValue;
typedef KernelRunContext NnopbaseKernelRunContext;
typedef gert::OpImplKernelRegistry::CompileInfoDeleterFunc NnopbaseCompileInfoDeleterFunc;

enum CoreType { kMix = 0, kAicore, kVectorcore, kMixAiCore, kMixAiv, kCoreTypeEnd };

enum InputsAppend {
    kInputsCompileInfo,
    kInputsPlatformInfo,
    kInputsTilingFunc,
    kInputsDeterministic,
    kInputsDeterministicLevel,
    kInputsAppendEnd
};

// CompileTimeTensorDesc class
using NnopbaseCompileTimeTensorDesc = gert::CompileTimeTensorDesc;

typedef struct {
    const NnopbaseChar* nodeType;
    const NnopbaseChar* nodeName;
    size_t irInputsNum;
    size_t inputsNum;
    size_t outputsNum;
    size_t irOutputsNum;
    size_t attrSize;
    uint8_t reserved[24]; // Reserved field, 8-byte aligned
    // following by AnchorInstanceInfo, inputs-outputs-CompileTimeTensorDesc, RuntimeAttrs
    uint64_t place_holder;
} NnopbaseComputeNodeInfo;

// TilingData class
using NnopbaseTilingData = gert::TilingData;
static constexpr int32_t NNOPBASE_TILING_DATA_STRUCT_SIZE = sizeof(NnopbaseTilingData);

typedef unsigned int (*TilingFun)(gert::TilingContext*);
typedef unsigned int(TilingFunPrototype)(gert::TilingContext*);
typedef ge::graphStatus (*GenSimplifiedKeyFun)(gert::TilingContext*, ge::char_t*);

#ifdef __cplusplus
}
#endif
#endif
