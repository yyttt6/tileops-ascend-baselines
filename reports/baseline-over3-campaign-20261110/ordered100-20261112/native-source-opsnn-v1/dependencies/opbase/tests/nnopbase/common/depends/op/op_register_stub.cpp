/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "gtest/gtest.h"
#include <iostream>
#include <stdlib.h>

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/tiling_parse_context.h"

// ---- op register stub begin ----
struct AxpyTilingStruct {
    int dummy;
};

ge::graphStatus AxpyTiling(gert::TilingContext* context)
{
    context->SetBlockDim(16);
    context->SetTilingKey(1234);
    context->SetNeedAtomic(false);
    gert::TilingData* tiling_data = context->GetRawTilingData();
    tiling_data->Append(9876);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AxpyTilingParse(gert::TilingParseContext* context)
{
    auto ci = context->GetCompiledInfo<AxpyTilingStruct>();
    ci->dummy = 100;
    return ge::GRAPH_SUCCESS;
}

struct SortTilingStruct {
    int dummy;
};

ge::graphStatus SortTiling(gert::TilingContext* context)
{
    context->SetBlockDim(16);
    context->SetTilingKey(1234);
    context->SetNeedAtomic(true);
    gert::TilingData* tiling_data = context->GetRawTilingData();
    size_t* ws = context->GetWorkspaceSizes(3);
    ws[0] = 32;
    ws[1] = 32;
    ws[2] = 32;

    tiling_data->Append(9876);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SortTilingParse(gert::TilingParseContext* context)
{
    auto ci = context->GetCompiledInfo<SortTilingStruct>();
    ci->dummy = 100;
    return ge::GRAPH_SUCCESS;
}

struct MemSetTilingStruct {
    int dummy;
};

ge::graphStatus MemSetTiling(gert::TilingContext* context)
{
    context->SetBlockDim(16);
    context->SetTilingKey(1234);
    context->SetNeedAtomic(false);
    gert::TilingData* tiling_data = context->GetRawTilingData();
    tiling_data->Append(9876);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MemSetTilingParse(gert::TilingParseContext* context)
{
    auto ci = context->GetCompiledInfo<MemSetTilingStruct>();
    ci->dummy = 100;
    return ge::GRAPH_SUCCESS;
}

struct AddNTilingStruct {
    int dummy;
};

ge::graphStatus AddNTiling(gert::TilingContext* context)
{
    context->SetBlockDim(16);
    context->SetTilingKey(1234);
    context->SetNeedAtomic(false);
    gert::TilingData* tiling_data = context->GetRawTilingData();
    tiling_data->Append(9876);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddNTilingParse(gert::TilingParseContext* context)
{
    auto ci = context->GetCompiledInfo<AddNTilingStruct>();
    ci->dummy = 100;
    return ge::GRAPH_SUCCESS;
}

struct MulTilingStruct {
    int dummy;
};

ge::graphStatus MulTiling(gert::TilingContext* context)
{
    context->SetBlockDim(17);
    context->SetTilingKey(1234567);
    context->SetNeedAtomic(false);
    gert::TilingData* tiling_data = context->GetRawTilingData();
    tiling_data->Append(8910);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MulTilingParse(gert::TilingParseContext* context)
{
    auto ci = context->GetCompiledInfo<AxpyTilingStruct>();
    ci->dummy = 100;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(Axpy).Tiling(AxpyTiling).TilingParse<AxpyTilingStruct>(AxpyTilingParse);
IMPL_OP(Sort).Tiling(SortTiling).TilingParse<SortTilingStruct>(SortTilingParse);
IMPL_OP(MemSet).Tiling(MemSetTiling).TilingParse<SortTilingStruct>(MemSetTilingParse);
IMPL_OP(AddN).Tiling(AddNTiling).TilingParse<SortTilingStruct>(AddNTilingParse);
IMPL_OP(Mul).Tiling(MulTiling).TilingParse<MulTilingStruct>(MulTilingParse);

// ---- op register stub end ----
