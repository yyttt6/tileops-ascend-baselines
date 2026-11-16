/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <mutex>
#include "graph/ge_error_codes.h"
#include "exe_graph/runtime/kernel_context.h"
#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/tiling_parse_context.h"
#include "register/op_impl_kernel_registry.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "opdev/op_log.h"

#ifdef __cplusplus
extern "C" {
#endif

extern unsigned int TilingForBn(gert::TilingContext* context);

#ifdef __cplusplus
}
#endif

struct CompileInfoStructStub {
    int dummy;
};

ge::graphStatus TilingFuncStub(gert::TilingContext* context)
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

ge::graphStatus Rt2TilingFuncStub(gert::TilingContext* context)
{
    size_t* ws = context->GetWorkspaceSizes(2);
    ws[0] = 100;
    ws[1] = 100;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus EmptyTilingFuncStub(gert::TilingContext* context) { return TilingForBn(context); }

ge::graphStatus MemSetTilingFuncStub(gert::TilingContext* context)
{
    context->SetBlockDim(16);
    context->SetTilingKey(1234);
    context->SetNeedAtomic(false);
    gert::TilingData* tiling_data = context->GetRawTilingData();
    tiling_data->Append(9876);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AtomicTilingFuncStub(gert::TilingContext* context)
{
    context->SetNeedAtomic(true);
    size_t* workspace_size = context->GetWorkspaceSizes(1);
    workspace_size[0] = 100;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MulTilingFuncStub(gert::TilingContext* context)
{
    context->SetBlockDim(17);
    context->SetTilingKey(1234567);
    context->SetNeedAtomic(false);
    gert::TilingData* tiling_data = context->GetRawTilingData();
    tiling_data->Append(8910);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingParseFuncStub(gert::KernelContext* context)
{
    gert::TilingParseContext* tilingParseCtx = reinterpret_cast<gert::TilingParseContext*>(context);
    auto ci = tilingParseCtx->GetCompiledInfo<CompileInfoStructStub>();
    ci->dummy = 100;
    return ge::GRAPH_SUCCESS;
}

// 框架传给genSimplifiedKey的key缓冲大小。本文件是metadef的桩，不便反向依赖nnopbase内部头，
// 故就地定义，取值须与NNOPBASE_CUS_KEY_LEN一致
constexpr size_t GEN_SIMPLIFIED_KEY_BUF_LEN = 256U;

ge::graphStatus GenSimplifiedKeyFuncStub(gert::TilingContext* context, ge::char_t* simplilfiedKey)
{
    const char* source = "diy,99";
    strcpy_s(simplilfiedKey, GEN_SIMPLIFIED_KEY_BUF_LEN, source);
    return ge::GRAPH_SUCCESS;
}

struct StubCompileInfo {
    int64_t stub_ = 2;
};

void* CreateCompileInfoFuncStub() { return new StubCompileInfo(); }

void DeleteCompileInfoFuncStub(void* compile_info) { delete reinterpret_cast<StubCompileInfo*>(compile_info); }

ge::graphStatus InferShapeFuncStub(gert::InferShapeContext* context) { return ge::GRAPH_SUCCESS; }

namespace gert {

std::map<std::string, OpImplKernelRegistry::OpImplFunctionsV2> g_opTypesToImplFunc;

void InitOpImplFunc(std::string opName, OpImplRegisterV2::TilingKernelFunc tilingFunc)
{
    g_opTypesToImplFunc[opName].tiling = tilingFunc;
    g_opTypesToImplFunc[opName].tiling_parse = TilingParseFuncStub;
    g_opTypesToImplFunc[opName].compile_info_creator = CreateCompileInfoFuncStub;
    g_opTypesToImplFunc[opName].compile_info_deleter = DeleteCompileInfoFuncStub;
}

void InitInferShapeFunc(std::string opName, OpImplRegisterV2::InferShapeKernelFunc inferShapeFunc)
{
    g_opTypesToImplFunc[opName].infer_shape = inferShapeFunc;
}

void InitGenSimplifiedKeyFunc(std::string opName, OpImplRegisterV2::GenSimplifiedKeyKernelFunc genFunc)
{
    g_opTypesToImplFunc[opName].gen_simplifiedkey = genFunc;
}

void InitKernelRegistryImplFunc()
{
    // common
    InitOpImplFunc("Sort", TilingFuncStub);
    InitOpImplFunc("Axpy", TilingFuncStub);
    InitInferShapeFunc("Axpy", InferShapeFuncStub);
    InitOpImplFunc("AddN", TilingFuncStub);
    InitOpImplFunc("Mul", MulTilingFuncStub);

    // nnopbase
    InitOpImplFunc("MemSet", MemSetTilingFuncStub);

    std::cout << "Start injecting tiling func " << std::endl;
    InitOpImplFunc("AddTik2", Rt2TilingFuncStub);
    InitOpImplFunc("custom_op", Rt2TilingFuncStub);
    InitOpImplFunc("custom_op1", Rt2TilingFuncStub);
    InitOpImplFunc("custom_op2", Rt2TilingFuncStub);
    InitOpImplFunc("NonFiniteCheck", Rt2TilingFuncStub);
    InitOpImplFunc("bninference_d_kernel", TilingFuncStub);
    InitOpImplFunc("AddCustom", EmptyTilingFuncStub);
    InitOpImplFunc("Flash", Rt2TilingFuncStub);
    InitOpImplFunc("1971_for_mix_normal", Rt2TilingFuncStub);
    InitOpImplFunc("310p_for_mix_aic_normal", Rt2TilingFuncStub);
    InitOpImplFunc("310p_for_mix_aiv_normal", Rt2TilingFuncStub);
    InitOpImplFunc("test_register", Rt2TilingFuncStub);
    InitOpImplFunc("test_profiling_mix_aiv", TilingFuncStub);
    InitOpImplFunc("test_profiling_mix_aic", TilingFuncStub);
    InitOpImplFunc("1971_for_mix_unnormal", TilingFuncStub);
    InitOpImplFunc("AutomicClean", AtomicTilingFuncStub);
    InitOpImplFunc("MemSetV2", AtomicTilingFuncStub);
    InitGenSimplifiedKeyFunc("MemSetV2", GenSimplifiedKeyFuncStub);
    InitOpImplFunc("TestDavidCustom", AtomicTilingFuncStub);
    InitOpImplFunc("NativeSparseAttention", Rt2TilingFuncStub);
    // StubCustomOp: customizedKey模式，genSimplifiedKey产出后缀simplifiedKey "diy,99"
    InitOpImplFunc("StubCustomOp", TilingFuncStub);
    InitGenSimplifiedKeyFunc("StubCustomOp", GenSimplifiedKeyFuncStub);
    // StubCustomOpNoGen: customizedKey模式但故意不注册genSimplifiedKey，用于加载期gen指针为空场景
    InitOpImplFunc("StubCustomOpNoGen", TilingFuncStub);
    // StubMixKeyOp: 内置包声明 mode=2、自定义包声明 mode=0。gen 产出的 "diy,99" 与内置包 json 的
    // "diy,zzz" 不匹配，故 customizedKey 必然未命中而回落
    InitOpImplFunc("StubMixKeyOp", TilingFuncStub);
    InitGenSimplifiedKeyFunc("StubMixKeyOp", GenSimplifiedKeyFuncStub);
}

void InitSpaceRegistryOpImpl()
{
    static std::once_flag flag;
    std::call_once(flag, [&]() -> void { InitKernelRegistryImplFunc(); });
}

const OpImplKernelRegistry::OpImplFunctionsV2* OpImplSpaceRegistryV2::GetOpImpl(const char* op_type) const
{
    OP_LOGI("[space_registry_stub] OpImplSpaceRegistryV2::GetOpImpl: %s", op_type);
    InitSpaceRegistryOpImpl();
    std::string opTypeStr = op_type;
    return &g_opTypesToImplFunc[opTypeStr];
}

} // namespace gert

namespace ge {
class AnyValue {
public:
    AnyValue() = default;
    enum ValueType { VT_NONE = 0 };
};
} // namespace ge