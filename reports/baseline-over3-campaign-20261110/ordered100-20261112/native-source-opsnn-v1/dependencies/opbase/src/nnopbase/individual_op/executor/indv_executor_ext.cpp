/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "indv_executor.h"
#include "indv_tilingcontext_builder.h"
#include "indv_args_pool.h"
#include "indv_collector.h"
#include "utils/thread_var_container.h"
#include "utils/indv_soc.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "register/op_binary_resource_manager.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace {
static constexpr size_t NNOPBASE_VERKEY_EXT_BUF_LEN = 16U;
static constexpr size_t NNOPBASE_REPLAY_VERKEY_EXT_BUF_LEN = 250U;
static constexpr size_t NNOPBASE_VERKEY_TENSOR_DESC_NUM = 2U;
static constexpr size_t NNOPBASE_OP_VERB_HEAD_LEN = 2U;
static constexpr uint32_t NNOPBASE_DTYPE_AND_FORMAT_SIZE = 16U;
std::unordered_map<void*, NnopbaseStreamForCombineExecution> g_nnopbaseStreamMap;
std::unordered_map<void*, std::shared_ptr<std::mutex>> g_nnopbaseStreamMtxMap;
static bool g_isNnopbaseRegisteredCallback = false;
static std::mutex g_nnopbaseStreamMtx;

void PrintTime(const NnopbaseExecutor* const executor, const char* const info, const unsigned short startIndex,
               const unsigned short endIndex)
{
    unsigned long time = 0U;
    if (startIndex < NnopbaseTimeIdx::kNum && endIndex < NnopbaseTimeIdx::kNum &&
        executor->dfx.timeStamp.tp[endIndex].tv_nsec != 0) {
        if (executor->dfx.timeStamp.tp[startIndex].tv_sec != executor->dfx.timeStamp.tp[endIndex].tv_sec) {
            // 1000000000 for time stamp
            time = executor->dfx.timeStamp.tp[endIndex].tv_nsec + 1000000000 -
                   executor->dfx.timeStamp.tp[startIndex].tv_nsec;
        } else {
            time = executor->dfx.timeStamp.tp[endIndex].tv_nsec - executor->dfx.timeStamp.tp[startIndex].tv_nsec;
        }
    }
    OP_EVENT("Nnopbase time %s : %f us", info, time / 1000.0); // 1000.0 for time us
}

void PrintInitTime(const timespec* const tp, const char* const info, const unsigned short startIndex,
                   const unsigned short endIndex)
{
    unsigned long time = 0U;
    if (startIndex < NnopbaseCollectorTimeIdx::kEnd && endIndex < NnopbaseCollectorTimeIdx::kEnd &&
        tp[endIndex].tv_nsec != 0) {
        if (tp[startIndex].tv_sec != tp[endIndex].tv_sec) {
            // 1000000000 for time stamp
            time = tp[endIndex].tv_nsec + 1000000000 - tp[startIndex].tv_nsec;
        } else {
            time = tp[endIndex].tv_nsec - tp[startIndex].tv_nsec;
        }
    }
    OP_EVENT("Nnopbase init time %s : %f us", info, time / 1000.0); // 1000.0 for time us
}

void NnopbaseExecutorClearAttrs(NnopbaseAttrs* attrs)
{
    attrs->totalSize = 0;
    attrs->totalDataLen = 0;
    for (size_t i = 0U; i < attrs->num; i++) {
        attrs->attrs[i].addr.addr = nullptr;
        attrs->attrs[i].addr.size = 0;
    }
}

void NnopbaseExecutorReset(NnopbaseExecutor* executor)
{
    executor->dfx.Reset();
    executor->isOutEmpty = false;
    executor->workspaces.num = 0U;
    executor->hasTiling = true;
    executor->ownArgs.enableCache = true;
    executor->userHandle = nullptr;
    executor->mc2.ClearRuntimeState();
    executor->inUnContExe = nullptr;
    executor->viewCopyExe = nullptr;
    executor->inUncontWsSize = 0U;
    executor->repeatFlag = false;
    executor->hasMemset = false;
    executor->formatCheckOption = kNnopbaseDefault;
    executor->isCachedArgs = false;
    executor->matchArgsV2 = false;
    executor->deterministicLevel = 0U;
    executor->deterministic = false;
}
} // namespace

aclnnStatus NnopbaseExecutorConvertScalarType(std::vector<uint8_t>& scalarValue, const aclScalar* scalar,
                                              ge::DataType dtype, const size_t offset)
{
    switch (dtype) {
        case ge::DataType::DT_FLOAT:
            *op::internal::PtrCastTo<float>(scalarValue.data() + offset) = scalar->ToFloat();
            return OK;
        case ge::DataType::DT_FLOAT16:
            *op::internal::PtrCastTo<uint16_t>(scalarValue.data() + offset) = scalar->ToFp16();
            return OK;
        case ge::DataType::DT_BF16:
            *op::internal::PtrCastTo<uint16_t>(scalarValue.data() + offset) = scalar->ToBf16();
            return OK;
        case ge::DataType::DT_INT8:
            *op::internal::PtrCastTo<int8_t>(scalarValue.data() + offset) = scalar->ToInt8();
            return OK;
        case ge::DataType::DT_INT16:
            *op::internal::PtrCastTo<int16_t>(scalarValue.data() + offset) = scalar->ToInt16();
            return OK;
        case ge::DataType::DT_INT32:
            *op::internal::PtrCastTo<int32_t>(scalarValue.data() + offset) = scalar->ToInt32();
            return OK;
        case ge::DataType::DT_INT64:
            *op::internal::PtrCastTo<int64_t>(scalarValue.data() + offset) = scalar->ToInt64();
            return OK;
        case ge::DataType::DT_UINT8:
            *op::internal::PtrCastTo<uint8_t>(scalarValue.data() + offset) = scalar->ToUint8();
            return OK;
        case ge::DataType::DT_UINT16:
            *op::internal::PtrCastTo<uint16_t>(scalarValue.data() + offset) = scalar->ToUint16();
            return OK;
        case ge::DataType::DT_UINT32:
            *op::internal::PtrCastTo<uint32_t>(scalarValue.data() + offset) = scalar->ToUint32();
            return OK;
        case ge::DataType::DT_UINT64:
            *op::internal::PtrCastTo<uint64_t>(scalarValue.data() + offset) = scalar->ToUint64();
            return OK;
        case ge::DataType::DT_BOOL:
            *op::internal::PtrCastTo<bool>(scalarValue.data() + offset) = scalar->ToBool();
            return OK;
        case ge::DataType::DT_DOUBLE:
            *op::internal::PtrCastTo<double>(scalarValue.data() + offset) = scalar->ToDouble();
            return OK;
        case ge::DataType::DT_COMPLEX64:
            *op::internal::PtrCastTo<std::complex<float>>(scalarValue.data() + offset) = scalar->ToComplex64();
            return OK;
        case ge::DataType::DT_COMPLEX128:
            *op::internal::PtrCastTo<std::complex<double>>(scalarValue.data() + offset) = scalar->ToComplex128();
            return OK;
        default:
            OP_LOGE(ACL_ERROR_API_NOT_SUPPORT, "Unsupported data type[%s].", op::ToString(dtype).GetString());
            return ACLNN_ERR_PARAM_INVALID;
    }
}

static void NnopbaseExecutorClearExtTensors(NnopbaseTensors* tensors)
{
    tensors->paramDescs.activeInputCount = tensors->paramDescs.count;
    tensors->paramDescs.emptyNum = 0U;
    tensors->hostInputNum = 0U;
    tensors->dynamicCnt = 0U;
    tensors->hostInputSize = 0U;
    tensors->expectIndex = 0U;
    tensors->outPutShapeSize = 0U;
    tensors->outPutShapeArgsOffset = 0U;
    tensors->outPutShapeMap.clear();
    if (!tensors->hasDynamic) {
        for (uint32_t i = 0U; i < tensors->arrayLen; i++) {
            tensors->extTensors[i].isNull = true;
            tensors->extTensors[i].valueDepend = false;
        }
    } else {
        tensors->usedNum = 0U;
        tensors->num = tensors->nonDynamicCnt;
        for (uint32_t i = 0U; i < tensors->arrayLen; i++) {
            tensors->extTensors[i].isNull = true;
            tensors->extTensors[i].isRequired = false;
            tensors->extTensors[i].valueDepend = false;
        }
    }
    NnopbaseClearParamInstance(tensors);
}

static void NnopbaseExecutorInitIoTensor(NnopbaseTensors* tensors, const NnopbaseChar* paramDesc, const uint32_t num)
{
    tensors->nonDynamicCnt = 0U;
    tensors->requiredCnt = 0U;
    tensors->dynamicCnt = 0U;
    tensors->dynamicNum = 0U;
    for (uint32_t i = 0U; i < num; i++) {
        if (paramDesc[i] == 1) {
            tensors->nonDynamicCnt++;
            tensors->requiredCnt++;
        } else if (paramDesc[i] == 0) {
            tensors->nonDynamicCnt++;
        } else {
            tensors->dynamicNum++;
        }
    }
    if (tensors->nonDynamicCnt < num) {
        tensors->hasDynamic = true;
        tensors->arrayLen = NNOPBASE_NORM_DEF_IO_NUMS + num;
        tensors->usedNum = 0U;
    } else {
        tensors->hasDynamic = false;
        tensors->arrayLen = num;
        tensors->usedNum = num;
    }
    tensors->num = tensors->nonDynamicCnt;
    tensors->extTensors = std::vector<NnopbaseTensor>(tensors->arrayLen);
}

aclnnStatus NnopbaseExecutorInitIoCaches(NnopbaseTensors* tensors, const NnopbaseChar* paramDesc, const uint32_t num)
{
    NnopbaseExecutorInitIoTensor(tensors, paramDesc, num);
    NnopbaseParamDesc* irInsts = &tensors->paramDescs;
    irInsts->instances = std::vector<NnopbaseParamInstance>(num);
    irInsts->count = num;
    irInsts->emptyNum = 0U;
    for (uint32_t i = 0U; i < tensors->arrayLen; i++) {
        tensors->extTensors[i].isOptional = false;
    }
    if (!tensors->hasDynamic) {
        for (uint32_t i = 0U; i < num; i++) {
            irInsts->instances[i].num = 1U;
            irInsts->instances[i].startIndex = i;
            irInsts->instances[i].cfgNum = paramDesc[i];
            irInsts->instances[i].isDynamic = false;
            irInsts->instances[i].isDisable = false;
            tensors->extTensors[i].isRequired = (paramDesc[i] == 1) ? true : false;
            tensors->extTensors[i].isOptional = (paramDesc[i] == 0) ? true : false;
        }
    } else {
        for (uint32_t i = 0U; i < num; i++) {
            irInsts->instances[i].startIndex = 0U;
            irInsts->instances[i].cfgNum = paramDesc[i];
            irInsts->instances[i].isDisable = false;
            if (paramDesc[i] <= 1) {
                irInsts->instances[i].isDynamic = false;
                irInsts->instances[i].num = 1U;
                tensors->extTensors[i].isOptional = (paramDesc[i] == 0) ? true : false;
            } else {
                irInsts->instances[i].isDynamic = true;
                irInsts->instances[i].num = 0U;
            }
        }
    }
    NnopbaseExecutorClearExtTensors(tensors);
    return OK;
}

static aclnnStatus NnopbaseExecutorInitAttrsCaches(NnopbaseExecutor* executor, const NnopbaseChar* attrsDesc,
                                                   const uint32_t attrsNum)
{
    NnopbaseAttrs* attrs = &executor->attrs;
    if (attrsNum == 0U) {
        return OK;
    }
    attrs->attrs = (NnopbaseAttr*)malloc(attrsNum * sizeof(NnopbaseAttr));
    NNOPBASE_ASSERT_NOTNULL_RETVAL(attrs->attrs);
    attrs->num = attrsNum;

    for (size_t i = 0U; i < attrsNum; i++) {
        if (attrsDesc[i] == 0) {
            attrs->attrs[i].addr.isOptional = true;
        }
    }
    return OK;
}

static void NnopbaseExecutorDeInitAttrsCaches(NnopbaseExecutor* executor) { FREE(executor->attrs.attrs); }

aclnnStatus NnopbaseExecutorInit(NnopbaseExecutor* executor, const NnopbaseOpInfo opInfo)
{
    NNOPBASE_ASSERT_OK_RETVAL(
        NnopbaseExecutorInitIoCaches(&executor->ownArgs.inputs, opInfo.inputsDesc, opInfo.inputNum));
    NNOPBASE_ASSERT_OK_RETVAL(
        NnopbaseExecutorInitIoCaches(&executor->ownArgs.outputs, opInfo.outputsDesc, opInfo.outputNum));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseExecutorInitAttrsCaches(executor, opInfo.attrsDesc, opInfo.attrsNum));

    executor->hasMemset = false;
    executor->args = nullptr;
    executor->tiling.Init();
    executor->collector = nullptr;
    executor->workspaces.num = 0U;
    executor->workspaces.length = 0U;
    executor->opType = nullptr;
    executor->space = nullptr;
    executor->binInfoKey.len = 0U;
    executor->binInfoKey.bufLen = 0U;
    executor->hasTiling = true;
    executor->isWork = false;
    executor->dfx.Init();
    executor->opCompatibility = {};
    executor->opTypeId = 0U;
    executor->poolIndex = -1;
    executor->userHandle = nullptr;
    executor->isOutEmpty = false;
    executor->mc2.Reset();
    executor->inUnContExe = nullptr;
    executor->viewCopyExe = nullptr;
    executor->inUncontWsSize = 0U;
    executor->repeatFlag = false;
    executor->formatCheckOption = kNnopbaseDefault;
    (void)NnopbaseTilingContextInit(executor);
    return OK;
}

aclnnStatus NnopbaseExecutorExtendIoCaches(NnopbaseTensors* tensors)
{
    const uint32_t num = tensors->num * 2U;
    tensors->extTensors.resize(num);
    tensors->arrayLen = num;
    return OK;
}

void StreamMapClear(aclrtStream stream)
{
    if (g_nnopbaseStreamMap.find(stream) != g_nnopbaseStreamMap.end()) {
        OP_LOGI("Start clear stream. Main stream is %p, subStream %p, eventA %p, eventB %p", stream,
                g_nnopbaseStreamMap[stream].stream, g_nnopbaseStreamMap[stream].eventA,
                g_nnopbaseStreamMap[stream].eventB);
        OP_CHECK_NO_RETURN(
            aclrtDestroyStream(g_nnopbaseStreamMap[stream].stream) == ACL_SUCCESS,
            OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "Failed to destroy stream %p.", g_nnopbaseStreamMap[stream].stream));
        OP_CHECK_NO_RETURN(
            aclrtDestroyEvent(g_nnopbaseStreamMap[stream].eventA) == ACL_SUCCESS,
            OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "Failed to destroy event %p.", g_nnopbaseStreamMap[stream].eventA));
        OP_CHECK_NO_RETURN(
            aclrtDestroyEvent(g_nnopbaseStreamMap[stream].eventB) == ACL_SUCCESS,
            OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "Failed to destroy event %p.", g_nnopbaseStreamMap[stream].eventB));
        g_nnopbaseStreamMap.erase(stream);
    }
    return;
}

void NnopbaseExecutorDeInit(NnopbaseExecutor* executor)
{
    NnopbaseExecutorDeInitAttrsCaches(executor);
    NnopbaseTilingContextDeInit(executor);
    if (executor->opCompatibility.ownsSupportedSocList) {
        delete[] executor->opCompatibility.supportedSocList;
    }
    executor->opCompatibility = {};
    FREE(executor->opType);
}

static void NnopbaseExecutorClearArgs(NnopbaseExecutorArgs* const args)
{
    args->workspaceLen = 0U;
    args->workspaceNum = 0U;
    args->isOutEmpty = false;
    args->hasTiling = true;
}

void NnopbaseExecutorClear(NnopbaseExecutor* executor)
{
    NnopbaseExecutorReset(executor);
    NnopbaseExecutorClearExtTensors(&executor->ownArgs.inputs);
    NnopbaseExecutorClearExtTensors(&executor->ownArgs.outputs);
    NnopbaseExecutorClearUnContiguousTensors(&executor->ownArgs.inputs);
    NnopbaseExecutorClearArgs(&executor->ownArgs);
    NnopbaseExecutorClearAttrs(&executor->attrs);
    RecordNnopbaseTime(executor, NnopbaseTimeIdx::kRunWithWsEnd);
    PrintNnopbaseAllTimeStampInfo(executor);
    nnopbase::ArgsPool::GetInstance().ReleaseArgs(executor->args);
    const std::lock_guard<std::mutex> lock(executor->space->spaceMtx);
    executor->isWork = false; /* this isWork field MUST BE the tail of this function. */
}

void NnopbaseExecutorFixCache(NnopbaseExecutor* executor)
{
    if (executor == nullptr || executor->args == nullptr) {
        return;
    }
    nnopbase::ArgsPool::GetInstance().FixCache(executor->args);
}

// NnopbaseInit内部调用，调用前已保证设置锁
aclnnStatus NnopbaseExecutorSetGlobalConfig()
{
    g_nnopbaseSysCfgParams.enableTimeStamp = nnopbase::EnableNnopbaseTimeStamp();   // 是否开启性能打点
    g_nnopbaseSysCfgParams.enableArgsCache = nnopbase::EnableNnopbaseArgsCache();   // 是否开启args(tiling)缓存
    g_nnopbaseSysCfgParams.deterministicLevel = nnopbase::GetGlobalDeterministic(); // 确定性设置
    g_nnopbaseSysCfgParams.implMode = nnopbase::NnopbaseGetImplMode(
        g_nnopbaseSysCfgParams.precision);                                          // 高性能/高精度模式设置
    NNOPBASE_ASSERT_OK_RETVAL(
        nnopbase::NnopbaseSetOverFlowAddr(g_nnopbaseSysCfgParams.overflowAddr));    // 设置溢出检测地址
    g_nnopbaseSysCfgParams.enableDebugKernel = nnopbase::GetDebugKernel();

    OP_LOGD("Set global config timeStamp enable = %d, argsCache enable = %d, deterministicLevel = %d,"
            "precision = %d, overflowAddr = %p, debugKernel = %d.",
            g_nnopbaseSysCfgParams.enableTimeStamp, g_nnopbaseSysCfgParams.enableArgsCache,
            g_nnopbaseSysCfgParams.deterministicLevel, g_nnopbaseSysCfgParams.precision,
            g_nnopbaseSysCfgParams.overflowAddr, g_nnopbaseSysCfgParams.enableDebugKernel);
    return OK;
}

aclnnStatus NnopbaseExecutorClearSpace(NnopbaseExecutorSpace* space)
{
    const std::lock_guard<std::mutex> lock(space->spaceMtx);
    for (auto& executor : space->executors) {
        if (executor != nullptr) {
            NnopbaseExecutorDeInit(executor);
        }
    }
    space->executors.clear();
    return ACLNN_SUCCESS;
}

aclnnStatus NnopbaseExecutorGcSpace(void* data)
{
    NnopbaseExecutorSpace* space = (NnopbaseExecutorSpace*)data;
    NnopbaseExecutorClearSpace(space);
    delete space;
    return ACLNN_SUCCESS;
}

aclnnStatus SetTensorDataSizeToInitValue(NnopbaseExecutor* executor)
{
    for (uint32_t i = 0; i < executor->args->binInfo->initValues.size(); i++) {
        size_t index = executor->args->binInfo->initValues[i].irIndex;
        CHECK_COND(index >= executor->args->inputs.paramDescs.count, ACLNN_ERR_PARAM_INVALID,
                   "Only support init outputs, init_value index %zu is small than inputs num %u.", index,
                   executor->args->inputs.paramDescs.count);
        size_t irIndex = index - executor->args->inputs.paramDescs.count;
        CHECK_COND(irIndex < executor->args->outputs.paramDescs.count, ACLNN_ERR_PARAM_INVALID,
                   "Only support init outputs, does not support init workspace.");
        CHECK_COND(!executor->args->outputs.paramDescs.instances[irIndex].isDynamic, ACLNN_ERR_PARAM_INVALID,
                   "Does not support init dynamic output[%zu].", irIndex);
        // 将算子的输出的IrIndex转化为输入IrIndex
        executor->args->binInfo->initValues[i].irIndex = irIndex;
        const uint32_t startIndex = executor->args->outputs.paramDescs.instances[irIndex].startIndex;
        if (!executor->args->outputs.extTensors[startIndex].isNull) {
            GertTensor* tensor = &executor->args->outputs.extTensors[startIndex].rt2Tensor;
            if (nnopbase::IndvSoc::GetInstance().NeedAlignInitValues()) {
                executor->args->binInfo->initValues[i].tensorDataSize = op::internal::AlignSize(tensor->GetSize(),
                                                                                                NNOPBASE_BLOCK_SIZE);
                OP_LOGI("Size is %zu bytes, dtypeSize is %d bytes, alignSize is %zu bytes.", tensor->GetSize(),
                        ge::GetSizeByDataType(tensor->GetDataType()),
                        op::internal::AlignSize(tensor->GetSize(), NNOPBASE_BLOCK_SIZE));
            } else {
                executor->args->binInfo->initValues[i].tensorDataSize = tensor->GetSize();
            }
            OP_LOGI("InitValues[%zu] outputIrIndex:%zu size is %zu bytes, dtypeSize is %d bytes, TensorDataSize is %zu "
                    "bytes.",
                    index, irIndex, tensor->GetSize(), ge::GetSizeByDataType(tensor->GetDataType()),
                    executor->args->binInfo->initValues[i].tensorDataSize);
        }
    }
    return OK;
}

static aclnnStatus NnopnbaseMemsetTiling(NnopbaseExecutor* executor)
{
    const size_t tilingDataSize = executor->args->binInfo->memsetInfo->binInfo->opParaSize == 0U ?
                                      NNOPBASE_MAX_TILING_DATA_LEN :
                                      executor->args->binInfo->memsetInfo->binInfo->opParaSize;
    // memset算子的输入是其他算子的输出，都是device地址, 2 is tiling and overflow
    const size_t tilingDataOffset = (executor->args->binInfo->initValues.size() + 2U) * sizeof(void*);
    const size_t argSize = tilingDataOffset + tilingDataSize + NNOPBASE_TILING_DATA_OFFSET;
    /* |initvalue0, initvalue1, ... initvalueN, tilingPtr, overFlowPtr, tilingData, NNOPBASE_TILING_DATA_OFFSET| */
    OP_LOGI("Memset tilingDataSize is %zu bytes, tilingDataOffset is %zu bytes.", tilingDataSize, tilingDataOffset);
    if (argSize > executor->args->memsetArgs.size()) {
        executor->args->memsetArgs.resize(argSize);
    }
    auto tilingData = op::internal::PtrCastTo<NnopbaseTilingData>(executor->args->binInfo->memsetInfo->tilingData);
    tilingData->Init(executor->args->binInfo->memsetInfo->binInfo->opParaSize,
                     &(executor->args->memsetArgs[tilingDataOffset]));

    auto tilingFunc = executor->args->binInfo->memsetInfo->tilingFunc;
    CHECK_COND((tilingFunc != nullptr), ACLNN_ERR_INNER_TILING_ERROR, "Failed to find tiling function of MemSet!");
    NNOPBASE_ASSERT_RETVAL(tilingFunc(op::internal::PtrCastTo<gert::TilingContext>(
                               executor->args->binInfo->memsetInfo->contextExt.context)) == ge::GRAPH_SUCCESS,
                           ACLNN_ERR_INNER_TILING_ERROR);

    NNOPBASE_ASSERT_NOTNULL_RETVAL(executor->args->binInfo->memsetInfo->numBlocks);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(executor->args->binInfo->memsetInfo->tilingKey);
    OP_LOGI("Memset op numBlocks is %u, tilingKey is %llu.", *(executor->args->binInfo->memsetInfo->numBlocks),
            *(executor->args->binInfo->memsetInfo->tilingKey));
    return OK;
}

aclnnStatus NnopbaseExecutorInsertMemsetOp(NnopbaseExecutor* executor)
{
    executor->hasMemset = executor->args->binInfo->isStaticShape ? true : executor->args->tilingInfo.needAtomic;
    if (executor->hasMemset) {
        // 构造 memset tiling 上下文
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseBuildMemsetTilingContext(executor));
        NNOPBASE_ASSERT_OK_RETVAL(NnopnbaseMemsetTiling(executor));
    }
    return OK;
}

static inline void NnopbaseExecutorPreportMemsetProfiling(const NnopbaseExecutor* const executor, uint32_t numBlocks,
                                                          const uint32_t taskType, const uint64_t launchBeginTime)
{
    NnopbaseInnerReportLaunchInfo(launchBeginTime, executor->dfx.memsetItemId);
    NnopbaseReportMemsetAdditionInfo(executor, numBlocks, taskType, launchBeginTime + 1U);
    return;
}

static aclnnStatus NnopbaseExecutorPrepareMemsetArgs(NnopbaseExecutor* executor)
{
    // 2 is for tiling, overflow
    const size_t argsNum = executor->args->binInfo->initValues.size() + 2U;
    auto tilingData = op::internal::PtrCastTo<NnopbaseTilingData>(executor->args->binInfo->memsetInfo->tilingData);
    // 通过tilingData指针反推Arg排布的起始位置
    void* args = op::internal::PtrCastTo<NnopbaseUChar>(tilingData->GetData()) - argsNum * sizeof(void*);
    executor->args->binInfo->memsetInfo->argsExt.args = args;
    void** addr = (void**)args;
    const size_t tilingDataSize = tilingData->GetDataSize();
    const size_t alignTilingDataSize = ((tilingDataSize % 8U) != 0) ? (tilingDataSize / 8U + 1U) * 8U :
                                                                      tilingDataSize; // 8byte对齐
    // 从索引0开始填充args的输入Tensor指针
    for (size_t i = 0U; i < executor->args->binInfo->initValues.size(); i++) {
        size_t irIndex = executor->args->binInfo->initValues[i].irIndex;
        const uint32_t startIndex = executor->args->outputs.paramDescs.instances[irIndex].startIndex;
        *addr = executor->args->outputs.extTensors[startIndex].rt2Tensor.GetAddr();
        addr++;
    }
    *addr = tilingData->GetData();
    executor->args->binInfo->memsetInfo->argsExt.tilingAddrOffset = static_cast<uint32_t>(
        op::internal::PtrCastTo<NnopbaseUChar>(addr) -
        op::internal::PtrCastTo<NnopbaseUChar>(
            executor->args->binInfo->memsetInfo->argsExt.args)); // 结果即initvalue.size() * sizeof(void *)
    executor->args->binInfo->memsetInfo->argsExt.tilingDataOffset = static_cast<uint32_t>(
        op::internal::PtrCastTo<NnopbaseUChar>(tilingData->GetData()) -
        op::internal::PtrCastTo<NnopbaseUChar>(executor->args->binInfo->memsetInfo->argsExt.args));
    OP_LOGI("Tiling address offset for memset is %u, tiling data offset is %u",
            executor->args->binInfo->memsetInfo->argsExt.tilingAddrOffset,
            executor->args->binInfo->memsetInfo->argsExt
                .tilingDataOffset); // 结果即 (initvalue.size() + 2U) * sizeof(void *)
    addr++;
    *addr = g_nnopbaseSysCfgParams.overflowAddr;
    addr++;
    executor->args->binInfo->memsetInfo->argsExt.hasTiling = true;
    executor->args->binInfo->memsetInfo->argsExt.hostInputInfoPtr = nullptr;
    executor->args->binInfo->memsetInfo->argsExt.hostInputInfoNum = 0U;
    executor->args->binInfo->memsetInfo->argsExt.argsSize = static_cast<uint32_t>(
        op::internal::PtrCastTo<NnopbaseUChar>(addr) -
        op::internal::PtrCastTo<NnopbaseUChar>(executor->args->binInfo->memsetInfo->argsExt.args) +
        alignTilingDataSize); // 结果即 (sizeof(initValues) + 2U) * sizeof(void*) + tilingDataSize

    OP_LOGI("Aligned tilingDataSize is %zu bytes, argsSize is %u.", alignTilingDataSize,
            executor->args->binInfo->memsetInfo->argsExt.argsSize);

    CHECK_COND((executor->args->binInfo->memsetInfo->argsExt.argsSize <= executor->args->memsetArgs.size()),
               ACLNN_ERR_PARAM_INVALID, "Memset args is out of memory. argsSize is %u, argsBuf size is %zu",
               executor->args->binInfo->memsetInfo->argsExt.argsSize, executor->args->memsetArgs.size());
    return OK;
}

aclnnStatus NnopbaseLaunchMemsetTask(NnopbaseExecutor* executor, aclrtStream stream)
{
    const uint64_t launchBeginTime = NnopbaseMsprofSysTime();
    auto memsetInfo = executor->args->binInfo->memsetInfo;
    const uint32_t numBlocks = *(memsetInfo->numBlocks);
    const uint32_t dynUBufSize = *(memsetInfo->dynUBufSize);
    const uint64_t tilingKey = *(memsetInfo->tilingKey);
    const uint8_t scheMode = static_cast<uint8_t>(*(memsetInfo->scheMode));
    OP_LOGI("Start to launch Memset task, numBlocks is %u, tilingKey is %llu, dynUBufSize is %u, scheMode is %u.",
            numBlocks, tilingKey, dynUBufSize, *(memsetInfo->scheMode));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseExecutorPrepareMemsetArgs(executor));
    aclrtFuncHandle funcHandle;
    aclrtLaunchKernelCfg* cfgPtr = nullptr;
    aclrtLaunchKernelCfg cfg;
    std::vector<aclrtLaunchKernelAttr> attrs;
    if (memsetInfo->binInfo->loadFuncHandleByTilingKey) {
        OP_LOGI("Get memset operator function handle by tilingKey.");
        NNOPBASE_ASSERT_OK_RETVAL(GetFuncHandleByEntry(memsetInfo->binInfo->binHandle, tilingKey, &funcHandle));
        if (*(memsetInfo->scheMode) != 0U) {
            aclrtLaunchKernelAttr schemModeAttr = {
                .id = ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE,
                .value = {.schemMode = scheMode},
            };
            attrs.push_back(schemModeAttr);
        }
        if (dynUBufSize != 0U) {
            aclrtLaunchKernelAttr simtModeAttr = {
                .id = static_cast<aclrtLaunchKernelAttrId>(INDV_LAUNCH_KERNEL_RTS_DYN_UBUF_SIZE),
                .value = {.localMemorySize = dynUBufSize},
            };
            attrs.push_back(simtModeAttr);
        }
        if (!attrs.empty()) {
            cfg.attrs = attrs.data();
            cfg.numAttrs = attrs.size();
            cfgPtr = &cfg;
        }
    } else {
        OP_LOGI("Get memset operator function handle by kernel name %s.", memsetInfo->binInfo->kernelName.c_str());
        NNOPBASE_ASSERT_RTOK_RETVAL(GetFuncHandleByKernelName(memsetInfo->binInfo->binHandle,
                                                              memsetInfo->binInfo->kernelName.c_str(), &funcHandle));
    }
    auto rtsHostPlaceHolder = NnopbaseGetRTSPlaceHolder(&memsetInfo->argsExt);
    NNOPBASE_ASSERT_RTOK_RETVAL(aclrtLaunchKernelWithHostArgs(funcHandle, numBlocks, stream, cfgPtr,
                                                              memsetInfo->argsExt.args, memsetInfo->argsExt.argsSize,
                                                              rtsHostPlaceHolder.data(), rtsHostPlaceHolder.size()));
    const uint32_t taskType = NnopbaseExecutorGetTaskType(memsetInfo->binInfo->coreType, kRationEnd);
    NnopbaseExecutorPreportMemsetProfiling(executor, numBlocks, taskType, launchBeginTime);
    OP_LOGI("Launch memset operator successfully.");
    return OK;
}

void NnopbaseExecutorReportProfiling(NnopbaseExecutor* const executor, uint32_t numBlocks, const uint32_t taskType,
                                     const uint64_t launchBeginTime, aclrtStream stream)
{
    NnopbaseInnerReportLaunchInfo(launchBeginTime, executor->dfx.itemId);
    NnopbaseReportAdditionInfo(executor, numBlocks, taskType, launchBeginTime + 1U);
    NnopbaseReportAttrAndHostInfo(executor, launchBeginTime + 1U);
    NnopbaseReportCacheOpInfo(executor, numBlocks, taskType, stream);
    return;
}

aclnnStatus NnopbaseExecutorGetCoreTypeAndTaskRation(NnopbaseExecutor* executor, CoreType& coreType,
                                                     NnopbaseTaskRation& taskRation)
{
    // 获取kernelType
    const uint64_t tilingKey = executor->args->tilingInfo.tilingKey;
    if (executor->args->binInfo->tilingKeyInfo.find(tilingKey) != executor->args->binInfo->tilingKeyInfo.end()) {
        coreType = executor->args->binInfo->tilingKeyInfo[tilingKey].coreType;
        taskRation = executor->args->binInfo->tilingKeyInfo[tilingKey].taskRation;
    } else {
        OP_LOGE(ACLNN_ERR_INNER_TILING_ERROR, "Cannot find tilingKey[%lu] in kernel json file of %s.", tilingKey,
                executor->opType);
        return ACLNN_ERR_PARAM_INVALID;
    }
    OP_LOGI("OP [%s] tilingKey is [%lu], coreType is %d, taskRation is [%d].", executor->opType, tilingKey, coreType,
            taskRation);
    return OK;
}

void PrintNnopbaseAllTimeStampInfo(NnopbaseExecutor* const executor)
{
    if (g_nnopbaseSysCfgParams.enableTimeStamp) {
        thread_local static std::atomic<std::uint64_t> index = 0;
        const unsigned long long opIdx = index++;
        OP_EVENT("test case opType: %s, index: %llu", executor->opType, opIdx);
        PrintTime(executor, "add io and attr", NnopbaseTimeIdx::kAfterCreateExecutor, NnopbaseTimeIdx::kGetWsStart);
        PrintTime(executor, "before match cache", NnopbaseTimeIdx::kGetWsStart, NnopbaseTimeIdx::kMatchCacheStart);
        PrintTime(executor, "match cache", NnopbaseTimeIdx::kMatchCacheStart, NnopbaseTimeIdx::kMatchCacheEnd);
        PrintTime(executor, "create cache", NnopbaseTimeIdx::kMatchCacheEnd, NnopbaseTimeIdx::kCreateCacheEnd);
        PrintTime(executor, "aging cache", NnopbaseTimeIdx::kCreateCacheEnd, NnopbaseTimeIdx::kAgingCacheEnd);
        PrintTime(executor, "check static bin", NnopbaseTimeIdx::kAgingCacheEnd, NnopbaseTimeIdx::kFindStaticEnd);
        PrintTime(executor, "find bin", NnopbaseTimeIdx::kFindStaticEnd, NnopbaseTimeIdx::kFindBinEnd);
        PrintTime(executor, "build tiling context", NnopbaseTimeIdx::kFindBinEnd, NnopbaseTimeIdx::kTilingStart);
        PrintTime(executor, "tiling", NnopbaseTimeIdx::kTilingStart, NnopbaseTimeIdx::kTilingEnd);
        PrintTime(executor, "before launch", NnopbaseTimeIdx::kRunWithWsStart, NnopbaseTimeIdx::kBeforeLaunch);
        PrintTime(executor, "launch", NnopbaseTimeIdx::kBeforeLaunch, NnopbaseTimeIdx::kAfterLaunch);
        PrintTime(executor, "after launch", NnopbaseTimeIdx::kAfterLaunch, NnopbaseTimeIdx::kRunWithWsEnd);
        (void)memset_s(executor->dfx.timeStamp.tp, sizeof(executor->dfx.timeStamp.tp), 0,
                       sizeof(executor->dfx.timeStamp.tp));
    }
}

void PrintNnopbaseInitTimeStampInfo()
{
    if (g_nnopbaseSysCfgParams.enableTimeStamp) {
        PrintInitTime(gBinCollector->collectorTp, "collector init", NnopbaseCollectorTimeIdx::kCollectorInitStart,
                      NnopbaseCollectorTimeIdx::kCollectorInitEnd);
        PrintInitTime(gBinCollector->collectorTp, "get base path", NnopbaseCollectorTimeIdx::kCollectorInitEnd,
                      NnopbaseCollectorTimeIdx::kGetBasePathEnd);
        PrintInitTime(gBinCollector->collectorTp, "load tiling so", NnopbaseCollectorTimeIdx::kGetBasePathEnd,
                      NnopbaseCollectorTimeIdx::kLoadTilingSoEnd);
        PrintInitTime(gBinCollector->collectorTp, "load debug kernel", NnopbaseCollectorTimeIdx::kLoadTilingSoEnd,
                      NnopbaseCollectorTimeIdx::kLoadDebugKernelEnd);
        PrintInitTime(gBinCollector->collectorTp, "load static kernel", NnopbaseCollectorTimeIdx::kLoadDebugKernelEnd,
                      NnopbaseCollectorTimeIdx::kLoadStaticKernelEnd);
        PrintInitTime(gBinCollector->collectorTp, "load dynamic kernel", NnopbaseCollectorTimeIdx::kLoadStaticKernelEnd,
                      NnopbaseCollectorTimeIdx::kLoadDynamicKernelEnd);
    }
}

aclnnStatus NnopbaseExecutorSetRegInfo(NnopbaseExecutor* executor, const NnopbaseChar* opType)
{
    const size_t len = strlen(opType);
    const uint64_t hashKey = static_cast<uint64_t>(
                                 NnopbaseHashBinary(op::internal::PtrCastTo<const NnopbaseUChar>(opType), len)) %
                             NNOPBASE_NORM_MAX_BIN_BUCKETS;
    executor->regInfo = NnopbaseCollectorFindRegInfoInTbl(executor->collector, opType, hashKey);
    if (executor->regInfo == nullptr) {
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseUpdateStaticBinJsonInfos(executor->collector, opType));
        OP_LOGI("Find register info in table again, opType: %s", opType);
        executor->regInfo = NnopbaseCollectorFindRegInfoInTbl(executor->collector, opType, hashKey); // 再次查找
    }
    if (executor->regInfo == nullptr) {
        std::string socVersion = nnopbase::IndvSoc::GetInstance().GetCurSocVersion();
        std::string
            errMsg = "1.SoC version " + socVersion +
                     " verification failed. This SoC is not configured through the AddConfig API of the OpDef class. ";
        errMsg += "2.The binary_info_config.json file to which the " + std::string(opType) +
                  " operator belongs fails to "
                  "be loaded because the file is damaged or does not exist or the user does not have sufficient "
                  "permissions. ";
        errMsg += "3.The operator package to which the " + std::string(opType) + " operator belongs is not installed";
        OP_LOGE_FOR_EXECUTION_ERROR_WITHOUT_SOLUTION(errMsg.c_str());
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (executor->opType == nullptr) {
        executor->opType = (NnopbaseChar*)malloc(len + 1U);
        NNOPBASE_ASSERT_NOTNULL_RETVAL(executor->opType);
        NNOPBASE_ASSERT_TRUE_RETVAL(strcpy_s(executor->opType, len + 1U, opType) == EOK);
        executor->opTypeHash = hashKey;
    }
    executor->dfx.itemId = MsprofGetHashId(opType, len);
    const std::string aicpuOpType = std::string(opType) + NNOPBASE_MC2_AICPU_SUFFIX;
    executor->dfx.aicpuItemId = MsprofGetHashId(&aicpuOpType[0], aicpuOpType.length());
    executor->dfx.memsetItemId = MsprofGetHashId(&NNOPBASE_MEMSET_OP_NAME[0], NNOPBASE_MEMSET_OP_NAME.length());
    OP_LOGI("OpType is %s, hashKey is %lu, item id is %lu, aicpu item id is %lu, memset item id is %lu.", opType,
            hashKey, executor->dfx.itemId, executor->dfx.aicpuItemId, executor->dfx.memsetItemId);
    return OK;
}

NnopbaseUChar* NnopbaseExecutor8ByteCopy(const size_t totalSize, NnopbaseUChar* verKey, const NnopbaseUChar* addr)
{
    const size_t length = totalSize / 8U;
    if (length != 0) {
        verKey = NnopbaseAppendBinary(verKey, sizeof(uint64_t) * length, addr, sizeof(uint64_t) * length);
        addr += sizeof(uint64_t) * length;
    }

    if ((totalSize % 8U) != 0U) { // 8 + 16 字节 false
        const size_t leftBit = totalSize % 8U;
        uint64_t littleEndian = 0U;
        for (size_t i = 0U; i < leftBit; i++) {
            littleEndian |= littleEndian | (static_cast<uint64_t>(*(addr + i)) << (sizeof(uint64_t) * i));
        }
        verKey = NnopbaseAppend8Byte(verKey, littleEndian);
    }
    return verKey;
}

static inline void NnopbaseExecutorSetFormatAndShape(GertTensor* rt2Tensor, int64_t shapeSize)
{
    if (shapeSize == 0) {
        rt2Tensor->MutableOriginShape() = {};
        rt2Tensor->MutableStorageShape() = {};
    } else {
        rt2Tensor->MutableOriginShape() = {shapeSize};
        rt2Tensor->MutableStorageShape() = {shapeSize};
    }
    rt2Tensor->SetOriginFormat(ge::FORMAT_ND);
    rt2Tensor->SetStorageFormat(ge::FORMAT_ND);
    rt2Tensor->MutableTensorData().SetPlacement(gert::kOnHost);
    rt2Tensor->MutableStride().SetDimNum(0U);
    rt2Tensor->SetOffset(0);
}

static void NnopbaseExecutorGetScalarDType(NnopbaseTensors* tensors, ge::DataType& dataType, const ge::DataType dtype,
                                           const int32_t srcIndex, const uint32_t index)
{
    if (srcIndex != -1) {
        if (static_cast<uint32_t>(srcIndex) > index) {
            tensors->paramDescs.instances[srcIndex].scalarIndex = static_cast<int32_t>(index);
        } else {
            auto srcTensor = &tensors->extTensors[tensors->paramDescs.instances[srcIndex].startIndex];
            dataType = srcTensor->rt2Tensor.GetDataType();
        }
    } else if (dtype != ge::DT_UNDEFINED) {
        dataType = dtype;
    }
}

aclnnStatus NnopbaseExecutorAddScalarInput(NnopbaseTensors* tensors, const aclScalar* scalar, const uint32_t index,
                                           const int32_t srcIndex, const ge::DataType dtype)
{
    tensors->paramDescs.instances[index].isInput = true;
    if (scalar != nullptr) {
        tensors->paramDescs.instances[index].num = 1U;
        ge::DataType dataType = scalar->GetDataType();
        NnopbaseExecutorGetScalarDType(tensors, dataType, dtype, srcIndex, index);
        auto tensor = &tensors->extTensors[tensors->paramDescs.instances[index].startIndex];
        tensor->isNull = false;
        GertTensor* rt2Tensor = &tensor->rt2Tensor;
        NNOPBASE_ASSERT_NOTNULL_RETVAL(rt2Tensor);
        NnopbaseExecutorSetFormatAndShape(rt2Tensor, 0);
        if (scalar->GetDataType() == ge::DataType::DT_DOUBLE) {
            dataType = ge::DataType::DT_FLOAT;
            tensors->paramDescs.instances[index].scalarValue.resize(sizeof(float));
            *op::internal::PtrCastTo<float>(
                tensors->paramDescs.instances[index].scalarValue.data()) = scalar->ToFloat();
            NNOPBASE_ASSERT_OK_RETVAL(rt2Tensor->MutableTensorData().SetAddr(
                tensors->paramDescs.instances[index].scalarValue.data(), nullptr));
        } else if (scalar->GetDataType() != dataType) {
            tensors->paramDescs.instances[index].scalarValue.resize(op::TypeSize(dataType));
            NNOPBASE_ASSERT_OK_RETVAL(NnopbaseExecutorConvertScalarType(
                tensors->paramDescs.instances[index].scalarValue, scalar, dataType, 0U));
            NNOPBASE_ASSERT_OK_RETVAL(rt2Tensor->MutableTensorData().SetAddr(
                tensors->paramDescs.instances[index].scalarValue.data(), nullptr));
        } else {
            NNOPBASE_ASSERT_OK_RETVAL(rt2Tensor->MutableTensorData().SetAddr(scalar->GetData(), nullptr));
        }
        rt2Tensor->SetDataType(dataType);
        rt2Tensor->SetSize(op::TypeSize(dataType));
        tensors->hostInputNum++;
        tensors->hostInputSize += op::TypeSize(dataType);
    } else {
        tensors->paramDescs.instances[index].num = 0U; // for null option
        tensors->paramDescs.emptyNum++;
    }
    OP_LOGD("Executor add scalar input[%u] successfully.", index);
    return OK;
}

aclnnStatus NnopbaseExecutorAddScalarListInput(NnopbaseTensors* tensors, const aclScalarList* scalarList,
                                               const uint32_t index, const int32_t srcIndex, const ge::DataType dtype)
{
    if ((scalarList == nullptr) || (scalarList->Size() == 0U)) {
        return OK;
    }
    tensors->paramDescs.instances[index].isInput = true;
    ge::DataType dataType = (*scalarList)[0]->GetDataType();
    NnopbaseExecutorGetScalarDType(tensors, dataType, dtype, srcIndex, index);

    auto tensor = &tensors->extTensors[tensors->paramDescs.instances[index].startIndex];
    NNOPBASE_ASSERT_NOTNULL_RETVAL(tensor);
    tensor->isNull = false;
    GertTensor* rt2Tensor = &tensor->rt2Tensor;
    NNOPBASE_ASSERT_NOTNULL_RETVAL(rt2Tensor);
    NnopbaseExecutorSetFormatAndShape(rt2Tensor, static_cast<int64_t>(scalarList->Size()));

    const size_t elementSize = op::TypeSize(dataType);
    const size_t tensorSize = elementSize * static_cast<size_t>(scalarList->Size());

    OP_LOGI("ScalarList input[%u] size is %lu, dType is %s, scalarListSize is %zu", index, scalarList->Size(),
            op::ToString(dataType).GetString(), tensors->paramDescs.instances[index].scalarValue.size());
    if (tensors->paramDescs.instances[index].scalarValue.size() < tensorSize) {
        tensors->paramDescs.instances[index].scalarValue.resize(tensorSize);
    }

    size_t offset = 0U;
    for (uint64_t i = 0U; i < scalarList->Size(); i++) {
        NNOPBASE_ASSERT_NOTNULL_RETVAL((*scalarList)[i]);
        if ((*scalarList)[i]->GetDataType() == ge::DataType::DT_DOUBLE) {
            dataType = ge::DataType::DT_FLOAT;
            *op::internal::PtrCastTo<float>(tensors->paramDescs.instances[index].scalarValue.data() +
                                            offset) = (*scalarList)[i]->ToFloat();
            offset += op::TypeSize(ge::DataType::DT_FLOAT);
        } else if ((*scalarList)[i]->GetDataType() != dataType) {
            NNOPBASE_ASSERT_OK_RETVAL(NnopbaseExecutorConvertScalarType(
                tensors->paramDescs.instances[index].scalarValue, (*scalarList)[i], dataType, offset));
            offset += op::TypeSize(dataType);
        } else {
            NNOPBASE_ASSERT_TRUE_RETVAL(memcpy_s(tensors->paramDescs.instances[index].scalarValue.data() + offset,
                                                 tensorSize - offset, (*scalarList)[i]->GetData(), elementSize) == EOK);
            offset += elementSize;
        }
    }

    rt2Tensor->SetDataType(dataType);
    const size_t scalarListSize = op::TypeSize(dataType) * scalarList->Size();
    rt2Tensor->SetSize(scalarListSize);
    tensors->hostInputSize += scalarListSize;

    NNOPBASE_ASSERT_OK_RETVAL(
        rt2Tensor->MutableTensorData().SetAddr(tensors->paramDescs.instances[index].scalarValue.data(), nullptr));
    tensors->hostInputNum++;

    OP_LOGD("Executor add scalarList input[%u] successfully.", index);
    return OK;
}

void NnopbaseDestroyStreamCallback(aclrtStream stream, const bool isCreate)
{
    if (isCreate) {
        return;
    }
    StreamMapClear(stream);
    return;
}

aclnnStatus NnopbaseExecutorGetStreamAndEvent(const aclrtStream stream, aclrtStream* subStream, aclrtEvent* evtA,
                                              aclrtEvent* evtB, std::shared_ptr<std::mutex>& streamLckPtr)
{
    const std::lock_guard<std::mutex> lock(g_nnopbaseStreamMtx);
    aclrtStream mainStream = stream;
    if (stream == nullptr) {
        OP_LOGI("Main stream is nullptr.");
        NNOPBASE_ASSERT_RTOK_RETVAL(aclrtCtxGetCurrentDefaultStream(&mainStream));
    }
    if (g_nnopbaseStreamMtxMap.find(mainStream) == g_nnopbaseStreamMtxMap.cend()) {
        g_nnopbaseStreamMtxMap[mainStream] = std::make_shared<std::mutex>();
        NNOPBASE_ASSERT_NOTNULL_RETVAL(g_nnopbaseStreamMtxMap[mainStream]);
    }
    streamLckPtr = g_nnopbaseStreamMtxMap[mainStream];
    if (g_nnopbaseStreamMap.find(mainStream) != g_nnopbaseStreamMap.end()) {
        *subStream = g_nnopbaseStreamMap[mainStream].stream;
        *evtA = g_nnopbaseStreamMap[mainStream].eventA;
        *evtB = g_nnopbaseStreamMap[mainStream].eventB;
        OP_LOGI("Found main stream is %p, subStream %p, eventA %p, eventB %p", mainStream, *subStream, *evtA, *evtB);
    } else {
        CHECK_COND(
            aclrtCreateStreamWithConfig(subStream, 0U, ACL_STREAM_FAST_LAUNCH | ACL_STREAM_FAST_SYNC) == ACL_SUCCESS,
            ACLNN_ERR_RUNTIME_ERROR, "Failed to create stream %p.", subStream);
        CHECK_COND(aclrtCreateEventExWithFlag(evtA, ACL_EVENT_SYNC) == ACL_SUCCESS, ACLNN_ERR_RUNTIME_ERROR,
                   "Failed to create event %p.", evtA);
        CHECK_COND(aclrtCreateEventExWithFlag(evtB, ACL_EVENT_SYNC) == ACL_SUCCESS, ACLNN_ERR_RUNTIME_ERROR,
                   "Failed to create event %p.", evtB);
        g_nnopbaseStreamMap[mainStream] = {*subStream, *evtA, *evtB};
    }
    OP_LOGI("Main stream is %p, subStream %p, eventA %p, eventB %p.", mainStream, *subStream, *evtA, *evtB);

    if (g_isNnopbaseRegisteredCallback) {
        return OK;
    }
    NNOPBASE_ASSERT_RTOK_RETVAL(rtRegStreamStateCallback("NnopbaseDestroyStream", NnopbaseDestroyStreamCallback));

    g_isNnopbaseRegisteredCallback = true;
    return OK;
}

aclnnStatus NnopbaseSetMc2Tiling(NnopbaseExecutor* executor, const char* const group)
{
    HcclComm commHandle = nullptr;
    NNOPBASE_ASSERT_RTOK_RETVAL(nnopbase::IndvHcclWrapper::GetInstance().HcomGetCommHandleByGroup(group, &commHandle));
    executor->mc2.commHandles.push_back(commHandle);
    NNOPBASE_ASSERT_RTOK_RETVAL(nnopbase::utils::ThreadVarContainer::SetCurMc2RankIdInThread(commHandle));
    return OK;
}

static NnopbaseUChar* NnopbaseInitDynamicKeyBuffer(NnopbaseExecutor* const executor,
                                                   const NnopbaseTensors* const inputs,
                                                   const NnopbaseTensors* const outputs)
{
    NnopbaseUChar* verKey = nullptr;
    if (executor->binInfoKey.verbose.empty()) {
        const uint32_t len = inputs->requiredCnt * NNOPBASE_VERKEY_TENSOR_DESC_NUM +
                             inputs->dynamicNum * NNOPBASE_VERKEY_TENSOR_DESC_NUM +
                             outputs->requiredCnt * NNOPBASE_VERKEY_TENSOR_DESC_NUM +
                             outputs->dynamicNum * NNOPBASE_VERKEY_TENSOR_DESC_NUM +
                             static_cast<uint32_t>(executor->regInfo->key.opType.size()) + NNOPBASE_OP_VERB_HEAD_LEN;
        executor->binInfoKey.verbose = std::vector<NnopbaseUChar>(len + NNOPBASE_VERKEY_EXT_BUF_LEN, '\0');
        verKey = &(executor->binInfoKey.verbose[0U]);
        executor->binInfoKey.bufLen = static_cast<uint32_t>(len + NNOPBASE_VERKEY_EXT_BUF_LEN);
        executor->binInfoKey.len = len;
        verKey = NnopbaseAppendBinary(verKey, static_cast<size_t>(executor->binInfoKey.bufLen),
                                      &(executor->regInfo->key.opType[0U]), executor->regInfo->key.opType.size());
    } else {
        verKey = &(executor->binInfoKey.verbose[0U]) + executor->regInfo->key.opType.size();
    }
    return verKey;
}

static NnopbaseUChar* NnopbaseAppendDynamicKeyHead(const NnopbaseExecutor* const executor, NnopbaseUChar* verKey)
{
    uint8_t determinLevel = (executor->deterministicLevel == 0U) ? 0U : 1U;
    verKey = NnopbaseAppend1Byte(verKey, static_cast<NnopbaseUChar>(determinLevel));
    verKey = NnopbaseAppend1Byte(verKey, g_nnopbaseSysCfgParams.precision);
    OP_LOGI("Verbose opType is %s, determinLevel is %d, highPrecision is %d, verboseLen is %u",
            executor->regInfo->key.opType.c_str(), executor->deterministicLevel, g_nnopbaseSysCfgParams.precision,
            executor->binInfoKey.len);
    return verKey;
}

static NnopbaseUChar* NnopbaseAppendDynamicTensorsKey(const NnopbaseExecutor* const executor, NnopbaseUChar* verKey,
                                                      const NnopbaseTensors* const tensors, const char* const ioType)
{
    for (uint32_t i = 0U; i < tensors->paramDescs.count; i++) {
        if (tensors->paramDescs.instances[i].cfgNum == 0) {
            continue;
        }
        const uint32_t startIndex = tensors->paramDescs.instances[i].startIndex;
        verKey = NnopbaseAppend1Byte(
            verKey, static_cast<NnopbaseUChar>(tensors->extTensors[startIndex].rt2Tensor.GetDataType()));
        verKey = NnopbaseAppend1Byte(
            verKey, static_cast<NnopbaseUChar>(tensors->extTensors[startIndex].rt2Tensor.GetStorageFormat()));
        OP_LOGI("Op %s %s[%u] datatype is %s, format is %s.", executor->opType, ioType, startIndex,
                op::ToString(tensors->extTensors[startIndex].rt2Tensor.GetDataType()).GetString(),
                op::ToString(tensors->extTensors[startIndex].rt2Tensor.GetStorageFormat()).GetString());
    }
    return verKey;
}

void NnopbaseExecutorGenDynamicKey(NnopbaseExecutor* const executor)
{
    const NnopbaseTensors* const inputs = &executor->ownArgs.inputs;
    const NnopbaseTensors* const outputs = &executor->ownArgs.outputs;
    // verb key format: op_type/Deterministic,Precision/input num/output num/tensor desc...
    NnopbaseUChar* verKey = NnopbaseInitDynamicKeyBuffer(executor, inputs, outputs);
    verKey = NnopbaseAppendDynamicKeyHead(executor, verKey);
    verKey = NnopbaseAppendDynamicTensorsKey(executor, verKey, inputs, "input");
    verKey = NnopbaseAppendDynamicTensorsKey(executor, verKey, outputs, "output");
    executor->binInfoKey.len = static_cast<uint32_t>(verKey - &(executor->binInfoKey.verbose[0U]));
    OP_LOGI("Op %s binInfoKey size %u", executor->opType, executor->binInfoKey.len);
    executor->binInfoKey.hashKey = NnopbaseHashBinary(&(executor->binInfoKey.verbose[0U]), executor->binInfoKey.len) %
                                   NNOPBASE_NORM_MAX_BIN_BUCKETS;
}

aclnnStatus NnopbaseExecutorGenCustomizedDynamicKey(NnopbaseExecutor* const executor)
{
    NNOPBASE_ASSERT_NOTNULL_RETVAL(executor->regInfo);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(executor->opType);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(executor->args);
    if (executor->regInfo->genSimplifiedKey == nullptr) {
        OP_LOGW("Failed to generate simplifiedKey, genSimplifiedKey function is nullptr.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseTilingContextBuild(executor));

    const size_t opTypeLen = strlen(executor->opType);
    NnopbaseUChar* verbKey = nullptr;
    if (executor->binInfoKey.verbose.empty()) {
        executor->binInfoKey.verbose = std::vector<NnopbaseUChar>(NNOPBASE_VEB_KEY_LEN, '\0');
        verbKey = &(executor->binInfoKey.verbose[0U]);
        executor->binInfoKey.bufLen = static_cast<uint32_t>(NNOPBASE_VEB_KEY_LEN);
        verbKey = NnopbaseAppendBinary(verbKey, static_cast<size_t>(NNOPBASE_VEB_KEY_LEN), executor->opType, opTypeLen);
    } else {
        verbKey = &(executor->binInfoKey.verbose[0U]) + opTypeLen;
    }
    verbKey = NnopbaseAppendDynamicKeyHead(executor, verbKey);

    char simplifiedKey[NNOPBASE_CUS_KEY_LEN] = {'\0'};
    auto ret = executor->regInfo->genSimplifiedKey(
        op::internal::PtrCastTo<gert::TilingContext>(executor->tiling.contextExt.context), simplifiedKey);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGW("Failed to generate simplifiedKey, ret is %u.", ret);
        return ACLNN_ERR_PARAM_INVALID;
    }
    const size_t writtenLen = static_cast<size_t>(verbKey - &(executor->binInfoKey.verbose[0U]));
    const size_t remainLen = static_cast<size_t>(executor->binInfoKey.bufLen) - writtenLen;
    verbKey = NnopbaseAppendBinary(verbKey, remainLen, simplifiedKey, strlen(simplifiedKey));

    executor->binInfoKey.len = static_cast<uint32_t>(verbKey - &(executor->binInfoKey.verbose[0U]));
    executor->binInfoKey.hashKey = NnopbaseHashBinary(&(executor->binInfoKey.verbose[0U]), executor->binInfoKey.len) %
                                   NNOPBASE_NORM_MAX_BIN_BUCKETS;

    OP_LOGI("Generate customizedKey verbose key successfully, simplifiedKey is %s, key length is %u, "
            "hashKey is %zu.",
            simplifiedKey, executor->binInfoKey.len, executor->binInfoKey.hashKey);
    return OK;
}

static bool NnopbaseExecutorFindBinInfoByCustomizedKey(NnopbaseExecutor* executor)
{
    const aclnnStatus ret = NnopbaseExecutorGenCustomizedDynamicKey(executor);
    if (ret != OK) {
        OP_LOGW("Failed to generate customizedKey verbose key, ret is %d, try to find dynamic kernel by "
                "tensor descriptor.",
                ret);
        return false;
    }
    executor->args->binInfo = NnopbaseCollectorFindBinInfo(
        executor->regInfo, executor->binInfoKey.hashKey, &(executor->binInfoKey.verbose[0U]), executor->binInfoKey.len);
    if (executor->args->binInfo != nullptr) {
        return true;
    }
    OP_LOGW("Failed to find dynamic kernel by customizedKey, try to find by tensor descriptor.");
    return false;
}

bool NnopbaseExecutorGetDynamicBinInfo(NnopbaseExecutor* const executor)
{
    const bool isCustomizedKey = executor->regInfo->customizedSimplifiedKey;
    if (isCustomizedKey && NnopbaseExecutorFindBinInfoByCustomizedKey(executor)) {
        OP_LOGI("Find dynamic kernel by customizedKey successfully, hashKey is %zu, key length is %u.",
                executor->binInfoKey.hashKey, executor->binInfoKey.len);
        return true;
    }

    if (isCustomizedKey) {
        executor->binInfoKey.verbose = std::vector<NnopbaseUChar>();
        executor->binInfoKey.len = 0U;
        executor->binInfoKey.bufLen = 0U;
        executor->binInfoKey.hashKey = 0U;
    }

    NnopbaseExecutorGenDynamicKey(executor);
    executor->args->binInfo = NnopbaseCollectorFindBinInfo(
        executor->regInfo, executor->binInfoKey.hashKey, &(executor->binInfoKey.verbose[0U]), executor->binInfoKey.len);
    if (executor->args->binInfo == nullptr) {
        OP_LOGW("Failed to find dynamic kernel by %s.",
                isCustomizedKey ? "either customizedKey or tensor descriptor" : "tensor descriptor");
        return false;
    }
    OP_LOGI("Find dynamic kernel by tensor descriptor successfully, hashKey is %zu, key length is %u.",
            executor->binInfoKey.hashKey, executor->binInfoKey.len);
    return true;
}

NnopbaseUChar* NnopbaseExecutorGenTensorsKey(NnopbaseUChar* verKey, const NnopbaseTensors* const tensors,
                                             const size_t tensorNum, const bool usingStride)
{
    const NnopbaseUChar* addr;
    size_t length;
    for (size_t i = 0U; i < tensorNum; i++) {
        if (tensors->extTensors[i].isNull) {
            OP_LOGI("Tensor[%zu] is null.", i);
            verKey = NnopbaseAppend8Byte(verKey, '_');
            continue;
        }
        if (tensors->extTensors[i].isRequired || tensors->extTensors[i].isOptional) {
            verKey = NnopbaseAppend8Byte(verKey, tensors->extTensors[i].rt2Tensor.GetDataType());
            verKey = NnopbaseAppend8Byte(verKey, tensors->extTensors[i].rt2Tensor.GetStorageFormat());
            OP_LOGI("Tensor[%zu] datatype is %d, format is %d, isRequired is %s, isOptional is %s.", i,
                    static_cast<int32_t>(tensors->extTensors[i].rt2Tensor.GetDataType()),
                    static_cast<int32_t>(tensors->extTensors[i].rt2Tensor.GetStorageFormat()),
                    tensors->extTensors[i].isRequired ? "true" : "false",
                    tensors->extTensors[i].isOptional ? "true" : "false");
        }
        const GertShape& shape = tensors->extTensors[i].rt2Tensor.GetStorageShape();
        const size_t dimNum = shape.GetDimNum();
        for (size_t j = 0U; j < dimNum; j++) {
            verKey = NnopbaseAppend8Byte(verKey, static_cast<uint64_t>(shape.GetDim(j)));
            OP_LOGI("Tensor[%zu] storageShape[%zu] is %ld", i, j, shape.GetDim(j));
        }
        if (usingStride) {
            const auto& stride = tensors->extTensors[i].rt2Tensor.GetStride();
            const size_t strideNum = stride.GetDimNum();
            for (size_t k = 0U; k < strideNum; k++) {
                verKey = NnopbaseAppend8Byte(verKey, static_cast<uint64_t>(stride.GetStride(k)));
            }
            verKey = NnopbaseAppend8Byte(verKey, static_cast<uint64_t>(tensors->extTensors[i].rt2Tensor.GetOffset()));
        }
        if (tensors->extTensors[i].valueDepend) {
            addr = op::internal::PtrCastTo<const NnopbaseUChar>(tensors->extTensors[i].rt2Tensor.GetAddr());
            const auto dtype = tensors->extTensors[i].rt2Tensor.GetDataType();
            length = tensors->extTensors[i].rt2Tensor.GetSize();
            const size_t typeSize = op::TypeSize(dtype);
            const size_t elementNum = length / typeSize;
            for (size_t k = 0; k < elementNum; k++) {
                verKey = NnopbaseExecutor8ByteCopy(typeSize, verKey, addr + typeSize * k);
            }
        }
    }
    return verKey;
}

NnopbaseUChar* NnopbaseExecutorGenAttrsKey(const NnopbaseAttrs* const attrs, NnopbaseUChar* verKey)
{
    const NnopbaseUChar* addr = nullptr;
    size_t length = 0;
    for (size_t j = 0; j < attrs->num; j++) {
        // 传入时已校验 attrs[j].addr.addr 不为空
        if (!attrs->attrs[j].addr.isVector) {
            if (attrs->attrs[j].dtype == NnopbaseAttrDtype::kNnopbaseString && attrs->attrs[j].addr.size == 1U) {
                OP_LOGW("For Attr %zu, this is a string type and actual value is empty, skip concating verKey.", j);
                continue;
            }
            addr = op::internal::PtrCastTo<const NnopbaseUChar>(attrs->attrs[j].addr.addr);
            length = attrs->attrs[j].addr.size;
            verKey = NnopbaseExecutor8ByteCopy(length, verKey, addr);
        } else {
            const size_t elementSize = attrs->attrs[j].addr.elementSize;
            if (elementSize == 0) {
                continue;
            }
            for (size_t i = 0U; i < attrs->attrs[j].addr.size / elementSize; i++) {
                addr = op::internal::PtrCastTo<const NnopbaseUChar>(attrs->attrs[j].addr.addr) + elementSize * i;
                verKey = NnopbaseExecutor8ByteCopy(elementSize, verKey, addr);
            }
        }
    }
    return verKey;
}

static size_t NnopbaseComputeStaticTensorsKeySize(const NnopbaseTensors* const tensors, const size_t tensorNum,
                                                  const bool usingStride, const bool withValueDepend)
{
    size_t len = 0U;
    for (size_t i = 0U; i < tensorNum; i++) {
        if (tensors->extTensors[i].isNull) {
            len += sizeof(uint64_t); // '_' 占位
            continue;
        }
        if (tensors->extTensors[i].isRequired || tensors->extTensors[i].isOptional) {
            len += NNOPBASE_DTYPE_AND_FORMAT_SIZE;
        }
        const GertShape& shape = tensors->extTensors[i].rt2Tensor.GetStorageShape();
        len += shape.GetDimNum() * sizeof(uint64_t);
        if (usingStride) {
            len += tensors->extTensors[i].rt2Tensor.GetStride().GetDimNum() * sizeof(uint64_t) + sizeof(uint64_t);
        }
        if (withValueDepend && tensors->extTensors[i].valueDepend) {
            len += (tensors->extTensors[i].rt2Tensor.GetSize() /
                    (op::TypeSize(tensors->extTensors[i].rt2Tensor.GetDataType()))) *
                   sizeof(uint64_t);
        }
    }
    return len;
}

static size_t NnopbaseComputeStaticAttrsKeySize(const NnopbaseAttrs* const attrs)
{
    size_t len = 0U;
    for (size_t i = 0U; i < attrs->num; i++) {
        if (!attrs->attrs[i].addr.isVector) {
            NnopbaseExecutorGet8ByteSize(attrs->attrs[i].addr.size, op::internal::PtrCastTo<uint32_t>(&len));
        } else {
            len += (attrs->attrs[i].addr.size / attrs->attrs[i].addr.elementSize) * sizeof(uint64_t);
        }
    }
    return len;
}

size_t NnopbaseExecutorComputeGenKeySize(const NnopbaseExecutor* const executor, bool usingStride)
{
    const NnopbaseTensors* const inputs = &executor->ownArgs.inputs;
    const NnopbaseAttrs* const attrs = &executor->attrs;
    const NnopbaseTensors* const outputs = &executor->ownArgs.outputs;
    size_t len = executor->regInfo->key.opType.size() + NNOPBASE_OP_VERB_HEAD_LEN * sizeof(uint64_t);
    OP_LOGD("Op %s after add [optype, deterministic and precision or performance], length is %zu.", executor->opType,
            len);
    const size_t inputTensorNum = static_cast<size_t>(inputs->nonDynamicCnt + inputs->dynamicCnt);
    len += NnopbaseComputeStaticTensorsKeySize(inputs, inputTensorNum, usingStride, true);
    OP_LOGD("Op %s after add input, length is %zu.", executor->opType, len);
    const size_t outputTensorNum = outputs->nonDynamicCnt + outputs->dynamicCnt;
    len += NnopbaseComputeStaticTensorsKeySize(outputs, outputTensorNum, usingStride, false);
    OP_LOGD("Op %s after add output, length is %zu.", executor->opType, len);
    len += NnopbaseComputeStaticAttrsKeySize(attrs);
    OP_LOGD("Op %s after add attr, length is %zu.", executor->opType, len);
    return len;
}

static NnopbaseUChar* NnopbaseInitStaticKeyBuffer(NnopbaseExecutor* const executor, const size_t len)
{
    NnopbaseUChar* verKey = nullptr;
    if (len > executor->binInfoKey.bufLen) {
        executor->binInfoKey.verbose = std::vector<NnopbaseUChar>(len + NNOPBASE_REPLAY_VERKEY_EXT_BUF_LEN, '\0');
        verKey = &(executor->binInfoKey.verbose[0U]);
        executor->binInfoKey.bufLen = static_cast<uint32_t>(len) + NNOPBASE_REPLAY_VERKEY_EXT_BUF_LEN;
        verKey = NnopbaseAppendBinary(verKey, static_cast<size_t>(executor->binInfoKey.bufLen),
                                      &(executor->regInfo->key.opType[0U]), executor->regInfo->key.opType.size());
    } else {
        verKey = &(executor->binInfoKey.verbose[0U]) + executor->regInfo->key.opType.size();
    }
    return verKey;
}

static NnopbaseUChar* NnopbaseAppendStaticKeyHead(const NnopbaseExecutor* const executor, NnopbaseUChar* verKey)
{
    uint64_t deterministicLevel = (executor->deterministicLevel == 0U) ? 0U : 1U;
    verKey = NnopbaseAppend8Byte(verKey, deterministicLevel);
    verKey = NnopbaseAppend8Byte(verKey, static_cast<uint64_t>(g_nnopbaseSysCfgParams.precision));
    OP_LOGI("Op %s HighPrecision is %d, Deterministic is %d, binInfoKey length is %u.", executor->opType,
            g_nnopbaseSysCfgParams.precision, deterministicLevel, executor->binInfoKey.bufLen);
    return verKey;
}

aclnnStatus NnopbaseExecutorGenStaticKey(NnopbaseExecutor* const executor, const bool usingStride)
{
    const NnopbaseTensors* const inputs = &executor->ownArgs.inputs;
    const NnopbaseAttrs* const attrs = &executor->attrs;
    const NnopbaseTensors* const outputs = &executor->ownArgs.outputs;
    /*
        when usingStride = false:
            verb key format = op_type/Deterministic/Precision/{dtype, format, (shape)}/.../attr binary
        when usingStride = true:
            verb key format = op_type/Deterministic/Precision/{dtype, format, (shape),(stride),offset}/.../attr binary
    */
    const size_t len = NnopbaseExecutorComputeGenKeySize(executor, usingStride);
    OP_LOGI("Op %s staticGenKey size %lu, usingStride = %d", executor->opType, len, usingStride);
    if (len > NNOPBASE_MAX_STATICKEY_LEN) {
        OP_LOGW("Op %s staticGenKey size is too large[%lu].", executor->opType, len);
        return ACLNN_ERR_PARAM_INVALID;
    }

    NnopbaseUChar* verKey = NnopbaseInitStaticKeyBuffer(executor, len);
    verKey = NnopbaseAppendStaticKeyHead(executor, verKey);
    const size_t inputTensorNum = inputs->nonDynamicCnt + inputs->dynamicCnt;
    verKey = NnopbaseExecutorGenTensorsKey(verKey, inputs, inputTensorNum, usingStride);
    const size_t outputTensorNum = static_cast<size_t>(outputs->nonDynamicCnt + outputs->dynamicCnt);
    verKey = NnopbaseExecutorGenTensorsKey(verKey, outputs, outputTensorNum, usingStride);
    verKey = NnopbaseExecutorGenAttrsKey(attrs, verKey);
    executor->binInfoKey.len = static_cast<uint32_t>(verKey - &(executor->binInfoKey.verbose[0U]));
    executor->binInfoKey.hashKey = NnopbaseHashBinary(&(executor->binInfoKey.verbose[0U]), executor->binInfoKey.len) %
                                   NNOPBASE_NORM_MAX_BIN_BUCKETS;
    return OK;
}

aclnnStatus NnopbaseExecutorSetUnContExecutor(NnopbaseExecutor* executor, aclOpExecutor* inExe, const size_t inWsSize)
{
    OP_LOGI("Op %s set UnContExe %p, executor addr is %p.", executor->opType, inExe, executor);
    executor->inUnContExe = inExe;
    executor->inUncontWsSize = inWsSize;
    if (inExe != nullptr) {
        if (executor->isCachedArgs) {
            executor->args->inputs.unContiguousTensors.workspaceOffsets = inExe->GetWorkspaceOffsets();
        } else {
            executor->ownArgs.inputs.unContiguousTensors.workspaceOffsets = inExe->GetWorkspaceOffsets();
        }
    }
    return OK;
}

aclnnStatus NnopbaseExecutorSetViewCopyExecutor(NnopbaseExecutor* executor, aclOpExecutor* exe)
{
    OP_LOGI("Op %s set viewCopyExe %p.", executor->opType, exe);
    executor->viewCopyExe = exe;
    return OK;
}

#ifdef __cplusplus
}
#endif
