/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tiling_context_to_json.h"
#include <map>
#include <string>
#include <cctype>
#include "register/op_impl_registry.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "graph/utils/type_utils.h"
#include "utils/indv_lib_wrapper.h"
#include "opdev/op_log.h"
#include "opdev/op_errno.h"
#include "opdev/platform.h"
#include "bin_to_json.h"

namespace aclnnOpInfoRecord {
namespace {
constexpr size_t STR_PRE = 3UL;
const char* const REQUIRED_STR = "required";
const char* const OPTIONAL_STR = "optional";
const char* const DYNAMIC_STR = "dynamic";
const char* const ATTR_DTYPE_STR = "str";
const char* const MC2_ATTR_GROUP_STR = "group";
constexpr size_t MAX_HCCL_NET_LAYER_NUM = 3U;

nlohmann::json StrAttrsToJson(const gert::RuntimeAttrs* attrs, size_t index)
{
    nlohmann::json j;
    if (attrs == nullptr) {
        OP_LOGW("Attr[%zu] is nullptr, skip dump.");
        return j;
    }
    j["value"] = attrs->GetStr(index);
    return j;
}

nlohmann::json BoolAttrsToJson(const gert::RuntimeAttrs* attrs, size_t index)
{
    nlohmann::json j;
    if (attrs == nullptr) {
        OP_LOGW("Attr[%zu] is nullptr, skip dump.");
        return j;
    }
    j["value"] = *attrs->GetBool(index);
    return j;
}

nlohmann::json IntAttrsToJson(const gert::RuntimeAttrs* attrs, size_t index)
{
    nlohmann::json j;
    if (attrs == nullptr) {
        OP_LOGW("Attr[%zu] is nullptr, skip dump.");
        return j;
    }
    j["value"] = *attrs->GetInt(index);
    return j;
}

nlohmann::json ListIntAttrsToJson(const gert::RuntimeAttrs* attrs, size_t index)
{
    nlohmann::json j;
    if (attrs == nullptr) {
        OP_LOGW("Attr[%zu] is nullptr, skip dump.");
        return j;
    }
    const auto* listInt = attrs->GetListInt(index);
    const int64_t* listIntPtr = listInt->GetData();
    std::vector<int64_t> vecInt;
    for (size_t idx = 0UL; idx < listInt->GetSize(); idx++) {
        vecInt.emplace_back(*(listIntPtr + idx));
    }
    j["value"] = vecInt;
    return j;
}

nlohmann::json FloatAttrsToJson(const gert::RuntimeAttrs* attrs, size_t index)
{
    nlohmann::json j;
    if (attrs == nullptr) {
        OP_LOGW("Attr[%zu] is nullptr, skip dump.");
        return j;
    }
    j["value"] = *attrs->GetFloat(index);
    return j;
}

nlohmann::json ListFloatAttrsToJson(const gert::RuntimeAttrs* attrs, size_t index)
{
    nlohmann::json j;
    if (attrs == nullptr) {
        OP_LOGW("Attr[%zu] is nullptr, skip dump.");
        return j;
    }
    const auto* listFloat = attrs->GetListFloat(index);
    const float* listFloatPtr = listFloat->GetData();
    std::vector<float> vecFloat;
    for (size_t idx = 0UL; idx < listFloat->GetSize(); idx++) {
        vecFloat.emplace_back(*(listFloatPtr + idx));
    }
    j["value"] = vecFloat;
    return j;
}

nlohmann::json ListBoolAttrsToJson(const gert::RuntimeAttrs* attrs, size_t index)
{
    nlohmann::json j;
    if (attrs == nullptr) {
        OP_LOGW("Attr[%zu] is nullptr, skip dump.");
        return j;
    }
    const auto* listBool = attrs->GetAttrPointer<gert::TypedContinuousVector<bool>>(index);
    const bool* listBoolPtr = listBool->GetData();
    std::vector<bool> vecBool;
    for (size_t idx = 0UL; idx < listBool->GetSize(); idx++) {
        vecBool.emplace_back(*(listBoolPtr + idx));
    }
    j["value"] = vecBool;
    return j;
}

const std::map<std::string, std::function<nlohmann::json(const gert::RuntimeAttrs*, size_t)>> ATTRS_TO_JSON = {
    {"str", StrAttrsToJson},
    {"string", StrAttrsToJson},
    {"bool", BoolAttrsToJson},
    {"int", IntAttrsToJson},
    {"int32", IntAttrsToJson},
    {"int64", IntAttrsToJson},
    {"float", FloatAttrsToJson},
    {"float32", FloatAttrsToJson},
    {"list_int", ListIntAttrsToJson},
    {"listInt", ListIntAttrsToJson},
    {"list_float", ListFloatAttrsToJson},
    {"list_bool", ListBoolAttrsToJson},
    {"listBool", ListBoolAttrsToJson},
};

std::vector<std::pair<std::string, std::string>> GetBuiltinIrFromJson(const nlohmann::json& supportInfoJsonConfig,
                                                                      bool isInput)
{
    std::vector<std::pair<std::string, std::string>> irDesc;
    std::string keyword = isInput ? "inputs" : "outputs";
    // keyword definitely exists
    try {
        for (size_t i = 0U; i < supportInfoJsonConfig[keyword].size(); ++i) {
            auto ele = supportInfoJsonConfig[keyword][i];
            if (ele.is_null()) {
                OP_LOGD("The %zuth element %s in supportInfo is null, skip.", i, keyword.c_str());
                irDesc.push_back({});
                continue;
            }
            if (ele.is_array()) {
                std::string paramType = ele[0].contains("paramType") ? ele[0]["paramType"] : "";
                irDesc.push_back({ele[0]["name"], paramType});
                OP_LOGD("The %zuth element %s in supportInfo is {name : %s, paramType: %s}", i, keyword.c_str(),
                        ele[0]["name"].get<std::string>().c_str(), paramType.c_str());
            } else {
                std::string paramType = ele.contains("paramType") ? ele["paramType"] : "";
                irDesc.push_back({ele["name"], paramType});
                OP_LOGD("The %zuth element %s in supportInfo is {name : %s, paramType: %s}", i, keyword.c_str(),
                        ele["name"].get<std::string>().c_str(), paramType.c_str());
            }
        }
    } catch (const nlohmann::json::exception& e) {
        OP_LOGE(ACLNN_ERR_INNER_LOAD_JSON_FAILED, "Read jsonfile failed, reason %s", e.what());
        return {};
    }
    return irDesc;
}

std::vector<std::pair<std::string, std::string>> GetBuiltinAttrFromJson(const nlohmann::json& supportInfoJsonConfig)
{
    if (!supportInfoJsonConfig.contains("attrs") || supportInfoJsonConfig["attrs"].size() == 0) {
        OP_LOGD("No Attrs in supportInfo.");
        return {};
    }
    std::vector<std::pair<std::string, std::string>> attrNames;
    try {
        auto& attrs = supportInfoJsonConfig["attrs"];
        for (size_t i = 0U; i < attrs.size(); ++i) {
            auto ele = attrs[i];
            if (ele.is_null()) {
                continue;
            }
            attrNames.push_back({ele["name"], ele["dtype"]});
        }
    } catch (const nlohmann::json::exception& e) {
        OP_LOGE(ACLNN_ERR_INNER_LOAD_JSON_FAILED, "Read jsonfile failed, reason %s", e.what());
        return {};
    }
    return attrNames;
}

struct OpIrDesc {
    std::string irName;
    size_t irLoc = 0U;
    std::string irType;
    bool isInput = true;
};

nlohmann::json TensorToJson(const OpIrDesc& opIrDesc, const gert::CompileTimeTensorDesc* cpTdInfo,
                            const size_t instanceLoc, const gert::TilingContext* ctx)
{
    auto opShape = opIrDesc.isInput ? ctx->GetInputShape(instanceLoc) : ctx->GetOutputShape(instanceLoc);
    if (cpTdInfo == nullptr || opShape == nullptr) {
        return nullptr;
    }
    nlohmann::json j;
    j["name"] = opIrDesc.irName;
    j["index"] = opIrDesc.irLoc;
    j["param_type"] = opIrDesc.irType;
    std::string dType = ge::TypeUtils::DataTypeToSerialString(cpTdInfo->GetDataType()).c_str() + STR_PRE;
    (void)std::transform(dType.begin(), dType.end(), dType.begin(), tolower);
    j["dtype"] = dType;

    j["shape"] = nlohmann::json::array();
    const auto& storageShape = opShape->GetStorageShape();
    for (size_t idx = 0UL; idx < storageShape.GetDimNum(); idx++) {
        j["shape"].emplace_back(storageShape.GetDim(idx));
    }

    j["format"] = ge::TypeUtils::FormatToSerialString(
        static_cast<ge::Format>(ge::GetPrimaryFormat(cpTdInfo->GetStorageFormat())));

    if (ge::HasSubFormat(cpTdInfo->GetStorageFormat())) {
        j["sub_format"] = ge::GetSubFormat(cpTdInfo->GetStorageFormat());
    }

    if (ge::HasC0Format(cpTdInfo->GetStorageFormat())) {
        j["c0_format"] = ge::GetC0Format(cpTdInfo->GetStorageFormat());
    }

    j["origin_shape"] = nlohmann::json::array();
    const auto& originShape = opShape->GetOriginShape();
    for (size_t idx = 0UL; idx < originShape.GetDimNum(); idx++) {
        j["origin_shape"].emplace_back(originShape.GetDim(idx));
    }

    j["origin_format"] = ge::TypeUtils::FormatToSerialString(cpTdInfo->GetOriginFormat());
    j["value_depend"] = false;
    bool isView = opIrDesc.isInput ? ctx->InputIsView(instanceLoc) : ctx->OutputIsView(instanceLoc);
    j["is_view"] = isView;
    auto stride = opIrDesc.isInput ? ctx->GetInputStride(instanceLoc) : ctx->GetOutputStride(instanceLoc);
    if (stride == nullptr) {
        OP_LOGW("Stride is nullptr! Detail: IsInput is %d, tensor instanceLoc is %zu.",
                static_cast<int32_t>(opIrDesc.isInput), instanceLoc);
        return j;
    }
    if (isView) {
        j["stride"] = nlohmann::json::array();
        for (size_t idx = 0UL; idx < stride->GetDimNum(); idx++) {
            j["stride"].emplace_back(stride->GetStride(idx));
        }
        j["offset"] = opIrDesc.isInput ? ctx->GetInputOffset(instanceLoc) : ctx->GetOutputOffset(instanceLoc);
    }
    return j;
}

nlohmann::json TransInputInstanceToJson(const gert::TilingContext* ctx, size_t& instanceLoc, const OpIrDesc& opIrDesc)
{
    const std::string irType = opIrDesc.irType;
    const std::size_t irLoc = opIrDesc.irLoc;
    const std::string irName = opIrDesc.irName;
    const auto computeNodeInfo = ctx->GetComputeNodeInfo();
    const auto inputInstanceInfo = computeNodeInfo->GetInputInstanceInfo(irLoc);
    OP_LOGD("ctx:%p, instanceLoc:%zu, irLoc:%zu, irName:%s, irType:%s, computeNodeInfo:%p, inputInstanceInfo:%p", ctx,
            instanceLoc, irLoc, irName.c_str(), irType.c_str(), computeNodeInfo, inputInstanceInfo);
    if (inputInstanceInfo != nullptr) {
        OP_LOGD("instanceNum:%zu, inputTdInfo:%p, inputShape:%p", inputInstanceInfo->GetInstanceNum(),
                computeNodeInfo->GetInputTdInfo(instanceLoc), ctx->GetInputShape(instanceLoc));
    }
    if (irName.empty() || irType.empty() || irType.compare(REQUIRED_STR) == 0 || irType.compare(OPTIONAL_STR) == 0) {
        if (inputInstanceInfo != nullptr && inputInstanceInfo->GetInstanceNum() != 0) {
            nlohmann::json tmpJ = TensorToJson(opIrDesc, computeNodeInfo->GetInputTdInfo(instanceLoc), instanceLoc,
                                               ctx);
            instanceLoc += inputInstanceInfo->GetInstanceNum();
            return tmpJ;
        }
        if (computeNodeInfo->GetInputTdInfo(instanceLoc) == nullptr || ctx->GetInputShape(instanceLoc) == nullptr) {
            instanceLoc++;
        }
        return nullptr;
    }

    if (irType.compare(DYNAMIC_STR) == 0) {
        nlohmann::json arrayJson = nlohmann::json::array();
        if (inputInstanceInfo == nullptr) {
            return arrayJson;
        }
        for (size_t idx = 0; idx < inputInstanceInfo->GetInstanceNum(); idx++) {
            arrayJson.emplace_back(
                TensorToJson(opIrDesc, computeNodeInfo->GetInputTdInfo(instanceLoc + idx), instanceLoc + idx, ctx));
        }
        instanceLoc += inputInstanceInfo->GetInstanceNum();
        return arrayJson;
    }
    OP_LOGE(ACLNN_ERR_INNER, "TransInputInstanceToJson not support input ir type[%s]!", irType.c_str());
    return nullptr;
}

const gert::OpImplKernelRegistry::OpImplFunctionsV2* GetOppImplement(const std::string& nodeType)
{
    // Start by searching in the default library kOpp.
    auto& registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    OP_CHECK(registry != nullptr, OP_LOGE(ACLNN_ERR_INNER, "Cannot get registry from gert with opp version."),
             return nullptr);
    const gert::OpImplKernelRegistry::OpImplFunctionsV2* opImplFunc = registry->GetOpImpl(nodeType.c_str());
    if (opImplFunc != nullptr) {
        return opImplFunc;
    }
    // Finally, search from Library kOppKernel.
    OP_LOGW("Start searching %s from opp kernel version registry.", nodeType.c_str());
    auto& kernelRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry(
        gert::OppImplVersionTag::kOppKernel);
    OP_CHECK(kernelRegistry != nullptr,
             OP_LOGE(ACLNN_ERR_INNER, "Cannot get registry from gert with opp kernel version."), return nullptr);
    return kernelRegistry->GetOpImpl(nodeType.c_str());
}

nlohmann::json TransOutputInstanceToJson(const gert::TilingContext* ctx, size_t& instanceLoc, const OpIrDesc& opIrDesc)
{
    const std::string irType = opIrDesc.irType;
    const std::size_t irLoc = opIrDesc.irLoc;
    const std::string irName = opIrDesc.irName;
    const auto computeNodeInfo = ctx->GetComputeNodeInfo();
    const auto outputInstanceInfo = computeNodeInfo->GetOutputInstanceInfo(irLoc);
    if (irName.empty() || irType.empty() || irType.compare(REQUIRED_STR) == 0 || irType.compare(OPTIONAL_STR) == 0) {
        if (outputInstanceInfo == nullptr) {
            instanceLoc++;
            return nullptr;
        }
        nlohmann::json tmpJ = TensorToJson(opIrDesc, computeNodeInfo->GetOutputTdInfo(instanceLoc), instanceLoc, ctx);
        instanceLoc += outputInstanceInfo->GetInstanceNum();
        return tmpJ;
    }
    if (irType.compare(DYNAMIC_STR) == 0) {
        nlohmann::json arrayJson = nlohmann::json::array();
        if (outputInstanceInfo == nullptr) {
            return arrayJson;
        }
        for (size_t idx = 0; idx < outputInstanceInfo->GetInstanceNum(); idx++) {
            arrayJson.emplace_back(
                TensorToJson(opIrDesc, computeNodeInfo->GetOutputTdInfo(instanceLoc + idx), instanceLoc + idx, ctx));
        }
        instanceLoc += outputInstanceInfo->GetInstanceNum();
        return arrayJson;
    }
    OP_LOGE(ACLNN_ERR_INNER, "TransOutputInstanceToJson not support output ir type[%s]!", irType.c_str());
    return nullptr;
}

aclnnStatus ConstructInputOutputJson(const gert::TilingContext* ctx, const nlohmann::json& supportInfoJsonConfig,
                                     const gert::ComputeNodeInfo* computeNodeInfo, nlohmann::json& j)
{
    auto opImplFunc = GetOppImplement(computeNodeInfo->GetNodeType());
    OP_CHECK(
        opImplFunc != nullptr,
        OP_LOGE(ACLNN_ERR_INNER, "[%s]TilingContextToJson get op impl func failed!", computeNodeInfo->GetNodeType()),
        return ACLNN_ERR_INNER);
    std::vector<std::pair<std::string, std::string>> inputIr = GetBuiltinIrFromJson(supportInfoJsonConfig, true);
    size_t inputInstanceLoc = 0;
    j["inputs"] = nlohmann::json::array();
    for (size_t irLoc = 0UL; irLoc < inputIr.size(); irLoc++) {
        const OpIrDesc opIrDesc{inputIr[irLoc].first, irLoc, inputIr[irLoc].second, true};
        nlohmann::json tmpJ = TransInputInstanceToJson(ctx, inputInstanceLoc, opIrDesc);
        if (!tmpJ.is_null()) {
            OP_LOGD("Check Tensor irLoc: %zu, inputInstanceLoc % zu"
                    " is InputDataDepend? %d, is TilingInputDataDepend %d.",
                    irLoc, inputInstanceLoc, opImplFunc->IsInputDataDependency(irLoc),
                    opImplFunc->IsTilingInputDataDependency(irLoc));
            if (tmpJ.is_object() &&
                (opImplFunc->IsInputDataDependency(irLoc) || opImplFunc->IsTilingInputDataDependency(irLoc)) &&
                inputInstanceLoc >= 1) {
                tmpJ["value_depend"] = true;
                ge::DataType dType;
                const void* tensorAddr = nullptr;
                size_t inputSize = 0U;
                if (ctx->InputIsView(inputInstanceLoc - 1)) {
                    const gert::TensorV2* tensor = reinterpret_cast<const gert::TensorV2*>(
                        ctx->GetInputTensor(inputInstanceLoc - 1));
                    if (tensor == nullptr) {
                        OP_LOGW("ValueDepend Tensor irLoc: %zu, inputInstanceLoc: %zu is nullptr!", irLoc,
                                inputInstanceLoc - 1);
                        return ACLNN_ERR_INNER;
                    }
                    dType = tensor->GetDataType();
                    tensorAddr = tensor->GetAddr();
                    inputSize = tensor->GetSize();
                } else {
                    const gert::Tensor* tensor = ctx->GetInputTensor(inputInstanceLoc - 1);
                    if (tensor == nullptr) {
                        OP_LOGW("ValueDepend Tensor irLoc: %zu, inputInstanceLoc: %zu is nullptr!", irLoc,
                                inputInstanceLoc - 1);
                        return ACLNN_ERR_INNER;
                    }
                    dType = tensor->GetDataType();
                    tensorAddr = tensor->GetAddr();
                    inputSize = tensor->GetSize();
                }
                const auto funcIter = BIN_TO_JSON.find(dType);
                if (funcIter == BIN_TO_JSON.cend()) {
                    OP_LOGE(ACLNN_ERR_INNER, "[%s]Not support type[%s] in const data translate!",
                            computeNodeInfo->GetNodeType(), ge::TypeUtils::DataTypeToSerialString(dType).c_str());
                    return ACLNN_ERR_INNER;
                }
                if (tensorAddr == nullptr) {
                    OP_LOGW("Current operator [%s] is empty!", computeNodeInfo->GetNodeType());
                    tmpJ["const_data"] = "NULL";
                } else {
                    tmpJ["const_data"] = funcIter->second(tensorAddr, inputSize);
                }
            }
        }
        OP_LOGD("[%s]inputs[%zu][%s]!", computeNodeInfo->GetNodeType(), irLoc, tmpJ.dump().c_str());
        j["inputs"].emplace_back(std::move(tmpJ));
    }
    std::vector<std::pair<std::string, std::string>> outputIr = GetBuiltinIrFromJson(supportInfoJsonConfig, false);
    size_t outputInstanceLoc = 0;
    j["outputs"] = nlohmann::json::array();
    for (size_t irLoc = 0UL; irLoc < outputIr.size(); irLoc++) {
        const OpIrDesc opIrDesc{outputIr[irLoc].first, irLoc, outputIr[irLoc].second, false};
        nlohmann::json tmpJ = TransOutputInstanceToJson(ctx, outputInstanceLoc, opIrDesc);
        OP_LOGD("[%s]outputs[%zu][%s]!", computeNodeInfo->GetNodeType(), irLoc, tmpJ.dump().c_str());
        j["outputs"].emplace_back(std::move(tmpJ));
    }
    return OK;
}

nlohmann::json AttrsToJson(const gert::ComputeNodeInfo* computeNodeInfo, const nlohmann::json& supportInfoJsonConfig)
{
    nlohmann::json nullJ(nullptr);
    const auto srcAttrs = computeNodeInfo->GetAttrs();
    if (srcAttrs == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER, "[%s]RuntimeAttrs is nullptr.", computeNodeInfo->GetNodeType());
        return nullJ;
    }
    auto irAttrNameTypes = GetBuiltinAttrFromJson(supportInfoJsonConfig);
    // The inconsistency between the RuntimeAttrs quantity and the prototype library quantity is not verified.
    size_t recordAttrLen = std::min(irAttrNameTypes.size(), srcAttrs->GetAttrNum());
    OP_LOGD("[%s]RuntimeAttrs size[%zu];irAttrNameTypes size[%zu].", computeNodeInfo->GetNodeType(),
            srcAttrs->GetAttrNum(), irAttrNameTypes.size());

    std::vector<nlohmann::json> attrs;
    for (size_t idx = 0UL; idx < recordAttrLen; idx++) {
        const auto funcIter = ATTRS_TO_JSON.find(irAttrNameTypes[idx].second);
        if (funcIter == ATTRS_TO_JSON.cend()) {
            OP_LOGE(ACLNN_ERR_INNER, "[%s]Not support type in attrs[%s][%zu]!", computeNodeInfo->GetNodeType(),
                    irAttrNameTypes[idx].second.c_str(), idx);
            return nullJ;
        }
        nlohmann::json tmpJ = funcIter->second(srcAttrs, idx);
        tmpJ["name"] = irAttrNameTypes[idx].first;
        tmpJ["dtype"] = irAttrNameTypes[idx].second;
        tmpJ["index"] = idx;
        OP_LOGD("[%s]attrs[%zu][%s]!", computeNodeInfo->GetNodeType(), idx, tmpJ.dump().c_str());
        attrs.emplace_back(std::move(tmpJ));
    }
    return attrs;
}

std::vector<std::pair<std::string, std::string>> GetMc2AttrGroupName(const nlohmann::json& attrs)
{
    std::vector<std::pair<std::string, std::string>> mc2AttrActualGroupNames;
    for (size_t i = 0U; i < attrs.size(); ++i) {
        const auto& attr = attrs[i];
        if (attr.is_null()) {
            continue;
        }
        std::string attrDtype = attr["dtype"].get<std::string>();
        if (attrDtype.compare(ATTR_DTYPE_STR) != 0) {
            continue;
        }
        const std::string attrName = attr["name"].get<std::string>();
        if (attrName.find(MC2_ATTR_GROUP_STR) != 0) {
            continue;
        }
        const std::string actualGroupName = attr["value"].get<std::string>();
        if (actualGroupName.empty()) {
            OP_LOGW("Mc2 operator attribute %s has no value.", attrName.c_str());
            continue;
        }
        OP_LOGI("Add (%s, %s) to get hccl topology information.", attrName.c_str(), actualGroupName.c_str());
        mc2AttrActualGroupNames.push_back({attrName, actualGroupName});
    }
    return mc2AttrActualGroupNames;
}

void AddExtraInfoToOpJson(nlohmann::json& opJson, const nlohmann::json& attrs)
{
    nlohmann::json topoInfosJson;
    auto mc2AttrActualGroupNames = GetMc2AttrGroupName(attrs);
    for (const auto& it : mc2AttrActualGroupNames) {
        auto attrName = it.first;
        auto actualGroupName = it.second;
        nlohmann::json topoInfoJson;
        HcclComm commHandle;
        OP_CHECK(nnopbase::IndvHcclWrapper::GetInstance().HcomGetCommHandleByGroup(actualGroupName.c_str(),
                                                                                   &commHandle) == OK,
                 OP_LOGE(ACLNN_ERR_INNER, "Call HcomGetCommHandleByGroup failed during dumping topo info of attr %s.",
                         attrName.c_str()),
                 return);
        uint32_t topoType = 0U;
        uint32_t netLayerNum = 0U;
        uint32_t gRankSize = 0U;
        uint32_t* netLayer = nullptr;
        OP_CHECK(
            nnopbase::IndvHcclWrapper::GetInstance().HcclRankGraphGetLayers(commHandle, &netLayer, &netLayerNum) == OK,
            OP_LOGE(ACLNN_ERR_INNER, "Call HcclRankGraphGetLayers failed during dumping topo info of attr %s.",
                    attrName.c_str()),
            return);
        nlohmann::json topoLevelDescs = nlohmann::json::array();
        for (uint32_t j = 0U; j < MAX_HCCL_NET_LAYER_NUM; j++) {
            nlohmann::json topoLevelDesc;
            if (j < netLayerNum) {
                OP_CHECK(nnopbase::IndvHcclWrapper::GetInstance().HcclRankGraphGetRankSizeByLayer(
                             commHandle, netLayer[j], &gRankSize) == OK,
                         OP_LOGE(ACLNN_ERR_INNER,
                                 "Call HcclRankGraphGetRankSizeByLayer failed during dumping topo info of attr %s.",
                                 attrName.c_str()),
                         return);
                OP_CHECK(nnopbase::IndvHcclWrapper::GetInstance().HcclRankGraphGetTopoTypeByLayer(
                             commHandle, netLayer[j], &topoType) == OK,
                         OP_LOGE(ACLNN_ERR_INNER,
                                 "Call HcclRankGraphGetTopoTypeByLayer failed during dumping topo info of attr %s.",
                                 attrName.c_str()),
                         return);
                topoLevelDesc["comm_sets"] = static_cast<uint32_t>(topoType);
                topoLevelDesc["rank_size"] = gRankSize;
            } else {
                // 无效部分填充0
                topoLevelDesc["comm_sets"] = 0U;
                topoLevelDesc["rank_size"] = 0U;
            }
            topoLevelDescs.push_back(topoLevelDesc);
        }
        uint64_t localWindowSize;
        void* buffer = nullptr;
        OP_CHECK(
            nnopbase::IndvHcclWrapper::GetInstance().HcclGetHcclBuffer(commHandle, &buffer, &localWindowSize) == OK,
            OP_LOGE(ACLNN_ERR_INNER, "Call HcclGetHcclBuffer failed during dumping topo info of attr %s.",
                    attrName.c_str()),
            return);
        uint32_t rankSize = 0U;
        OP_CHECK(nnopbase::IndvHcclWrapper::GetInstance().HcclGetRankSize(commHandle, &rankSize) == OK,
                 OP_LOGE(ACLNN_ERR_INNER, "Call HcclGetHcclBuffer failed during dumping topo info of attr %s.",
                         attrName.c_str()),
                 return);
        topoInfoJson["topo_level_descs"] = topoLevelDescs;
        topoInfoJson["local_window_size"] = localWindowSize / 2; // cclBufferSize需要做除2处理
        topoInfoJson["rank_size"] = rankSize;
        topoInfosJson[attrName] = topoInfoJson;
    }
    if (topoInfosJson.is_null()) {
        return;
    }
    opJson["extra_params"] = topoInfosJson;
    return;
}
} // namespace

nlohmann::json TilingContextToJson(const gert::TilingContext* ctx,
                                   const std::map<std::string, std::string>& iniConfigMap,
                                   const nlohmann::json& supportInfoJsonConfig, bool isMc2)
{
    const auto computeNodeInfo = ctx->GetComputeNodeInfo();
    nlohmann::json nullJ(nullptr);
    if (computeNodeInfo == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER, "TilingContextToJson computeNodeInfo is nullptr!");
        return nullJ;
    }
    if (computeNodeInfo->GetNodeType() == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER, "TilingContextToJson node type is nullptr!");
        return nullJ;
    }
    std::string opType = computeNodeInfo->GetNodeType();
    nlohmann::json opJson;
    opJson["op_type"] = opType;
    OP_LOGI("TilingContextToJson get op[%s] to serialize!", computeNodeInfo->GetNodeName());

    const auto& iter = iniConfigMap.find(opType);
    if (iter != iniConfigMap.cend()) {
        opJson["tune_mode"] = iter->second;
    } else {
        opJson["tune_mode"] = "all";
    }

    auto ret = ConstructInputOutputJson(ctx, supportInfoJsonConfig, computeNodeInfo, opJson);
    if (ret != OK) {
        OP_LOGE(ACLNN_ERR_INNER, "Failed to construct input/output/attr.");
        return nullJ;
    }
    nlohmann::json attrs = AttrsToJson(computeNodeInfo, supportInfoJsonConfig);
    if (attrs.is_null()) {
        return nullJ;
    }
    opJson["attrs"] = attrs;
    if (isMc2) {
        AddExtraInfoToOpJson(opJson, attrs);
    }
    return opJson;
}
} // namespace aclnnOpInfoRecord