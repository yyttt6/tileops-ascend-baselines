/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "op_kernel.h"
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <cerrno>
#include <system_error>

#include "acl/acl_base.h"
#include "acl/acl_rt.h"
#include "register/op_binary_resource_manager.h"

#include "opdev/data_type_utils.h"
#include "kernel_utils.h"
#include "opdev/op_errno.h"
#include "opdev/shape_utils.h"
#include "op_info_serialize.h"
#include "nnopbase_error_msg.h"

namespace op {
namespace internal {
using std::ifstream;
using std::string;
using json = nlohmann::json;
namespace {
constexpr const char* NULL_STR = "null";
constexpr const char* BIN_SUFFIX = ".o";
constexpr const char* JSON_SUFFIX = ".json";
constexpr const char* BIN_LIST = "binList";
constexpr const char* BIN_INFO = "binInfo";
constexpr const char* JSON_FILE_PATH = "jsonFilePath";
constexpr const char* BIN_PATH = "binPath";
constexpr const char* INPUTS = "inputs";
constexpr const char* OUTPUTS = "outputs";
constexpr const char* ATTRS = "attrs";
constexpr const char* PARAM_TYPE = "paramType";
constexpr const char* TYPE = "type";
constexpr const char* DATA_TYPE = "dtype";
constexpr const char* SHAPE = "shape";
constexpr const char* DTYPE_MATCH_MODE = "dtype_match_mode";
constexpr const char* DTYPE_BYTE = "DtypeByte";
constexpr const char* FORMAT = "format";
constexpr const char* DETERMINISTIC = "deterministic";
constexpr const char* IMPL_MODE = "implMode";
constexpr const char* SIMPLIFIED_KEY_MODE = "simplifiedKeyMode";
constexpr const char* SIMPLIFIED_KEY = "simplifiedKey";
constexpr const char* NAME = "name";
constexpr const char* VALUE = "value";
constexpr const char* ALL = "ALL";
constexpr const char* OP_KERNEL_TRUE = "true";
constexpr const char* OP_KERNEL_FALSE = "false";
constexpr const char* IGNORE = "ignore";
constexpr const char* STATIC_LIST = "staticList";
constexpr const char* VALUE_DEPEND_INDEX = "valueDependIndex";
constexpr const char* OPTIONAL_INPUT_MODE = "optionalInputMode";
constexpr const char* GEN_PLACEHOLDER = "gen_placeholder";
constexpr const char* DYNAMIC_PARAM_MODE = "dynamicParamMode";
constexpr const char* FOLDED_WITH_DESC = "folded_with_desc";
constexpr const char* OPERATOR = "/";

constexpr const char* INT = "int";
constexpr const char* FLOAT = "float";
constexpr const char* BOOL = "bool";
constexpr const char* STRING = "string";
constexpr const char* LIST_INT = "list_int";
constexpr const char* LIST_FLOAT = "list_float";

constexpr size_t MAX_BIN_MATCH_ARG_KEY_COUNT = 2; // datatype, formats
constexpr uint32_t PRINT_ARGS_STEP = 4;
constexpr int64_t UNKNOWN_DIM_NUM = -2;
constexpr int64_t CUSTOMIZED_SIMPLIFIED_KEY = 2;
constexpr int64_t IGNORE_ATTR_SIMPLIFIED_KEY = 0;
constexpr size_t MAX_CUSTOMIZED_SIMPLIFIED_KEY_LEN = 256;
constexpr char const*
    ALL_PRECISION_MODE = "high_performance,high_precision,enable_float_32_execution,enable_hi_float_32_execution";

constexpr uint64_t OP_API_HASH_SEED = 0x9e3779b9U;
constexpr uint32_t KERNEL_RATION_TWO = 2;

} // namespace

const std::map<std::string, TensorType> STR_2_TENSOR_TYPE{{"required", TensorType::REQUIRED},
                                                          {"optional", TensorType::OPTIONAL},
                                                          {"dynamic", TensorType::DYNAMIC},
                                                          {"folded_with_desc", TensorType::DYNAMIC_FOLDED}};

const std::map<std::string, ge::DataType> STR_2_DATA_TYPE{{"float", ge::DT_FLOAT},
                                                          {"float32", ge::DT_FLOAT},
                                                          {"float16", ge::DT_FLOAT16},
                                                          {"bfloat16", ge::DT_BF16},
                                                          {"int8", ge::DT_INT8},
                                                          {"int16", ge::DT_INT16},
                                                          {"uint16", ge::DT_UINT16},
                                                          {"uint8", ge::DT_UINT8},
                                                          {"int32", ge::DT_INT32},
                                                          {"int64", ge::DT_INT64},
                                                          {"uint32", ge::DT_UINT32},
                                                          {"uint64", ge::DT_UINT64},
                                                          {"bool", ge::DT_BOOL},
                                                          {"double", ge::DT_DOUBLE},
                                                          {"string", ge::DT_STRING},
                                                          {"bf16", ge::DT_BF16},
                                                          {"int4", ge::DT_INT4},
                                                          {"uint1", ge::DT_UINT1},
                                                          {"int2", ge::DT_INT2},
                                                          {"uint2", ge::DT_UINT2},
                                                          {"complex32", ge::DT_COMPLEX32},
                                                          {"complex64", ge::DT_COMPLEX64},
                                                          {"complex128", ge::DT_COMPLEX128},
                                                          {"hifloat8", ge::DT_HIFLOAT8},
                                                          {"float8_e5m2", ge::DT_FLOAT8_E5M2},
                                                          {"float8_e4m3fn", ge::DT_FLOAT8_E4M3FN},
                                                          {"float8_e8m0", DataType::DT_FLOAT8_E8M0},
                                                          {"float6_e3m2", DataType::DT_FLOAT6_E3M2},
                                                          {"float6_e2m3", DataType::DT_FLOAT6_E2M3},
                                                          {"float4_e2m1", DataType::DT_FLOAT4_E2M1},
                                                          {"float4_e1m2", DataType::DT_FLOAT4_E1M2}};

const std::map<ge::DataType, std::string> DATA_TYPE_2_STR{{ge::DT_FLOAT, "float"},
                                                          {ge::DT_FLOAT, "float32"},
                                                          {ge::DT_FLOAT16, "float16"},
                                                          {ge::DT_INT8, "int8"},
                                                          {ge::DT_INT16, "int16"},
                                                          {ge::DT_UINT16, "uint16"},
                                                          {ge::DT_UINT8, "uint8"},
                                                          {ge::DT_INT32, "int32"},
                                                          {ge::DT_INT64, "int64"},
                                                          {ge::DT_UINT32, "uint32"},
                                                          {ge::DT_UINT64, "uint64"},
                                                          {ge::DT_BOOL, "bool"},
                                                          {ge::DT_DOUBLE, "double"},
                                                          {ge::DT_STRING, "string"},
                                                          {ge::DT_BF16, "bf16"},
                                                          {ge::DT_INT4, "int4"},
                                                          {ge::DT_UINT1, "uint1"},
                                                          {ge::DT_INT2, "int2"},
                                                          {ge::DT_UINT2, "uint2"},
                                                          {ge::DT_COMPLEX32, "complex32"},
                                                          {ge::DT_COMPLEX64, "complex64"},
                                                          {ge::DT_COMPLEX128, "complex128"},
                                                          {ge::DT_HIFLOAT8, "hifloat8"},
                                                          {ge::DT_FLOAT8_E5M2, "float8_e5m2"},
                                                          {ge::DT_FLOAT8_E4M3FN, "float8_e4m3fn"},
                                                          {DataType::DT_FLOAT8_E8M0, "float8_e8m0"},
                                                          {DataType::DT_FLOAT6_E3M2, "float6_e3m2"},
                                                          {DataType::DT_FLOAT6_E2M3, "float6_e2m3"},
                                                          {DataType::DT_FLOAT4_E2M1, "float4_e2m1"},
                                                          {DataType::DT_FLOAT4_E1M2, "float4_e1m2"}};

const std::map<std::string, ge::Format> STR_2_FORMAT{{"ND", ge::FORMAT_ND},
                                                     {"ALL", ge::FORMAT_ND},
                                                     {"NCHW", ge::FORMAT_NCHW},
                                                     {"NHWC", ge::FORMAT_NHWC},
                                                     {"HWCN", ge::FORMAT_HWCN},
                                                     {"NDHWC", ge::FORMAT_NDHWC},
                                                     {"NCDHW", ge::FORMAT_NCDHW},
                                                     {"DHWCN", ge::FORMAT_DHWCN},
                                                     {"NDC1HWC0", ge::FORMAT_NDC1HWC0},
                                                     {"FRACTAL_NZ", ge::FORMAT_FRACTAL_NZ},
                                                     {"NC1HWC0", ge::FORMAT_NC1HWC0},
                                                     {"FRACTAL_Z", ge::FORMAT_FRACTAL_Z},
                                                     {"C1HWNCoC0", ge::FORMAT_C1HWNCoC0},
                                                     {"FRACTAL_Z_C04", ge::FORMAT_FRACTAL_Z_C04},
                                                     {"FRACTAL_Z_3D", ge::FORMAT_FRACTAL_Z_3D},
                                                     {"FRACTAL_NZ_C0_16", ge::FORMAT_FRACTAL_NZ_C0_16},
                                                     {"FRACTAL_NZ_C0_32", ge::FORMAT_FRACTAL_NZ_C0_32},
                                                     {"FRACTAL_NZ_C0_2", ge::FORMAT_FRACTAL_NZ_C0_2},
                                                     {"FRACTAL_NZ_C0_4", ge::FORMAT_FRACTAL_NZ_C0_4},
                                                     {"FRACTAL_NZ_C0_8", ge::FORMAT_FRACTAL_NZ_C0_8}};

const std::map<ge::Format, std::string> FORMAT_2_STR{{ge::FORMAT_ND, "ND"},
                                                     {ge::FORMAT_NCHW, "NCHW"},
                                                     {ge::FORMAT_NHWC, "NHWC"},
                                                     {ge::FORMAT_HWCN, "HWCN"},
                                                     {ge::FORMAT_NCDHW, "NCDHW"},
                                                     {ge::FORMAT_NDHWC, "NDHWC"},
                                                     {ge::FORMAT_DHWCN, "DHWCN"},
                                                     {ge::FORMAT_NDC1HWC0, "NDC1HWC0"},
                                                     {ge::FORMAT_FRACTAL_NZ, "FRACTAL_NZ"},
                                                     {ge::FORMAT_NC1HWC0, "NC1HWC0"},
                                                     {ge::FORMAT_FRACTAL_Z, "FRACTAL_Z"},
                                                     {ge::FORMAT_C1HWNCoC0, "C1HWNCoC0"},
                                                     {ge::FORMAT_FRACTAL_Z_C04, "FRACTAL_Z_C04"},
                                                     {ge::FORMAT_FRACTAL_Z_3D, "FRACTAL_Z_3D"},
                                                     {ge::FORMAT_FRACTAL_NZ_C0_16, "FRACTAL_NZ_C0_16"},
                                                     {ge::FORMAT_FRACTAL_NZ_C0_32, "FRACTAL_NZ_C0_32"},
                                                     {ge::FORMAT_FRACTAL_NZ_C0_2, "FRACTAL_NZ_C0_2"},
                                                     {ge::FORMAT_FRACTAL_NZ_C0_4, "FRACTAL_NZ_C0_4"},
                                                     {ge::FORMAT_FRACTAL_NZ_C0_8, "FRACTAL_NZ_C0_8"}};

thread_local int OpKernelBin::currDevId_;
thread_local std::vector<MemSetTensorInfo> OpKernelBin::memSetValueCtx_;

ge::DataType GetDataType(const string& dataType)
{
    auto iter = STR_2_DATA_TYPE.find(dataType);
    if (iter != STR_2_DATA_TYPE.end()) {
        return iter->second;
    }
    OP_LOGW("Data type %s is invalid.", dataType.c_str());
    return ge::DT_UNDEFINED;
}

ge::Format GetFormat(const string& format)
{
    auto iter = STR_2_FORMAT.find(format);
    if (iter != STR_2_FORMAT.end()) {
        return iter->second;
    }
    OP_LOGW("Format %s is invalid.", format.c_str());
    return ge::FORMAT_RESERVED;
}

TensorType GetTensorType(const string& tensorType)
{
    auto iter = STR_2_TENSOR_TYPE.find(tensorType);
    if (iter != STR_2_TENSOR_TYPE.end()) {
        return iter->second;
    }
    OP_LOGW("Tensor type %s is invalid.", tensorType.c_str());
    return TensorType::BOTTOM;
}

/* This is a special operation for node TransData because:
 * the TransData of format NCHW <---> FRACTAL_Z need to designate a
 * C0 value and it currently uses last dimension of shape to differentiate two binary.
 * We need to get the last dimension of shape and use it as a key for
 * binary matching. */
static void ParseC0(const nlohmann::json& tensor, std::string& key)
{
    auto shape = tensor.find(SHAPE);
    if (shape != tensor.end() && shape->is_array() && shape->size() == FRACTAL_Z_SIZE) {
        wchar_t c0 = static_cast<wchar_t>(shape->get<FVector<int64_t, MAX_DIM_NUM>>().at(FRACTAL_Z_SIZE - 1));
        key += (MAX_VALID_DTYPE_FORMAT_KEY + c0);
        OP_LOGD("C0 = %ld.", shape->get<FVector<int64_t, MAX_DIM_NUM>>().at(FRACTAL_Z_SIZE - 1));
    }
}

static void ParseShapeDimNum(const TensorInfo& tensorInfo, const nlohmann::json& tensor, std::string& key)
{
    if (tensorInfo.shapeSupportType == ShapeSupportType::NOT_SUPPORT_ALL) {
        auto shape = tensor.find(SHAPE);
        if (shape != tensor.end() && shape->is_array()) {
            auto dimNum = static_cast<wchar_t>(shape->size());
            key += (MAX_VALID_DTYPE_FORMAT_KEY + dimNum);
            OP_LOGD("Shape dim num = %zu.", shape->size());
        } else {
            OP_LOGW("shape is invalid.");
        }
    }
}

aclnnStatus OpKernel::AssembleKeyForSingleTensor(const nlohmann::json& tensor, std::string& key, TensorInfo& tensorInfo)
{
    if (tensorInfo.dtMatchMode == DtMatchMode::NORMAL) {
        auto dtypeMatchMode = tensor.find(DTYPE_MATCH_MODE);
        if (dtypeMatchMode != tensor.end()) {
            if (static_cast<string>(*dtypeMatchMode) == DTYPE_BYTE) {
                tensorInfo.dtMatchMode = DtMatchMode::DTYPE_BYTE;
            }
        }
    }

    auto dataType = tensor.find(DATA_TYPE);
    if (dataType != tensor.end()) {
        auto dataTypeEnum = GetDataType(static_cast<string>(*dataType));
        if (dataTypeEnum == ge::DT_UNDEFINED) {
            OP_LOGW("Dtype is invalid for tensor %s.", tensor.dump().c_str());
            return ACLNN_ERR_INNER_JSON_DTYPE_INVALID;
        }

        if (tensorInfo.dtMatchMode == DtMatchMode::DTYPE_BYTE) {
            auto typeSize = op::TypeSize(dataTypeEnum);
            OP_LOGD("Tensor is in dtype byte mode, its type size is %zu, dtype %d.", typeSize, dataTypeEnum);
            if (typeSize <= MAX_VALID_DTYPE_SIZE) {
                key += static_cast<char>(typeSize);
            } else {
                key += static_cast<char>(typeSize - kDataTypeSizeBitOffset + MAX_VALID_DTYPE_SIZE);
            }
        } else {
            key += static_cast<char>(dataTypeEnum);
        }
    }

    auto format = tensor.find(FORMAT);
    ge::Format formatEnum = ge::FORMAT_RESERVED;
    if (format != tensor.end()) {
        formatEnum = GetFormat(static_cast<string>(*format));
        if (formatEnum == ge::FORMAT_RESERVED) {
            OP_LOGW("Format is invalid for tensor %s.", tensor.dump().c_str());
            return ACLNN_ERR_INNER_JSON_FORMAT_INVALID;
        }
        tensorInfo.fmtInfo.supportFormats.emplace(formatEnum);
        key += static_cast<char>(formatEnum);
    }

    if (opType_ == BinConfigJsonDict::transDataId_ && formatEnum == ge::FORMAT_FRACTAL_Z) {
        ParseC0(tensor, key);
    }

    ParseShapeDimNum(tensorInfo, tensor, key);
    return ACLNN_SUCCESS;
}

aclnnStatus OpKernelBin::BinLoad()
{
    CHECK_RET_CODE(aclrtGetDevice(&currDevId_), "aclrtGetDevice failed.");
    if (currDevId_ >= MAX_DEV_NUM) {
        OP_LOGE(ACLNN_ERR_INNER, "Invalid DeviceId %d. Max DeviceID: %d", currDevId_, MAX_DEV_NUM);
        return ACLNN_ERR_INNER;
    }

    auto f = [this](aclrtBinHandle& hdl) -> aclnnStatus { return BinLoadImpl(hdl); };
    return binHandle_[currDevId_].InitVar(f);
}

aclnnStatus OpKernelBin::GetBinData()
{
    std::tuple<nlohmann::json, nnopbase::Binary> binInfo;
    auto ret = nnopbase::OpBinaryResourceManager::GetInstance().GetOpBinaryDescByPath(relativeJsonPath_.c_str(),
                                                                                      binInfo);
    if (ret == ACLNN_SUCCESS) {
        auto f = [&binInfo](std::string& binData) -> aclnnStatus {
            auto& binary = std::get<1>(binInfo);
            if (binary.content == nullptr || binary.len <= 0) {
                return ACLNN_ERR_INNER;
            }
            try {
                binData = std::string(reinterpret_cast<const char*>(binary.content), static_cast<size_t>(binary.len));
            } catch (const std::bad_alloc&) {
                return ACLNN_ERR_INNER;
            }
            return ACLNN_SUCCESS;
        };
        ret = binData_.InitVar(f);
        if (ret != ACLNN_SUCCESS) {
            OP_LOGE(ACLNN_ERR_INNER, "Failed to load builtin kernel binary [%s], data is null/empty or out of memory.",
                    relativeJsonPath_.c_str());
            return ret;
        }
        OP_LOGI("Get builtin op kernel bin obj [%s]", relativeJsonPath_.c_str());
        return ACLNN_SUCCESS;
    }
    OP_LOGW("No builtin op kernel bin obj [%s]", relativeJsonPath_.c_str());

    auto f = [this](std::string& binData) -> aclnnStatus { return ReadFile2String(binPath_.c_str(), binData); };
    ret = binData_.InitVar(f);
    if (ret != ACLNN_SUCCESS) {
        std::string errMsg = "[Errno " + std::to_string(errno) + "] " + std::generic_category().message(errno);
        OP_LOGE_FOR_FILE_OPERATION_ERROR_PARSE(binPath_.c_str(), errMsg.c_str());
        return ret;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus OpKernelBin::BinLoadImpl(aclrtBinHandle& binHandle)
{
    auto ret = GetBinData();
    CHECK_COND(ret == ACLNN_SUCCESS, ACLNN_ERR_INNER, "Failed to get op kernel bin data [%s]", binPath_.c_str());

    static std::mutex rtsLock;

    // binary loading rts api are reported not-thread-safe by tsan.
    const std::lock_guard<std::mutex> lock(rtsLock);

    uint32_t magic = ACL_RT_BINARY_MAGIC_ELF_AICORE;
    auto& opJson = binJson_.GetVar();
    if (opJson.contains("magic")) {
        const auto& str = opJson["magic"].get<std::string>();
        if (str == "RT_DEV_BINARY_MAGIC_ELF_AICUBE") {
            magic = ACL_RT_BINARY_MAGIC_ELF_CUBE_CORE;
        } else if (str == "RT_DEV_BINARY_MAGIC_ELF_AIVEC" || str == "FFTS_BINARY_MAGIC_ELF_MIX_AIC" ||
                   str == "FFTS_BINARY_MAGIC_ELF_MIX_AIV") {
            magic = ACL_RT_BINARY_MAGIC_ELF_VECTOR_CORE;
        }
    }

    aclrtBinaryLoadOption magicLoadOption;
    magicLoadOption.type = ACL_RT_BINARY_LOAD_OPT_MAGIC;
    magicLoadOption.value.magic = magic;
    aclrtBinaryLoadOptions loadOptions = {&magicLoadOption, 1};

    CHECK_COND(aclrtBinaryLoadFromData(binData_.GetVar().c_str(), binData_.GetVar().size(), &loadOptions, &binHandle) ==
                   ACL_SUCCESS,
               ACLNN_ERR_RUNTIME_ERROR, "aclrtBinaryLoadFromData failed, bin path: %s", binPath_.c_str());

    OP_LOGD("Register Kernel succ on dev[%d], binHandle: %p, bin path: %s", currDevId_, binHandle, binPath_.c_str());
    return OK;
}

aclnnStatus OpKernelBin::InitFunctionHandle(bool isLaunchWithTilingKey, uint64_t tilingKey)
{
    const std::lock_guard<std::mutex> getFuncHandleLock(funcHandleMutex_);
    if (isLaunchWithTilingKey) {
        auto f = [this, tilingKey](aclrtFuncHandle& hdl) -> aclnnStatus {
            aclrtBinHandle binHandle = binHandle_[currDevId_].GetVar();
            CHECK_COND(aclrtBinaryGetFunctionByEntry(binHandle, tilingKey, &hdl) == ACL_SUCCESS,
                       ACLNN_ERR_RUNTIME_ERROR,
                       "aclrtBinaryGetFunctionByEntry failed, kernel name: %s, tiling key: %lu",
                       op::OpTypeDict::ToString(opType_).GetString(), tilingKey);
            OP_LOGI("Get function handle by tiling key [%lu] successfully, function handle: %p", tilingKey, hdl);
            return ACLNN_SUCCESS;
        };
        return funcHandleWithTilingKey_[currDevId_][tilingKey].InitVar(f);
    }
    auto f = [this](aclrtFuncHandle& hdl) -> aclnnStatus {
        aclrtBinHandle binHandle = binHandle_[currDevId_].GetVar();
        auto& opJson = binJson_.GetVar();
        if (!opJson.contains("kernelName")) {
            std::string reason = "The operator JSON file does not contain the kernel name";
            OP_LOGE_FOR_FILE_OPERATION_ERROR_PARSE_WITH_INVALID_CONTENT(jsonPath_.c_str(), reason.c_str());
            return ACLNN_ERR_INNER_JSON_VALUE_NOT_FOUND;
        }
        kernelNameOfNoFatBin_ = opJson["kernelName"].get<std::string>();
        CHECK_COND(aclrtBinaryGetFunction(binHandle, kernelNameOfNoFatBin_.c_str(), &hdl) == ACL_SUCCESS,
                   ACLNN_ERR_RUNTIME_ERROR, "aclrtBinaryGetFunction failed, kernel name: %s",
                   kernelNameOfNoFatBin_.c_str());
        OP_LOGI("Get function handle by kernel name [%s] successfully, function handle: %p",
                kernelNameOfNoFatBin_.c_str(), hdl);
        return ACLNN_SUCCESS;
    };
    return funcHandleWithKernelName_[currDevId_].InitVar(f);
}

void GetTaskInfoMultiKernelInfo(TaskInfo& info, const nlohmann::json& elem)
{
    const string kernelType = elem["kernelType"].get<std::string>();
    const string taskRation = elem["taskRation"].get<std::string>();
    const uint32_t crossCoreSync = (elem["crossCoreSync"].is_number_integer()) ?
                                       (elem["crossCoreSync"].get<uint32_t>()) :
                                       0;
    OP_LOGI("get info from json, kernelType is: %s, taskRation is: %s, crossCoreSync is: %u", kernelType.c_str(),
            taskRation.c_str(), crossCoreSync);
    if (taskRation == "1:0" || taskRation == "0:1") {
        if (crossCoreSync == 1) {
            info.type = (taskRation == "0:1") ? MSPROF_GE_TASK_TYPE_MIX_AIV : MSPROF_GE_TASK_TYPE_MIX_AIC;
        } else {
            info.type = (taskRation == "0:1") ? MSPROF_GE_TASK_TYPE_AIV : MSPROF_GE_TASK_TYPE_AI_CORE;
        }
        info.ration = 0;
    } else {
        if (kernelType == "MIX_AIC") {
            info.type = MSPROF_GE_TASK_TYPE_MIX_AIC;
        } else {
            info.type = MSPROF_GE_TASK_TYPE_MIX_AIV;
        }
        info.ration = (taskRation == "1:2") ? KERNEL_RATION_TWO : 1;
    }
}

void GetOpExecModeForTaskInfo(TaskInfo& info, OpArgContext* args)
{
    if (args == nullptr || !args->ContainsOpArgType(op::OP_EXEC_MODE_ARG)) {
        return;
    }
    auto& argList = *args->GetOpArg(op::OP_EXEC_MODE_ARG);
    OP_CHECK((argList.count == 1), OP_LOGW("OP_EXEC_MODE_ARG must has only one value."), return);
    auto& arg = argList[0];
    OP_CHECK((arg.type == OpArgType::OPARG_UINT), OP_LOGW("OP_EXEC_MODE_ARG must use OpExecMode enum."), return);
    info.execMode = static_cast<OpExecMode>(arg->value);
    OP_LOGI("Get task op exec mode %u", static_cast<uint32_t>(info.execMode));
}

MsprofGeTaskType GetTaskTypeSingleKernelType(const nlohmann::json& opJson)
{
    MsprofGeTaskType taskType = MSPROF_GE_TASK_TYPE_AI_CORE;
    if (opJson.contains("magic")) {
        const auto& str = opJson["magic"].get<std::string>();
        OP_LOGI("get task type, kernel magic: %s", str.c_str());
        if (str == "RT_DEV_BINARY_MAGIC_ELF_AICUBE") {
            taskType = MSPROF_GE_TASK_TYPE_AI_CORE;
        } else if (str == "RT_DEV_BINARY_MAGIC_ELF_AIVEC" || str == "FFTS_BINARY_MAGIC_ELF_MIX_AIC" ||
                   str == "FFTS_BINARY_MAGIC_ELF_MIX_AIV") {
            taskType = MSPROF_GE_TASK_TYPE_AIV;
        }
        if (opJson.contains("taskRation")) {
            const auto& taskRation = opJson["taskRation"].get<std::string>();
            OP_LOGI("get task type, task ration: %s", taskRation.c_str());
            if (!taskRation.empty() && (taskRation[0] == '0')) {
                taskType = MSPROF_GE_TASK_TYPE_MIX_AIV;
            } else {
                taskType = MSPROF_GE_TASK_TYPE_MIX_AIC;
            }
        }
    }
    return taskType;
}

void OpKernelBin::GetTaskRationForSingleBinMutilKernel(TaskInfo& info, const nlohmann::json& opJson, uint64_t tilingkey,
                                                       MsprofGeTaskType taskType)
{
    OP_LOGI("single-bin multi-kernel");
    if (!opJson.contains("kernelList")) {
        OP_LOGW("json parse error. does not contain kernellist.");
        info.type = taskType;
        info.ration = 0;
        return;
    }
    for (const auto& elem : opJson["kernelList"]) {
        if (!elem.contains("tilingKey")) {
            OP_LOGW("json parse error. does not contain tilingKey.");
            continue;
        }
        if (elem["tilingKey"] != tilingkey) {
            continue;
        }
        if (elem.contains("kernelType") && (elem["kernelType"].get<std::string>() == "MIX_AIC" ||
                                            elem["kernelType"].get<std::string>() == "MIX_AIV")) {
            if (!elem.contains("taskRation")) {
                OP_LOGW("json parse error. json file: %s does not contain taskRation.", jsonPath_.c_str());
                info.type = taskType;
                info.ration = 0;
                return;
            }
            GetTaskInfoMultiKernelInfo(info, elem);
            OP_LOGI("single-bin multi-kernel %u, %u", info.type, info.ration);
        } else {
            OP_LOGW("json parse error. kernelType parse error.");
            info.type = taskType;
            info.ration = 0;
        }
    }
}

TaskInfo OpKernelBin::GetTaskInfo(uint64_t tilingkey, OpArgContext* args)
{
    auto& opJson = binJson_.GetVar();
    MsprofGeTaskType taskType = MSPROF_GE_TASK_TYPE_AI_CORE;
    TaskInfo info;
    GetOpExecModeForTaskInfo(info, args);
    info.ration = 0;
    if (!(opJson.contains("taskRation") && opJson["taskRation"].get<std::string>() == "tilingKey")) {
        OP_LOGI("Non-single-bin multi-kernel");
        taskType = GetTaskTypeSingleKernelType(opJson);
        if (mixKernel.find(tilingkey) != mixKernel.end()) {
            taskType = MSPROF_GE_TASK_TYPE_MIX_AIC; // Currently only matmul has AIC mix kernel in fatbin
        }
        info.type = taskType;
        info.ration = (taskType == MSPROF_GE_TASK_TYPE_MIX_AIC) ? KERNEL_RATION_TWO : 0;
        OP_LOGI("non-single-bin multi-kernel %u, %u", info.type, info.ration);
        return info;
    } else {
        GetTaskRationForSingleBinMutilKernel(info, opJson, tilingkey, taskType);
        return info;
    }
}

aclnnStatus OpKernelBin::InitTilingParseCtx()
{
    aclnnStatus ret = ACLNN_SUCCESS;
    OpRunContextMgr::InitOpFunctions(opType_);
    ThreadCoreNum key(GetThreadLocalContext().opConfigInfo_.aicNum_, GetThreadLocalContext().opConfigInfo_.aivNum_);
    auto f = [&ret, &key, this]() {
        auto p = std::make_unique<TilingParseCtxHolder>();
        if (p->BuildTilingParseCtx(opType_, OpRunContextMgr::GetOpTilingFuncs(opType_), binJson_.GetVar(),
                                   SocContext::GetPlatformInfo(), keyAndDetail_.implMode,
                                   aclnnOpInfoRecord::OpKernelInfo(binPath_, static_cast<int8_t>(binType_))) != OK) {
            ret = ACLNN_ERR_RUNTIME_ERROR;
            return;
        }
        tilingParseCtxHolder_[key] = std::move(p);
        ret = ACLNN_SUCCESS;
        return;
    };
    std::call_once(getFlagForKey(key), f);
    return ret;
}

static void GetParamtersValue(const nlohmann::json& elem, op::DataType& dtype, int64_t& valuei, float32_t& valuef)
{
    if (elem["dtype"] == "float16") {
        dtype = op::DataType::DT_FLOAT16;
        valuef = elem["init_value"].get<float32_t>();
    } else if (elem["dtype"] == "float32") {
        dtype = op::DataType::DT_FLOAT;
        valuef = elem["init_value"].get<float32_t>();
    } else if (elem["dtype"] == "int32") {
        dtype = op::DataType::DT_INT32;
        valuei = static_cast<int64_t>(elem["init_value"].get<int32_t>());
    } else if (elem["dtype"] == "uint32") {
        dtype = op::DataType::DT_UINT32;
        valuei = static_cast<int64_t>(elem["init_value"].get<uint32_t>());
    } else if (elem["dtype"] == "int64") {
        dtype = op::DataType::DT_INT64;
        valuei = elem["init_value"].get<int64_t>();
    } else if (elem["dtype"] == "uint64") {
        dtype = op::DataType::DT_UINT64;
        valuei = static_cast<int64_t>(elem["init_value"].get<uint64_t>());
    } else if (elem["dtype"] == "int16") {
        dtype = op::DataType::DT_INT16;
        valuei = static_cast<int64_t>(elem["init_value"].get<int16_t>());
    } else if (elem["dtype"] == "uint16") {
        dtype = op::DataType::DT_UINT16;
        valuei = static_cast<int64_t>(elem["init_value"].get<uint16_t>());
    } else if (elem["dtype"] == "int8") {
        dtype = op::DataType::DT_INT8;
        valuei = static_cast<int64_t>(elem["init_value"].get<int8_t>());
    } else if (elem["dtype"] == "uint8") {
        dtype = op::DataType::DT_UINT8;
        valuei = static_cast<int64_t>(elem["init_value"].get<uint8_t>());
    } else {
        OP_LOGW("unknown dtype: %s", elem["dtype"].get<std::string>().c_str());
    }
}

void OpKernelBin::SetMemSetFlagFromJson()
{
    auto& opJson = binJson_.GetVar();
    if (!opJson.contains("parameters") || !opJson["parameters"].is_array()) {
        return;
    }

    for (size_t i = 0; i < opJson["parameters"].size(); i++) {
        const auto& elem = opJson["parameters"][i];
        if (!elem.is_null()) {
            OP_LOGD("Need atomic clean in op json");
            op::DataType dtype = op::DataType::DT_FLOAT;
            int64_t valuei = 0;
            float32_t valuef = 0;
            if (!elem.contains("dtype") || !elem.contains("init_value")) {
                memSetValue_.emplace_back(MemSetTensorInfo{i, dtype, valuef, valuei, 0, 0, OpArgType::OPARG_ACLTENSOR,
                                                           nullptr, nullptr, nullptr});
                OP_LOGW("not contain dtype, index: %zu", i);
                continue;
            }
            GetParamtersValue(elem, dtype, valuei, valuef);

            memSetValue_.emplace_back(MemSetTensorInfo{i, dtype, valuef, valuei, 0, 0, OpArgType::OPARG_ACLTENSOR,
                                                       nullptr, nullptr, nullptr});
            OP_LOGD("memSetValue_ emplace: [index: %zu, dtype: %s, valuef: %f, valuei: %ld]", i,
                    op::ToString(dtype).GetString(), valuef, valuei);
        }
    }
}

uint64_t OpKernelBin::GetAttrId(OpArgContext* args)
{
    uint64_t aclGraphAttrId = 0;
    if (GetThreadLocalContext().cacheOpInfoSwitch_ && opKernel_ != nullptr) {
        std::string attrStrId;
        if (args->ContainsOpArgType(op::OP_ATTR_ARG)) {
            op::internal::ReportAttrInfo(*args->GetOpArg(op::OP_ATTR_ARG), attrStrId,
                                         static_cast<OpKernel*>(opKernel_)->attrInfos_);
            OP_LOGI("attrStrId is %s after add attr value", attrStrId.c_str());
        }
        OpArgList input = *args->GetOpArg(op::OP_INPUT_ARG);
        input.VisitByNoReturn([&attrStrId](size_t idx, OpArg& elem) { SummaryAttrArg(idx, elem, attrStrId); });
        OP_LOGI("attrStrId is %s after add input tensor", attrStrId.c_str());
        attrStrId += std::string("IsStaticKernel:") +
                     (binType_ == BinType::STATIC_BIN ? std::string("true") : std::string("false"));
        aclGraphAttrId = MsprofGetHashId(attrStrId.c_str(), attrStrId.size());
    }
    return aclGraphAttrId;
}

void ParseImplModeByJson(const nlohmann::json& singleBinJson, const std::string& jsonPath,
                         FVector<OpImplMode>& implModes)
{
    auto iter = singleBinJson.find(IMPL_MODE);
    if (iter != singleBinJson.end()) {
        if (*iter == ALL_PRECISION_MODE) {
            OP_LOGD("jsonPath %s support all op impl mode.", jsonPath.c_str());
            implModes = {OpImplMode::IMPL_MODE_HIGH_PERFORMANCE, OpImplMode::IMPL_MODE_HIGH_PRECISION,
                         OpImplMode::IMPL_MODE_ENABLE_FLOAT32_EXECUTION,
                         OpImplMode::IMPL_MODE_ENABLE_HI_FLOAT32_EXECUTION};
        } else {
            OpImplMode mode = ToOpImplMode(*iter);
            if (mode != OpImplMode::IMPL_MODE_RESERVED) {
                OP_LOGD("Impl mode in json: %u", static_cast<uint32_t>(mode));
                implModes.emplace_back(mode);
            } else {
                implModes.emplace_back(OpImplMode::IMPL_MODE_HIGH_PRECISION);
                OP_LOGW("Invalid op impl mode %s in json %s.", static_cast<string>(*iter).c_str(), jsonPath.c_str());
            }
        }
    } else {
        OP_LOGW("Cannot find impl mode in json %s.", jsonPath.c_str());
        implModes.emplace_back(OpImplMode::IMPL_MODE_HIGH_PRECISION);
    }
}

aclnnStatus OpKernel::ParseContext(const nlohmann::json& singleBinJson, const std::string& jsonPath,
                                   KeyParams& keyParams)
{
    // simp_key: "deterministic/overflow/precision_mode + inputs + outputs"
    /* 1. Generate context. */
    // deterministic: 1, 2
    // overflow: default as 2
    // precision_mode:
    //  "high_performance" :1;
    //  "high_precision" : 2
    //  "super_performance" : 3,
    //  "support_out_of_bound_index": 4
    //  "enable_float_32_execution": 5
    //  "enable_hi_float_32_execution": 6
    auto dtm = singleBinJson.find(DETERMINISTIC);
    FVector<char> determinFlags;
    if (dtm != singleBinJson.end()) {
        const auto& dtmStr = dtm->get<std::string>();
        if (dtmStr == OP_KERNEL_TRUE) {
            determinFlags.emplace_back(DETERMINISTIC_VALUE);
        } else if (dtmStr == OP_KERNEL_FALSE) {
            determinFlags.emplace_back(NON_DETERMINISTIC_VALUE);
        } else {
            determinFlags.emplace_back(DETERMINISTIC_VALUE);
            determinFlags.emplace_back(NON_DETERMINISTIC_VALUE);
        }
    } else {
        determinFlags.emplace_back(DETERMINISTIC_VALUE);
        determinFlags.emplace_back(NON_DETERMINISTIC_VALUE);
    }

    FVector<OpImplMode> implModes;
    if (jsonPath.find(ToString(OpImplMode::IMPL_MODE_HIGH_PERFORMANCE).GetString()) != string::npos) {
        implModes.emplace_back(OpImplMode::IMPL_MODE_HIGH_PERFORMANCE);
    } else if (jsonPath.find(ToString(OpImplMode::IMPL_MODE_HIGH_PRECISION).GetString()) != string::npos) {
        implModes.emplace_back(OpImplMode::IMPL_MODE_HIGH_PRECISION);
    } else if (jsonPath.find(ToString(OpImplMode::IMPL_MODE_SUPPORT_OUT_OF_BOUND_INDEX).GetString()) != string::npos) {
        implModes.emplace_back(OpImplMode::IMPL_MODE_SUPPORT_OUT_OF_BOUND_INDEX);
    } else {
        ParseImplModeByJson(singleBinJson, jsonPath, implModes);
    }

    for (const auto& dtmFlag : determinFlags) {
        for (const auto& implMode : implModes) {
            keyParams.keys.emplace_back();
            auto& back = keyParams.keys.back();
            back.key += dtmFlag;
            back.key += "/";
            back.key += ToIndexChar(implMode);
            back.key += "/";

            keyParams.keysWithoutAttr.emplace_back(back.key);

            back.deterministicFlag = (dtmFlag != NON_DETERMINISTIC_VALUE);
            back.implMode = ToString(implMode).GetString();
            AppendImplModeBm(implMode);
        }
    }
    keyParams.len.ctxLen = keyParams.keys.back().key.length();
    return ACLNN_SUCCESS;
}

aclnnStatus OpKernel::AssembleKeyByTensor(const nlohmann::json& inOrOuts,
                                          std::array<TensorInfo, MAX_TENSOR_SIZE>& tensorInfos, std::string& key,
                                          KeyParams& keyParams)
{
    size_t cnt = 0;
    for (auto& tensor : inOrOuts) {
        if (keyParams.genPlaceholder) {
            auto paramType = tensor.find(PARAM_TYPE);
            if (paramType != tensor.end() && GetTensorType(*paramType) == TensorType::OPTIONAL) {
                OP_LOGD("genPlaceholder, ignore optional tensor");
                tensorInfos[cnt++].tensorType = TensorType::OPTIONAL;
                continue;
            }
        }
        /* 1. Load tensor type. */
        if (tensor.is_null()) {
            tensorInfos[cnt++].tensorType = TensorType::OPTIONAL;
            continue;
        }

        if (tensor.is_array() && !tensor.empty()) {
            auto ret = AssembleKeyForSingleTensor(tensor.at(0), key, tensorInfos[cnt]);
            if (ret != ACLNN_SUCCESS) {
                return ret;
            }
            if (keyParams.hasDevPtrArg) {
                key += (MAX_VALID_DTYPE_FORMAT_KEY + static_cast<wchar_t>(1));
                tensorInfos[cnt++].tensorType = TensorType::DYNAMIC_FOLDED;
                OP_LOGD("Tensor is dynamic_folded, number is 1");
            } else {
                key += (MAX_VALID_DTYPE_FORMAT_KEY + static_cast<wchar_t>(tensor.size()));
                tensorInfos[cnt++].tensorType = TensorType::DYNAMIC;
                OP_LOGD("Tensor is dynamic, number is %zu", tensor.size());
            }
            continue;
        }

        auto type = tensor.find(TYPE);
        if (type != tensor.end() && *type == NULL_STR) {
            tensorInfos[cnt++].tensorType = TensorType::OPTIONAL;
            continue;
        }

        if (tensorInfos[cnt].tensorType != TensorType::BOTTOM) {
            auto paramType = tensor.find(PARAM_TYPE);
            if (paramType == tensor.end()) {
                tensorInfos[cnt].tensorType = TensorType::REQUIRED;
            } else {
                tensorInfos[cnt].tensorType = GetTensorType(*paramType);
            }
        }

        /* 2. Use dtype and format to assemble simplified key. */
        AssembleKeyForSingleTensor(tensor, key, tensorInfos[cnt]);
        ++cnt;
    }

    return ACLNN_SUCCESS;
}

void OpKernel::UpdateFormatType(std::array<TensorInfo, MAX_TENSOR_SIZE>& tensorInfos, size_t validTensorNum) const
{
    for (size_t i = 0; i < validTensorNum; i++) {
        auto& formatInfo = tensorInfos[i].fmtInfo;
        if (formatInfo.supportFormats.count(ge::FORMAT_ND) == 0) {
            formatInfo.fmtType = FormatType::NOT_SUPPORT_ND;
        } else {
            if (formatInfo.supportFormats.size() == 1) {
                formatInfo.fmtType = FormatType::ONLY_SUPPORT_ND;
            } else {
                formatInfo.fmtType = FormatType::SUPPORT_ND;
            }
        }
        OP_LOGD("Format Type of node %s is %d.", OpTypeDict::ToString(opType_).GetString(),
                static_cast<int32_t>(formatInfo.fmtType));
    }
}

aclnnStatus OpKernel::ParseAttributes(const nlohmann::json& singleBinJson, string& key)
{
    if (ingnoreAttrSimplifiedKeyMode_) {
        return ACLNN_SUCCESS;
    }

    auto attrsIter = singleBinJson.find(ATTRS);
    if (attrsIter == singleBinJson.end()) {
        return ACLNN_SUCCESS;
    }

    size_t attrCnt = 0;
    for (auto& attr : *attrsIter) {
        if (attrCnt >= attrInfos_.size()) {
            OP_LOGE(ACLNN_ERR_INNER, "Attr cnt %zu is larger than attrInfos_ size %zu!", attrCnt, attrInfos_.size());
            return ACLNN_ERR_INNER_ATTR_NUM_OUT_OF_BOUND;
        }

        auto& attrInfo = attrInfos_.at(attrCnt);
        ++attrCnt;
        if (attrInfo.supportAll) {
            continue;
        }

        /* Parse value for attributes. */
        auto value = attr.find(VALUE);
        if (value == attr.end()) {
            OP_LOGE(ACLNN_ERR_INNER, "Attr %s does not contains value!", attrInfo.attrName.c_str());
            return ACLNN_ERR_INNER;
        }

        key += SLASH;
        if (value->is_boolean()) {
            key += static_cast<char>(value->get<bool>());
        } else if (value->is_number_float()) {
            float valueFloat = value->get<float>();
            for (size_t i = 0; i < attrInfo.realSize; i++) {
                key += static_cast<char>(*(reinterpret_cast<uint8_t*>(&valueFloat) + i));
            }
        } else if (value->is_number()) {
            int64_t valueInt64 = value->get<int64_t>();
            for (size_t i = 0; i < attrInfo.realSize; i++) {
                key += static_cast<char>(*(reinterpret_cast<uint8_t*>(&valueInt64) + i));
            }
        } else if (value->is_array()) {
            key += '[';
            key += static_cast<char>(value->size());
            key += ']';
        } else {
            key += *value;
        }
    }
    return ACLNN_SUCCESS;
}

bool OpKernel::IsGenInputPlaceholder(const nlohmann::json& singleBinJson)
{
    auto optionalInputMode = singleBinJson.find(OPTIONAL_INPUT_MODE);
    if (optionalInputMode == singleBinJson.end()) {
        return false;
    }
    const auto& value = optionalInputMode->get<std::string>();
    if (value == GEN_PLACEHOLDER) {
        return true;
    }
    return false;
}

bool OpKernel::HasDevPtrArg(const nlohmann::json& singleBinJson) const
{
    auto dynamicParamMode = singleBinJson.find(DYNAMIC_PARAM_MODE);
    if (dynamicParamMode == singleBinJson.end()) {
        return false;
    }
    const auto& value = dynamicParamMode->get<std::string>();
    if (value == FOLDED_WITH_DESC) {
        return true;
    }
    return false;
}

aclnnStatus OpKernel::GenerateKeyBySimplifiedKey(const nlohmann::json& singleBinJson, KeyParams& keyParams)
{
    auto simplifiedKey = singleBinJson.find(SIMPLIFIED_KEY);
    if (simplifiedKey == singleBinJson.end()) {
        OP_LOGE(ACLNN_ERR_INNER_JSON_VALUE_NOT_FOUND, "simplifiedKey not found.");
        return ACLNN_ERR_INNER_JSON_VALUE_NOT_FOUND;
    }

    if (!simplifiedKey->is_array() || simplifiedKey->empty()) {
        OP_LOGE(ACLNN_ERR_INNER_OP_FILE_INVALID, "simplifiedKey is invalid.");
        return ACLNN_ERR_INNER_OP_FILE_INVALID;
    }

    for (auto& key : *simplifiedKey) {
        // exclude op type str in simplified key
        // "ZerosLike/d=0,p=0/12,2/12,2" -> "d=0,p=0/12,2/12,2"
        std::string keyStr = key.get<std::string>().substr(opTypeStr_.size() + 1);
        OP_LOGD("simplified key string %s.", keyStr.c_str());
        uint32_t deterministicFlag;
        uint32_t mode;
        constexpr int goodRes = 2;
        int rc = sscanf_s(keyStr.c_str(), "d=%u,p=%u/", &deterministicFlag, &mode);
        if (rc != goodRes) {
            OP_LOGE(ACLNN_ERR_INNER_OP_FILE_INVALID, "simplifiedKey %s is invalid", keyStr.c_str());
            return ACLNN_ERR_INNER_OP_FILE_INVALID;
        }
        OpImplMode implMode = static_cast<OpImplMode>((1U << mode));
        if (implMode == OpImplMode::IMPL_MODE_DEFAULT && opTypeStr_ != "MemSetV2") {
            continue;
        }
        keyParams.keys.emplace_back();
        auto& back = keyParams.keys.back();
        back.key = keyStr;
        back.deterministicFlag = (deterministicFlag != 0);
        back.implMode = ToString(implMode).GetString();
        AppendImplModeBm(implMode);
        OP_LOGD("%s append simplified key %s deterministic(%u) implMode(%u)", opTypeStr_.c_str(), back.key.c_str(),
                back.deterministicFlag, static_cast<uint32_t>(implMode));
    }
    keyParams.genPlaceholder = IsGenInputPlaceholder(singleBinJson);
    keyParams.hasDevPtrArg = HasDevPtrArg(singleBinJson);
    return ACLNN_SUCCESS;
}

/* Generate mapping of {simplified_key -> the path of json and bin} */
aclnnStatus OpKernel::GenerateKeyByJson(const nlohmann::json& singleBinJson, const std::string& jsonPath,
                                        KeyParams& keyParams)
{
    /* 1. Generate key by context. */
    ParseContext(singleBinJson, jsonPath, keyParams);
    keyParams.genPlaceholder = IsGenInputPlaceholder(singleBinJson);
    keyParams.hasDevPtrArg = HasDevPtrArg(singleBinJson);
    string tensorKey;
    /* 2. Generate inputs and outputs */
    auto inputs = singleBinJson.find(INPUTS);
    if (inputs != singleBinJson.end()) {
        if (inputs->size() > inputNum_) {
            inputNum_ = inputs->size();
        }
        auto ret = AssembleKeyByTensor(*inputs, inputInfos_, tensorKey, keyParams);
        if (ret != ACLNN_SUCCESS) {
            return ret;
        }
        UpdateFormatType(inputInfos_, inputNum_);
    }

    auto outputs = singleBinJson.find(OUTPUTS);
    if (outputs != singleBinJson.end()) {
        if (outputs->size() > outputNum_) {
            outputNum_ = outputs->size();
        }
        auto ret = AssembleKeyByTensor(*outputs, outputInfos_, tensorKey, keyParams);
        if (ret != ACLNN_SUCCESS) {
            return ret;
        }
        UpdateFormatType(outputInfos_, outputNum_);
    }

    keyParams.len.tensorLen = tensorKey.length();
    keyParams.len.ctxAndtensorLen = keyParams.len.ctxLen + keyParams.len.tensorLen;
    for (auto& ele : keyParams.keysWithoutAttr) {
        ele += tensorKey;
    }

    auto ret = ParseAttributes(singleBinJson, tensorKey);
    auto attrLenOfThisBin = tensorKey.length() - keyParams.len.tensorLen;
    if (attrLenOfThisBin > maxAttrLen_) {
        maxAttrLen_ = attrLenOfThisBin;
    }

    for (auto& key : keyParams.keys) {
        key.key += tensorKey;
    }
    return ret;
}

void OpKernel::GetReadableFmtDtypeKey(wchar_t ele, int32_t& cnt, string& debug_str) const
{
    if (cnt % MAX_BIN_MATCH_ARG_KEY_COUNT == 0) {
        if (ele <= ge::DT_MAX) {
            auto iter = DATA_TYPE_2_STR.find(static_cast<ge::DataType>(ele));
            if (iter != DATA_TYPE_2_STR.end()) {
                debug_str += (iter->second);
                debug_str += SLASH;
                ++cnt;
                return;
            }
        }

        if (opType_ == BinConfigJsonDict::transDataId_) {
            debug_str += "c0:";
        } else {
            debug_str += "dynamic num:";
        }
        debug_str += std::to_string(ele - MAX_VALID_DTYPE_FORMAT_KEY);
        debug_str += SLASH;
    } else if (cnt % MAX_BIN_MATCH_ARG_KEY_COUNT == 1) {
        auto iter = FORMAT_2_STR.find(static_cast<ge::Format>(ele));
        if (iter != FORMAT_2_STR.end()) {
            debug_str += (iter->second);
            debug_str += SLASH;
            ++cnt;
        }
    }
}

string OpKernel::GetReadableKey(const string& key, KeyLength& len) const
{
    if (customizedSimplifiedKeyMode_) {
        return key;
    }

    if (key.length() < len.ctxLen) {
        OP_LOGD("Key %s is not qualified.", key.c_str());
        return key;
    }

    string keyNew;
    for (char i : key) {
        keyNew += std::to_string(i);
        keyNew += ",";
    }
    if (len.tensorLen == 0 && len.ctxAndtensorLen >= len.ctxLen) {
        len.tensorLen = len.ctxAndtensorLen - len.ctxLen;
    }
    OP_LOGD("key is: tensorLen[%zu], ctxtLen[%zu]: %s", len.tensorLen, len.ctxLen, keyNew.c_str());

    string debugStr = key.substr(0, len.ctxLen);
    debugStr += '|';
    int32_t cnt = 0;
    if (len.tensorLen > 0) {
        for (auto ele : key.substr(len.ctxLen, len.tensorLen)) {
            GetReadableFmtDtypeKey(ele, cnt, debugStr);
        }
    }

    for (auto ele : key.substr(len.ctxLen + len.tensorLen)) {
        if (ele == SLASH) {
            debugStr += '|';
            continue;
        }

        if (ele == LEFT_BRACKET || ele == RIGHT_BRACKET) {
            debugStr += ele;
            continue;
        }
        if ((ele >= 'a' && ele <= 'z') || (ele >= 'A' && ele <= 'Z')) {
            debugStr += ele;
        } else {
            debugStr += std::to_string(ele);
        }
        debugStr += ',';
    }

    return debugStr;
}

size_t OpKernel::HashBinary(const char* addr, uint32_t len) const
{
    const std::hash<uint64_t> hasher;
    uint32_t size = len >> 3;
    uint32_t rem = len % sizeof(uint64_t);
    uint32_t i = 0;
    size_t seed = 0;
    auto* ptr = reinterpret_cast<const uint64_t*>(addr);

    while (i < size) {
        seed ^= hasher(*ptr) + OP_API_HASH_SEED + (seed << 6U) + (seed >> 2U);
        ptr++;
        i++;
    }
    if (rem != 0) {
        uint64_t val = 0U;
        OP_CHECK(memcpy_s(&val, sizeof(uint64_t), ptr, rem) == EOK, OP_LOGW("Failed to memcpy."), ;);
        seed ^= hasher(val) + OP_API_HASH_SEED + (seed << 6U) + (seed >> 2U);
    }
    return seed;
}

aclnnStatus OpKernel::HashAndInsert(const string& binAndJsonDir, const string& binOrJsonPath, const size_t& pos,
                                    KeyParams& keyParams)
{
    bool enableDebug = false;
    op::internal::systemConfig.GetEnableDebugKernelFlag(enableDebug);
    for (const auto& key : keyParams.keys) {
        string binPath = binAndJsonDir;
        string binOrJsonPathPrefix = binOrJsonPath.substr(0, pos);
        binPath.append(binOrJsonPathPrefix);
        binPath.append(BIN_SUFFIX);
        string jsonPath = binAndJsonDir;
        jsonPath.append(binOrJsonPathPrefix);
        jsonPath.append(JSON_SUFFIX);
        if (keyParams.binType == BinType::STATIC_BIN) {
            auto hash = HashBinary(binPath.c_str(), binPath.size());
            const std::lock_guard<std::mutex> lock(staticKernelsMutex_);
            if (staticBins_.find(hash) != staticBins_.end()) {
                OP_LOGD("This static bin has been added, key: %s", binPath.c_str());
                continue;
            }
            staticBins_.emplace(
                hash, std::make_unique<OpKernelBin>(opType_, jsonPath, binOrJsonPath, binPath, key, hash,
                                                    keyParams.binType, keyParams.genPlaceholder, false, this));
            OP_LOGD("Static bin: key: %s, json: %s, bin: %s", key.key.c_str(), jsonPath.c_str(), binPath.c_str());
            continue;
        }

        OP_LOGD("Add key %s into bins_.", GetReadableKey(key.key, keyParams.len).c_str());
        OP_LOGD("result json path [%s]; bin file path[%s]", jsonPath.c_str(), binPath.c_str());
        auto hash = HashBinary(key.key.c_str(), key.key.size());
        OP_LOGD("Hash key %zu origin size %zu into bins_;", hash, key.key.size());
        auto binsIter = bins_.find(hash);
        if (binsIter != bins_.end()) {
            if (key.key != binsIter->second->GetKeyAndDetail().key) {
                OP_LOGE(ACLNN_ERR_INNER, "Two bin [%s & %s] has same hash but their integral key is diff[%s, %s].",
                        binOrJsonPath.c_str(), binsIter->second->jsonPath_.c_str(),
                        GetReadableKey(key.key, keyParams.len).c_str(),
                        GetReadableKey(binsIter->second->GetKeyAndDetail().key, keyParams.len).c_str());
                return ACLNN_ERR_INNER;
            } else {
                if (enableDebug) {
                    binsIter->second->SetJsonPath(jsonPath);
                    binsIter->second->SetBinPath(binPath);
                } else {
                    OP_LOGW("Two different bin [%s & %s] has same integral key %s.", binOrJsonPath.c_str(),
                            binsIter->second->jsonPath_.c_str(), GetReadableKey(key.key, keyParams.len).c_str());
                    continue;
                }
            }
        } else {
            bins_.emplace(hash, std::make_unique<OpKernelBin>(opType_, jsonPath, binOrJsonPath, binPath, key, hash,
                                                              keyParams.binType, keyParams.genPlaceholder,
                                                              keyParams.hasDevPtrArg, this));
        }
    }
    return ACLNN_SUCCESS;
}

void HandleAttrValue(const nlohmann::json& value, AttrInfo& attrInfo)
{
    OP_LOGD("Check attr [%s].", attrInfo.attrName.c_str());
    if (attrInfo.values == nullptr) {
        if (value.is_number_float()) {
            OP_LOGD("Float attr [%s], value: [%f, %u].", attrInfo.attrName.c_str(), value.get<float>(),
                    value.get<uint32_t>());
            attrInfo.values = std::make_unique<uint8_t[]>(sizeof(float));
            attrInfo.realSize = sizeof(float);

            float valueFloat = value.get<float>();
            for (size_t i = 0; i < attrInfo.realSize; ++i) {
                OP_LOGD("Attr value compare %zu [%x: %x].", i, attrInfo.values[i], *((uint8_t*)(&valueFloat) + i));
            }
            OP_CHECK(memcpy_s(attrInfo.values.get(), sizeof(float), &valueFloat, sizeof(float)) == EOK,
                     OP_LOGW("Failed to memcpy."),
                     ;);
        } else if (value.is_number()) {
            OP_LOGD("Int attr [%s], value: [%ld].", attrInfo.attrName.c_str(), value.get<int64_t>());
            int64_t valueInt64 = value.get<int64_t>();
            attrInfo.values = std::make_unique<uint8_t[]>(sizeof(int64_t));
            OP_CHECK(memcpy_s(attrInfo.values.get(), sizeof(int64_t), &valueInt64, sizeof(int64_t)) == EOK,
                     OP_LOGW("Failed to memcpy."),
                     ;);
            attrInfo.realSize = sizeof(int64_t);
            for (size_t i = 0; i < attrInfo.realSize; ++i) {
                OP_LOGD("Attr value compare %zu [%u: %u].", i, attrInfo.values[i], *((uint8_t*)(&valueInt64) + i));
            }
        } else if (value.is_array()) {
            if (value.at(0).is_array()) {
                OP_LOGD("Do not support list list int attr [%s].", attrInfo.attrName.c_str());
                return;
            }
            OP_LOGD("List Int attr [%s], value: [%s].", attrInfo.attrName.c_str(),
                    op::ToString(value.get<FVector<int64_t, MAX_DIM_NUM>>()).GetString());
            attrInfo.values = std::make_unique<uint8_t[]>(1);
            attrInfo.realSize = 1;
            attrInfo.values[0] = value.size();
        }
        return;
    }
    /* If current value is not equal to the history value, we know this attribute
    contains multiple different values, and it does not support all. We need to
    match it in binary searching. */
    if (value.is_number_float()) {
        float valueFloat = value.get<float>();
        for (size_t i = 0; i < attrInfo.realSize; ++i) {
            OP_LOGD("Attr value compare %zu [%x: %x].", i, attrInfo.values[i], *((uint8_t*)(&valueFloat) + i));
        }
        if (memcmp(attrInfo.values.get(), &valueFloat, attrInfo.realSize) != 0) {
            OP_LOGD("Float attr [%s] need to be compared.", attrInfo.attrName.c_str());
            attrInfo.supportAll = false;
        } else {
            OP_LOGD("Float attr [%s] currently do not need to be compared.", attrInfo.attrName.c_str());
        }
    } else if (value.is_number()) {
        int64_t valueInt64 = value.get<int64_t>();
        for (size_t i = 0; i < attrInfo.realSize; ++i) {
            OP_LOGD("Attr value compare %zu [%u: %u].", i, attrInfo.values[i], *((uint8_t*)(&valueInt64) + i));
        }
        if (memcmp(attrInfo.values.get(), &valueInt64, attrInfo.realSize) != 0) {
            OP_LOGD("Int64 attr [%s] need to be compared.", attrInfo.attrName.c_str());
            attrInfo.supportAll = false;
        } else {
            OP_LOGD("Int64 attr [%s] currently do not need to be compared.", attrInfo.attrName.c_str());
        }
    } else if (value.is_array()) {
        if (attrInfo.values[0] != value.size()) {
            OP_LOGD("Int64 attr [%s] need to be compared.", attrInfo.attrName.c_str());
            attrInfo.supportAll = false;
        } else {
            OP_LOGD("Int64 attr [%s] currently do not need to be compared.", attrInfo.attrName.c_str());
        }
    }
    return;
}

static void RecordDimNumAndCheck(TensorInfo& inputInfo, size_t dimNum, int64_t inputsCnt)
{
    if (inputInfo.shapeDimension == INVALID_MAX_DIM_NUM) {
        inputInfo.shapeDimension = dimNum;
    } else if (inputInfo.shapeDimension != dimNum) {
        OP_LOGD("Shape of %ld dimension is different, [%zu, %zu].", inputsCnt, inputInfo.shapeDimension, dimNum);
        inputInfo.shapeSupportType = ShapeSupportType::NOT_SUPPORT_ALL;
    }
}

aclnnStatus OpKernel::JudgeInputSupportAll(const nlohmann::json& binListJson)
{
    for (auto& binJson : binListJson) {
        auto inputsIter = binJson.find(INPUTS);
        if (inputsIter == binJson.end()) {
            return ACLNN_SUCCESS;
        }

        int64_t inputsCnt = -1;
        for (const auto& input : *inputsIter) {
            ++inputsCnt;
            if (inputsCnt >= static_cast<int64_t>(inputInfos_.size())) {
                OP_LOGE(ACLNN_ERR_INNER_INPUT_NUM_IN_JSON_TOO_LARGE, "%ld is too large", inputsCnt);
                return ACLNN_ERR_INNER_INPUT_NUM_IN_JSON_TOO_LARGE;
            }
            auto& inputInfo = inputInfos_[inputsCnt];
            if (inputInfo.shapeSupportType == ShapeSupportType::SUPPORT_ALL ||
                inputInfo.shapeSupportType == ShapeSupportType::NOT_SUPPORT_ALL) {
                continue;
            }

            if (input.is_null()) {
                continue;
            }

            auto shape = input.find(SHAPE);
            if (shape == input.end() || shape->is_null()) {
                continue;
            }

            if (!shape->is_array()) {
                continue;
            }

            size_t dimNum = shape->size();
            if (dimNum == 1 && shape->at(0) == UNKNOWN_DIM_NUM) {
                inputInfo.shapeSupportType = ShapeSupportType::SUPPORT_ALL;
                continue;
            }

            RecordDimNumAndCheck(inputInfo, dimNum, inputsCnt);
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus OpKernel::JudgeAttrSupportAll(const nlohmann::json& binListJson)
{
    for (auto& binJson : binListJson) {
        auto attrsIter = binJson.find(ATTRS);
        if (attrsIter == binJson.end()) {
            return ACLNN_SUCCESS;
        }
        size_t attrCnt = 0;
        for (const auto& attr : *attrsIter) {
            if (attr.is_null()) {
                OP_LOGE(ACLNN_ERR_INNER, "Attr is null!");
                return ACLNN_ERR_INNER;
            }
            /* 1. Check name. */
            auto name = attr.find(NAME);
            if (name == attr.end() || name->is_null()) {
                OP_LOGE(ACLNN_ERR_INNER, "Attr does not contains name!");
                return ACLNN_ERR_INNER;
            }

            if (attrCnt >= attrInfos_.size()) {
                attrInfos_.emplace_back();
                attrInfos_.back().attrName = *name;
                attrInfos_.back().supportAll = true;
            }
            auto& attrInfo = attrInfos_.at(attrCnt);
            ++attrCnt;
            if (!attrInfo.supportAll) {
                /* If this attribute is marked as not support all, we know that the value is differed
                   from each other, and we do not need to check more attributes. */
                continue;
            }
            if (customizedSimplifiedKeyMode_) {
                /* simplifiedkey scence only parse attrname, attrvalue may be value_range type */
                continue;
            }

            /* 2. Check value  */
            auto value = attr.find(VALUE);
            if (value == attr.end()) {
                OP_LOGE(ACLNN_ERR_INNER, "Attr %s does not contains value!", attrInfo.attrName.c_str());
                return ACLNN_ERR_INNER;
            }

            if (value->is_null() || *value == ALL || (value->is_array() && value->empty())) {
                continue;
            }

            if (value->is_boolean() || value->is_string()) {
                attrInfo.supportAll = false;
                continue;
            }

            HandleAttrValue(*value, attrInfo);
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus OpKernel::InitTensorInfo(const nlohmann::json& binListJson)
{
    for (auto& input : inputInfos_) {
        input.opType = opType_;
    }

    for (auto& output : outputInfos_) {
        output.opType = opType_;
    }
    return JudgeInputSupportAll(binListJson);
}

void OpKernel::JudgeCustomizedSimpliedKeyMode(const nlohmann::json& binListJson)
{
    for (auto& binJson : binListJson) {
        auto simplifiedKeyMode = binJson.find(SIMPLIFIED_KEY_MODE);
        if (simplifiedKeyMode != binJson.end() && simplifiedKeyMode->is_number() &&
            simplifiedKeyMode->get<int64_t>() == CUSTOMIZED_SIMPLIFIED_KEY) {
            OP_LOGI("op type %s has customized simplified key.", opTypeStr_.c_str());
            customizedSimplifiedKeyMode_ = true;
            break;
        }
        // if one bin dont has simplifiedKeyMode or simplifiedKeyMode is not 0, dont use ingnoreAttrSimplifiedKeyMode_
        if (simplifiedKeyMode != binJson.end() && simplifiedKeyMode->is_number() &&
            simplifiedKeyMode->get<int64_t>() == IGNORE_ATTR_SIMPLIFIED_KEY) {
            OP_LOGI("op type %s has ignore attr simplified key.", opTypeStr_.c_str());
            ingnoreAttrSimplifiedKeyMode_ = true;
        } else {
            OP_LOGI("op type %s remove ignore attr simplified key.", opTypeStr_.c_str());
            ingnoreAttrSimplifiedKeyMode_ = false;
            break;
        }
    }
}

aclnnStatus OpKernel::GetOpDescJson(bool debug)
{
    if (!debug) {
        auto ret = nnopbase::OpBinaryResourceManager::GetInstance().GetOpBinaryDesc(opTypeStr_.c_str(), configJson_);
        OP_CHECK(ret != ACLNN_SUCCESS, OP_LOGI("Get builtin op desc info [%s]", opTypeStr_.c_str()),
                 return ACLNN_SUCCESS);
        OP_LOGW("No builtin op desc info [%s]", opTypeStr_.c_str());
    }

    ifstream f(configJsonPath_);
#if !defined(NNOPBASE_UT) && !defined(NNOPBASE_ST)
    std::string errMsg = "[Errno " + std::to_string(errno) + "] " + std::generic_category().message(errno);
    OP_CHECK(f.is_open(), OP_LOGE_FOR_FILE_OPERATION_ERROR_OPEN(configJsonPath_.c_str(), errMsg.c_str()),
             return ACLNN_ERR_INNER);
#endif
    try {
        configJson_ = nlohmann::json::parse(f);
    } catch (nlohmann::json::exception& e) {
        OP_LOGE_FOR_FILE_OPERATION_ERROR_PARSE(configJsonPath_.c_str(), e.what());
        return ACLNN_ERR_INNER;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus OpKernel::AppendDynBin(const string& jsonPath, const string& binAndJsonDir, bool debug)
{
    configJsonPath_ = jsonPath;
    auto ret = GetOpDescJson(debug);
    CHECK_COND(ret == ACLNN_SUCCESS, ACLNN_ERR_INNER, "failed to get op desc info [%s]", opTypeStr_.c_str());

    auto binListIter = configJson_.find(BIN_LIST);
    if (binListIter == configJson_.end()) {
        std::string reason = "The operator JSON file does not contain the bin list";
        OP_LOGE_FOR_FILE_OPERATION_ERROR_PARSE_WITH_INVALID_CONTENT(jsonPath.c_str(), reason.c_str());
        return ACLNN_ERR_INNER;
    }

    JudgeCustomizedSimpliedKeyMode(*binListIter);
    ret = JudgeAttrSupportAll(*binListIter);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }

    ret = InitTensorInfo(*binListIter);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }

    for (auto& binJson : *binListIter) {
        auto binInfoIter = binJson.find(BIN_INFO);
        if (binInfoIter == binJson.end()) {
            continue;
        }
        auto jsonFilePathIter = binInfoIter->find(JSON_FILE_PATH);
        if (jsonFilePathIter == binInfoIter->end()) {
            continue;
        }

        std::string resultJsonPath = jsonFilePathIter->get<std::string>();
        // if repoName is not in resultJsonPath，need to modify resultJsonPath
        if (opsRepoName_ != "" && resultJsonPath.find(opsRepoName_.c_str()) == string::npos) {
            auto firstOpePos = resultJsonPath.find(OPERATOR);
            if (firstOpePos != string::npos) {
                resultJsonPath.insert(firstOpePos + 1, opsRepoName_ + "/");
                OP_LOGD("End for modify resultJsonPath %s.", resultJsonPath.c_str());
            }
        }
        auto pos = resultJsonPath.find(JSON_SUFFIX);
        if (pos == string::npos) {
            continue;
        }

        KeyParams keyParams;
        if (customizedSimplifiedKeyMode_) {
            GenerateKeyBySimplifiedKey(binJson, keyParams);
        } else {
            GenerateKeyByJson(binJson, resultJsonPath, keyParams);
        }
        if (HashAndInsert(binAndJsonDir, resultJsonPath, pos, keyParams) != ACLNN_SUCCESS) {
            return ACLNN_ERR_INNER_KEY_CONFLICT;
        }
        genPlaceholder_ = keyParams.genPlaceholder;
    }
    if (opTypeStr_ == "LayerNormBetaGammaBackpropV2") {
        attrInfos_[0].supportAll = true;

        attrInfos_.emplace_back();
        attrInfos_.back().attrName = "shape_gamma_private";
        attrInfos_.back().supportAll = false;
    }
    maxKeyLength_ = customizedSimplifiedKeyMode_ ? MAX_CUSTOMIZED_SIMPLIFIED_KEY_LEN + sizeof("d=1,p=1/") :
                                                   (inputNum_ + outputNum_) * MAX_BIN_MATCH_ARG_KEY_COUNT +
                                                       maxAttrLen_ + DEFAULT_CONTEXT_LEN + REDUNDANT_LEN;
    OP_LOGD("End for parsing config file %s. Max key length = %zu.", jsonPath.c_str(), maxKeyLength_);
    return ACLNN_SUCCESS;
}

aclnnStatus OpKernel::AppendStaticBin(const nlohmann::json& opJson, const string& binAndJsonDir)
{
    auto valueDependIter = opJson.find(VALUE_DEPEND_INDEX);
    if (valueDependIter != opJson.end() && !valueDependIter->is_null() && valueDependIter->is_array()) {
        valueDependIndex_ = valueDependIter->get<FVector<int64_t>>();
    }

    auto staticListIter = opJson.find(STATIC_LIST);
    if (staticListIter == opJson.end() || staticListIter->is_null()) {
        return ACLNN_SUCCESS;
    }

    for (const auto& singleJson : *staticListIter) {
        auto binPathIter = singleJson.find(BIN_PATH);
        if (binPathIter == singleJson.end()) {
            OP_LOGW("binPath does not exist.");
            continue;
        }

        auto simplifiedKeyIter = singleJson.find(SIMPLIFIED_KEY);
        if (simplifiedKeyIter == singleJson.end()) {
            OP_LOGW("simplifiedKey does not exist.");
            continue;
        }

        const std::string& binPath = binPathIter->get<std::string>();
        auto pos = binPath.find(BIN_SUFFIX);
        if (pos == string::npos) {
            OP_LOGW("binPath %s is not valid.", binPath.c_str());
            continue;
        }

        const std::string& simplifiedKey = simplifiedKeyIter->get<std::string>();

        KeyParams keyParams;
        keyParams.binType = BinType::STATIC_BIN;
        keyParams.keys.emplace_back();
        keyParams.keys.back().key = binPath;

        CHECK_COND(HashAndInsert(binAndJsonDir, binPath, pos, keyParams) == ACLNN_SUCCESS, ACLNN_ERR_INNER_KEY_CONFLICT,
                   "HashAndInsert failed");
    }
    return ACLNN_SUCCESS;
}

aclnnStatus OpKernel::AppendTensor(const aclTensor* tensor, FVector<const aclTensor*>& tensors,
                                   [[maybe_unused]] FVector<int64_t>& dynamicIndex,
                                   [[maybe_unused]] FVector<int64_t>& dynamicCount) const
{
    tensors.emplace_back(tensor);
    return ACLNN_SUCCESS;
}

aclnnStatus OpKernel::AppendTensor(const aclTensorList* tensorList, FVector<const aclTensor*>& tensors,
                                   FVector<int64_t>& dynamicIndex, FVector<int64_t>& dynamicCount) const
{
    dynamicIndex.emplace_back(static_cast<int64_t>(tensors.size()));
    dynamicCount.emplace_back(tensorList->Size());
    auto tensorsPtr = tensorList->GetData();
    for (size_t i = 0; i < tensorList->Size(); ++i) {
        tensors.emplace_back(*(tensorsPtr + i));
    }
    OP_LOGD("Append Tensor List with size %lu. Index %ld", tensorList->Size(), dynamicIndex.back());
    return ACLNN_SUCCESS;
}

aclnnStatus OpKernel::AppendTensor(OpArg& opArg, FVector<const aclTensor*>& tensors, FVector<int64_t>& dynamicIndex,
                                   FVector<int64_t>& dynamicCount) const
{
    if (opArg.type == OpArgType::OPARG_ACLTENSOR) {
        return AppendTensor(reinterpret_cast<aclTensor*>(opArg->pointer), tensors, dynamicIndex, dynamicCount);
    } else if (opArg.type == OpArgType::OPARG_ACLTENSOR_LIST) {
        return AppendTensor(reinterpret_cast<aclTensorList*>(opArg->pointer), tensors, dynamicIndex, dynamicCount);
    }
    return ACLNN_SUCCESS;
}

uint32_t OpKernel::GetOpType() const { return opType_; }

const string& OpKernel::GetOpTypeStr() const { return opTypeStr_; }

void OpKernel::SetOpType(uint32_t opType) { opType_ = opType; }

void OpKernel::SetOpType(const string& opTypeStr) { opTypeStr_ = opTypeStr; }

void OpKernel::SetOpsRepoName(const string& opsRepoName) { opsRepoName_ = opsRepoName; }

void OpKernelBin::SetJsonPath(const string& jsonPath) { jsonPath_ = jsonPath; }

void OpKernelBin::SetBinPath(const string& binPath) { binPath_ = binPath; }

void OpKernelBin::ReportOpAttrInfo(OpArgContext* args, uint64_t summaryId)
{
    // only level2 profiling need to report attr info
    if (!op::internal::opProfilingSwitch.level2ProfilingFlag) {
        return;
    }
    if (opKernel_ == nullptr) {
        return;
    }

    std::string attrStr;
    if (args->ContainsOpArgType(op::OP_ATTR_ARG)) {
        op::internal::ReportAttrInfo(*args->GetOpArg(op::OP_ATTR_ARG), attrStr,
                                     static_cast<OpKernel*>(opKernel_)->attrInfos_);
        OP_LOGI("attrStr is %s after add attr value", attrStr.c_str());
    }
    OpArgList input = *args->GetOpArg(op::OP_INPUT_ARG);
    input.VisitByNoReturn([&attrStr](size_t idx, OpArg& elem) { SummaryAttrArg(idx, elem, attrStr); });
    OP_LOGI("attrStr is %s after add input tensor", attrStr.c_str());
    attrStr += std::string("IsStaticKernel:") +
               (binType_ == BinType::STATIC_BIN ? std::string("true") : std::string("false"));
    ReportAttrInfo(attrStr, summaryId);
}

} // namespace internal
} // namespace op
