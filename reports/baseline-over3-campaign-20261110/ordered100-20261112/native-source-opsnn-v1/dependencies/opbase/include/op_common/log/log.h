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
 * \file log.h
 * \brief
 */
#ifndef OP_COMMON_LOG_LOG_H
#define OP_COMMON_LOG_LOG_H

#include <string>
#include <sstream>
#include <cinttypes>
#include <unistd.h>
#include <sys/syscall.h>
#include "exe_graph/runtime/shape.h"
#include "base/err_msg.h"
#include "graph/ge_error_codes.h"
#include "graph/types.h"
#include "op_common/op_host/util/opbase_export.h"

#ifndef DLOG_DEBUG
#define DLOG_DEBUG 0
#endif
#ifndef DLOG_INFO
#define DLOG_INFO 1
#endif
#ifndef DLOG_WARN
#define DLOG_WARN 2
#endif
#ifndef DLOG_ERROR
#define DLOG_ERROR 3
#endif

constexpr int32_t OP_MODULE_ID = 63;

extern "C" {
int32_t CheckLogLevel(int32_t moduleId, int32_t logLevel);
void DlogRecord(int32_t moduleId, int32_t level, const char* fmt, ...);
}

namespace Ops {
namespace Base {

#ifndef OP_SUBMOD_NAME
#define OP_SUBMOD_NAME "OPS_BASE"
#endif

static inline uint64_t GetTid()
{
    const uint64_t tid = static_cast<uint64_t>(syscall(__NR_gettid));
    return tid;
}

template <class T>
typename std::enable_if<std::is_same<const char*, typename std::decay<T>::type>::value, const char*>::type GetSafeStr(
    T name)
{
    return name;
}

template <class T>
typename std::enable_if<std::is_same<char*, typename std::decay<T>::type>::value, const char*>::type GetSafeStr(T name)
{
    return name;
}

template <class T>
typename std::enable_if<std::is_same<std::string, typename std::decay<T>::type>::value, const char*>::type GetSafeStr(
    const T& name)
{
    return name.c_str();
}

inline const std::string& GetOpInfo(const std::string& str) { return str; }

inline const char* GetOpInfo(const char* str) { return (str == nullptr) ? "nil" : str; }

template <class T>
constexpr bool IsContextType()
{
    return !std::is_same<const char*, typename std::decay<T>::type>::value &&
           !std::is_same<char*, typename std::decay<T>::type>::value &&
           !std::is_same<std::string, typename std::decay<T>::type>::value;
}

template <class T>
typename std::enable_if<IsContextType<T>(), std::string>::type GetOpInfo(T context)
{
    if (context == nullptr) {
        return "nil:nil";
    }
    std::string opInfo = context->GetNodeType() != nullptr ? context->GetNodeType() : "nil";
    opInfo += ":";
    opInfo += context->GetNodeName() != nullptr ? context->GetNodeName() : "nil";
    return opInfo;
}

#define OP_LOGE_LIBOPAPI_REPORT(opName, fmt, ...)                                             \
    do {                                                                                      \
        if (CheckLogLevel(static_cast<int>(OP_MODULE_ID), DLOG_ERROR) == 1) {                 \
            DlogRecord(static_cast<int>(OP_MODULE_ID), DLOG_ERROR,                            \
                       "[%s:%d][%s]"                                                          \
                       "[%s][%" PRIu64 "] OpName:[%s] " fmt,                                  \
                       __FILE__, __LINE__, OP_SUBMOD_NAME, __FUNCTION__, Ops::Base::GetTid(), \
                       Ops::Base::GetSafeStr(Ops::Base::GetOpInfo(opName)), ##__VA_ARGS__);   \
        }                                                                                     \
    } while (0)

// ============================================================================
// 以下接口仅供算子或aclnn实现使用
// ============================================================================

/**
 * 此接口仅供算子或aclnn实现使用
 * entityName:string - 算子名称或aclnn接口名称
 * index:int
 * incorrectShape:string
 * correctShape:string
 * errMessage:The [index]th input of [entityName] has incorrect shape [incorrectShape]. It should be
 * [correctShape].
 */
#define OP_LOGE_WITH_INVALID_INPUT_SHAPE(entityName, index, incorrectShape, correctShape)                              \
    do {                                                                                                               \
        std::string _safe_entityName_(entityName);                                                                     \
        std::string _safe_incorrectShape_(incorrectShape);                                                             \
        std::string _safe_correctShape_(correctShape);                                                                 \
        std::string index_str = std::to_string(index);                                                                 \
        OP_LOGE_LIBOPAPI_REPORT(                                                                                       \
            _safe_entityName_.c_str(), "The %sth input of %s has incorrect shape [%s]. It should be [%s].",            \
            index_str.c_str(), _safe_entityName_.c_str(), _safe_incorrectShape_.c_str(), _safe_correctShape_.c_str()); \
        const std::vector<const char*> msgKey = {"index", "op_name", "incorrect_shape", "correct_shape"};              \
        const std::vector<const char*> msgvalue = {index_str.c_str(), _safe_entityName_.c_str(),                       \
                                                   _safe_incorrectShape_.c_str(), _safe_correctShape_.c_str()};        \
        REPORT_PREDEFINED_ERR_MSG("EZ0001", msgKey, msgvalue);                                                         \
    } while (0)

/**
 * 此接口仅供算子或aclnn实现使用
 * entityName:string - 算子名称或aclnn接口名称
 * attrName:string
 * incorrectVal:string
 * correctVal:string
 * errMessage:Attribute [attrName] of [entityName] has incorrect value [incorrectVal]. It should be
 * [correctVal].
 */
#define OP_LOGE_WITH_INVALID_ATTR(entityName, attrName, incorrectVal, correctVal)                                \
    do {                                                                                                         \
        std::string _safe_entityName_(entityName);                                                               \
        std::string _safe_attrName_(attrName);                                                                   \
        std::string _safe_incorrectVal_(incorrectVal);                                                           \
        std::string _safe_correctVal_(correctVal);                                                               \
        OP_LOGE_LIBOPAPI_REPORT(_safe_entityName_.c_str(),                                                       \
                                "Attribute %s of %s has incorrect value %s. It should be %s.",                   \
                                _safe_attrName_.c_str(), _safe_entityName_.c_str(), _safe_incorrectVal_.c_str(), \
                                _safe_correctVal_.c_str());                                                      \
        const std::vector<const char*> msgKey = {"attr_name", "op_name", "incorrect_val", "correct_val"};        \
        const std::vector<const char*> msgvalue = {_safe_attrName_.c_str(), _safe_entityName_.c_str(),           \
                                                   _safe_incorrectVal_.c_str(), _safe_correctVal_.c_str()};      \
        REPORT_PREDEFINED_ERR_MSG("EZ0002", msgKey, msgvalue);                                                   \
    } while (0)

/**
 * 此接口仅供算子或aclnn实现使用
 * entityName:string - 算子名称或aclnn接口名称
 * attrName:string
 * incorrectSize:string
 * correctSize:string
 * errMessage:Attribute [attrName] of [entityName] has incorrect size [incorrectSize]. It should be
 * [correctSize].
 */
#define OP_LOGE_WITH_INVALID_ATTR_SIZE(entityName, attrName, incorrectSize, correctSize)                               \
    do {                                                                                                               \
        std::string _safe_entityName_(entityName);                                                                     \
        std::string _safe_attrName_(attrName);                                                                         \
        std::string _safe_incorrectSize_(incorrectSize);                                                               \
        std::string _safe_correctSize_(correctSize);                                                                   \
        OP_LOGE_LIBOPAPI_REPORT(_safe_entityName_.c_str(),                                                             \
                                "Attribute %s of %s has incorrect size %s. It should be %s.", _safe_attrName_.c_str(), \
                                _safe_entityName_.c_str(), _safe_incorrectSize_.c_str(), _safe_correctSize_.c_str());  \
        const std::vector<const char*> msgKey = {"attr_name", "op_name", "incorrect_size", "correct_size"};            \
        const std::vector<const char*> msgvalue = {_safe_attrName_.c_str(), _safe_entityName_.c_str(),                 \
                                                   _safe_incorrectSize_.c_str(), _safe_correctSize_.c_str()};          \
        REPORT_PREDEFINED_ERR_MSG("EZ0003", msgKey, msgvalue);                                                         \
    } while (0)

/**
 * 此接口仅供算子或aclnn实现使用
 * entityName:string - 算子名称或aclnn接口名称
 * paramName:string
 * errMessage:Parameter [paramName] of [entityName] is required, but it is empty.
 */
#define OP_LOGE_WITH_INVALID_INPUT(entityName, paramName)                                                      \
    do {                                                                                                       \
        std::string _safe_entityName_(entityName);                                                             \
        std::string _safe_paramName_(paramName);                                                               \
        OP_LOGE_LIBOPAPI_REPORT(_safe_entityName_.c_str(), "Parameter %s of %s is required, but it is empty.", \
                                _safe_paramName_.c_str(), _safe_entityName_.c_str());                          \
        const std::vector<const char*> msgKey = {"param_name", "op_name"};                                     \
        const std::vector<const char*> msgvalue = {_safe_paramName_.c_str(), _safe_entityName_.c_str()};       \
        REPORT_PREDEFINED_ERR_MSG("EZ0004", msgKey, msgvalue);                                                 \
    } while (0)

/**
 * 此接口仅供算子或aclnn实现使用
 * entityName:string - 算子名称或aclnn接口名称
 * index:int
 * incorrectSize:string
 * correctSize:string
 * errMessage:The [index]th input of [entityName] has incorrect shape size [incorrectSize]. It should be
 * [correctSize].
 */
#define OP_LOGE_WITH_INVALID_INPUT_SHAPESIZE(entityName, index, incorrectSize, correctSize)                          \
    do {                                                                                                             \
        std::string _safe_entityName_(entityName);                                                                   \
        std::string _safe_incorrectSize_(incorrectSize);                                                             \
        std::string _safe_correctSize_(correctSize);                                                                 \
        std::string index_str = std::to_string(index);                                                               \
        OP_LOGE_LIBOPAPI_REPORT(                                                                                     \
            _safe_entityName_.c_str(), "The %sth input of %s has incorrect shape size %s. It should be %s.",         \
            index_str.c_str(), _safe_entityName_.c_str(), _safe_incorrectSize_.c_str(), _safe_correctSize_.c_str()); \
        const std::vector<const char*> msgKey = {"index", "op_name", "incorrect_size", "correct_size"};              \
        const std::vector<const char*> msgvalue = {index_str.c_str(), _safe_entityName_.c_str(),                     \
                                                   _safe_incorrectSize_.c_str(), _safe_correctSize_.c_str()};        \
        REPORT_PREDEFINED_ERR_MSG("EZ0005", msgKey, msgvalue);                                                       \
    } while (0)

/**
 * 此接口仅供算子或aclnn实现使用
 * entityName:string - 算子名称或aclnn接口名称
 * paramName:string
 * dataFormat:string
 * expectedFormatList:string
 * errMessage:Input parameter [paramName] of [entityName] has incorrect format [dataFormat]. It should
 * be [expectedFormatList].
 */
#define OP_LOGE_WITH_INVALID_INPUT_FORMAT(entityName, paramName, dataFormat, expectedFormatList)                  \
    do {                                                                                                          \
        std::string _safe_entityName_(entityName);                                                                \
        std::string _safe_paramName_(paramName);                                                                  \
        std::string _safe_dataFormat_(dataFormat);                                                                \
        std::string _safe_expectedFormatList_(expectedFormatList);                                                \
        OP_LOGE_LIBOPAPI_REPORT(_safe_entityName_.c_str(),                                                        \
                                "Input parameter %s of %s has incorrect format %s. It should be %s.",             \
                                _safe_paramName_.c_str(), _safe_entityName_.c_str(), _safe_dataFormat_.c_str(),   \
                                _safe_expectedFormatList_.c_str());                                               \
        const std::vector<const char*> msgKey = {"param_name", "op_name", "data_format", "expected_format_list"}; \
        const std::vector<const char*> msgvalue = {_safe_paramName_.c_str(), _safe_entityName_.c_str(),           \
                                                   _safe_dataFormat_.c_str(), _safe_expectedFormatList_.c_str()}; \
        REPORT_PREDEFINED_ERR_MSG("EZ0006", msgKey, msgvalue);                                                    \
    } while (0)

/**
 * 此接口仅供算子或aclnn实现使用
 * entityName:string - 算子名称或aclnn接口名称
 * paramName:string
 * dataDtype:string
 * expectedDtypeList:string
 * errMessage:Input parameter [paramName] of [entityName] has incorrect dtype [dataDtype]. It should
 * be [expectedDtypeList].
 */
#define OP_LOGE_WITH_INVALID_INPUT_DTYPE(entityName, paramName, dataDtype, expectedDtypeList)                   \
    do {                                                                                                        \
        std::string _safe_entityName_(entityName);                                                              \
        std::string _safe_paramName_(paramName);                                                                \
        std::string _safe_dataDtype_(dataDtype);                                                                \
        std::string _safe_expectedDtypeList_(expectedDtypeList);                                                \
        OP_LOGE_LIBOPAPI_REPORT(_safe_entityName_.c_str(),                                                      \
                                "Input parameter %s of %s has incorrect dtype %s. It should be %s.",            \
                                _safe_paramName_.c_str(), _safe_entityName_.c_str(), _safe_dataDtype_.c_str(),  \
                                _safe_expectedDtypeList_.c_str());                                              \
        const std::vector<const char*> msgKey = {"param_name", "op_name", "data_dtype", "expected_dtype_list"}; \
        const std::vector<const char*> msgvalue = {_safe_paramName_.c_str(), _safe_entityName_.c_str(),         \
                                                   _safe_dataDtype_.c_str(), _safe_expectedDtypeList_.c_str()}; \
        REPORT_PREDEFINED_ERR_MSG("EZ0007", msgKey, msgvalue);                                                  \
    } while (0)

// ============================================================================
// General Parameter Validation Macros (EZ0xxx)
// These macros do not distinguish between input/output and use generic "parameter" naming
// ============================================================================

/**
 * 此接口仅供算子或aclnn实现使用
 * EZ0008: Parameter shape error (single parameter, with correct shape)
 * entityName:string - 算子名称或aclnn接口名称
 * paramName:string - Parameter name
 * incorrectShape:string - Actual shape, format like "[1,128,128]"
 * correctShape:string - Expected shape, format like "[1,256,256]"
 * errMessage: Parameter [paramName] of [entityName] has incorrect shape [incorrectShape]. It should
 * be [correctShape].
 */
#define OP_LOGE_FOR_INVALID_SHAPE(entityName, paramName, incorrectShape, correctShape)                              \
    do {                                                                                                            \
        std::string _safe_entityName_(entityName);                                                                  \
        std::string _safe_paramName_(paramName);                                                                    \
        std::string _safe_incorrectShape_(incorrectShape);                                                          \
        std::string _safe_correctShape_(correctShape);                                                              \
        OP_LOGE_LIBOPAPI_REPORT(_safe_entityName_.c_str(),                                                          \
                                "Parameter %s of %s has incorrect shape %s. It should be %s.",                      \
                                _safe_paramName_.c_str(), _safe_entityName_.c_str(), _safe_incorrectShape_.c_str(), \
                                _safe_correctShape_.c_str());                                                       \
        const std::vector<const char*> msgKey = {"param_name", "op_name", "incorrect_shape", "correct_shape"};      \
        const std::vector<const char*> msgvalue = {_safe_paramName_.c_str(), _safe_entityName_.c_str(),             \
                                                   _safe_incorrectShape_.c_str(), _safe_correctShape_.c_str()};     \
        REPORT_PREDEFINED_ERR_MSG("EZ0008", msgKey, msgvalue);                                                      \
    } while (0)

/**
 * 此接口仅供算子或aclnn实现使用
 * EZ0009: Parameter shape error (single parameter, with reason)
 * entityName:string - 算子名称或aclnn接口名称
 * paramName:string - Parameter name
 * incorrectShape:string - Actual incorrect shape, format like "[1,128]"
 * reason:string - Reason for the error
 * errMessage: Parameter [paramName] of [entityName] has incorrect shape [incorrectShape]. Reason: [reason].
 */
#define OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(entityName, paramName, incorrectShape, reason)                         \
    do {                                                                                                             \
        std::string _safe_entityName_(entityName);                                                                   \
        std::string _safe_paramName_(paramName);                                                                     \
        std::string _safe_incorrectShape_(incorrectShape);                                                           \
        std::string _safe_reason_(reason);                                                                           \
        OP_LOGE_LIBOPAPI_REPORT(_safe_entityName_.c_str(), "Parameter %s of %s has incorrect shape %s. Reason: %s.", \
                                _safe_paramName_.c_str(), _safe_entityName_.c_str(), _safe_incorrectShape_.c_str(),  \
                                _safe_reason_.c_str());                                                              \
        const std::vector<const char*> msgKey = {"param_name", "op_name", "incorrect_shape", "reason"};              \
        const std::vector<const char*> msgvalue = {_safe_paramName_.c_str(), _safe_entityName_.c_str(),              \
                                                   _safe_incorrectShape_.c_str(), _safe_reason_.c_str()};            \
        REPORT_PREDEFINED_ERR_MSG("EZ0009", msgKey, msgvalue);                                                       \
    } while (0)

/**
 * 此接口仅供算子或aclnn实现使用
 * EZ0010: Parameters shape error (multiple parameters, with reason)
 * entityName:string - 算子名称或aclnn接口名称
 * paramNames:string - Parameter names
 * incorrectShapes:string - Actual incorrect shapes
 * reason:string - Reason for the error
 * errMessage: Parameters [paramNames] of [entityName] have incorrect shapes [incorrectShapes]. Reason:
 * [reason].
 */
#define OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(entityName, paramNames, incorrectShapes, reason)                       \
    do {                                                                                                              \
        std::string _safe_entityName_(entityName);                                                                    \
        std::string _safe_paramNames_(paramNames);                                                                    \
        std::string _safe_incorrectShapes_(incorrectShapes);                                                          \
        std::string _safe_reason_(reason);                                                                            \
        OP_LOGE_LIBOPAPI_REPORT(_safe_entityName_.c_str(),                                                            \
                                "Parameters %s of %s have incorrect shapes %s. Reason: %s.",                          \
                                _safe_paramNames_.c_str(), _safe_entityName_.c_str(), _safe_incorrectShapes_.c_str(), \
                                _safe_reason_.c_str());                                                               \
        const std::vector<const char*> msgKey = {"param_names", "op_name", "incorrect_shapes", "reason"};             \
        const std::vector<const char*> msgvalue = {_safe_paramNames_.c_str(), _safe_entityName_.c_str(),              \
                                                   _safe_incorrectShapes_.c_str(), _safe_reason_.c_str()};            \
        REPORT_PREDEFINED_ERR_MSG("EZ0010", msgKey, msgvalue);                                                        \
    } while (0)

/**
 * 此接口仅供算子或aclnn实现使用
 * EZ0011: Parameter shape dim error (single parameter, with correct dim)
 * entityName:string - 算子名称或aclnn接口名称
 * paramName:string - Parameter name
 * incorrectDim:string - Actual dimension, format like "2D" or "3D"
 * correctDim:string - Expected dimension, format like "4D"
 * errMessage: Parameter [paramName] of [entityName] has incorrect shape dim [incorrectDim]. It should
 * be [correctDim].
 */
#define OP_LOGE_FOR_INVALID_SHAPEDIM(entityName, paramName, incorrectDim, correctDim)                             \
    do {                                                                                                          \
        std::string _safe_entityName_(entityName);                                                                \
        std::string _safe_paramName_(paramName);                                                                  \
        std::string _safe_incorrectDim_(incorrectDim);                                                            \
        std::string _safe_correctDim_(correctDim);                                                                \
        OP_LOGE_LIBOPAPI_REPORT(_safe_entityName_.c_str(),                                                        \
                                "Parameter %s of %s has incorrect shape dim %s. It should be %s.",                \
                                _safe_paramName_.c_str(), _safe_entityName_.c_str(), _safe_incorrectDim_.c_str(), \
                                _safe_correctDim_.c_str());                                                       \
        const std::vector<const char*> msgKey = {"param_name", "op_name", "incorrect_dim", "correct_dim"};        \
        const std::vector<const char*> msgvalue = {_safe_paramName_.c_str(), _safe_entityName_.c_str(),           \
                                                   _safe_incorrectDim_.c_str(), _safe_correctDim_.c_str()};       \
        REPORT_PREDEFINED_ERR_MSG("EZ0011", msgKey, msgvalue);                                                    \
    } while (0)

/**
 * 此接口仅供算子或aclnn实现使用
 * EZ0012: Parameter shape dim error (single parameter, with reason)
 * entityName:string - 算子名称或aclnn接口名称
 * paramName:string - Parameter name
 * incorrectDim:string - Actual incorrect dimension
 * reason:string - Reason for the error
 * errMessage: Parameter [paramName] of [entityName] has incorrect shape dim [incorrectDim]. Reason: [reason].
 */
#define OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(entityName, paramName, incorrectDim, reason)                         \
    do {                                                                                                              \
        std::string _safe_entityName_(entityName);                                                                    \
        std::string _safe_paramName_(paramName);                                                                      \
        std::string _safe_incorrectDim_(incorrectDim);                                                                \
        std::string _safe_reason_(reason);                                                                            \
        OP_LOGE_LIBOPAPI_REPORT(                                                                                      \
            _safe_entityName_.c_str(), "Parameter %s of %s has incorrect shape dim %s. Reason: %s.",                  \
            _safe_paramName_.c_str(), _safe_entityName_.c_str(), _safe_incorrectDim_.c_str(), _safe_reason_.c_str()); \
        const std::vector<const char*> msgKey = {"param_name", "op_name", "incorrect_dim", "reason"};                 \
        const std::vector<const char*> msgvalue = {_safe_paramName_.c_str(), _safe_entityName_.c_str(),               \
                                                   _safe_incorrectDim_.c_str(), _safe_reason_.c_str()};               \
        REPORT_PREDEFINED_ERR_MSG("EZ0012", msgKey, msgvalue);                                                        \
    } while (0)

/**
 * 此接口仅供算子或aclnn实现使用
 * EZ0013: Parameters shape dim error (multiple parameters, with reason)
 * entityName:string - 算子名称或aclnn接口名称
 * paramNames:string - Parameter names
 * incorrectDims:string - Actual incorrect dimensions
 * reason:string - Reason for the error
 * errMessage: Parameters [paramNames] of [entityName] have incorrect shape dims [incorrectDims]. Reason:
 * [reason].
 */
#define OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(entityName, paramNames, incorrectDims, reason)                    \
    do {                                                                                                            \
        std::string _safe_entityName_(entityName);                                                                  \
        std::string _safe_paramNames_(paramNames);                                                                  \
        std::string _safe_incorrectDims_(incorrectDims);                                                            \
        std::string _safe_reason_(reason);                                                                          \
        OP_LOGE_LIBOPAPI_REPORT(_safe_entityName_.c_str(),                                                          \
                                "Parameters %s of %s have incorrect shape dims %s. Reason: %s.",                    \
                                _safe_paramNames_.c_str(), _safe_entityName_.c_str(), _safe_incorrectDims_.c_str(), \
                                _safe_reason_.c_str());                                                             \
        const std::vector<const char*> msgKey = {"param_names", "op_name", "incorrect_dims", "reason"};             \
        const std::vector<const char*> msgvalue = {_safe_paramNames_.c_str(), _safe_entityName_.c_str(),            \
                                                   _safe_incorrectDims_.c_str(), _safe_reason_.c_str()};            \
        REPORT_PREDEFINED_ERR_MSG("EZ0013", msgKey, msgvalue);                                                      \
    } while (0)

/**
 * 此接口仅供算子或aclnn实现使用
 * EZ0014: Parameter shape size error (single parameter, with correct size)
 * entityName:string - 算子名称或aclnn接口名称
 * paramName:string - Parameter name
 * incorrectSize:string - Actual size (total element count)
 * correctSize:string - Expected size (total element count)
 * errMessage: Parameter [paramName] of [entityName] has incorrect shape size [incorrectSize]. It
 * should be [correctSize].
 */
#define OP_LOGE_FOR_INVALID_SHAPESIZE(entityName, paramName, incorrectSize, correctSize)                           \
    do {                                                                                                           \
        std::string _safe_entityName_(entityName);                                                                 \
        std::string _safe_paramName_(paramName);                                                                   \
        std::string _safe_incorrectSize_(incorrectSize);                                                           \
        std::string _safe_correctSize_(correctSize);                                                               \
        OP_LOGE_LIBOPAPI_REPORT(_safe_entityName_.c_str(),                                                         \
                                "Parameter %s of %s has incorrect shape size %s. It should be %s.",                \
                                _safe_paramName_.c_str(), _safe_entityName_.c_str(), _safe_incorrectSize_.c_str(), \
                                _safe_correctSize_.c_str());                                                       \
        const std::vector<const char*> msgKey = {"param_name", "op_name", "incorrect_size", "correct_size"};       \
        const std::vector<const char*> msgvalue = {_safe_paramName_.c_str(), _safe_entityName_.c_str(),            \
                                                   _safe_incorrectSize_.c_str(), _safe_correctSize_.c_str()};      \
        REPORT_PREDEFINED_ERR_MSG("EZ0014", msgKey, msgvalue);                                                     \
    } while (0)

/**
 * 此接口仅供算子或aclnn实现使用
 * EZ0015: Parameter shape size error (single parameter, with reason)
 * entityName:string - 算子名称或aclnn接口名称
 * paramName:string - Parameter name
 * incorrectSize:string - Actual incorrect shape size
 * reason:string - Reason for the error
 * errMessage: Parameter [paramName] of [entityName] has incorrect shape size [incorrectSize]. Reason:
 * [reason].
 */
#define OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(entityName, paramName, incorrectSize, reason)                        \
    do {                                                                                                               \
        std::string _safe_entityName_(entityName);                                                                     \
        std::string _safe_paramName_(paramName);                                                                       \
        std::string _safe_incorrectSize_(incorrectSize);                                                               \
        std::string _safe_reason_(reason);                                                                             \
        OP_LOGE_LIBOPAPI_REPORT(                                                                                       \
            _safe_entityName_.c_str(), "Parameter %s of %s has incorrect shape size %s. Reason: %s.",                  \
            _safe_paramName_.c_str(), _safe_entityName_.c_str(), _safe_incorrectSize_.c_str(), _safe_reason_.c_str()); \
        const std::vector<const char*> msgKey = {"param_name", "op_name", "incorrect_size", "reason"};                 \
        const std::vector<const char*> msgvalue = {_safe_paramName_.c_str(), _safe_entityName_.c_str(),                \
                                                   _safe_incorrectSize_.c_str(), _safe_reason_.c_str()};               \
        REPORT_PREDEFINED_ERR_MSG("EZ0015", msgKey, msgvalue);                                                         \
    } while (0)

/**
 * 此接口仅供算子或aclnn实现使用
 * EZ0016: Parameters shape size error (multiple parameters, with reason)
 * entityName:string - 算子名称或aclnn接口名称
 * paramNames:string - Parameter names
 * incorrectSizes:string - Actual incorrect sizes
 * reason:string - Reason for the error
 * errMessage: Parameters [paramNames] of [entityName] have incorrect shape sizes [incorrectSizes]. Reason:
 * [reason].
 */
#define OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(entityName, paramNames, incorrectSizes, reason)                   \
    do {                                                                                                             \
        std::string _safe_entityName_(entityName);                                                                   \
        std::string _safe_paramNames_(paramNames);                                                                   \
        std::string _safe_incorrectSizes_(incorrectSizes);                                                           \
        std::string _safe_reason_(reason);                                                                           \
        OP_LOGE_LIBOPAPI_REPORT(_safe_entityName_.c_str(),                                                           \
                                "Parameters %s of %s have incorrect shape sizes %s. Reason: %s.",                    \
                                _safe_paramNames_.c_str(), _safe_entityName_.c_str(), _safe_incorrectSizes_.c_str(), \
                                _safe_reason_.c_str());                                                              \
        const std::vector<const char*> msgKey = {"param_names", "op_name", "incorrect_sizes", "reason"};             \
        const std::vector<const char*> msgvalue = {_safe_paramNames_.c_str(), _safe_entityName_.c_str(),             \
                                                   _safe_incorrectSizes_.c_str(), _safe_reason_.c_str()};            \
        REPORT_PREDEFINED_ERR_MSG("EZ0016", msgKey, msgvalue);                                                       \
    } while (0)

/**
 * 此接口仅供算子或aclnn实现使用
 * EZ0017: Parameter format error (single parameter, with correct format)
 * entityName:string - 算子名称或aclnn接口名称
 * paramName:string - Parameter name
 * incorrectFormat:string - Actual format (e.g., "NHWC")
 * correctFormat:string - Expected format (e.g., "NCHW")
 * errMessage: Parameter [paramName] of [entityName] has incorrect format [incorrectFormat]. It should
 * be [correctFormat].
 */
#define OP_LOGE_FOR_INVALID_FORMAT(entityName, paramName, incorrectFormat, correctFormat)                            \
    do {                                                                                                             \
        std::string _safe_entityName_(entityName);                                                                   \
        std::string _safe_paramName_(paramName);                                                                     \
        std::string _safe_incorrectFormat_(incorrectFormat);                                                         \
        std::string _safe_correctFormat_(correctFormat);                                                             \
        OP_LOGE_LIBOPAPI_REPORT(_safe_entityName_.c_str(),                                                           \
                                "Parameter %s of %s has incorrect format %s. It should be %s.",                      \
                                _safe_paramName_.c_str(), _safe_entityName_.c_str(), _safe_incorrectFormat_.c_str(), \
                                _safe_correctFormat_.c_str());                                                       \
        const std::vector<const char*> msgKey = {"param_name", "op_name", "incorrect_format", "correct_format"};     \
        const std::vector<const char*> msgvalue = {_safe_paramName_.c_str(), _safe_entityName_.c_str(),              \
                                                   _safe_incorrectFormat_.c_str(), _safe_correctFormat_.c_str()};    \
        REPORT_PREDEFINED_ERR_MSG("EZ0017", msgKey, msgvalue);                                                       \
    } while (0)

/**
 * 此接口仅供算子或aclnn实现使用
 * EZ0035: Parameter format error (single parameter, with reason)
 * entityName:string - 算子名称或aclnn接口名称
 * paramName:string - Parameter name
 * incorrectFormat:string - Actual incorrect format
 * reason:string - Reason for the error
 * errMessage: Parameter [paramName] of [entityName] has incorrect format [incorrectFormat]. Reason: [reason].
 */
#define OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(entityName, paramName, incorrectFormat, reason)                        \
    do {                                                                                                              \
        std::string _safe_entityName_(entityName);                                                                    \
        std::string _safe_paramName_(paramName);                                                                      \
        std::string _safe_incorrectFormat_(incorrectFormat);                                                          \
        std::string _safe_reason_(reason);                                                                            \
        OP_LOGE_LIBOPAPI_REPORT(_safe_entityName_.c_str(), "Parameter %s of %s has incorrect format %s. Reason: %s.", \
                                _safe_paramName_.c_str(), _safe_entityName_.c_str(), _safe_incorrectFormat_.c_str(),  \
                                _safe_reason_.c_str());                                                               \
        const std::vector<const char*> msgKey = {"param_name", "op_name", "incorrect_format", "reason"};              \
        const std::vector<const char*> msgvalue = {_safe_paramName_.c_str(), _safe_entityName_.c_str(),               \
                                                   _safe_incorrectFormat_.c_str(), _safe_reason_.c_str()};            \
        REPORT_PREDEFINED_ERR_MSG("EZ0035", msgKey, msgvalue);                                                        \
    } while (0)

/**
 * 此接口仅供算子或aclnn实现使用
 * EZ0018: Parameters format error (multiple parameters, with reason)
 * entityName:string - 算子名称或aclnn接口名称
 * paramNames:string - Parameter names
 * incorrectFormats:string - Actual incorrect formats
 * reason:string - Reason for the error
 * errMessage: Parameters [paramNames] of [entityName] have incorrect formats [incorrectFormats]. Reason:
 * [reason].
 */
#define OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(entityName, paramNames, incorrectFormats, reason)                      \
    do {                                                                                                               \
        std::string _safe_entityName_(entityName);                                                                     \
        std::string _safe_paramNames_(paramNames);                                                                     \
        std::string _safe_incorrectFormats_(incorrectFormats);                                                         \
        std::string _safe_reason_(reason);                                                                             \
        OP_LOGE_LIBOPAPI_REPORT(_safe_entityName_.c_str(),                                                             \
                                "Parameters %s of %s have incorrect formats %s. Reason: %s.",                          \
                                _safe_paramNames_.c_str(), _safe_entityName_.c_str(), _safe_incorrectFormats_.c_str(), \
                                _safe_reason_.c_str());                                                                \
        const std::vector<const char*> msgKey = {"param_names", "op_name", "incorrect_formats", "reason"};             \
        const std::vector<const char*> msgvalue = {_safe_paramNames_.c_str(), _safe_entityName_.c_str(),               \
                                                   _safe_incorrectFormats_.c_str(), _safe_reason_.c_str()};            \
        REPORT_PREDEFINED_ERR_MSG("EZ0018", msgKey, msgvalue);                                                         \
    } while (0)

/**
 * 此接口仅供算子或aclnn实现使用
 * EZ0019: Parameter dtype error (single parameter, with correct dtype)
 * entityName:string - 算子名称或aclnn接口名称
 * paramName:string - Parameter name
 * incorrectDtype:string - Actual data type (e.g., "INT32")
 * correctDtype:string - Expected data type (e.g., "FLOAT16")
 * errMessage: Parameter [paramName] of [entityName] has incorrect dtype [incorrectDtype]. It should
 * be [correctDtype].
 */
#define OP_LOGE_FOR_INVALID_DTYPE(entityName, paramName, incorrectDtype, correctDtype)                              \
    do {                                                                                                            \
        std::string _safe_entityName_(entityName);                                                                  \
        std::string _safe_paramName_(paramName);                                                                    \
        std::string _safe_incorrectDtype_(incorrectDtype);                                                          \
        std::string _safe_correctDtype_(correctDtype);                                                              \
        OP_LOGE_LIBOPAPI_REPORT(_safe_entityName_.c_str(),                                                          \
                                "Parameter %s of %s has incorrect dtype %s. It should be %s.",                      \
                                _safe_paramName_.c_str(), _safe_entityName_.c_str(), _safe_incorrectDtype_.c_str(), \
                                _safe_correctDtype_.c_str());                                                       \
        const std::vector<const char*> msgKey = {"param_name", "op_name", "incorrect_dtype", "correct_dtype"};      \
        const std::vector<const char*> msgvalue = {_safe_paramName_.c_str(), _safe_entityName_.c_str(),             \
                                                   _safe_incorrectDtype_.c_str(), _safe_correctDtype_.c_str()};     \
        REPORT_PREDEFINED_ERR_MSG("EZ0019", msgKey, msgvalue);                                                      \
    } while (0)

/**
 * 此接口仅供算子或aclnn实现使用
 * EZ0020: Parameter dtype error (single parameter, with reason)
 * entityName:string - 算子名称或aclnn接口名称
 * paramName:string - Parameter name
 * incorrectDtype:string - Actual incorrect data type
 * reason:string - Reason for the error
 * errMessage: Parameter [paramName] of [entityName] has incorrect dtype [incorrectDtype]. Reason: [reason].
 */
#define OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(entityName, paramName, incorrectDtype, reason)                         \
    do {                                                                                                             \
        std::string _safe_entityName_(entityName);                                                                   \
        std::string _safe_paramName_(paramName);                                                                     \
        std::string _safe_incorrectDtype_(incorrectDtype);                                                           \
        std::string _safe_reason_(reason);                                                                           \
        OP_LOGE_LIBOPAPI_REPORT(_safe_entityName_.c_str(), "Parameter %s of %s has incorrect dtype %s. Reason: %s.", \
                                _safe_paramName_.c_str(), _safe_entityName_.c_str(), _safe_incorrectDtype_.c_str(),  \
                                _safe_reason_.c_str());                                                              \
        const std::vector<const char*> msgKey = {"param_name", "op_name", "incorrect_dtype", "reason"};              \
        const std::vector<const char*> msgvalue = {_safe_paramName_.c_str(), _safe_entityName_.c_str(),              \
                                                   _safe_incorrectDtype_.c_str(), _safe_reason_.c_str()};            \
        REPORT_PREDEFINED_ERR_MSG("EZ0020", msgKey, msgvalue);                                                       \
    } while (0)

/**
 * 此接口仅供算子或aclnn实现使用
 * EZ0021: Parameters dtype error (multiple parameters, with reason)
 * entityName:string - 算子名称或aclnn接口名称
 * paramNames:string - Parameter names
 * incorrectDtypes:string - Actual incorrect data types
 * reason:string - Reason for the error
 * errMessage: Parameters [paramNames] of [entityName] have incorrect dtypes [incorrectDtypes]. Reason:
 * [reason].
 */
#define OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(entityName, paramNames, incorrectDtypes, reason)                       \
    do {                                                                                                              \
        std::string _safe_entityName_(entityName);                                                                    \
        std::string _safe_paramNames_(paramNames);                                                                    \
        std::string _safe_incorrectDtypes_(incorrectDtypes);                                                          \
        std::string _safe_reason_(reason);                                                                            \
        OP_LOGE_LIBOPAPI_REPORT(_safe_entityName_.c_str(),                                                            \
                                "Parameters %s of %s have incorrect dtypes %s. Reason: %s.",                          \
                                _safe_paramNames_.c_str(), _safe_entityName_.c_str(), _safe_incorrectDtypes_.c_str(), \
                                _safe_reason_.c_str());                                                               \
        const std::vector<const char*> msgKey = {"param_names", "op_name", "incorrect_dtypes", "reason"};             \
        const std::vector<const char*> msgvalue = {_safe_paramNames_.c_str(), _safe_entityName_.c_str(),              \
                                                   _safe_incorrectDtypes_.c_str(), _safe_reason_.c_str()};            \
        REPORT_PREDEFINED_ERR_MSG("EZ0021", msgKey, msgvalue);                                                        \
    } while (0)

/**
 * 此接口仅供算子或aclnn实现使用
 * EZ0022: Parameter tensor num error (single parameter, with correct num)
 * entityName:string - 算子名称或aclnn接口名称
 * paramName:string - Parameter name
 * incorrectNum:int64_t - Actual tensor count
 * correctNum:string - Expected tensor count
 * errMessage: Parameter [paramName] of [entityName] has invalid tensor num [incorrectNum]. It should
 * be [correctNum].
 */
#define OP_LOGE_FOR_INVALID_TENSORNUM(entityName, paramName, incorrectNum, correctNum)                     \
    do {                                                                                                   \
        std::string _safe_entityName_(entityName);                                                         \
        std::string _safe_paramName_(paramName);                                                           \
        std::string _safe_correctNum_(correctNum);                                                         \
        OP_LOGE_LIBOPAPI_REPORT(_safe_entityName_.c_str(),                                                 \
                                "Parameter %s of %s has invalid tensor num %ld. It should be %s.",         \
                                _safe_paramName_.c_str(), _safe_entityName_.c_str(),                       \
                                static_cast<int64_t>(incorrectNum), _safe_correctNum_.c_str());            \
        const std::vector<const char*> msgKey = {"param_name", "op_name", "incorrect_num", "correct_num"}; \
        std::string incorrectNumStr = std::to_string(static_cast<int64_t>(incorrectNum));                  \
        const std::vector<const char*> msgvalue = {_safe_paramName_.c_str(), _safe_entityName_.c_str(),    \
                                                   incorrectNumStr.c_str(), _safe_correctNum_.c_str()};    \
        REPORT_PREDEFINED_ERR_MSG("EZ0022", msgKey, msgvalue);                                             \
    } while (0)

/**
 * 此接口仅供算子或aclnn实现使用
 * EZ0023: Parameters tensor num error (multiple parameters, with reason)
 * entityName:string - 算子名称或aclnn接口名称
 * paramNames:string - Parameter names
 * incorrectNums:string - Actual incorrect tensor counts
 * reason:string - Reason for the error
 * errMessage: Parameters [paramNames] of [entityName] have invalid tensor nums [incorrectNums]. Reason:
 * [reason].
 */
#define OP_LOGE_FOR_INVALID_TENSORNUMS_WITH_REASON(entityName, paramNames, incorrectNums, reason)                   \
    do {                                                                                                            \
        std::string _safe_entityName_(entityName);                                                                  \
        std::string _safe_paramNames_(paramNames);                                                                  \
        std::string _safe_incorrectNums_(incorrectNums);                                                            \
        std::string _safe_reason_(reason);                                                                          \
        OP_LOGE_LIBOPAPI_REPORT(_safe_entityName_.c_str(),                                                          \
                                "Parameters %s of %s have invalid tensor nums %s. Reason: %s.",                     \
                                _safe_paramNames_.c_str(), _safe_entityName_.c_str(), _safe_incorrectNums_.c_str(), \
                                _safe_reason_.c_str());                                                             \
        const std::vector<const char*> msgKey = {"param_names", "op_name", "incorrect_nums", "reason"};             \
        const std::vector<const char*> msgvalue = {_safe_paramNames_.c_str(), _safe_entityName_.c_str(),            \
                                                   _safe_incorrectNums_.c_str(), _safe_reason_.c_str()};            \
        REPORT_PREDEFINED_ERR_MSG("EZ0023", msgKey, msgvalue);                                                      \
    } while (0)

/**
 * 此接口仅供算子或aclnn实现使用
 * EZ0024: Parameter value error (single parameter, with correct value)
 * entityName:string - 算子名称或aclnn接口名称
 * paramName:string - Parameter name
 * incorrectValue:string - Actual incorrect value
 * correctValue:string - Expected correct value
 * errMessage: Parameter [paramName] of [entityName] has incorrect value [incorrectValue]. It should
 * be [correctValue].
 */
#define OP_LOGE_FOR_INVALID_VALUE(entityName, paramName, incorrectValue, correctValue)                              \
    do {                                                                                                            \
        std::string _safe_entityName_(entityName);                                                                  \
        std::string _safe_paramName_(paramName);                                                                    \
        std::string _safe_incorrectValue_(incorrectValue);                                                          \
        std::string _safe_correctValue_(correctValue);                                                              \
        OP_LOGE_LIBOPAPI_REPORT(_safe_entityName_.c_str(),                                                          \
                                "Parameter %s of %s has incorrect value %s. It should be %s.",                      \
                                _safe_paramName_.c_str(), _safe_entityName_.c_str(), _safe_incorrectValue_.c_str(), \
                                _safe_correctValue_.c_str());                                                       \
        const std::vector<const char*> msgKey = {"param_name", "op_name", "incorrect_value", "correct_value"};      \
        const std::vector<const char*> msgvalue = {_safe_paramName_.c_str(), _safe_entityName_.c_str(),             \
                                                   _safe_incorrectValue_.c_str(), _safe_correctValue_.c_str()};     \
        REPORT_PREDEFINED_ERR_MSG("EZ0024", msgKey, msgvalue);                                                      \
    } while (0)

/**
 * 此接口仅供算子或aclnn实现使用
 * EZ0025: Parameter list size error (single parameter, with correct size)
 * entityName:string - 算子名称或aclnn接口名称
 * paramName:string - Parameter name
 * incorrectSize:string - Actual incorrect list size
 * correctSize:string - Expected correct list size
 * errMessage: Parameter [paramName] of [entityName] has invalid list size [incorrectSize]. It should
 * be [correctSize].
 */
#define OP_LOGE_FOR_INVALID_LISTSIZE(entityName, paramName, incorrectSize, correctSize)                            \
    do {                                                                                                           \
        std::string _safe_entityName_(entityName);                                                                 \
        std::string _safe_paramName_(paramName);                                                                   \
        std::string _safe_incorrectSize_(incorrectSize);                                                           \
        std::string _safe_correctSize_(correctSize);                                                               \
        OP_LOGE_LIBOPAPI_REPORT(_safe_entityName_.c_str(),                                                         \
                                "Parameter %s of %s has incorrect element nums %s. It should be %s.",              \
                                _safe_paramName_.c_str(), _safe_entityName_.c_str(), _safe_incorrectSize_.c_str(), \
                                _safe_correctSize_.c_str());                                                       \
        const std::vector<const char*> msgKey = {"param_name", "op_name", "incorrect_size", "correct_size"};       \
        const std::vector<const char*> msgvalue = {_safe_paramName_.c_str(), _safe_entityName_.c_str(),            \
                                                   _safe_incorrectSize_.c_str(), _safe_correctSize_.c_str()};      \
        REPORT_PREDEFINED_ERR_MSG("EZ0025", msgKey, msgvalue);                                                     \
    } while (0)

/**
 * 此接口仅供算子或aclnn实现使用
 * EZ0026: Parameter value error (single parameter, with reason)
 * entityName:string - 算子名称或aclnn接口名称
 * paramName:string - Parameter name
 * incorrectValue:string - Actual incorrect value
 * reason:string - Reason for the error
 * errMessage: Parameter [paramName] of [entityName] has incorrect value [incorrectValue]. Reason: [reason].
 */
#define OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(entityName, paramName, incorrectValue, reason)                         \
    do {                                                                                                             \
        std::string _safe_entityName_(entityName);                                                                   \
        std::string _safe_paramName_(paramName);                                                                     \
        std::string _safe_incorrectValue_(incorrectValue);                                                           \
        std::string _safe_reason_(reason);                                                                           \
        OP_LOGE_LIBOPAPI_REPORT(_safe_entityName_.c_str(), "Parameter %s of %s has incorrect value %s. Reason: %s.", \
                                _safe_paramName_.c_str(), _safe_entityName_.c_str(), _safe_incorrectValue_.c_str(),  \
                                _safe_reason_.c_str());                                                              \
        const std::vector<const char*> msgKey = {"param_name", "op_name", "incorrect_value", "reason"};              \
        const std::vector<const char*> msgvalue = {_safe_paramName_.c_str(), _safe_entityName_.c_str(),              \
                                                   _safe_incorrectValue_.c_str(), _safe_reason_.c_str()};            \
        REPORT_PREDEFINED_ERR_MSG("EZ0026", msgKey, msgvalue);                                                       \
    } while (0)

/**
 * 此接口仅供算子或aclnn实现使用
 * EZ0027: Parameters value error (multiple parameters, with reason)
 * entityName:string - 算子名称或aclnn接口名称
 * paramNames:string - Parameter names
 * incorrectValues:string - Actual incorrect values
 * reason:string - Reason for the error
 * errMessage: Parameters [paramNames] of [entityName] have incorrect values [incorrectValues]. Reason:
 * [reason].
 */
#define OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(entityName, paramNames, incorrectValues, reason)                       \
    do {                                                                                                              \
        std::string _safe_entityName_(entityName);                                                                    \
        std::string _safe_paramNames_(paramNames);                                                                    \
        std::string _safe_incorrectValues_(incorrectValues);                                                          \
        std::string _safe_reason_(reason);                                                                            \
        OP_LOGE_LIBOPAPI_REPORT(_safe_entityName_.c_str(),                                                            \
                                "Parameters %s of %s have incorrect values %s. Reason: %s.",                          \
                                _safe_paramNames_.c_str(), _safe_entityName_.c_str(), _safe_incorrectValues_.c_str(), \
                                _safe_reason_.c_str());                                                               \
        const std::vector<const char*> msgKey = {"param_names", "op_name", "incorrect_values", "reason"};             \
        const std::vector<const char*> msgvalue = {_safe_paramNames_.c_str(), _safe_entityName_.c_str(),              \
                                                   _safe_incorrectValues_.c_str(), _safe_reason_.c_str()};            \
        REPORT_PREDEFINED_ERR_MSG("EZ0027", msgKey, msgvalue);                                                        \
    } while (0)

/**
 * EZ0028: Parameter stride error
 * entityName:string - 算子名称或aclnn接口名称
 * paramName:string - Parameter name
 * incorrectStride:string - Actual stride (e.g., "[1,1,1,1]")
 * correctStride:string - Expected stride (e.g., "[1,1,1,1]")
 * errMessage: Parameter [paramName] of [entityName] has incorrect stride [incorrectStride],
 *             it should be [correctStride].
 */
#define OP_LOGE_FOR_INVALID_STRIDE(entityName, paramName, incorrectStride, correctStride)                            \
    do {                                                                                                             \
        std::string _safe_entityName_(entityName);                                                                   \
        std::string _safe_paramName_(paramName);                                                                     \
        std::string _safe_incorrectStride_(incorrectStride);                                                         \
        std::string _safe_correctStride_(correctStride);                                                             \
        OP_LOGE_LIBOPAPI_REPORT(_safe_entityName_.c_str(),                                                           \
                                "Parameter %s of %s has incorrect stride %s, it should be %s.",                      \
                                _safe_paramName_.c_str(), _safe_entityName_.c_str(), _safe_incorrectStride_.c_str(), \
                                _safe_correctStride_.c_str());                                                       \
        const std::vector<const char*> msgKey = {"param_name", "op_name", "incorrect_stride", "correct_stride"};     \
        const std::vector<const char*> msgvalue = {_safe_paramName_.c_str(), _safe_entityName_.c_str(),              \
                                                   _safe_incorrectStride_.c_str(), _safe_correctStride_.c_str()};    \
        REPORT_PREDEFINED_ERR_MSG("EZ0028", msgKey, msgvalue);                                                       \
    } while (0)

/**
 * EZ0029: File operation path error
 * filePath:string - File path
 * reason:string - Reason for the error
 * errMessage: Input path [filePath] is invalid. Reason: [reason].
 */
#define OP_LOGE_FOR_FILE_PATH(opName, filePath, reason)                                             \
    do {                                                                                            \
        std::string _safe_opName_(opName);                                                          \
        std::string _safe_filePath_(filePath);                                                      \
        std::string _safe_reason_(reason);                                                          \
        OP_LOGE_LIBOPAPI_REPORT(_safe_opName_.c_str(), "Input path %s is invalid. Reason: %s.",     \
                                _safe_filePath_.c_str(), _safe_reason_.c_str());                    \
        const std::vector<const char*> msgKey = {"file_path", "reason"};                            \
        const std::vector<const char*> msgvalue = {_safe_filePath_.c_str(), _safe_reason_.c_str()}; \
        REPORT_PREDEFINED_ERR_MSG("EZ0029", msgKey, msgvalue);                                      \
    } while (0)

/**
 * EZ0030: File operation open error
 * filePath:string - File path
 * reason:string - Reason for the error
 * errMessage: Failed to open file [filePath]. Reason: [reason].
 */
#define OP_LOGE_FOR_FILE_OPEN(opName, filePath, reason)                                                                \
    do {                                                                                                               \
        std::string _safe_opName_(opName);                                                                             \
        std::string _safe_filePath_(filePath);                                                                         \
        std::string _safe_reason_(reason);                                                                             \
        OP_LOGE_LIBOPAPI_REPORT(_safe_opName_.c_str(), "Failed to open file %s. Reason: %s.", _safe_filePath_.c_str(), \
                                _safe_reason_.c_str());                                                                \
        const std::vector<const char*> msgKey = {"file_path", "reason"};                                               \
        const std::vector<const char*> msgvalue = {_safe_filePath_.c_str(), _safe_reason_.c_str()};                    \
        REPORT_PREDEFINED_ERR_MSG("EZ0030", msgKey, msgvalue);                                                         \
    } while (0)

/**
 * EZ0031: File operation parse error
 * fileName:string - File name
 * reason:string - Reason for the error
 * errMessage: Failed to parse file [fileName]. Reason: [reason].
 */
#define OP_LOGE_FOR_FILE_PARSE(opName, fileName, reason)                                            \
    do {                                                                                            \
        std::string _safe_opName_(opName);                                                          \
        std::string _safe_fileName_(fileName);                                                      \
        std::string _safe_reason_(reason);                                                          \
        OP_LOGE_LIBOPAPI_REPORT(_safe_opName_.c_str(), "Failed to parse file %s. Reason: %s.",      \
                                _safe_fileName_.c_str(), _safe_reason_.c_str());                    \
        const std::vector<const char*> msgKey = {"file_name", "reason"};                            \
        const std::vector<const char*> msgvalue = {_safe_fileName_.c_str(), _safe_reason_.c_str()}; \
        REPORT_PREDEFINED_ERR_MSG("EZ0031", msgKey, msgvalue);                                      \
    } while (0)

/**
 * EZ0032: Config error (single item, with expect)
 * value:string - Actual incorrect value
 * item:string - configuration item name
 * configFile:string - configuration file path
 * expect:string - Expected value (e.g., "3")
 * errMessage: Value [value] of configuration item [item] in configuration file [configFile] is invalid,
 *             it should be [expect].
 */
#define OP_LOGE_FOR_INVALID_CONFIG(opName, configFile, item, value, expect)                               \
    do {                                                                                                  \
        std::string _safe_opName_(opName);                                                                \
        std::string _safe_configFile_(configFile);                                                        \
        std::string _safe_item_(item);                                                                    \
        std::string _safe_value_(value);                                                                  \
        std::string _safe_expect_(expect);                                                                \
        OP_LOGE_LIBOPAPI_REPORT(                                                                          \
            _safe_opName_.c_str(),                                                                        \
            "Value %s of configuration item %s in configuration file %s is invalid, it should be %s.",    \
            _safe_value_.c_str(), _safe_item_.c_str(), _safe_configFile_.c_str(), _safe_expect_.c_str()); \
        const std::vector<const char*> msgKey = {"value", "item", "config_file", "expect"};               \
        const std::vector<const char*> msgvalue = {_safe_value_.c_str(), _safe_item_.c_str(),             \
                                                   _safe_configFile_.c_str(), _safe_expect_.c_str()};     \
        REPORT_PREDEFINED_ERR_MSG("EZ0032", msgKey, msgvalue);                                            \
    } while (0)

/**
 * EZ0033: Config error (single item, with reason)
 * value:string - Actual incorrect value
 * item:string - configuration item name
 * configFile:string - configuration file path
 * reason:string - Reason for the error
 * errMessage: Value [value] of configuration item [item] in
 *             configuration file [configFile] is invalid. Reason: [reason].
 */
#define OP_LOGE_FOR_INVALID_CONFIG_WITH_REASON(opName, configFile, item, value, reason)                               \
    do {                                                                                                              \
        std::string _safe_opName_(opName);                                                                            \
        std::string _safe_configFile_(configFile);                                                                    \
        std::string _safe_item_(item);                                                                                \
        std::string _safe_value_(value);                                                                              \
        std::string _safe_reason_(reason);                                                                            \
        OP_LOGE_LIBOPAPI_REPORT(_safe_opName_.c_str(),                                                                \
                                "Value %s of configuration item %s in configuration file %s is invalid. Reason: %s.", \
                                _safe_value_.c_str(), _safe_item_.c_str(), _safe_configFile_.c_str(),                 \
                                _safe_reason_.c_str());                                                               \
        const std::vector<const char*> msgKey = {"value", "item", "config_file", "reason"};                           \
        const std::vector<const char*> msgvalue = {_safe_value_.c_str(), _safe_item_.c_str(),                         \
                                                   _safe_configFile_.c_str(), _safe_reason_.c_str()};                 \
        REPORT_PREDEFINED_ERR_MSG("EZ0033", msgKey, msgvalue);                                                        \
    } while (0)

/**
 * EZ0034: Config error (multiple items, with reason)
 * values:string - Actual incorrect values
 * items:string - configuration item names
 * configFile:string - configuration file path
 * reason:string - Reason for the error
 * errMessage: Values [values] of configuration items [items] in
 *             configuration file [configFile] are invalid. Reason: [reason].
 */
#define OP_LOGE_FOR_INVALID_CONFIGS_WITH_REASON(opName, configFile, items, values, reason)                  \
    do {                                                                                                    \
        std::string _safe_opName_(opName);                                                                  \
        std::string _safe_configFile_(configFile);                                                          \
        std::string _safe_items_(items);                                                                    \
        std::string _safe_values_(values);                                                                  \
        std::string _safe_reason_(reason);                                                                  \
        OP_LOGE_LIBOPAPI_REPORT(                                                                            \
            _safe_opName_.c_str(),                                                                          \
            "Values %s of configuration items %s in configuration file %s are invalid. Reason: %s.",        \
            _safe_values_.c_str(), _safe_items_.c_str(), _safe_configFile_.c_str(), _safe_reason_.c_str()); \
        const std::vector<const char*> msgKey = {"values", "items", "config_file", "reason"};               \
        const std::vector<const char*> msgvalue = {_safe_values_.c_str(), _safe_items_.c_str(),             \
                                                   _safe_configFile_.c_str(), _safe_reason_.c_str()};       \
        REPORT_PREDEFINED_ERR_MSG("EZ0034", msgKey, msgvalue);                                              \
    } while (0)

/**
 * EZ0036: Graph node error (single parameter, with reason)
 * node_name:string - graph node name
 * reason:string - Reason for the error
 */
#define OP_LOGE_FOR_INVALID_GRAPH_NODE(nodeName, reason)                                                             \
    do {                                                                                                             \
        std::string _safe_nodeName_(nodeName);                                                                       \
        std::string _safe_reason_(reason);                                                                           \
        OP_LOGE_LIBOPAPI_REPORT(_safe_nodeName_.c_str(), "The input graph contains an invalid node %s. Reason: %s.", \
                                _safe_nodeName_.c_str(), _safe_reason_.c_str());                                     \
        const std::vector<const char*> msgKey = {"node_name", "reason"};                                             \
        const std::vector<const char*> msgvalue = {_safe_nodeName_.c_str(), _safe_reason_.c_str()};                  \
        REPORT_PREDEFINED_ERR_MSG("EZ0036", msgKey, msgvalue);                                                       \
    } while (0)

/**
 * EZ0037: Parameter argument error
 * entityName:string - 算子名称或aclnn接口名称
 * paramName:string - Parameter name
 * errMessage: Parameter [paramName] of [entityName] is invalid. Reason:
 * [reason].
 */
#define OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(entityName, paramName, reason)                              \
    do {                                                                                                     \
        std::string _safe_entityName_(entityName);                                                           \
        std::string _safe_paramName_(paramName);                                                             \
        std::string _safe_reason_(reason);                                                                   \
        OP_LOGE_LIBOPAPI_REPORT(_safe_entityName_.c_str(), "Parameter %s of %s is invalid. Reason: %s.",     \
                                _safe_paramName_.c_str(), _safe_entityName_.c_str(), _safe_reason_.c_str()); \
        const std::vector<const char*> msgKey = {"param_name", "op_name", "reason"};                         \
        const std::vector<const char*> msgvalue = {_safe_paramName_.c_str(), _safe_entityName_.c_str(),      \
                                                   _safe_reason_.c_str()};                                   \
        REPORT_PREDEFINED_ERR_MSG("EZ0037", msgKey, msgvalue);                                               \
    } while (0)

/**
 * 此接口仅供算子或aclnn实现使用
 * EZ0038: Parameter list size error (single parameter, with reason)
 * entityName:string - 算子名称或aclnn接口名称
 * paramName:string - Parameter name
 * incorrectSize:string - Actual incorrect list size
 * reason:string - Reason for the error
 * errMessage: Parameter [paramName] of [entityName] has invalid list size [incorrectSize]. Reason:
 * [reason].
 */
#define OP_LOGE_FOR_INVALID_LISTSIZE_WITH_REASON(entityName, paramName, incorrectSize, reason)                      \
    do {                                                                                                            \
        std::string _safe_entityName_(entityName);                                                                  \
        std::string _safe_paramNames_(paramNames);                                                                  \
        std::string _safe_incorrectSize_(incorrectSize);                                                            \
        std::string _safe_reason_(reason);                                                                          \
        OP_LOGE_LIBOPAPI_REPORT(_safe_entityName_.c_str(),                                                          \
                                "Parameter %s of %s has incorrect element nums %s. Reason: %s.",                    \
                                _safe_paramNames_.c_str(), _safe_entityName_.c_str(), _safe_incorrectSize_.c_str(), \
                                _safe_reason_.c_str());                                                             \
        const std::vector<const char*> msgKey = {"param_names", "op_name", "incorrect_size", "reason"};             \
        const std::vector<const char*> msgvalue = {_safe_paramNames_.c_str(), _safe_entityName_.c_str(),            \
                                                   _safe_incorrectSize_.c_str(), _safe_reason_.c_str()};            \
        REPORT_PREDEFINED_ERR_MSG("EZ0038", msgKey, msgvalue);                                                      \
    } while (0)

#ifndef OP_LOG_LIBOPAPI_ONLY
#define OpLogSub(moduleId, level, opInfo, fmt, ...)                                           \
    do {                                                                                      \
        if (CheckLogLevel(static_cast<int>(moduleId), level) == 1) {                          \
            DlogRecord(static_cast<int>(moduleId), level,                                     \
                       "[%s:%d][%s]"                                                          \
                       "[%s][%" PRIu64 "] OpName:[%s] " fmt,                                  \
                       __FILE__, __LINE__, OP_SUBMOD_NAME, __FUNCTION__, Ops::Base::GetTid(), \
                       Ops::Base::GetSafeStr(opInfo), ##__VA_ARGS__);                         \
        }                                                                                     \
    } while (0)

#define OpLogErrSub(moduleId, level, opInfo, fmt, ...)                                        \
    do {                                                                                      \
        if (CheckLogLevel(static_cast<int>(moduleId), level) == 1) {                          \
            DlogRecord(static_cast<int>(moduleId), level,                                     \
                       "[%s:%d][%s]"                                                          \
                       "[%s][%" PRIu64 "] OpName:[%s] " fmt,                                  \
                       __FILE__, __LINE__, OP_SUBMOD_NAME, __FUNCTION__, Ops::Base::GetTid(), \
                       Ops::Base::GetSafeStr(opInfo), ##__VA_ARGS__);                         \
        }                                                                                     \
    } while (0)

#define D_OP_LOGI(opName, fmt, ...) OpLogSub(OP_MODULE_ID, DLOG_INFO, opName, fmt, ##__VA_ARGS__)
#define D_OP_LOGW(opName, fmt, ...) OpLogSub(OP_MODULE_ID, DLOG_WARN, opName, fmt, ##__VA_ARGS__)
#define D_OP_LOGE(opName, fmt, ...) OpLogErrSub(OP_MODULE_ID, DLOG_ERROR, opName, fmt, ##__VA_ARGS__)
#define D_OP_LOGD(opName, fmt, ...) OpLogSub(OP_MODULE_ID, DLOG_DEBUG, opName, fmt, ##__VA_ARGS__)
#define unlikely(x) __builtin_expect((x), 0)
#define likely(x) __builtin_expect((x), 1)

#define OP_LOGI(opName, ...) D_OP_LOGI(Ops::Base::GetOpInfo(opName), __VA_ARGS__)
#define OP_LOGW(opName, ...) D_OP_LOGW(Ops::Base::GetOpInfo(opName), __VA_ARGS__)
#define OP_LOGE_WITHOUT_REPORT(opName, ...) OP_LOGE_LIBOPAPI_REPORT(opName, __VA_ARGS__)
#define OP_LOGE(opName, ...)                            \
    do {                                                \
        OP_LOGE_LIBOPAPI_REPORT(opName, ##__VA_ARGS__); \
        REPORT_INNER_ERR_MSG("EZ9999", ##__VA_ARGS__);  \
    } while (0)
#define OP_LOGD(opName, ...) D_OP_LOGD(Ops::Base::GetOpInfo(opName), __VA_ARGS__)
#define OP_CHECK_IF(condition, log, return_expr) \
    do {                                         \
        if (unlikely(condition)) {               \
            log;                                 \
            return_expr;                         \
        }                                        \
    } while (0)
#define OP_CHECK_NULL_WITH_CONTEXT(context, ptr)                                                           \
    do {                                                                                                   \
        if (unlikely((ptr) == nullptr)) {                                                                  \
            const char* name = (unlikely(((context) == nullptr) || (context)->GetNodeName() == nullptr)) ? \
                                   "nil" :                                                                 \
                                   (context)->GetNodeName();                                               \
            OP_LOGE(name, "%s is nullptr!", #ptr);                                                         \
            return ge::GRAPH_FAILED;                                                                       \
        }                                                                                                  \
    } while (0)
OPBASE_API std::string ToString(ge::DataType type);
OPBASE_API std::string ToString(ge::Format format);
OPBASE_API std::string ToString(const gert::Shape& shape);
OPBASE_API std::string ToString(const std::vector<const gert::Shape*>& v);
#endif
} // namespace Base
} // namespace Ops
#endif // OP_COMMON_LOG_LOG_H
