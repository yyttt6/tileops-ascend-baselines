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
#ifndef UTIL_LOG_H_
#define UTIL_LOG_H_

#if defined __CCE_KT_TEST__ || defined __ATP_UT__
#include <cxxabi.h>
#include <securec.h>
#include <memory>
#include <string>
#endif

namespace Ops {
namespace Base {
#define DO_JOIN_SYMBOL(symbol1, symbol2) symbol1##symbol2
#define JOIN_SYMBOL(symbol1, symbol2) DO_JOIN_SYMBOL(symbol1, symbol2)

#ifdef __COUNTER__
#define UNIQUE_ID __COUNTER__
#else
#define UNIQUE_ID __LINE__
#endif

#define UNIQUE_NAME(prefix) JOIN_SYMBOL(prefix, UNIQUE_ID)

#if defined __CCE_KT_TEST__ || defined __ATP_UT__
#ifdef __CCE_KT_TEST__
#define RUN_LOG_BASE(...)     \
    if (GetBlockIdx() == 0) { \
        printf(__VA_ARGS__);  \
    }
#else
#define RUN_LOG_BASE(...)    \
    do {                     \
        printf(__VA_ARGS__); \
    } while (0)
#endif

template <typename T>
const std::string GetTypeName()
{
    std::string result;
    const char* name = typeid(T).name();
    int status = -1;
    char* readableName = abi::__cxa_demangle(name, nullptr, nullptr, &status);
    if (status == 0 && readableName != nullptr) {
        result = std::string(readableName);
        std::free(readableName);
    } else {
        result = "[Undemangled]" + std::string(name);
    }
    return result;
}

#define PRINT_TYPE(T) Ops::Base::GetTypeName<T>().c_str()

#define RUN_LOG_ONE_BLOCK(...)                                                                \
    do {                                                                                      \
        const char* filename = strrchr(__FILE__, '/');                                        \
        if (!filename)                                                                        \
            filename = strrchr(__FILE__, '\\');                                               \
        filename = filename ? filename + 1 : __FILE__;                                        \
        if constexpr (sizeof(#__VA_ARGS__) <= 1) {                                            \
            printf("[INFO][Core0:%s:%d] (empty log)\n", filename, __LINE__);                  \
        } else {                                                                              \
            char buffer[1024];                                                                \
            int result = snprintf_s(buffer, sizeof(buffer), sizeof(buffer) - 1, __VA_ARGS__); \
            if (result < 0 || buffer[strlen(buffer) - 1] != '\n') {                           \
                printf("[INFO][Core0:%s:%d] %s\n", filename, __LINE__, buffer);               \
            } else {                                                                          \
                printf("[INFO][Core0:%s:%d] %s", filename, __LINE__, buffer);                 \
            }                                                                                 \
        }                                                                                     \
    } while (0)

#else
#define RUN_LOG_BASE(...)
#define PRINT_TYPE(T)
#define RUN_LOG_ONE_BLOCK(...)
#endif

/////////////////////////////////////////////////////////////////////////////

#ifdef __CCE_KT_TEST__
// CPU孪生调试时支持打印
#define RUN_LOG(...)                    \
    if (GetBlockIdx() == 0) {           \
        RUN_LOG_ONE_BLOCK(__VA_ARGS__); \
    }

template <typename T, T v>
struct Print2 {
    constexpr operator char() { return 1 + 0xFF; }
};
#define BUILD_LOG(...) char UNIQUE_NAME(print_value_) = Print2<__VA_ARGS__>()

#else // __CCE_KT_TEST__
#ifdef __ATP_UT__
#define RUN_LOG(...) RUN_LOG_ONE_BLOCK(__VA_ARGS__)
#else // __ATP_UT__
// 实际编译Kernel时不打印
#define RUN_LOG(...)
#endif // __ATP_UT__
#define BUILD_LOG(...)
#endif // __CCE_KT_TEST__
} // namespace Base
} // namespace Ops
#endif // UTIL_LOG_H_
