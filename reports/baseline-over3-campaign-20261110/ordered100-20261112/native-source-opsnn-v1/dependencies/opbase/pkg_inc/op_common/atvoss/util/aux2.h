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
 * \file aux2.h
 * \brief
 */
#ifndef UTIL_AUX_2_H_
#define UTIL_AUX_2_H_
#include "log.h"

namespace Ops {
namespace Base {

template <typename... Ts>
struct Elems;

namespace __aux {

template <class... _Types>
using Void_t = void;

template <class T, class U>
struct IsSameType {
    constexpr static bool Value = false;
};

template <class T>
struct IsSameType<T, T> {
    constexpr static bool Value = true;
};

template <typename T, template <typename, int...> typename U>
struct IsSameTemplateType {
    constexpr static bool Value = false;
};
template <typename U, template <typename, int...> typename T>
struct IsSameTemplateType<T<U>, T> {
    constexpr static bool Value = true;
};

template <bool v, typename True, typename False>
struct ConditionAux {};

template <typename True, typename False>
struct ConditionAux<true, True, False> {
    using Type = True;
};

template <typename True, typename False>
struct ConditionAux<false, True, False> {
    using Type = False;
};

template <bool v, typename True, typename False>
using Condition = typename ConditionAux<v, True, False>::Type;

template <int pos, int offset, typename... Ts>
struct GetElemAtAux {
    using Type = void;
};

template <int pos, int offset, typename T>
struct GetElemAtAux<pos, offset, T> {
    using Type = Condition<pos == offset, T, void>;
};

template <int pos, int offset, typename T, typename... Ts>
struct GetElemAtAux<pos, offset, T, Ts...> {
    using Type = Condition<pos == offset, T, typename GetElemAtAux<pos, offset + 1, Ts...>::Type>;
};

template <int pos, int offset, typename... Ts>
using GetElemAt = typename GetElemAtAux<pos, offset, Ts...>::Type;

template <typename Es, typename T>
struct ElemsExist {
    constexpr static bool Value = Es::template IsExist<T>();
};

template <typename Es, typename T>
struct ElemsNotExist {
    constexpr static bool Value = !Es::template IsExist<T>();
};

template <template <class> class F, class... Ts>
struct ForEachAux {};

template <template <class> class F, class T>
struct ForEachAux<F, T> {
    using Type = typename F<T>::Type;
};

template <template <class> class F, class T, class... Ts>
struct ForEachAux<F, T, Ts...> {
    using cur = typename ForEachAux<F, T>::Type;
    using Type = typename cur::template Concat<typename ForEachAux<F, Ts...>::Type>;
};

template <typename Tuple1, typename Tuple2>
struct ConcatEls {};

template <typename... Ts1, typename... Ts2>
struct ConcatEls<Elems<Ts1...>, Elems<Ts2...>> {
    using Type = Elems<Ts1..., Ts2...>;
};

template <typename T, typename UniqT = Elems<>>
struct UniqAux {};

template <typename UniqT>
struct UniqAux<Elems<>, UniqT> {
    using Type = UniqT;
};

template <typename T, typename... Ts, typename UniqT>
struct UniqAux<Elems<T, Ts...>, UniqT> {
    using next = Condition<UniqT::template IsExist<T>(), UniqT, typename ConcatEls<UniqT, Elems<T>>::Type>;
    using Type = typename UniqAux<Elems<Ts...>, next>::Type;
};

template <typename Tuple>
using UniqAuxT = typename UniqAux<Tuple>::Type;

template <typename Es, template <class, class> class Check, typename Filtered = Elems<>>
struct FilterAux {
    using Type = Filtered;
};

template <template <class, class> class Check, typename Filtered>
struct FilterAux<Elems<>, Check, Filtered> {
    using Type = Filtered;
};

template <typename T, typename... Ts, template <class, class> class Check, typename Filtered>
struct FilterAux<Elems<T, Ts...>, Check, Filtered> {
    using next = Condition<Check<Filtered, T>::Value, typename ConcatEls<Filtered, Elems<T>>::Type, Filtered>;
    using Type = typename FilterAux<Elems<Ts...>, Check, next>::Type;
};

template <int ReducePos, int pos>
struct CheckPre {
    constexpr static bool Value = pos <= ReducePos;
};

template <int ReducePos, int pos>
struct CheckPost {
    constexpr static bool Value = pos > ReducePos;
};

template <typename Es, int ReducePos, int pos, template <int, int> class Check, typename Filtered = Elems<>>
struct FilterDagAux {
    using Type = Filtered;
};

template <int ReducePos, int pos, template <int, int> class Check, typename Filtered>
struct FilterDagAux<Elems<>, ReducePos, pos, Check, Filtered> {
    using Type = Filtered;
};

template <typename T, typename... Ts, int ReducePos, int pos, template <int, int> class Check, typename Filtered>
struct FilterDagAux<Elems<T, Ts...>, ReducePos, pos, Check, Filtered> {
    using next = Condition<Check<ReducePos, pos>::Value, typename ConcatEls<Filtered, Elems<T>>::Type, Filtered>;
    using Type = typename FilterDagAux<Elems<Ts...>, ReducePos, pos + 1, Check, next>::Type;
};
// 删除Elems中首个节点。若不是相同节点，则返回元Elems
template <typename Es, typename ToPop>
struct PopFrontAux {};

template <typename ToPop>
struct PopFrontAux<Elems<>, ToPop> {
    using Type = Elems<>;
};

template <typename T>
struct PopFrontAux<Elems<T>, T> {
    using Type = Elems<>;
};

template <typename T, typename ToPop>
struct PopFrontAux<Elems<T>, ToPop> {
    using Type = Elems<T>;
};

template <typename T, typename... Ts>
struct PopFrontAux<Elems<T, Ts...>, T> {
    using Type = Elems<Ts...>;
};

template <typename T, typename... Ts, typename ToPop>
struct PopFrontAux<Elems<T, Ts...>, ToPop> {
    using Type = Elems<T, Ts...>;
};

} // namespace __aux
} // namespace Base
} // namespace Ops
#endif // UTIL_AUX_2_H_
