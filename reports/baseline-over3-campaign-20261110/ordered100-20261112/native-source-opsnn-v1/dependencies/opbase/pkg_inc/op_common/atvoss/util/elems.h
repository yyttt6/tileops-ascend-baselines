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
 * \file elems.h
 * \brief
 */
#ifndef UTIL_ELEMS_H_
#define UTIL_ELEMS_H_

#ifndef __CCE_AICORE__
#define __aicore__
#endif

#include <cstddef>
#include "aux2.h"
namespace Ops {
namespace Base {
template <typename... Ts>
struct Elems {
    using Type = Elems<Ts...>;
    constexpr static size_t Size = sizeof...(Ts);

    template <typename... T>
    using Append = Elems<Ts..., T...>;

    template <template <class...> class T>
    using Export = T<Ts...>;

    template <typename Es>
    using Concat = typename Es::template Export<Append>;

    template <int pos>
    using At = __aux::GetElemAt<pos, 0, Ts...>;

    template <typename T, int start = 0>
    __aicore__ constexpr static bool IsExist()
    {
        if constexpr (start < Size) {
            if constexpr (__aux::IsSameType<T, At<start>>::Value) {
                return true;
            }
            return IsExist<T, start + 1>();
        }
        return false;
    }

    template <typename T, int start = 0>
    __aicore__ constexpr static int GetIndex()
    {
        if constexpr (__aux::IsSameType<T, At<start>>::Value) {
            return start;
        } else if constexpr (start + 1 < Size) {
            return GetIndex<T, start + 1>();
        }
        return -1;
    }

    template <template <class, class> class Check, typename Filtered = Elems<>>
    using Filter = typename __aux::FilterAux<Elems<Ts...>, Check, Filtered>::Type;

    using Unique = __aux::UniqAuxT<Elems<Ts...>>;

    template <class Es>
    using Union = typename Concat<Es>::Unique;

private:
    template <class Es>
    struct _RemoveAux {
        template <typename E, typename T>
        struct Check {
            constexpr static bool Value = !Es::template IsExist<T>();
        };

        using Type = typename __aux::FilterAux<Elems<Ts...>, Check>::Type;
    };

public:
    template <class Es>
    using Remove = typename _RemoveAux<Es>::Type;

    template <template <class...> class F>
    using ForEach = Elems<typename F<Ts>::Type...>;

    template <typename ToPop>
    using PopFront = typename __aux::PopFrontAux<Elems<Ts...>, ToPop>::Type;
};
} // namespace Base
} // namespace Ops

#endif // UTIL_ELEMS_H_
