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
 * \file placeholder.h
 * \brief
 */
#ifndef UTIL_PLACEHOLDER_H_
#define UTIL_PLACEHOLDER_H_

#include <cstddef>
#include <cstdint>
#ifndef __CCE_AICORE__
#define __aicore__
#define __global__
#define __gm__
#endif
namespace Ops {
namespace Base {
namespace Placeholder {

#define CONCATENATE_DETAIL(name, idx) name##idx
#define MAKE_UNIQ_NAME(name, idx) CONCATENATE_DETAIL(name, idx)
#define MAKE_CONST(t, v)                                     \
    struct MAKE_UNIQ_NAME(ConcatValue_##t##_, __COUNTER__) { \
        constexpr static bool IsConstValue_ = true;          \
        static constexpr t value = v;                        \
    }

enum HolderScope { GM, VECTOR };

template <class T, class Attr_ = void, int cur = 0>
struct Holder {
    constexpr static int Pos = cur;
    constexpr static bool IsHolder = true;
    using DType = T;
    using Attr = Attr_;
};

template <HolderScope scope = HolderScope::GM, int isScalar = 0>
struct InAttr {
    constexpr static int IsScalar = isScalar;
    constexpr static HolderScope Scope = scope;
};

template <int isScalar = 0>
struct ScalarAttr : public InAttr<HolderScope::GM, isScalar> {};

template <HolderScope scope = HolderScope::GM>
struct ScopeAttr : public InAttr<scope, 0> {};

constexpr static int INPUT_HOLDER0 = 0;
constexpr static int INPUT_HOLDER1 = 1;
constexpr static int INPUT_HOLDER2 = 2;
constexpr static int INPUT_HOLDER3 = 3;
constexpr static int INPUT_HOLDER4 = 4;
constexpr static int INPUT_HOLDER5 = 5;
constexpr static int INPUT_HOLDER6 = 6;
constexpr static int INPUT_HOLDER7 = 7;
constexpr static int INPUT_HOLDER8 = 8;
constexpr static int INPUT_HOLDER9 = 9;
constexpr static int INPUT_HOLDER10 = 10;
constexpr static int INPUT_HOLDER11 = 11;

/** GM输入， 存在Scalar输入 */
template <class T, class Attr = InAttr<>>
struct In0 : public Holder<T, Attr, INPUT_HOLDER0> {};
template <class T, class Attr = InAttr<>>
struct In1 : public Holder<T, Attr, INPUT_HOLDER1> {};
template <class T, class Attr = InAttr<>>
struct In2 : public Holder<T, Attr, INPUT_HOLDER2> {};
template <class T, class Attr = InAttr<>>
struct In3 : public Holder<T, Attr, INPUT_HOLDER3> {};
template <class T, class Attr = InAttr<>>
struct In4 : public Holder<T, Attr, INPUT_HOLDER4> {};
template <class T, class Attr = InAttr<>>
struct In5 : public Holder<T, Attr, INPUT_HOLDER5> {};
template <class T, class Attr = InAttr<>>
struct In6 : public Holder<T, Attr, INPUT_HOLDER6> {};
template <class T, class Attr = InAttr<>>
struct In7 : public Holder<T, Attr, INPUT_HOLDER7> {};
template <class T, class Attr = InAttr<>>
struct In8 : public Holder<T, Attr, INPUT_HOLDER8> {};
template <class T, class Attr = InAttr<>>
struct In9 : public Holder<T, Attr, INPUT_HOLDER9> {};
template <class T, class Attr = InAttr<>>
struct In10 : public Holder<T, Attr, INPUT_HOLDER10> {};
template <class T, class Attr = InAttr<>>
struct In11 : public Holder<T, Attr, INPUT_HOLDER11> {};

template <class T>
struct IsInHolder {
    constexpr static bool Value = false;
};
template <class U, class Attr, template <class, class> class T>
struct IsInHolder<T<U, Attr>> {
    constexpr static bool Value = false;
};
template <class U, class Attr>
struct IsInHolder<In0<U, Attr>> {
    constexpr static bool Value = true;
};
template <class U, class Attr>
struct IsInHolder<In1<U, Attr>> {
    constexpr static bool Value = true;
};
template <class U, class Attr>
struct IsInHolder<In2<U, Attr>> {
    constexpr static bool Value = true;
};
template <class U, class Attr>
struct IsInHolder<In3<U, Attr>> {
    constexpr static bool Value = true;
};
template <class U, class Attr>
struct IsInHolder<In4<U, Attr>> {
    constexpr static bool Value = true;
};
template <class U, class Attr>
struct IsInHolder<In5<U, Attr>> {
    constexpr static bool Value = true;
};
template <class U, class Attr>
struct IsInHolder<In6<U, Attr>> {
    constexpr static bool Value = true;
};
template <class U, class Attr>
struct IsInHolder<In7<U, Attr>> {
    constexpr static bool Value = true;
};
template <class U, class Attr>
struct IsInHolder<In8<U, Attr>> {
    constexpr static bool Value = true;
};
template <class U, class Attr>
struct IsInHolder<In9<U, Attr>> {
    constexpr static bool Value = true;
};
template <class U, class Attr>
struct IsInHolder<In10<U, Attr>> {
    constexpr static bool Value = true;
};
template <class U, class Attr>
struct IsInHolder<In11<U, Attr>> {
    constexpr static bool Value = true;
};

/** 判断输入是否是Scalar输入 */
template <class T>
struct IsInScalar {
    constexpr static bool Value = false;
};
template <class U, class Attr, template <class, class> class T>
struct IsInScalar<T<U, Attr>> {
    constexpr static bool Value = Attr::IsScalar == 1;
};

constexpr static int OUTPUT_HOLDER0 = 0;
constexpr static int OUTPUT_HOLDER1 = 1;
constexpr static int OUTPUT_HOLDER2 = 2;
constexpr static int OUTPUT_HOLDER3 = 3;
constexpr static int OUTPUT_HOLDER4 = 4;
constexpr static int OUTPUT_HOLDER5 = 5;
constexpr static int OUTPUT_HOLDER6 = 6;
constexpr static int OUTPUT_HOLDER7 = 7;
constexpr static int OUTPUT_HOLDER8 = 8;
constexpr static int OUTPUT_HOLDER9 = 9;

/** GM输出，不存在Scalar输出 */
template <typename T, class Attr = void, class Parent_ = void>
struct Out0 : public Holder<T, Attr, OUTPUT_HOLDER0> {
    using Parent = Parent_;
};
template <typename T, class Attr = void, class Parent_ = void>
struct Out1 : public Holder<T, Attr, OUTPUT_HOLDER1> {
    using Parent = Parent_;
};
template <typename T, class Attr = void, class Parent_ = void>
struct Out2 : public Holder<T, Attr, OUTPUT_HOLDER2> {
    using Parent = Parent_;
};
template <typename T, class Attr = void, class Parent_ = void>
struct Out3 : public Holder<T, Attr, OUTPUT_HOLDER3> {
    using Parent = Parent_;
};
template <typename T, class Attr = void, class Parent_ = void>
struct Out4 : public Holder<T, Attr, OUTPUT_HOLDER4> {
    using Parent = Parent_;
};
template <typename T, class Attr = void, class Parent_ = void>
struct Out5 : public Holder<T, Attr, OUTPUT_HOLDER5> {
    using Parent = Parent_;
};
template <typename T, class Attr = void, class Parent_ = void>
struct Out6 : public Holder<T, Attr, OUTPUT_HOLDER6> {
    using Parent = Parent_;
};
template <typename T, class Attr = void, class Parent_ = void>
struct Out7 : public Holder<T, Attr, OUTPUT_HOLDER7> {
    using Parent = Parent_;
};
template <typename T, class Attr = void, class Parent_ = void>
struct Out8 : public Holder<T, Attr, OUTPUT_HOLDER8> {
    using Parent = Parent_;
};
template <typename T, class Attr = void, class Parent_ = void>
struct Out9 : public Holder<T, Attr, OUTPUT_HOLDER9> {
    using Parent = Parent_;
};

template <class T>
struct IsOutHolder {
    constexpr static bool Value = false;
};
template <class U, class Attr, class P, template <class, class, class> class T>
struct IsOutHolder<T<U, Attr, P>> {
    constexpr static bool Value = false;
};
template <class U, class Attr, class P>
struct IsOutHolder<Out0<U, Attr, P>> {
    constexpr static bool Value = true;
};
template <class U, class Attr, class P>
struct IsOutHolder<Out1<U, Attr, P>> {
    constexpr static bool Value = true;
};
template <class U, class Attr, class P>
struct IsOutHolder<Out2<U, Attr, P>> {
    constexpr static bool Value = true;
};
template <class U, class Attr, class P>
struct IsOutHolder<Out3<U, Attr, P>> {
    constexpr static bool Value = true;
};
template <class U, class Attr, class P>
struct IsOutHolder<Out4<U, Attr, P>> {
    constexpr static bool Value = true;
};
template <class U, class Attr, class P>
struct IsOutHolder<Out5<U, Attr, P>> {
    constexpr static bool Value = true;
};
template <class U, class Attr, class P>
struct IsOutHolder<Out6<U, Attr, P>> {
    constexpr static bool Value = true;
};
template <class U, class Attr, class P>
struct IsOutHolder<Out7<U, Attr, P>> {
    constexpr static bool Value = true;
};
template <class U, class Attr, class P>
struct IsOutHolder<Out8<U, Attr, P>> {
    constexpr static bool Value = true;
};
template <class U, class Attr, class P>
struct IsOutHolder<Out9<U, Attr, P>> {
    constexpr static bool Value = true;
};

template <class T, int cur, class Attr = void>
struct Var : public Holder<T, Attr, cur> {
    constexpr static bool IsVarValue = true;
};

#define CHECK_TYPE(NAME, V)                                                     \
    template <typename T>                                                       \
    struct NAME {                                                               \
        template <typename>                                                     \
        static uint8_t has(...);                                                \
        template <typename C>                                                   \
        static uint32_t has(decltype(C::V));                                    \
        constexpr static bool Value = sizeof(has<T>(NULL)) == sizeof(uint32_t); \
    }

CHECK_TYPE(IsVar, IsVarValue);
CHECK_TYPE(IsConstValue, IsConstValue_);

/** 值设置：定义值存储结构 */
template <class... Ts>
struct VarTypeStruct {};
template <>
struct VarTypeStruct<> {};
template <class T>
struct VarTypeStruct<T> {
    using DataType = typename T::DType;
    DataType value;
    template <int offset = 0>
    __aicore__ inline constexpr DataType Get()
    {
        return value;
    }
    template <int offset = 0>
    __aicore__ inline constexpr void Set(DataType v)
    {
        value = v;
    }
};

template <class T, class... Ts>
struct VarTypeStruct<T, Ts...> : public VarTypeStruct<Ts...> {
    using DataType = typename T::DType;
    DataType value;
    template <int offset = 0>
    __aicore__ inline constexpr DataType Get()
    {
        if constexpr (offset == 0) {
            return value;
        } else {
            return VarTypeStruct<Ts...>::template Get<offset - 1>();
        }
    }
    template <int offset = 0>
    __aicore__ inline constexpr void Set(DataType v)
    {
        if constexpr (offset == 0) {
            value = v;
        } else {
            VarTypeStruct<Ts...>::template Set<offset - 1>(v);
        }
    }
};

template <class... Ts>
struct VarTypeAux {
    using Type = VarTypeStruct<Ts...>;
};

} // namespace Placeholder
} // namespace Base
} // namespace Ops

#endif // UTIL_PLACEHOLDER_H_
