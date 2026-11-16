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
 * \file aux1.h
 * \brief
 */
#ifndef UTIL_AUX_1_H_
#define UTIL_AUX_1_H_

#include "aux2.h"
#include "elems.h"
#include "log.h"
#include "placeholder.h"
#include "vec.h"

namespace Ops {
namespace Base {
namespace __aux {

constexpr static int MAX_DTYPE_BYTES = 32;

template <typename Es, typename T>
struct TypeIsVar {
    constexpr static bool Value = Placeholder::IsVar<T>::Value;
};

template <typename Es, typename T>
struct TypeIsConst {
    constexpr static bool Value = Placeholder::IsConstValue<T>::Value;
};

/*
 * 是OutHolder，且不在Es中。
 * 模板参数：
 *   1. Es: 过滤后的集合
 *   2. T : 待确认对象
 */
template <typename Es, typename T>
struct TypeIsOutHolder {
    constexpr static bool Value = Placeholder::IsOutHolder<T>::Value && ElemsNotExist<Es, T>::Value;
};

/*
 * 是InHolder
 * 模板参数：
 *   1. Es: 过滤后的集合
 *   2. T : 待确认对象
 */
template <typename Es, typename T>
struct TypeIsInHolder {
    constexpr static bool Value = Placeholder::IsInHolder<T>::Value;
};

/*
 * 是否是TensorScalar
 * 模板参数：
 *   1. Es: 过滤后的集合
 *   2. T : 待确认对象
 */
template <typename Es, typename T>
struct TypeIsInScalarHolder {
    constexpr static bool Value = Placeholder::IsInScalar<T>::Value;
};

/*
 * 是否是Scalar操作类型的Bind
 * 模板参数：
 *   1. Es: 过滤后的集合
 *   2. T : 待确认对象
 */
template <typename Es, typename T>
struct TypeIsScalarBind {
    constexpr static bool Value = T::IsScalarOp;
};

/*
 * 过滤CopyInBrc操作类型的Bind
 * 模板参数：
 *   1. Es: 过滤后的集合
 *   2. T : 待确认对象
 */
template <typename Es, typename T>
struct TypeIsCopyInBrcBind {
    constexpr static bool Value = Vec::IsCopyInBrcOp<typename T::Fun>::Value;
};

/*
 * Filter ReduceOp Bind
 */
template <typename Es, typename T>
struct TypeIsReduceOpBind {
    constexpr static bool Value = Vec::IsReduceOp<typename T::Fun>::Value;
};

/*
 * 检测某个类是否是Bind模板
 */
template <class T, typename = void>
struct TypeIsFunBind {
    constexpr static bool Value = false;
};

template <class T>
struct TypeIsFunBind<T, Void_t<typename T::BindType>> {
    constexpr static bool Value = true;
};

template <typename Es, typename T>
struct TypeIsInFun {
    constexpr static bool Value = TypeIsFunBind<T>::Value;
};

/*
 * 判断T是否是输入参数。（输出OutHolder被排除在外）
 */
template <typename Es, typename T>
struct TypeIsInput {
    constexpr static bool Value = Placeholder::IsInHolder<T>::Value || TypeIsInFun<Es, T>::Value ||
                                  Placeholder::IsVar<T>::Value || Placeholder::IsConstValue<T>::Value;
};

/*
 * 获取输出 Outputs 的依赖树
 */
template <class... Ts>
struct FunListAux {};

template <>
struct FunListAux<> {
    using Type = Elems<>;
};

template <class T>
struct FunListAux<T> {
    using Type = typename T::DependFuns;
};

template <typename T, typename... Ts>
struct FunListAux<T, Ts...> {
    BUILD_LOG(int, T::InputSize);
    using Type = typename FunListAux<T>::Type::template Union<typename FunListAux<Ts...>::Type>;
};

template <typename Es>
struct GetDependFunsAux {
    BUILD_LOG(int, Es::Size);
    using Type = typename Es::template Export<FunListAux>::Type;
};

/*
 * 追溯依赖树中原始输入 PlaceHolder
 */
template <class... Ts>
struct SourceInHoldersListAux {};

template <>
struct SourceInHoldersListAux<> {
    using Type = Elems<>;
};

template <class T>
struct SourceInHoldersListAux<T> {
    using Type = typename T::SourceInHolders;
};

template <typename T, typename... Ts>
struct SourceInHoldersListAux<T, Ts...> {
    using Type = typename SourceInHoldersListAux<T>::Type::template Union<typename SourceInHoldersListAux<Ts...>::Type>;
};
template <typename Es>
struct GetSourceInHoldersAux {
    using Type = typename Es::template Export<SourceInHoldersListAux>::Type;
};

/*
 * 检查函数列表Es中有几个 PlaceHolder::in
 */
template <typename Es, int start = 0, typename Acc = Elems<>>
constexpr int GetInputSize()
{
    if constexpr (start < Es::Size) {
        // 获取当前Bind
        using func = typename Es::template At<start>;
        // 获取当前Bind的输入Holder
        using inHolders = typename func::InHolders;
        if constexpr (inHolders::Size > 0) {
            return GetInputSize<Es, start + 1, typename Acc::template Union<inHolders>>();
        } else {
            return GetInputSize<Es, start + 1, Acc>();
        }
    }
    return Acc::Size;
};

/*
 * 获取某个Bind上的InHolders
 */
template <class... Ts>
struct InHolderAux {};

template <>
struct InHolderAux<> {
    using Type = Elems<>;
};

template <class T>
struct InHolderAux<T> {
    using Type = typename T::InHolders;
};

/*
 * 获取Bind列表中所有InHolders
 */
template <typename T, typename... Ts>
struct InHolderAux<T, Ts...> {
    using Type = typename InHolderAux<T>::Type::template Union<typename InHolderAux<Ts...>::Type>;
};

template <typename Es>
struct GetInHolder {
    using Type = typename Es::template Export<InHolderAux>::Type::Unique;
};

/*
 * 获取某个Bind上的OutHolders
 */
template <class... Ts>
struct OutHolderAux {};

template <>
struct OutHolderAux<> {
    using Type = Elems<>;
};

template <class T>
struct OutHolderAux<T> {
    using Type = typename T::OutHolders;
};

/*
 * 获取Bind列表中所有OutHolders
 */
template <typename T, typename... Ts>
struct OutHolderAux<T, Ts...> {
    using Type = typename OutHolderAux<T>::Type::template Union<typename OutHolderAux<Ts...>::Type>;
};

template <typename Es>
struct GetOutHolder {
    using Type = typename Es::template Export<OutHolderAux>::Type::Unique;
};

/*
 * 获取某个Bind上的Var类型输入参数
 */
template <class... Ts>
struct VarAux {};

template <>
struct VarAux<> {
    using Type = Elems<>;
};

template <class T>
struct VarAux<T> {
    using Type = typename T::Vars;
};

/*
 * 获取Bind列表中所有Var类型输入参数
 */
template <typename T, typename... Ts>
struct VarAux<T, Ts...> {
    using Type = typename VarAux<T>::Type::template Union<typename VarAux<Ts...>::Type>;
};

template <typename Es>
struct GetVars {
    using Type = typename Es::template Export<VarAux>::Type::Unique;
};

/*
 * 检测节点@T是否是列表@Es中从@start开始的节点的输入
 * 模板参数：
 *   1. Es:    调用节点列表
 *   2. start：后续节点起始位置
 *   3. T：    待检查节点
 */
template <typename Es, int start, typename T>
__aicore__ constexpr bool CheckIsInput()
{
    if constexpr (start < Es::Size) {
        using func = typename Es::template At<start>;
        using inArgs = typename func::InArgs;
        if constexpr (inArgs::template IsExist<T>()) {
            return true;
        }
        return CheckIsInput<Es, start + 1, T>();
    }
    return false;
}

/*
 * 过滤出@Inputs中可以释放（删除）的节点
 * 模板参数：
 *   1. FunList: 调用节点列表
 *   2. RsvList: 永久存活节点列表
 *   3. start：  后续节点起始位置
 *   4. Inputs:  待筛选节点列表
 */
template <typename FunList, typename RsvList, int start, typename Inputs>
struct TryDelInOutAux {
protected:
    template <class Target, class T>
    struct NotUsing {
        // 是否是永久存活节点
        constexpr static bool isRsvd = RsvList::template IsExist<T>();
        // 检查是否是后续节点的输入
        constexpr static bool isIn = CheckIsInput<FunList, start, T>();
        constexpr static bool Value = !(isRsvd || isIn);
    };

public:
    using Type = typename Inputs::template Filter<NotUsing>;
};

/*
 * 过滤出@ToFilterList中，中间计算节点列表
 * 模板参数：
 *   1. OutList:      搬出节点列表
 *   2. ToFilterList: 待筛选节点列表
 *   3. SkipCopyInBrc: 是否将CopyInBrc节点当作普通VecOp节点
 */
template <typename OutList, typename ToFilterList, bool SkipCopyInBrc = true>
struct FilterTempCalcNode {
protected:
    template <class Target, class T>
    struct NotCopyInOrConnectToCopyOut {
        constexpr static bool isCopyIn = Vec::IsCopyInOp<typename T::Fun>::Value;
        constexpr static bool isCopyInBrc = Vec::IsCopyInBrcOp<typename T::Fun>::Value;
        constexpr static bool isConnectToCopyOut = CheckIsInput<OutList, 0, T>();
        constexpr static bool Value = !(isConnectToCopyOut || (SkipCopyInBrc ? (isCopyIn && !isCopyInBrc) : isCopyIn));
    };

public:
    using Type = typename ToFilterList::template Filter<NotCopyInOrConnectToCopyOut>;
};

/*
 * 最大存活节点等相关返回信息（剔除永久存活节点后）
 */
struct DagMaxAliveInfo {
    uint32_t aliveNode = 0;                   // 最大存活节点数量
    uint32_t tempCalcNode = 0;                // 中间计算临时占用节点数量
    uint32_t maxDtypeBytes = 0;               // 最大dtype字节数
    uint32_t minDtypeBytes = MAX_DTYPE_BYTES; // 最小dtype字节数
    uint32_t aliveNodeNoCopyBrcTmpBuf = 0;    // 不包含CopyInBrcOp临时变量的存活节点数量
    uint32_t tempCalcNodeNoCopyBrcTmpBuf = 0; // 不包含CopyInBrcOp临时变量的中间计算临时占用节点数量

    constexpr DagMaxAliveInfo()
        : aliveNode(0),
          tempCalcNode(0),
          maxDtypeBytes(0),
          minDtypeBytes(MAX_DTYPE_BYTES),
          aliveNodeNoCopyBrcTmpBuf(0),
          tempCalcNodeNoCopyBrcTmpBuf(0)
    {}

    constexpr DagMaxAliveInfo(const DagMaxAliveInfo& v)
        : aliveNode(v.aliveNode),
          tempCalcNode(v.tempCalcNode),
          maxDtypeBytes(v.maxDtypeBytes),
          minDtypeBytes(v.minDtypeBytes),
          aliveNodeNoCopyBrcTmpBuf(v.aliveNodeNoCopyBrcTmpBuf),
          tempCalcNodeNoCopyBrcTmpBuf(v.tempCalcNodeNoCopyBrcTmpBuf)
    {}
};

template <typename T>
static constexpr const T& Max(const T& a, const T& b)
{
    return a > b ? a : b;
}

/*
 * 计算最大存活节点
 *   1. 当前 func 的输入 + 输出，记为 aliveNode
 *   2. 依赖的函数B，如果依赖的函数的输入/输出可以删除的个数为 x，则 A 的aliveNode += B.aliveNode - x
 *   3. 遍历每一个 func，计算最大值
 * 模板参数：
 *   1. FunList: 调用顺序列表
 *   2. OutList: 真实搬出节点列表
 *   3. RsvList: 永久存活节点
 *   4. start:   递归过程中，当前节点索引
 *   5. Acc:     当前节点前存活的节点
 * 入参;
 *   1. info: 存放图遍历存活节点信息
 */
template <typename FunList, typename OutList, typename RsvList = Elems<>, int start = 0, typename Acc = Elems<>>
constexpr DagMaxAliveInfo MaxAliveNode(DagMaxAliveInfo info)
{
    if constexpr (start < FunList::Size) {
        using func = typename FunList::template At<start>;

        // 依赖节点： 输入 + 输出
        using inOutNodes = Condition<(Vec::IsCopyOutOp<typename func::Fun>::Value || func::IsScalarOp),
                                     typename func::InNonScalarFuns,
                                     typename func::InNonScalarFuns::template Append<typename func::RealBindType>>;
        // Union当前存活节点，刨去永久存活节点
        using aliveNodes = typename Acc::template Union<inOutNodes>::Unique::template Remove<RsvList>;
        // 存活节点中CopyInBrc节点
        using copyInBrcNodes = typename aliveNodes::template Filter<TypeIsCopyInBrcBind>;
        // 刨去输入/输出占用的Buffer节点，保留中间计算临时节点（非NDDMA场景下CopyInBrc视作普通节点）
        using tempCalcNodesWithCopyBrc = typename FilterTempCalcNode<OutList, aliveNodes, true>::Type;
        // 刨去输入/输出占用的Buffer节点，保留中间计算临时节点（NDDMA场景下视作CopyIn节点）
        using tempCalcNodes = typename FilterTempCalcNode<OutList, tempCalcNodesWithCopyBrc, false>::Type;

        constexpr uint32_t funcTmpSize = Vec::IsCopyInBrcOp<typename func::Fun>::Value ? 0 : func::Fun::TempSize;
        // alive node size considering CopyInBrc as CopyIn + VecBrc
        constexpr uint32_t aliveNodeSize = static_cast<uint32_t>(aliveNodes::Size - copyInBrcNodes::Size +
                                                                 2 * copyInBrcNodes::Size + funcTmpSize);
        constexpr uint32_t tempCalcNodeWithCopyBrcSize = static_cast<uint32_t>(tempCalcNodesWithCopyBrc::Size +
                                                                               funcTmpSize);
        info.aliveNode = Max<uint32_t>(aliveNodeSize, info.aliveNode);
        info.tempCalcNode = Max<uint32_t>(tempCalcNodeWithCopyBrcSize, info.tempCalcNode);

        if constexpr (Vec::IsCopyInBrcOp<typename func::Fun>::Value) {
            info.aliveNodeNoCopyBrcTmpBuf = Max<uint32_t>(static_cast<uint32_t>(aliveNodes::Size),
                                                          info.aliveNodeNoCopyBrcTmpBuf);
            info.tempCalcNodeNoCopyBrcTmpBuf = Max<uint32_t>(static_cast<uint32_t>(tempCalcNodes::Size),
                                                             info.tempCalcNodeNoCopyBrcTmpBuf);
        } else {
            constexpr uint32_t tempCalcNodeSize = static_cast<uint32_t>(tempCalcNodes::Size + func::Fun::TempSize);
            constexpr uint32_t aliveNodeSizeNddma = static_cast<uint32_t>(aliveNodes::Size + func::Fun::TempSize);
            info.aliveNodeNoCopyBrcTmpBuf = Max<uint32_t>(aliveNodeSizeNddma, info.aliveNodeNoCopyBrcTmpBuf);
            info.tempCalcNodeNoCopyBrcTmpBuf = Max<uint32_t>(tempCalcNodeSize, info.tempCalcNodeNoCopyBrcTmpBuf);
        }

        constexpr uint32_t curDtypeBytes = static_cast<uint32_t>(sizeof(typename func::OutDataType));
        info.maxDtypeBytes = Max<uint32_t>(curDtypeBytes, info.maxDtypeBytes);
        info.minDtypeBytes = curDtypeBytes < info.minDtypeBytes ? curDtypeBytes : info.minDtypeBytes;

        // 检查输入是否可以删除
        using delVar = typename TryDelInOutAux<FunList, RsvList, start + 1, typename func::InNonScalarFuns>::Type;
        // 删除不再在后续节点中使用的节点
        using next = typename aliveNodes::template Remove<delVar>;

        return MaxAliveNode<FunList, OutList, RsvList, start + 1, next>(info);
    }
    return info;
};

/*
 * 检测某个输入@InArg是否可以被释放（删除）
 * 模板参数：
 *   1. FunList:   调用顺序列表
 *   2. posInFuns: 递归过程中，当前节点索引
 *   3. InArg:     待检测的输入节点
 */
template <typename FunList, int posInFuns, typename InArg>
__aicore__ constexpr bool InputIsAbleToFreeAux()
{
    if constexpr (posInFuns < FunList::Size) {
        using func = typename FunList::template At<posInFuns>;
        using funArgs = typename func::InArgs;
        if constexpr (funArgs::template IsExist<InArg>()) {
            return false;
        }
        return InputIsAbleToFreeAux<FunList, posInFuns + 1, InArg>();
    }
    return true;
}

/*
 * 检测某个节点@curArg是否直连搬出节点
 * 模板参数：
 *   1. FunList:   调用列表
 *   2. posInFuns: 递归过程中，当前节点索引
 *   3. InArg:     待检测节点
 */
template <typename FunList, int posInFuns, typename InArg>
__aicore__ constexpr bool IsConnectOutput()
{
    if constexpr (posInFuns < FunList::Size) {
        using func = typename FunList::template At<posInFuns>;
        if constexpr (Vec::IsCopyOutOp<typename func::Fun>::Value) {
            using inFuns = typename func::InFuns;
            if constexpr (inFuns::template IsExist<InArg>()) {
                return true;
            } else {
                return IsConnectOutput<FunList, posInFuns + 1, InArg>();
            }
        } else {
            return IsConnectOutput<FunList, posInFuns + 1, InArg>();
        }
    } else {
        return false;
    }
}

/*
 * 获取某个搬入节点被后续哪个VecBrc节点依赖
 * 模板参数：
 *   1. VecBrcNodes : 图中所有VecBrc节点
 *   2. InArg : 某个待检测CopyIn节点，若是其他节点则直接返回 -1
 *   3. pos : 递归VecBrcNodes的索引
 * 返回值：
 *   > 0 : 依赖的VecBrc节点在VecBrcNodes中的索引
 *   -1:   不被任何VecBrc依赖
 */
template <typename VecBrcNodes, typename InArg, int pos = 0>
__aicore__ constexpr int GetDependByVecBrcIdx()
{
    if constexpr (!Vec::IsCopyInOp<typename InArg::Fun>::Value) {
        return -1;
    } else if constexpr (pos < VecBrcNodes::Size) {
        using func = typename VecBrcNodes::template At<pos>;
        using srcInHolders = typename func::SourceInHolders;
        using copyInHolder = typename InArg::InHolders::template At<0>;
        if constexpr (srcInHolders::template IsExist<copyInHolder>()) {
            return pos;
        } else {
            return GetDependByVecBrcIdx<VecBrcNodes, InArg, pos + 1>();
        }
    } else {
        return -1;
    }
}

template <typename FuncList, int pos = 0>
constexpr int32_t GetReducePosition()
{
    if constexpr (pos < FuncList::Size) {
        using func = typename FuncList::template At<pos>;
        if constexpr (Vec::IsReduceOp<typename func::Fun>::Value) {
            return pos;
        } else {
            return GetReducePosition<FuncList, pos + 1>();
        }
    } else {
        return -1;
    }
}

} // namespace __aux
} // namespace Base
} // namespace Ops
#endif // UTIL_AUX_1_H_
