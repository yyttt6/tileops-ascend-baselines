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
 * \file dfx.h
 * \brief
 */
#ifndef UTIL_DFX_H_
#define UTIL_DFX_H_
#include "log.h"

namespace Ops {
namespace Base {
class DagDfx {
public:
    /**
     * 打印计算图结构，输出markdown mermaid graph
     * @tparam Dag DAG图
     * @tparam pos 图的起始偏移
     * @return
     */
    template <class Dag, int pos = 0>
    __aicore__ inline void DumpGraph()
    {
        if constexpr (pos == 0) {
            RUN_LOG("DAG BufferNum: %d", Dag::BufferNum);
            RUN_LOG("DAG MaxDtypeBytes: %d", Dag::MaxDtypeBytes);
            RUN_LOG("DAG MinDtypeBytes: %d", Dag::MinDtypeBytes);
            RUN_LOG_BASE("<<<<<<<<<<<<<<<<<<<<<<<< DumpGraph Start >>>>>>>>>>>>>>>>>>>>>>>>>> \n");
            RUN_LOG_BASE("```mermaid\n graph TD; \n");
        }
        using Op = typename Dag::FunList::template At<pos>;
        PrintGraph<Dag, Op>();
        if constexpr (pos + 1 < Dag::FunList::Size) {
            return DumpGraph<Dag, pos + 1>();
        } else {
            RUN_LOG_BASE("```\n");
            RUN_LOG_BASE("<<<<<<<<<<<<<<<<<<<<<<<< DumpGraph End >>>>>>>>>>>>>>>>>>>>>>>>>> \n");
        }
    }

private:
    template <class Dag, class Op, int pos = 0>
    __aicore__ constexpr static inline int GetFunOutputPos()
    {
        if constexpr (std::is_same<typename Dag::FunList::template At<pos>, Op>::value) {
            return pos;
        } else if constexpr (pos + 1 < Dag::FunList::Size) {
            return GetFunOutputPos<Dag, Op, pos + 1>();
        }
        return -1;
    }

    template <class Dag, typename Op, int pos = 0>
    __aicore__ inline void PrintGraph()
    {
        using InArg = typename Op::Args::template At<pos>;
        using Fun = typename Op::Fun;
        int funcPos = GetFunOutputPos<Dag, Op>();
        if constexpr (__aux::TypeIsFunBind<InArg>::Value) {
            using ArgFun = typename InArg::Fun;
            int argFunPos = GetFunOutputPos<Dag, InArg>();
            RUN_LOG_BASE("%d[\"%d %s\"] --> %d[\"%d %s\"] \n", argFunPos, argFunPos, PRINT_TYPE(ArgFun), funcPos,
                         funcPos, PRINT_TYPE(Fun));
        } else {
            RUN_LOG_BASE("%d[\"%s\"] --> %d[\"%d %s\"] \n", (funcPos + 1) * 1000 + pos, PRINT_TYPE(InArg), funcPos,
                         funcPos, PRINT_TYPE(Fun));
        }
        if constexpr (pos + 1 < Op::Args::Size) {
            PrintGraph<Dag, Op, pos + 1>();
        }
    }
};
} // namespace Base
} // namespace Ops

#endif // UTIL_DFX_H_
