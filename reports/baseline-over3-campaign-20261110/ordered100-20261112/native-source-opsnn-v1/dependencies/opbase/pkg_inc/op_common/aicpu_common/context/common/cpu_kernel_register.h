/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AICPU_CONTEXT_INC_REGISTAR_H
#define AICPU_CONTEXT_INC_REGISTAR_H

#include <map>
#include <string>

#include "cpu_context.h"
#include "cpu_kernel.h"

namespace aicpu {
class AICPU_VISIBILITY CpuKernelRegister {
public:
    /*
     * get instance.
     * @return CpuKernelRegister &: CpuKernelRegister instance
     */
    static CpuKernelRegister& Instance();

    /*
     * get cpu kernel.
     * param op_type: the op type of kernel
     * @return shared_ptr<CpuKernel>: cpu kernel ptr
     */
    std::shared_ptr<CpuKernel> GetCpuKernel(const std::string& op_type);

    /*
     * get cpu kernel V2.
     * param op_type: the op type of kernel
     * @return shared_ptr<CpuKernel>: cpu kernel ptr
     *
     * V2 查找仅访问 creatorMapV2_, 未命中返回 nullptr, 不回退到 V1.
     * 本函数不打印"未注册"日志, 由调用方根据命中情况输出更有信息量的日志
     * (例如命中哪个 so).
     */
    std::shared_ptr<CpuKernel> GetCpuKernelV2(const std::string& op_type);

    /*
     * check whether the op type is registered in V2 without instantiating kernel.
     * param op_type: the op type of kernel
     * @return bool: true if registered in V2, otherwise false
     *
     * 轻量查询接口: 仅查 creatorMapV2_ 是否存在该 op_type, 不会通过
     * creator 函数构造 kernel. 供上层在"V2 优先, V1 兜底"的路由场景使用,
     * 避免重复构造带来的开销.
     */
    bool IsRegisteredV2(const std::string& op_type) const;

    /*
     * get all cpu kernel registered op types.
     * @return std::vector<string>: all cpu kernel registered op type
     */
    std::vector<std::string> GetAllRegisteredOpTypes() const;

    /*
     * get all cpu kernel registered op types V2.
     * @return std::vector<string>: all cpu kernel registered op type
     */
    std::vector<std::string> GetAllRegisteredOpTypesV2() const;

    /*
     * run cpu kernel.
     * param ctx: context of kernel
     * @return uint32_t: 0->success other->failed
     */
    uint32_t RunCpuKernel(CpuKernelContext& ctx);

    /*
     * run cpu kernel V2.
     * param ctx: context of kernel
     * @return uint32_t: 0->success other->failed
     *
     * V2 执行仅从 creatorMapV2_ 查找, 不回退到 V1.
     */
    uint32_t RunCpuKernelV2(CpuKernelContext& ctx);

    /*
     * run async cpu kernel.
     * @param ctx: context of kernel
     * @param wait_type : event wait type
     * @param wait_id : event wait id
     * @param cb : callback function
     * @return uint32_t: 0->success other->failed
     */
    uint32_t RunCpuKernelAsync(CpuKernelContext& ctx, const uint8_t wait_type, const uint32_t wait_id,
                               std::function<uint32_t()> cb);

    /*
     * run async cpu kernel V2.
     * @param ctx: context of kernel
     * @param wait_type : event wait type
     * @param wait_id : event wait id
     * @param cb : callback function
     * @return uint32_t: 0->success other->failed
     */
    uint32_t RunCpuKernelAsyncV2(CpuKernelContext& ctx, const uint8_t wait_type, const uint32_t wait_id,
                                 std::function<uint32_t()> cb);

    // CpuKernel registration function to register different types of kernel to
    // the factory
    class Registerar {
    public:
        Registerar(const std::string& type, const KERNEL_CREATOR_FUN& fun);
        ~Registerar() = default;

        Registerar(const Registerar&) = delete;
        Registerar(Registerar&&) = delete;
        Registerar& operator=(const Registerar&) = delete;
        Registerar& operator=(Registerar&&) = delete;
    };

    // CpuKernel registration function V2 to register different types of kernel to
    // the factory
    class RegisterarV2 {
    public:
        RegisterarV2(const std::string& type, const KERNEL_CREATOR_FUN& fun);
        ~RegisterarV2() = default;

        RegisterarV2(const RegisterarV2&) = delete;
        RegisterarV2(RegisterarV2&&) = delete;
        RegisterarV2& operator=(const RegisterarV2&) = delete;
        RegisterarV2& operator=(RegisterarV2&&) = delete;
    };

protected:
    CpuKernelRegister() = default;
    ~CpuKernelRegister() = default;

    CpuKernelRegister(const CpuKernelRegister&) = delete;
    CpuKernelRegister(CpuKernelRegister&&) = delete;
    CpuKernelRegister& operator=(const CpuKernelRegister&) = delete;
    CpuKernelRegister& operator=(CpuKernelRegister&&) = delete;

    // register creator, this function will call in the constructor
    void Register(const std::string& type, const KERNEL_CREATOR_FUN& fun);

    // register creator V2, this function will call in the constructor
    void RegisterV2(const std::string& type, const KERNEL_CREATOR_FUN& fun);

private:
    uint32_t RunCpuKernelCommon(CpuKernelContext& ctx, const std::string type, const std::shared_ptr<CpuKernel> kernel);
    uint32_t SetAsyncKernelContext(const std::string& type, const uint8_t wait_type, const uint32_t wait_id);
    uint32_t RunCpuKernelAsyncCommon(CpuKernelContext& ctx, const uint8_t wait_type, const uint32_t wait_id,
                                     std::function<uint32_t()> cb, const std::shared_ptr<CpuKernel> kernel);
    std::map<std::string, KERNEL_CREATOR_FUN> creatorMap_;   // kernel map
    std::map<std::string, KERNEL_CREATOR_FUN> creatorMapV2_; // kernel map V2
};
} // namespace aicpu
#endif // AICPU_CONTEXT_INC_REGISTAR_H_
