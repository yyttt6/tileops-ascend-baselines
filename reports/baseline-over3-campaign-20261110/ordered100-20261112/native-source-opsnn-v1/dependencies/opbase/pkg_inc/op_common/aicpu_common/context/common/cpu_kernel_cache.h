/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AICPU_CPU_KERNEL_CACHE_H_
#define AICPU_CPU_KERNEL_CACHE_H_

#include <map>
#include <memory>

#include "aicpu_task_struct.h"
#include "fwk_adpt_struct.h"
#include "cpu_context.h"
#include "cpu_node_def.h"
#include "kernel_cache.h"
#include "device_cpu_kernel.h"

namespace aicpu {
struct ExtInfoMsg {
    bool has_sess_info = false;
    uint64_t kernel_id = 0U;
    bool unknown_shape = false;
    bool async_flag = false;
    bool devicetype_host_flag = false;
    uint8_t wait_type = 0U;
    uint32_t wait_id = 0U;
    std::vector<FWKAdapter::ShapeAndType*> input_shape_and_type;
    std::vector<FWKAdapter::ShapeAndType*> output_shape_and_type;
    std::map<uint32_t, uint64_t> unknown_shape_input_index_addr;
    std::map<uint32_t, uint64_t> unknown_shape_output_index_addr;
    uint64_t workspace_size = 0UL;
    uint64_t workspace_addr = 0UL;
};

struct CpuCacheData {
    std::shared_ptr<NodeDef> proto = nullptr;
    std::shared_ptr<CpuKernelContext> context = nullptr;
    CpuCacheData(std::shared_ptr<NodeDef> pro, std::shared_ptr<CpuKernelContext> ctx) : proto(pro), context(ctx) {}
};

class CpuKernelCache : public KernelCache<CpuCacheData> {
public:
    CpuKernelCache() = default;
    ~CpuKernelCache() = default;

    /*
     * Init kernel cache.
     * @return int32_t: 0 indicates success, while the others fail
     */
    int32_t InitParameter() override;

    /*
     * run kernel.
     * @param param: kernel context
     * @return int32_t: 0 indicates success, whilWe the others fail
     */
    int32_t RunKernel(void* param) override;

    /*
     * run kernel with blockDimInfo.
     * @param param: kernel context and blkDimInfo
     * @return int32_t: 0 indicates success, whilWe the others fail
     */
    int32_t RunCpuKernelWithBlock(void* param, struct BlkDimInfo* blkdim_info) override;

private:
    CpuKernelCache(const CpuKernelCache&) = delete;
    CpuKernelCache(CpuKernelCache&&) = delete;
    CpuKernelCache& operator=(const CpuKernelCache&) = delete;
    CpuKernelCache& operator=(CpuKernelCache&&) = delete;

    /*
     * update framework output tensor shape.
     * @return uint32_t: 0 indicates success, while the others fail
     */
    uint32_t UpdateFWKOutputShape(ExtInfoMsg& ext_info_msg, const CpuKernelContext& ctx) const;

    /*
     * get shape information from framework.
     * @param dims: shape information
     */
    void GetDimsFromShapeAndType(const FWKAdapter::ShapeAndType* shape_and_type, std::vector<int64_t>& dims) const;

    /*
     * get shape information from arrays.
     * @param dims: shape information
     */
    void GetDimsFromArrays(const int64_t* shape, size_t len, std::vector<int64_t>& dims) const;

    /*
     * update tensor information.
     * @param ctx: kernel context
     * @return uint32_t: 0 indicates success, while the others fail
     */
    uint32_t UpdateTensor(const std::vector<uint64_t>& io_addrs, ExtInfoMsg& ext_info_msg, CpuKernelContext& ctx) const;

    /*
     * parse extend tensor shape types information.
     * @param ext_info: extend information
     * @return uint32_t: 0 indicates success, while the others fail
     */
    uint32_t ParseExtShapeType(const FWKAdapter::ExtInfo* ext_info, bool& unknown_shape) const;

    /*
     * parse extend tensor bitmap information.
     * @param ext_info: extend information
     * @return uint32_t: 0 indicates success, while the others fail
     */
    uint32_t ParseExtBitMap(const FWKAdapter::ExtInfo* ext_info, bool& unknown_shape) const;

    /*
     * parse extend tensor topictype 16bit devicetype information.
     * @param ext_info: extend information
     * @return uint32_t: 0 indicates success, while the others fail
     */
    uint32_t ParseExtTopicTypeDeviceType(const FWKAdapter::ExtInfo* ext_info, bool& devicetype_host_flag) const;

    /*
     * parse extend tensor shape and types information.
     * @param ext_info: extend information
     * @param shape_and_type: shape and types from extend information
     * @return uint32_t: 0 indicates success, while the others fail
     */
    uint32_t ParseExtShapeAndType(bool unknown_shape, FWKAdapter::ExtInfo* ext_info,
                                  std::vector<FWKAdapter::ShapeAndType*>& shape_and_type) const;

    /*
     * parse extend unknown shape index information.
     * @param ext_info: extend information
     * @param unknown_shape_index_addr: unknown shape index and addr map
     * @return uint32_t: 0 indicates success, while the others fail
     */
    uint32_t ParseExtUnknownShapeIndex(FWKAdapter::ExtInfo* ext_info,
                                       std::map<uint32_t, uint64_t>& unknown_shape_index_addr) const;

    /*
     * parse extend session information.
     * @param ext_info: extend information
     * @param kernel_id: kernel id from extend information
     * @return uint32_t: 0 indicates success, while the others fail
     */
    uint32_t ParseExtSessionInfo(FWKAdapter::ExtInfo* ext_info, uint64_t& kernel_id) const;

    /*
     * parse extend async wait info
     * @param ext_info : extend infomation
     * @param wait_type: event wait type
     * @param wait_id : event wait id
     * @return uint32_t: 0 indicates success, while the others fail
     */
    uint32_t ParseAsyncWait(FWKAdapter::ExtInfo* ext_info, uint8_t& wait_type, uint32_t& wait_id) const;

    uint32_t ParseExtWorkSpaceInfo(FWKAdapter::ExtInfo* ext_info, uint64_t& workspace_size,
                                   uint64_t& workspace_addr) const;

    /*
     * parse extend information.
     * @param param_head: kernel context
     * @param ext_info_msg: extend info msg
     * @return uint32_t: 0 indicates success, while the others fail
     */
    uint32_t ParseExtMsg(AicpuParamHead* param_head, ExtInfoMsg& ext_info_msg) const;

    /*
     * parse io address.
     * @param param_head: kernel context
     * @param io_addrs: kernel inputs and outputs adress
     * @param nodedef: kernel node def
     * @param nodedef_len: kernel node def length
     * @return uint32_t: 0 indicates success, while the others fail
     */
    uint32_t ParseIoAddr(AicpuParamHead* param_head, std::vector<uint64_t>& io_addrs, char*& nodedef,
                         uint32_t& nodedef_len) const;

    /*
     * get cpu kernel context from cache
     * @param has_sess_info: whether has session info
     * @param kernel_id: kernel id, the key of cache
     * @return uint32_t: 0 indicates success, while the others fail
     */
    std::shared_ptr<CpuKernelContext> GetCpuKernelContext(std::shared_ptr<ExtInfoMsg> ext_info_msg, const char* nodedef,
                                                          uint32_t nodedef_len,
                                                          std::shared_ptr<NodeDef>& nodedef_proto);

    /*
     * get cpu kernel context from cache
     * @param has_sess_info: whether has session info
     * @param kernel_id: kernel id, the key of cache
     * @param blkDimInfo: kernel blockdim info
     * @return uint32_t: 0 indicates success, while the others fail
     */
    std::shared_ptr<CpuKernelContext> GetCpuKernelContextWithBlock(std::shared_ptr<ExtInfoMsg> ext_info_msg,
                                                                   const char* nodedef, uint32_t nodedef_len,
                                                                   std::shared_ptr<NodeDef>& nodedef_proto,
                                                                   struct BlkDimInfo* blkdim_info);

    /*
     * get bit status on pos
     * @param num: input number
     * @param pos: bit pos
     * @return bool: bit is 1 or 0
     */
    bool GetBitStatus(uint64_t num, uint64_t pos) const;

    uint32_t CheckTensorParam(const std::vector<uint64_t>& io_addrs, ExtInfoMsg& ext_info_msg,
                              CpuKernelContext& ctx) const;

    uint32_t UpdateInputTensor(const std::vector<uint64_t>& io_addrs, ExtInfoMsg& ext_info_msg, CpuKernelContext& ctx,
                               size_t& addr_index) const;

    /*
     * parse run kernel parameters: io addrs and ExtInfoMsg.
     * @return int32_t: 0 indicates success, while the others fail
     */
    int32_t ParseRunKernelParam(void* param, std::vector<uint64_t>& io_addrs, char*& node_def, uint32_t& node_def_len,
                                std::shared_ptr<ExtInfoMsg>& ext_info_msg) const;

    /*
     * dispatch cpu kernel based on V2/V1 registry.
     * @return uint32_t: raw kernel status code
     */
    uint32_t DispatchCpuKernel(CpuKernelContext& ctx, ExtInfoMsg& ext_info_msg) const;
};
} // namespace aicpu
#endif // AICPU_CPU_KERNEL_CACHE_H_
