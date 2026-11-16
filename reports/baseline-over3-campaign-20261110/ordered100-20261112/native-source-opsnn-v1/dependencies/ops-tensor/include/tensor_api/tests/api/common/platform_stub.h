/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef _PLATFORM_STUB_H_
#define _PLATFORM_STUB_H_

#include <string>
#include "exe_graph/runtime/tiling_context.h"

namespace fe {
enum class LocalMemType {
  L0_A = 0,
  L0_B = 1,
  L0_C = 2,
  L1 = 3,
  L2 = 4,
  UB = 5,
  HBM = 6,
  RESERVED
};

class PlatFormInfos {
public:
uint32_t GetCoreNum(void) const;
uint32_t GetCoreNumAiv(void) const;
bool GetPlatformRes(const std::string &label, const std::string &key, std::string &val);
bool GetPlatformResWithLock(const std::string &label, const std::string &key, std::string &val);
void GetLocalMemSize(const LocalMemType &mem_type, uint64_t &size);
void GetLocalMemBw(const LocalMemType &mem_type, uint64_t &bw_size);
void SetCoreNumByCoreType(const std::string &core_type);
};

inline gert::TilingContext* GetFakeTilingContext()
{
  size_t kernelInputNum = 1;
  size_t kernelOutputNum = 1;
  size_t size = sizeof(KernelRunContext) + sizeof(AsyncAnyValue *) * (kernelInputNum + kernelOutputNum);
  std::unique_ptr<uint8_t[]> contextHolder = std::unique_ptr<uint8_t[]>(new uint8_t[size]);
  memset(contextHolder.get(), 0xff, size);
  std::vector<AsyncAnyValue> valueHolder(kernelInputNum + kernelOutputNum);
  auto context = reinterpret_cast<KernelRunContext *>(contextHolder.get());
  context->input_size = kernelInputNum;
  context->output_size = kernelOutputNum;
  context->compute_node_info = nullptr;
  for (size_t i = 0; i < kernelInputNum + kernelOutputNum; ++i) {
    context->values[i] = &valueHolder[i];
  }
  auto context1 = reinterpret_cast<gert::TilingContext *>(contextHolder.get());
  return context1;
}
} // namespace fe

#endif // _PLATFORM_STUB_H_