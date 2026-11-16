/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <memory>

std::unique_ptr<TypesToImpl[]> gert::OpImplRegistryHolder::GetOpImplFunctionsByHandle(const void* handle,
                                                                                      const std::string& so_path,
                                                                                      size_t& impl_num) const
{
    if (handle != nullptr && !so_path.empty()) {
        impl_num = 0;
    } else {
        impl_num = 0;
    }
    std::unique_ptr<TypesToImpl[]> ret(new TypesToImpl[1]);
    return ret;
};