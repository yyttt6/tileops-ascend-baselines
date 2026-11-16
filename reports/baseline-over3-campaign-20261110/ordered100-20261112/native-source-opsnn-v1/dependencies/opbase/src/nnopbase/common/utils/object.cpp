/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "opdev/object.h"
#include "opdev/op_arg_def.h"
#include "bridge_pool.h"

namespace op {

void* Object::operator new(size_t size) throw() { return op::internal::Allocate(size); }
void* Object::operator new[](size_t size) throw() { return op::internal::Allocate(size); }

void* Object::operator new(size_t size, [[maybe_unused]] const std::nothrow_t& tag) throw()
{
    return op::internal::Allocate(size);
}
void* Object::operator new[](size_t size, [[maybe_unused]] const std::nothrow_t& tag) throw()
{
    return op::internal::Allocate(size);
}

void Object::operator delete(void* addr) { op::internal::DeAllocate(addr); }
void Object::operator delete[](void* addr) { op::internal::DeAllocate(addr); }

} // namespace op
