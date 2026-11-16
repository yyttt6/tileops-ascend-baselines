/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_OP_API_TLS_GUARDIAN_API_H_
#define OP_API_OP_API_TLS_GUARDIAN_API_H_
#ifdef TLS_GUARDIAN_
#include <thread>

namespace fixglibc27136 {

std::thread::id DummyTls___();

}

#endif // TLS_GUARDIAN_
#endif // OP_API_OP_API_TLS_GUARDIAN_API_H_