# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
set(TLS_GUARDIAN_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/tls_guardian.cpp")

add_library(dummy_tls SHARED ${TLS_GUARDIAN_SRCS})
target_compile_definitions(dummy_tls PRIVATE TLS_GUARDIAN_)
target_compile_options(dummy_tls PRIVATE -O0)
target_link_libraries(dummy_tls
    PRIVATE
    intf_pub
)