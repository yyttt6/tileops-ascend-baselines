# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
project(ascendc_tests)
cmake_minimum_required(VERSION 3.16.0)

add_compile_options(-Werror)

set(CMAKE_CXX_STANDARD 17)
set(TEST_MOD all CACHE STRING "Tensor API UT target filter")

include(cmake/config.cmake)
include(cmake/intf.cmake)
include(cmake/func.cmake)

add_subdirectory(api)
