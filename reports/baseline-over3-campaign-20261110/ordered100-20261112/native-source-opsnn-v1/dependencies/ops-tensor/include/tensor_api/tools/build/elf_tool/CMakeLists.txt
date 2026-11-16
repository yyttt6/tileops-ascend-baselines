# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
add_library(elf_tool_obj OBJECT ascendc_elf_tool.c)

target_compile_options(elf_tool_obj PRIVATE
  -Wextra
  -Wfloat-equal
  -Werror
  )

if(BUILD_OPEN_PROJECT)
  target_include_directories(elf_tool_obj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/build/elf_tool
  )
else()
  target_include_directories(elf_tool_obj PRIVATE
    ${TOP_DIR}/asc/asc-devkit/tools/build/elf_tool
  )
endif()

target_link_libraries(elf_tool_obj PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    c_sec_static
)