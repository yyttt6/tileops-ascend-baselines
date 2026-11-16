# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
project (asc_opc)

#build opc whl pkg
message(STATUS "CMAKE_BINARY_DIR=" ${CMAKE_BINARY_DIR})
message(STATUS "CMAKE_CURRENT_SOURCE_DIR=" ${CMAKE_CURRENT_SOURCE_DIR})

set(OPC_WHL_NAME "asc_opc_tool-0.1.0-py3-none-any.whl")
set(OPC_DIR "./opc_python")
set(OPC_BASE "${CMAKE_CURRENT_SOURCE_DIR}")

add_custom_target(${OPC_WHL_NAME} ALL
  COMMAND echo "[OPC] Build target ${OPC_WHL_NAME}" &&
    mkdir -p ${OPC_DIR}/asc_opc_tool && rm -rf ${OPC_DIR}/asc_opc_tool/* &&
    cp -rf ${OPC_BASE}/python/asc_opc_tool/* ${OPC_DIR}/asc_opc_tool/ &&
    cp -rf ${OPC_BASE}/setup.py ${OPC_DIR}/ &&
    cd ${OPC_DIR} && ${HI_PYTHON} setup.py bdist_wheel && cd - &&
    ls && pwd &&
    mkdir ${CMAKE_INSTALL_PREFIX}/lib -p &&
    ${CMAKE_COMMAND} -E rename  ${OPC_DIR}/dist/asc_opc*.whl ${CMAKE_INSTALL_PREFIX}/lib/${OPC_WHL_NAME} &&
    echo "[OPC] Build target ${OPC_WHL_NAME} end"
)

if(BUILD_OPEN_PROJECT)
  install(FILES ${OPC_BASE}/asc_opc
    DESTINATION ${INSTALL_LIBRARY_DIR}/bin
  )
endif()