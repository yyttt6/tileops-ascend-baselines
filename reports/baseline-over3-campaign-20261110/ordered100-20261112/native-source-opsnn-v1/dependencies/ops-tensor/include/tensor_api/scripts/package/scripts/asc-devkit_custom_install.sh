#!/bin/sh
# Perform custom_install script for asc-devkit package
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

sourcedir="$PWD/asc-devkit"
curpath=$(dirname $(readlink -f "$0"))
common_func_path="${curpath}/common_func.inc"
devkit_func_path="${curpath}/asc-devkit_func.sh"

. "${common_func_path}"
. "${devkit_func_path}"

common_parse_dir=""
logfile=""
stage=""
is_quiet="n"
hetero_arch="n"

while true; do
    case "$1" in
    --install-path=*)
        pkg_install_path=$(echo "$1" | cut -d"=" -f2-)
        shift
        ;;
    --common-parse-dir=*)
        common_parse_dir=$(echo "$1" | cut -d"=" -f2-)
        shift
        ;;
    --version-dir=*)
        pkg_version_dir=$(echo "$1" | cut -d"=" -f2-)
        shift
        ;;
    --logfile=*)
        logfile=$(echo "$1" | cut -d"=" -f2)
        shift
        ;;
    --stage=*)
        stage=$(echo "$1" | cut -d"=" -f2)
        shift
        ;;
    --quiet=*)
        is_quiet=$(echo "$1" | cut -d"=" -f2)
        shift
        ;;
    --hetero-arch=*)
        hetero_arch=$(echo "$1" | cut -d"=" -f2)
        shift
        ;;
    -*)
        shift
        ;;
    *)
        break
        ;;
    esac
done

WHL_INSTALL_DIR_PATH="${common_parse_dir}/python/site-packages"
PYTHON_ASC_OP_COMPILE_BASE_NAME="asc_op_compile_base"
PYTHON_ASC_OP_COMPILE_BASE_WHL_PATH="${sourcedir}/../lib/asc_op_compile_base-0.1.0-py3-none-any.whl"
PYTHON_ASC_OPC_TOOL_NAME="asc_opc_tool"
PYTHON_ASC_OPC_TOOL_WHL_PATH="${sourcedir}/../lib/asc_opc_tool-0.1.0-py3-none-any.whl"

# 写日志
log() {
    local cur_date="$(date +'%Y-%m-%d %H:%M:%S')"
    local log_type="$1"
    shift
    echo "[AscDevkit] [$cur_date] [$log_type]: $*" 1> /dev/null
    echo "[AscDevkit] [$cur_date] [$log_type]: $*" >> "$logfile"
}

install_whl_package() {
    local _package_path="$1"
    local _package_name="$2"
    local _pythonlocalpath="$3"
    log "INFO" "start install python module package ${_package_name}."
    if [ -f "$_package_path" ]; then
        pip3 install --disable-pip-version-check --upgrade --no-deps --force-reinstall "${_package_path}" -t "${_pythonlocalpath}" 1> /dev/null
        local ret=$?
        if [ $ret -ne 0 ]; then
            log "WARNING" "install ${_package_name} failed, error code: $ret."
            exit 1
        else
            log "INFO" "${_package_name} installed successfully!"
        fi
    else
        log "ERROR" "ERR_NO:0x0080;ERR_DES:install ${_package_name} failed, can not find the matched package for this platform."
        exit 1
    fi
}

custom_install() {
    if [ -z "$common_parse_dir/share/info/asc-devkit" ]; then
        log "ERROR" "ERR_NO:0x0001;ERR_DES:asc-devkit directory is empty"
        exit 1
    elif [ "$hetero_arch" != "y" ]; then
        install_whl_package "${PYTHON_ASC_OP_COMPILE_BASE_WHL_PATH}" "${PYTHON_ASC_OP_COMPILE_BASE_NAME}" "${WHL_INSTALL_DIR_PATH}"
        install_whl_package "${PYTHON_ASC_OPC_TOOL_WHL_PATH}" "${PYTHON_ASC_OPC_TOOL_NAME}" "${WHL_INSTALL_DIR_PATH}"
    fi

    # create softlinks for stub libs in devlib/linux/${ARCH}
    create_stub_softlink "$common_parse_dir"

    return 0
}

custom_install
if [ $? -ne 0 ]; then
    exit 1
fi
exit 0
