#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
SHELL_DIR=$(cd "$(dirname "$0")" || exit;pwd)
COMMON_SHELL_PATH="$SHELL_DIR/common.sh"
LOG_PATH="/var/log/ascend_seclog/ascend_install.log"
PACKAGE=asc_tools
LEVEL_INFO="INFO"
LEVEL_WARN="WARNING"
LEVEL_ERROR="ERROR"
MIN_PIP_VERSION=18

source "${COMMON_SHELL_PATH}"

username=$(id -un)
usergroup=$(groups | cut -d" " -f1)

log() {
    local content=`echo "$@" | cut -d" " -f2-`
    cur_date=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[Toolkit] [${cur_date}] [$1]: $content" >> "${log_file}"
}

log_and_print() {
    local content=`echo "$@" | cut -d" " -f2-`
    cur_date=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[Toolkit] [${cur_date}] [$1]: $content"
    echo "[Toolkit] [${cur_date}] [$1]: $content" >> "${log_file}"
}

checkPip() {
    pip_info=$(pip3 --version 2>/dev/null)
    if [ $? -ne 0 ] || [ -z "${pip_info}" ]; then
        log_and_print $LEVEL_ERROR "execute pip3 --version failed, please check."
        return 1
    fi
    version=$(echo "${pip_info}" | cut -d" " -f2 | cut -d"." -f1)
    if [ ${version} -lt ${MIN_PIP_VERSION} ]; then
        log_and_print $LEVEL_WARN "pip3 version low than ${MIN_PIP_VERSION}.0"
        log_and_print $LEVEL_WARN "Current pip3 version may cause the installation problems."
    fi
    curpath="$(dirname ${BASH_SOURCE:-$0})"
    install_dir="$(realpath $curpath/..)"
    common_interface=$(realpath $install_dir/script*/common_interface.bash)
    if [ -f "$common_interface" ]; then
        . "$common_interface"
        py_version_check
    else
        log_and_print $LEVEL_ERROR "Failed to find common_interface.bash file to check python version."
        return 1
    fi
    return 0
}

installWhlPackage() {
    local _package=$1
    local _pythonlocalpath=$2

    changeFileMode 750 ${_pythonlocalpath}
    if [ ! -f "${_package}" ]; then
        log_and_print ${LEVEL_ERROR} "ERR_NO:0x0080;ERR_DES: The ${_package} does not exist."
        return 1
    fi
    if [ "-${pylocal}" = "-y" ]; then
        pip3 install --upgrade --no-deps --force-reinstall --disable-pip-version-check "${_package}" -t "${_pythonlocalpath}" >> "${log_file}" 2>&1
    else
        if [ "$(id -u)" -ne 0 ]; then
            pip3 install --upgrade --no-deps --force-reinstall --disable-pip-version-check "${_package}" --user >> "${log_file}" 2>&1
        else
            pip3 install --upgrade --no-deps --force-reinstall --disable-pip-version-check "${_package}" >> "${log_file}" 2>&1
        fi
    fi
    if [ $? -ne 0 ]; then
        log_and_print ${LEVEL_ERROR} "Install ${_package} failed. For more details, please read ${log_file} file."
        return 1
    fi
    log ${LEVEL_INFO} "install ${_package} succeed."
    return 0
}

installMsprofWhlPackage() {
    local _package=$1
    local _python_local_path=$2

    log ${LEVEL_INFO} "start to begin install ${_package}."
    log ${LEVEL_INFO} "The installation path ${_python_local_path} of whl package"
    if [ ! -f "${_package}" ]; then
        log_and_print ${LEVEL_ERROR} "ERR_NO:0x0080;ERR_DES: The ${_package} does not exist."
        return 1
    fi

    pip3 install --upgrade --no-deps --force-reinstall --disable-pip-version-check "${_package}" -t "${_python_local_path}" > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        log_and_print ${LEVEL_ERROR} "Install ${_package} failed."
        return 1
    fi
    changeDirMode 550 ${_python_local_path}
    changeFileMode 550 ${_python_local_path}
    log ${LEVEL_INFO} "install ${_package} succeed."
    return 0
}

installOpPython() {
    checkAscendcFeature ${feature_type}
    [ $? -ne 0 ] && return 0

    log_and_print $LEVEL_INFO "begin to install ascendc python whl."
    if [ "-${pylocal}" = "-y" ]; then
        createPythonLocalDir "$install_path"
        [ $? -ne 0 ] && return 1
        installOpPythonLocal
        [ $? -ne 0 ] && return 1
        return 0
    fi

    installWhlPackage "${install_path}/${PACKAGE}/tools/msobjdump-0.1.0-py3-none-any.whl"
    if [ $? -ne 0 ]; then
        return 1
    fi

    installWhlPackage "${install_path}/${PACKAGE}/tools/show_kernel_debug_data-0.1.0-py3-none-any.whl"
    if [ $? -ne 0 ]; then
        return 1
    fi

    return 0
}

installAllPython() {
    checkAllFeature ${feature_type}
    [ $? -ne 0 ] && return 0

    if [ "-${pylocal}" = "-y" ]; then
        createPythonLocalDir "$install_path"
        [ $? -ne 0 ] && return 1
        installAllPythonLocal
        [ $? -ne 0 ] && return 1
        return 0
    fi

    installWhlPackage "${install_path}/${PACKAGE}/tools/msobjdump-0.1.0-py3-none-any.whl"
    if [ $? -ne 0 ]; then
        return 1
    fi

    installWhlPackage "${install_path}/${PACKAGE}/tools/show_kernel_debug_data-0.1.0-py3-none-any.whl"
    if [ $? -ne 0 ]; then
        return 1
    fi

    return 0
}

installPython() {
    checkPip
    [ $? -ne 0 ] && return 1

    installOpPython
    [ $? -ne 0 ] && return 1

    installAllPython
    [ $? -ne 0 ] && return 1

    return 0
}

createPythonLocalDir() {
    local _dst_path="$1"
    local _py_path="$_dst_path/python"
    local _py_pkg_path="$_dst_path/python/site-packages"

    if [ ! -d "$_py_path" ]; then
        createFolder "$_py_path" "$username:$usergroup" 750
        if [ $? -ne 0 ]; then
            log_and_print $LEVEL_ERROR "Create $_py_path folder failed."
            return 1
        fi
    fi
    if [ ! -d "$_py_pkg_path" ]; then
        createFolder "$_py_pkg_path" "$username:$usergroup" 750
        if [ $? -ne 0 ]; then
            log_and_print $LEVEL_ERROR "Create $_py_pkg_path folder failed."
            return 1
        fi
    fi
    return 0
}

createPythonLocalDirSoftlink() {
    local _py_dirs="$@"
    [ -z "$_py_dirs" ] && return

    createPythonLocalDir "$install_path/$PACKAGE"
    # create softlink python/site-packages to toolkit/python/site-packages
    local _py_pkg_path="$install_path/$PACKAGE/python/site-packages"
    for item in ${_py_dirs[@]}; do
        if [ -d "${_py_pkg_path}/${item}" ] || [ -f "${_py_pkg_path}/${item}" ]; then
            rm -f "${_py_pkg_path}/${item}"
        fi
        ln -s "../../../python/site-packages/${item}" "${_py_pkg_path}/${item}"
    done
}

installPyWhlLocal() {
    local _whl_name="$1"
    local _py_whl_path="$install_path/$PACKAGE/tools/$_whl_name"
    local _py_install_path="$install_path/python/site-packages"

    local _temp_install_path="$_py_install_path/temp"
    # install python package to temp
    installWhlPackage "$_py_whl_path" "$_temp_install_path"
    [ $? -ne 0 -o ! -d "$_temp_install_path" ] && return 1

    changeDirMode 750 "$_temp_install_path"
    changeFileMode 550 "$_temp_install_path"
    chown -RP "$username:$usergroup" "$_temp_install_path">/dev/null 2>&1

    # copy temp to dest path
    local _py_dirs=$(ls "$_temp_install_path")
    for dir in ${_py_dirs[@]}; do
        if [ "-$dir" = "-bin" ]; then
            if [ -d "$_py_install_path/bin" ]; then
                cp -Rfp "$_temp_install_path/bin"/* "$_py_install_path/bin"
                continue
            fi
        fi
        cp -Rfp "$_temp_install_path/$dir" "$_py_install_path"
    done

    # create softlink python/site-packages to toolkit/python/site-packages
    createPythonLocalDirSoftlink "$_py_dirs"
    # remove temp
    rm -rf "$_temp_install_path"
    return 0
}

installOpPythonLocal() {
    installPyWhlLocal "msobjdump-0.1.0-py3-none-any.whl"
    [ $? -ne 0 ] && return 1

    installPyWhlLocal "show_kernel_debug_data-0.1.0-py3-none-any.whl"
    [ $? -ne 0 ] && return 1

    log $LEVEL_INFO "Install opeator python package succeed."
    return 0
}

installAllPythonLocal() {

    installPyWhlLocal "hccl_parser-0.1-py3-none-any.whl"
    [ $? -ne 0 ] && return 1

    installPyWhlLocal "msobjdump-0.1.0-py3-none-any.whl"
    [ $? -ne 0 ] && return 1
    installPyWhlLocal "show_kernel_debug_data-0.1.0-py3-none-any.whl"
    [ $? -ne 0 ] && return 1

    log $LEVEL_INFO "Install all module python package succeed."
    return 0
}

init() {
    [ ! -d "${install_path}" ] && exit 1

    if [ ! -z "${version_dir}" ]; then
        install_path="${install_path}/${version_dir}"
        [ ! -d "${install_path}" ] && exit 1
    fi

    if [ $(id -u) -eq 0 ]; then
        log_file=${LOG_PATH}
    else
        local _home_path=$(eval echo "~")
        log_file="${_home_path}/${LOG_PATH}"
    fi
}

log_file=""
is_quiet=n
pylocal=n
install_path=""
version_dir=""
feature_type=""

while true; do
    case "$1" in
    --install-path=*)
        install_path=$(echo "$1" | cut -d"=" -f2-)
        [ -z "${install_path}" ] && exit 1
        shift
        ;;
    --version-dir=*)
        version_dir=$(echo "$1" | cut -d"=" -f2-)
        shift
        ;;
    --quiet=*)
        is_quiet=$(echo "$1" | cut -d"=" -f2)
        shift
        ;;
    --pylocal=*)
        pylocal=$(echo "$1" | cut -d"=" -f2)
        shift
        ;;
    --feature=*)
        feature_type=$(echo "$1" | cut -d"=" -f2)
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

init

installPython
[ $? -ne 0 ] && exit 1

exit 0
