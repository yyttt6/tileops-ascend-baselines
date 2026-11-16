#!/bin/bash
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------


if [ "$(id -u)" != "0" ]; then
    _LOG_PATH=$(echo "${HOME}")"/var/log/ascend_seclog"
    _INSTALL_LOG_FILE="${_LOG_PATH}/ascend_install.log"
    _OPERATE_LOG_FILE="${_LOG_PATH}/operation.log"
else
    _LOG_PATH="/var/log/ascend_seclog"
    _INSTALL_LOG_FILE="${_LOG_PATH}/ascend_install.log"
    _OPERATE_LOG_FILE="${_LOG_PATH}/operation.log"
fi

# log functions
getdate() {
    _cur_date=$(date +"%Y-%m-%d %H:%M:%S")
    echo "${_cur_date}"
}

logandprint() {
    is_error_level=$(echo $1 | grep -E 'ERROR|WARN|INFO')
    if [ "${is_quiet}" != "y" ] || [ "${is_error_level}" != "" ]; then
        echo "[OpsBase] [$(getdate)] ""$1"
    fi
    echo "[OpsBase] [$(getdate)] ""$1" >> "${_INSTALL_LOG_FILE}"
}

# create opapi soft link
createrelativelysoftlink() {
    local src_path_="$1"
    local dst_path_="$2"
    local dst_parent_path_=$(dirname ${dst_path_})
    # echo "dst_parent_path_: ${dst_parent_path_}"
    local relative_path_=$(realpath --relative-to="$dst_parent_path_" "$src_path_")
    # echo "relative_path_: ${relative_path_}"
    if [ -L "$2" ]; then
        return 0
    fi
    ln -s "${relative_path_}" "${dst_path_}" 2> /dev/null
    if [ "$?" != "0" ]; then
        return 1
    else
        return 0
    fi
}

_CURR_PATH=$(dirname $(readlink -f $0))
SCENE_FILE="${_CURR_PATH}""/../scene.info"

get_pkg_arch_name() {
    if [ ! -f "$SCENE_FILE" ]; then
        echo "[OpsBase] [$(getdate)] [ERROR]: $SCENE_FILE file cannot be found!"
        exit 1
    fi
    local arch="$(grep -iw arch "$SCENE_FILE" | cut -d"=" -f2- | awk '{print tolower($0)}')"
    if [ -z "$arch" ]; then
        echo "[OpsBase] [$(getdate)] [ERROR]: var arch cannot be found in file $SCENE_FILE!"
        exit 1
    fi
    echo $arch
}

get_stub_libs_from_filelist() {
    awk -v arch_name="$arch_name" 'BEGIN {
        FS=","
        prefix=sprintf("^%s-linux/devlib/", arch_name)
        pat=sprintf("^%s-linux/devlib/(linux/%s/[^/]+\\.(so|a)$)", arch_name, arch_name)
    }
    {
        if (match($4, pat)) {
            sub(prefix, "", $4)
            print($4)
        }
    }' $_CURR_PATH/filelist.csv
}

create_stub_softlink() {
    local install_path="$1"
    if [ ! -d "$install_path" ]; then
        return
    fi
    local arch_name="$pkg_arch_name"
    ([ -d "$install_path/${arch_name}-linux/devlib" ] && cd "$install_path/${arch_name}-linux/devlib" && {
        chmod u+w . && \
        for lib in $(get_stub_libs_from_filelist); do
            [ -f "$lib" ] && ln -sf "$lib" "$(basename $lib)"
        done
        chmod u-w .
    })
}

remove_stub_softlink() {
    local install_path="$1"
    if [ ! -d "$install_path" ]; then
        return
    fi
    local arch_name="$pkg_arch_name"
    ([ -d "$install_path/${arch_name}-linux/devlib" ] && cd "$install_path/${arch_name}-linux/devlib" && {
        chmod u+w . && basename --multiple $(get_stub_libs_from_filelist) | xargs --no-run-if-empty rm -rf
        chmod u-w .
    })
}

pkg_arch_name="$(get_pkg_arch_name)"