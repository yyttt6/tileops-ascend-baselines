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
set -e

while [[ $# -gt 0 ]]; do
    case $1 in
    -s)
        src=$2
        shift 2
        ;;
    -d)
        dst=$2
        shift 2
        ;;
    *)
        break
        ;;
    esac
done

if [  -n "${dst}" ]; then
    rm -rf ${dst}
fi

if [ ! -d "${dst}" ]; then
    mkdir -p ${dst}
fi

for arg in "$@"
do
    relative_file=$(realpath --relative-to="${src}" "${arg}")
    dst_file=${dst}/${relative_file}
    dst_dir=$(dirname ${dst_file})
    if [ ! -d "${dst_dir}" ]; then
        mkdir -p ${dst_dir}
    fi
    cp -v ${arg} ${dst_dir}
done

