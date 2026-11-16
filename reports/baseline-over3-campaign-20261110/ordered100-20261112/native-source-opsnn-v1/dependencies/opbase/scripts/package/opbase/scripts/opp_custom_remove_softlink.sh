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
curpath=$(dirname $(readlink -f "$0"))
SCENE_FILE="${curpath}""/../scene.info"
OPP_COMMON="${curpath}""/opp_common.sh"
common_func_path="${curpath}/common_func.inc"
. "${OPP_COMMON}"
. "${common_func_path}"

while true; do
    case "$1" in
    --install-path=*)
        install_path=$(echo "$1" | cut -d"=" -f2-)
        shift
        ;;
    --version-dir=*)
        version_dir=$(echo "$1" | cut -d"=" -f2)
        shift
        ;;
    --latest-dir=*)
        latest_dir=$(echo "$1" | cut -d"=" -f2)
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

get_version_dir "opp_kernel_version_dir" "$install_path/$version_dir/opp_kernel/version.info"

if [ -z "$opp_kernel_version_dir" ]; then
    latestSoftlinksRemove "$install_path/$version_dir"
fi

# if opp exists, restore the softlink of the opp version
get_version_dir "kernel_latest_dir" "${install_path}/$latest_dir/opp_kernel/version.info"

if [ -n "$kernel_latest_dir" ]; then
    createOpapiLatestSoftlink ${install_path}/${kernel_latest_dir} opp_kernel
fi

level2_dir="$install_path/$latest_dir/include/aclnnop/level2"
if [ -d "$level2_dir" ] && [ "$(ls -A "$level2_dir")" = "" ]; then
    rm -rf "$level2_dir"
fi
 
aclnnop_dir="$install_path/$latest_dir/include/aclnnop"
if [ -d "$aclnnop_dir" ] && [ "$(ls -A "$aclnnop_dir")" = "" ]; then
    rm -rf "$aclnnop_dir"
fi
 
rm -rf "$install_path/$latest_dir/ops"
rm -rf "$install_path/$latest_dir/opp"
