#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# 生成版本号头文件

set -e

parse_version() {
    local _outvar1="$1"
    local _outvar2="$2"
    local _version="$3"

    if [[ "$_version" == *-* ]]; then
        IFS="." read -ra "$_outvar1" <<< "${_version%%-*}"
        read -r "$_outvar2" <<< "${_version#*-}"
    else
        # 如果没有 '-' 但通过 '.' 分隔后超过3段（如：1.2.3.rc1）
        # 将其拆分为数组处理
        IFS="." read -ra "$_outvar1" <<< "$(cut -d. -f-3 <<< "$_version")"
        read -r "$_outvar2" <<< "$(cut -d. -f4- <<< "$_version")"
    fi
}

calc_pre_release_expr() {
    local _outvar="$1"
    local _pre_release="$2"
    local _pre_type_weight _num_value

    _pre_type_weight=400
    if [[ "$_pre_release" == rc* ]]; then
        _pre_type_weight=100
    elif [[ "$_pre_release" == beta* ]]; then
        _pre_type_weight=200
    elif [[ "$_pre_release" == alpha* ]]; then
        _pre_type_weight=300
    fi
    # 移除pre_release中非数字字符，转为10进制数，防止以0开头被识别为8进制数
    _num_value="$(( "10#$(sed 's/[^0-9]//g' <<< "$_pre_release")" ))"
    read -r "$_outvar" <<< "- ${_pre_type_weight} + ${_num_value}"
}

gen_version_header() {
    local package_name="$1"
    local version="$2"
    local -a release_part 
    local pre_release base_expr pre_release_expr

    if ! grep -Eq "^[0-9]+\.[0-9]+\.[0-9]+" <<< "$version"; then
        echo "error: Invalid version number: $version! (package_name=${package_name})"
        return 1
    fi

    package_name="$(tr "-" "_" <<< ${package_name^^})"
    version="${version%%+*}"

    parse_version "release_part" "pre_release" "$version"

    echo "#ifndef ${package_name}_VERSION_H"
    echo "#define ${package_name}_VERSION_H"
    echo ""
    echo "#define ${package_name}_VERSION_STR \"${version}\""
    echo "#define ${package_name}_MAJOR ${release_part[0]}"
    echo "#define ${package_name}_MINOR ${release_part[1]}"
    echo "#define ${package_name}_PATCH ${release_part[2]}"
    echo "#define ${package_name}_PRERELEASE \"${pre_release}\""

    base_expr="(${release_part[0]} * 10000000) + (${release_part[1]} * 100000) + (${release_part[2]} * 1000)"
    if [[ -z "$pre_release" ]]; then
        echo "#define ${package_name}_VERSION_NUM (${base_expr})"
        echo ""
        echo "#endif /* ${package_name}_VERSION_H */"
        return 0
    fi

    calc_pre_release_expr "pre_release_expr" "$pre_release"
    echo "#define ${package_name}_VERSION_NUM (${base_expr} ${pre_release_expr})"
    echo ""
    echo "#endif /* ${package_name}_VERSION_H */"
    return 0
}

parse_args() {
    parsed_args=$(getopt -n "$BASH_SOURCE" -a -o "" -l output: -- "$@") || {
        return 1
    }
    eval set -- "$parsed_args"

    while true; do
        case "$1" in
        --output)
            output=$2
            shift 2
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Undefined option: $1"
            return 1
            ;;
        esac
    done
    package_name="$1"
    version="$2"

    if [[ -z "$package_name" ]]; then
        echo "error: package_name is empty!"
        return 1
    fi

    if [[ -z "$version" ]]; then
        echo "error: version is empty!"
        return 1
    fi
}

main() {
    local output="" output_dir package_name version
    local -a cmd

    parse_args "$@"
    cmd=("gen_version_header" "$package_name" "$version")
    if [[ -z "$output" ]]; then
        "${cmd[@]}"
    else
        output_dir="$(dirname "$output")"
        if [[ ! -d "$output_dir" ]]; then
            mkdir -p "$output_dir"
        fi
        "${cmd[@]}" > "$output"
    fi
}

main "$@"
