#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# Kernel UT 覆盖率报告生成脚本
# 用法: bash generate_cpp_cov.sh <build_dir> <cov_info_file> <html_out_dir> <cann_pkg_path> [path_prefix]
#
# 仅看护 include/blaze 下的代码覆盖率，submodule 及其他代码不纳入报告。

set -e

CUR_DIR="$(cd "$(dirname "$0")" && pwd)"

logging() {
    echo "[COV] $1"
}

mk_dir() {
    mkdir -p "$1"
}

# 采集覆盖率数据
# 参数: source_dir coverage_file cann_pkg_path
generate_coverage() {
    local _source_dir="$1"
    local _coverage_file="$2"
    local _cann_pkg_path="$3"

    if [[ -z "${_source_dir}" ]]; then
        logging "directory required to find the .da/.gcda files"
        exit 1
    fi

    if [[ ! -d "${_source_dir}" ]]; then
        logging "directory does not exist, please check ${_source_dir}"
        exit 1
    fi

    if [[ -z "${_coverage_file}" ]]; then
        _coverage_file="coverage.info"
        logging "using default file name to generate coverage"
    fi

    if ! command -v lcov >/dev/null 2>&1; then
        logging "lcov is required to generate coverage data, please install"
        exit 1
    fi

    local _path_to_gen="$(dirname ${_coverage_file})"
    if [[ ! -d "${_path_to_gen}" ]]; then
        mk_dir "${_path_to_gen}"
    fi

    # lcov 版本兼容: >=2.0 需要额外参数忽略行号不一致错误
    local LCOV_MAJOR=$(lcov --version 2>/dev/null | grep -oE '[0-9]+' | head -n 1)
    local REMOVE_ARGS=""
    local EXTRA_ARGS=""
    if [ -n "$LCOV_MAJOR" ] && [ "$LCOV_MAJOR" -ge 2 ]; then
        REMOVE_ARGS="--ignore-errors unused --ignore-errors mismatch --ignore-errors source"
        EXTRA_ARGS="--ignore-errors mismatch --ignore-errors source"
    fi

    logging "Collecting coverage data from ${_source_dir}..."
    lcov -c -d "${_source_dir}" -o "${_coverage_file}" $EXTRA_ARGS

    # 仅保留 include/blaze/ 下的代码，其余全部移除
    logging "Extracting include/blaze coverage only..."
    lcov --extract "${_coverage_file}" "*/include/blaze/*" \
        -o "${_coverage_file}" $REMOVE_ARGS

    logging "Generated coverage file: ${_coverage_file}"
}

# 生成 HTML 报告
generate_html() {
    local _filtered_file="$1"
    local _out_path="$2"
    local _path_prefix="$3"

    if ! command -v genhtml >/dev/null 2>&1; then
        logging "genhtml is required to generate coverage html report, please install"
        exit 1
    fi

    local _path_to_gen="$(dirname "${_out_path}")"
    if [[ ! -d "${_out_path}" ]]; then
        mk_dir "${_out_path}"
    fi

    local _custom_css="${CUR_DIR}/gcov_custom.css"
    local _html_args=()
    if [[ -f "${_custom_css}" ]]; then
        _html_args+=(--css-file "${_custom_css}")
        logging "Using custom CSS: ${_custom_css}"
    fi

    if [[ -n "${_path_prefix}" ]]; then
        _html_args+=(--prefix "${_path_prefix}")
        logging "Stripping path prefix: ${_path_prefix}"
    fi

    logging "Generating HTML report..."
    genhtml "${_filtered_file}" -o "${_out_path}" "${_html_args[@]}"
    logging "HTML report generated at: ${_out_path}/index.html"
}

# 对比 include/blaze 全量文件与 coverage.info 中已出现的文件，输出未被看护的文件
report_uncovered_files() {
    local _cov_file="$1"
    local _path_prefix="$2"
    local _out_path="$3"

    local _blaze_dir="${_path_prefix}/ops-tensor/include/blaze"
    if [[ ! -d "${_blaze_dir}" ]]; then
        logging "Blaze directory not found: ${_blaze_dir}, skip uncovered file report"
        return 0
    fi

    local _all_files=$(find "${_blaze_dir}" -name "*.h" -o -name "*.hpp" | sort)
    local _cov_files=$(grep "^SF:" "${_cov_file}" | sed 's/SF://' | sort)

    local _uncovered=$(comm -23 \
        <(echo "${_all_files}") \
        <(echo "${_cov_files}"))

    local _total=$(echo "${_all_files}" | grep -c .)
    local _covered=$(echo "${_cov_files}" | grep -c . || true)
    local _uncov_count=$(echo "${_uncovered}" | grep -c . || true)

    local _report_file="${_out_path}/uncovered_files.txt"

    {
        echo "Uncovered blaze files (${_uncov_count}/${_total} files not guarded by this kernel UT)"
        echo "Generated: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "=========================================="
        if [[ ${_uncov_count} -eq 0 ]]; then
            echo "All blaze files are covered."
        else
            echo "${_uncovered}" | while read -r f; do
                local _rel="${f#${_path_prefix}/ops-tensor/}"
                echo "[MISS] ${_rel}"
            done
        fi
        echo "=========================================="
    } > "${_report_file}"

    logging "=========================================="
    logging "Uncovered blaze files (${_uncov_count}/${_total} files not guarded by this kernel UT):"
    if [[ ${_uncov_count} -eq 0 ]]; then
        logging "  All blaze files are covered."
    else
        echo "${_uncovered}" | while read -r f; do
            local _rel="${f#${_path_prefix}/ops-tensor/}"
            logging "  [MISS] ${_rel}"
        done
    fi
    logging "=========================================="
    logging "Uncovered files report: ${_report_file}"
}

# 主流程
if [[ $# -lt 4 || $# -gt 5 ]]; then
    logging "Usage: $0 BUILD_DIR COV_FILE OUT_PATH CANN_PATH [PATH_PREFIX]"
    exit 1
fi

_src="$1"
_cov_file="$2"
_out="$3"
_cann_path="$4"
_path_prefix="${5:-}"

generate_coverage "${_src}" "${_cov_file}" "${_cann_path}"
generate_html     "${_cov_file}" "${_out}" "${_path_prefix}"

# 输出覆盖率摘要
logging "=========================================="
logging "Coverage report summary:"
lcov --summary "${_cov_file}" 2>/dev/null || true
logging "=========================================="
logging "HTML report: ${_out}/index.html"
logging "Coverage data: ${_cov_file}"

report_uncovered_files "${_cov_file}" "${_path_prefix}" "${_out}"
