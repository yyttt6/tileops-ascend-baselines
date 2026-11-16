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

CUR_DIR="$(cd "$(dirname "$0")" && pwd)"

source ${CUR_DIR}/util.sh

# using lcov to generate coverage for cpp files
generate_coverage() {
  local _source_dir="$1"
  local _coverage_file="$2"
  local _cann_pkg_path="$3"

  if [[ -z "${_source_dir}" ]]; then
    logging "directory required to find the .da files"
    exit 1
  fi

  if [[ ! -d "${_source_dir}" ]]; then
    logging "directory is not exist, please check ${_source_dir}"
    exit 1
  fi

  if [[ -z "${_coverage_file}" ]]; then
    _coverage_file="coverage.info"
    logging "using default file name to generate coverage"
  fi

  \which lcov >/dev/null 2>&1
  if [[ $? -ne 0 ]]; then
    logging "lcov is required to generate coverage data, please install"
    exit 1
  fi

  local _path_to_gen="$(dirname ${_coverage_file})"
  if [[ ! -d "${_path_to_gen}" ]]; then
    mk_dir "${_path_to_gen}"
  fi

  # 获取lcov版本号
  LCOV_MAJOR=$(lcov --version 2>/dev/null | grep -oE '[0-9]+' | head -n 1)
  # 初始化额外参数字符串
  REMOVE_ARGS=""
  EXTRA_ARGS=""
  # 版本比较，>=2.0，加参数忽略编译和运行行号不一致导致的lcov执行失败
  if [ -n "$LCOV_MAJOR" ] && [ "$LCOV_MAJOR" -ge 2 ]; then
    REMOVE_ARGS="--ignore-errors unused"
    EXTRA_ARGS="--ignore-errors mismatch --ignore-errors source"
  fi

  lcov -c -d "${_source_dir}" -o "${_coverage_file}" $EXTRA_ARGS 2>/dev/null
  lcov -r "${_coverage_file}" "${_cann_pkg_path}/*" "/home/jenkins/opensource/*" "${_src}/build/*" "${_src}/build_out/*" "${_src}/output/*" "${_src}/tests/*" -o "${_coverage_file}" $REMOVE_ARGS 2>/dev/null
  logging "generated coverage file ${_coverage_file}"
}

# filter out some unused directories or files
filter_coverage() {
  local _coverage_file="$1"
  local _filtered_file="$2"

  if [[ ! -f "${_coverage_file}" ]]; then
    logging "coverage data file required"
    exit 1
  fi

  \which lcov >/dev/null 2>&1
  if [[ $? -ne 0 ]]; then
    logging "lcov is required to generate coverage data, please install"
    exit 1
  fi

  # 获取lcov版本号
  LCOV_MAJOR=$(lcov --version 2>/dev/null | grep -oE '[0-9]+' | head -n 1)
  # 初始化额外参数字符串
  REMOVE_ARGS=""
  # 版本比较，>=2.0，加参数忽略编译和运行行号不一致导致的lcov执行失败
  if [ -n "$LCOV_MAJOR" ] && [ "$LCOV_MAJOR" -ge 2 ]; then
    REMOVE_ARGS="--ignore-errors unused"
  fi

  lcov --remove "${_coverage_file}" '/usr/include/*' '/usr/local/include/*' -o "${_filtered_file}" $REMOVE_ARGS 2>/dev/null
}

# generate html report
generate_html() {
  local _filtered_file="$1"
  local _out_path="$2"

  \which genhtml >/dev/null 2>&1
  if [[ $? -ne 0 ]]; then
    logging "genhtml is required to generate coverage html report, please install"
    exit 1
  fi

  local _path_to_gen="$(dirname ${_out_path})"
  if [[ ! -d "${_out_path}" ]]; then
    mk_dir "${_out_path}"
  fi
  genhtml "${_filtered_file}" -o "${_out_path}"
}


if [[ $# -ne 4 ]]; then
  logging "Usage: $0 DIR COV_FILE OUT_PATH CANN_PATH"
  exit 0
fi

_src="$1"
_cov_file="$2"
_out="$3"
_cann_path="$4"

generate_coverage "${_src}" "${_cov_file}" "${_cann_path}"
filter_coverage   "${_cov_file}" "${_cov_file}_filtered"
generate_html     "${_cov_file}_filtered" "${_out}"
