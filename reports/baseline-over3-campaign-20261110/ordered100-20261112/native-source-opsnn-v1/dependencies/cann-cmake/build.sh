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

set -e
BASEPATH=$(cd "$(dirname "$0")"; pwd)
OUTPUT_PATH="${BASEPATH}/build_out"
BUILD_PATH="${BASEPATH}/build"

# print usage message
usage() {
  echo "Usage:"
  echo "  sh build.sh --pkgs=<PACKAGES> [-h | --help] [-v | --verbose] [-j<N>]"
  echo "              [--binary-pkgs=<PACKAGES>]"
  echo "              [--superbuild-config=<PATH>] [--p=<PATH> | --cann_path=<PATH>]"
  echo "              [--cann_3rd_lib_path=<PATH>]"
  echo "              [--build_host_only] [--build-type=<TYPE>] [--pkg-type=<TYPE>]"
  echo "              [--asan] [--cov]"
  echo "              [--sign-script <PATH>] [--enable-sign]"
  echo "              [--rule-launch | --rule_launch]"
  echo ""
  echo "Options:"
  echo "    -h, --help     Print usage"
  echo "    --pkgs=<PACKAGES>"
  echo "                   Packages to be built, separate the package names with commas"
  echo "    -j<N>          Set the number of threads used for building, default is the number of processors"
  echo "    -v, --verbose  Display build command"
  echo "    --binary-pkgs=<PACKAGES>"
  echo "                   Use the binary in the package"
  echo "    --superbuild-config=<PATH>"
  echo "                   Config path for superbuild"
  echo "    -p, --cann_path=<PATH>"
  echo "                   Set ascend package install path, default /usr/local/Ascend/cann"
  echo "    --cann_3rd_lib_path=<PATH>"
  echo "                   Set ascend third_party package install path, default ./output/third_party"
  echo "    --build_host_only"
  echo "                   Only build host target"
  echo "    --build-type=<TYPE>"
  echo "                   Specify build type (TYPE options: Release/Debug), Default: Release"
  echo "    --pkg-type=<TYPE>"
  echo "                   Specify pkg type （TYPE options: run/rpm/deb, Default: run"
  echo "    --asan         Enable AddressSanitizer"
  echo "    --cov          Enable Coverage"
  echo "    --sign-script <PATH>"
  echo "                   Set sign-script's path to <PATH>"
  echo "    --enable-sign"
  echo "                   Enable to sign"
  echo "    --rule-launch/--rule_launch"
  echo "                   Set compiler launcher rule"
  echo "    --cmake-extra-args"
  echo "                   Set cmake extra arguments"
  echo "    --host-toolchain=<PATH>"
  echo "                   Set host CMAKE_TOOLCHAIN_FILE (absolute path or filename under toolchain/)"
  echo "    --host-toolchain-dir=<PATH>"
  echo "                   Set host TOOLCHAIN_DIR (compiler root referenced by toolchain file)"
  echo "    --device-toolchain=<PATH>"
  echo "                   Set device CMAKE_TOOLCHAIN_FILE (absolute path or filename under toolchain/)"
  echo "    --device-toolchain-dir=<PATH>"
  echo "                   Set device TOOLCHAIN_DIR (compiler root referenced by toolchain file)"
  echo ""
}

trans_commas() {
  printf -v "$1" '%s' "${2//,/;}"
}

resolve_toolchain_file() {
  local input="$1"
  if [[ "$input" = /* ]]; then
    echo "$input"
  else
    echo "$BASEPATH/toolchain/$input"
  fi
}

# parse and set options
checkopts() {
  VERBOSE="0"
  THREAD_NUM=$(grep -c ^processor /proc/cpuinfo)
  ENABLE_GCOV=""
  ENABLE_ASAN=""
  CANN_3RD_LIB_PATH="$BASEPATH/output/third_party"
  BUILD_TYPE="Release"
  CUSTOM_SIGN_SCRIPT=""
  ENABLE_SIGN=""
  ENABLE_BUILD_DEVICE=""
  CANN_PACKAGES=""
  CANN_BINARY_PACKAGES=""
  CANN_SUPERBUILD_CONFIG=""
  LAUNCH_RULE=""
  MAKE_PROFILER=""
  CMAKE_EXTRA_ARGS=()
  CMAKE_EXTRA_ARGS_STR=""
  PACKAGE_TYPE="run"
  HOST_TOOLCHAIN=""
  HOST_TOOLCHAIN_DIR=""
  DEVICE_TOOLCHAIN=""
  DEVICE_TOOLCHAIN_DIR=""

  # Process the options
  parsed_args=$(getopt -a -o j:hp:v -l help,pkgs:,superbuild-config:,binary-pkgs:,verbose,cov,build_host_only,cann_path:,build-type:,pkg-type:,cann_3rd_lib_path:,asan,sign-script:,enable-sign,rule-launch:,rule_launch:,make-profiler:,cmake-extra-args:,host-toolchain:,host-toolchain-dir:,device-toolchain:,device-toolchain-dir: -- "$@") || {
    usage
    exit 1
  }
  eval set -- "$parsed_args"

  while true; do
    case "$1" in
      -h | --help)
        usage
        exit 0
        ;;
      -j)
        THREAD_NUM="$2"
        shift 2
        ;;
      -v | --verbose)
        VERBOSE="1"
        shift
        ;;
      --pkgs)
        trans_commas "CANN_PACKAGES" "$2"
        shift 2
        ;;
      --superbuild-config)
        CANN_SUPERBUILD_CONFIG="$2"
        shift 2
        ;;
      --binary-pkgs)
        trans_commas "CANN_BINARY_PACKAGES" "$2"
        shift 2
        ;;
      --asan)
        ENABLE_ASAN="on"
        shift
        ;;
      --cov)
        ENABLE_GCOV="on"
        shift
        ;;
      --build_host_only)
        ENABLE_BUILD_DEVICE="OFF"
        shift
        ;;
      --build-type)
        BUILD_TYPE=$2
        shift 2
        ;;
      --pkg-type)
        PACKAGE_TYPE=$2
        shift 2
        ;;
      --cann_path | -p)
        CANN_PATH="$(realpath "$2")"
        shift 2
        ;;
      --cann_3rd_lib_path)
        CANN_3RD_LIB_PATH="$(realpath "$2")"
        shift 2
        ;;
      --sign-script)
        CUSTOM_SIGN_SCRIPT=$2
        shift 2
        ;;
      --enable-sign)
        ENABLE_SIGN="ON"
        shift
        ;;
      --rule-launch | --rule_launch)
        LAUNCH_RULE="$2"
        shift 2
        ;;
      --make-profiler)
        MAKE_PROFILER="$2"
        shift 2
        ;;
      --cmake-extra-args)
        CMAKE_EXTRA_ARGS_STR="$2"
        shift 2
        ;;
      --host-toolchain)
        HOST_TOOLCHAIN=$(resolve_toolchain_file "$2")
        shift 2
        ;;
      --host-toolchain-dir)
        HOST_TOOLCHAIN_DIR="$2"
        shift 2
        ;;
      --device-toolchain)
        DEVICE_TOOLCHAIN=$(resolve_toolchain_file "$2")
        shift 2
        ;;
      --device-toolchain-dir)
        DEVICE_TOOLCHAIN_DIR="$2"
        shift 2
        ;;
      --)
        shift
        break
        ;;
      *)
        echo "Undefined option: $1"
        usage
        exit 1
        ;;
    esac
  done

  if [[ -z "$CANN_PACKAGES" ]]; then
    echo "error: --pkgs option is required."
    exit 1
  fi

  case "$BUILD_TYPE" in
    Release|Debug) ;;
    *) echo "error: invalid --build-type '$BUILD_TYPE', must be Release or Debug."; exit 1 ;;
  esac

  case "$PACKAGE_TYPE" in
    run|rpm|deb|deb,rpm|all) ;;
    *) echo "error: invalid --pkg-type '$PACKAGE_TYPE', must be run, rpm, deb, deb,rpm or all."; exit 1 ;;
  esac

  set_env
  parse_cmake_extra_args
}

parse_cmake_extra_args() {
  local kv_pairs kv_pair key value
  IFS=',' read -ra kv_pairs <<< "$CMAKE_EXTRA_ARGS_STR"

  for kv_pair in "${kv_pairs[@]}"; do
    if [[ -z "$kv_pair" ]]; then
        continue
    fi
    key="${kv_pair%%=*}"
    value="${kv_pair#*=}"
    CMAKE_EXTRA_ARGS+=("-D${key}=${value}")
  done
}

set_env() {
  if [ "$(id -u)" != "0" ]; then
    DEFAULT_TOOLKIT_INSTALL_DIR="${HOME}/Ascend/cann"
    DEFAULT_INSTALL_DIR="${HOME}/Ascend/cann"
  else
    DEFAULT_TOOLKIT_INSTALL_DIR="/usr/local/Ascend/cann"
    DEFAULT_INSTALL_DIR="/usr/local/Ascend/cann"
  fi

  ASCEND_CANN_PACKAGE_PATH=""
  if [ -n "${CANN_PATH}" ];then
    ASCEND_CANN_PACKAGE_PATH=${CANN_PATH}
  elif [ -n "${ASCEND_HOME_PATH}" ];then
    ASCEND_CANN_PACKAGE_PATH=${ASCEND_HOME_PATH}
  elif [ -n "${ASCEND_OPP_PATH}" ];then
    ASCEND_CANN_PACKAGE_PATH=$(dirname "${ASCEND_OPP_PATH}")
  elif [ -d "${DEFAULT_TOOLKIT_INSTALL_DIR}" ];then
    ASCEND_CANN_PACKAGE_PATH=${DEFAULT_TOOLKIT_INSTALL_DIR}
  elif [ -d "${DEFAULT_INSTALL_DIR}" ];then
    ASCEND_CANN_PACKAGE_PATH=${DEFAULT_INSTALL_DIR}
  fi
}

mk_dir() {
  local create_dir="$1"  # the target to make
  mkdir -pv "${create_dir}"
  echo "created ${create_dir}"
}

# create build path
build_project() {
  local -a cmake_cmd
  echo "create build directory and build";
  mk_dir "${BUILD_PATH}"
  mk_dir "${OUTPUT_PATH}"
  mkdir -p "$BUILD_PATH/.cmake/api/v1/query"
  touch "$BUILD_PATH/.cmake/api/v1/query/codemodel-v2"

  CMAKE_ARGS=(
    "${CMAKE_EXTRA_ARGS[@]}"
    "-DENABLE_OPEN_SRC=TRUE"
    "-DENABLE_UNIFIED_BUILD=TRUE"
    "-DCANN_PACKAGES=${CANN_PACKAGES}"
    "-DCMAKE_BUILD_TYPE=${BUILD_TYPE}"
    "-DCMAKE_INSTALL_PREFIX=${OUTPUT_PATH}"
    "-DASCEND_CANN_PACKAGE_PATH=${ASCEND_CANN_PACKAGE_PATH}"
    "-DCANN_3RD_LIB_PATH=${CANN_3RD_LIB_PATH}"
    "-DPACKAGE_TYPE=${PACKAGE_TYPE}"
  )

  if [ -n "${CANN_BINARY_PACKAGES}" ]; then
    CMAKE_ARGS+=("-DCANN_BINARY_PACKAGES=${CANN_BINARY_PACKAGES}")
  fi
  if [ -n "${CANN_SUPERBUILD_CONFIG}" ]; then
    CMAKE_ARGS+=("-DCANN_SUPERBUILD_CONFIG=${CANN_SUPERBUILD_CONFIG}")
  fi
  if [ -n "${ENABLE_GCOV}" ]; then
    CMAKE_ARGS+=("-DENABLE_GCOV=${ENABLE_GCOV}")
  fi
  if [ -n "${ENABLE_ASAN}" ]; then
    CMAKE_ARGS+=("-DENABLE_ASAN=${ENABLE_ASAN}")
  fi
  if [ -n "${ENABLE_SIGN}" ]; then
    CMAKE_ARGS+=("-DENABLE_SIGN=${ENABLE_SIGN}")
  fi
  if [ -n "${ENABLE_BUILD_DEVICE}" ]; then
    CMAKE_ARGS+=("-DENABLE_BUILD_DEVICE=${ENABLE_BUILD_DEVICE}")
  fi
  if [ -n "${CUSTOM_SIGN_SCRIPT}" ]; then
    CMAKE_ARGS+=("-DCUSTOM_SIGN_SCRIPT=${CUSTOM_SIGN_SCRIPT}")
  fi
  if [ -n "${HOST_TOOLCHAIN}" ]; then
    CMAKE_ARGS+=("-DCMAKE_TOOLCHAIN_FILE=${HOST_TOOLCHAIN}")
  fi
  if [ -n "${HOST_TOOLCHAIN_DIR}" ]; then
    CMAKE_ARGS+=("-DTOOLCHAIN_DIR=${HOST_TOOLCHAIN_DIR}")
  fi
  if [ -n "${DEVICE_TOOLCHAIN}" ]; then
    CMAKE_ARGS+=("-DDEVICE_TOOLCHAIN_FILE=${DEVICE_TOOLCHAIN}")
  fi
  if [ -n "${DEVICE_TOOLCHAIN_DIR}" ]; then
    CMAKE_ARGS+=("-DDEVICE_TOOLCHAIN_DIR=${DEVICE_TOOLCHAIN_DIR}")
  fi

  if [ -n "${LAUNCH_RULE}" ]; then
    CMAKE_ARGS+=("-DLAUNCH_COMPILE_TOOL=${LAUNCH_RULE}")
    CMAKE_ARGS+=("-DLAUNCH_LINK_TOOL=${LAUNCH_RULE}")
  fi

  cmake_cmd=(cmake -S "$BASEPATH/superbuild" -B "$BUILD_PATH" "${CMAKE_ARGS[@]}")
  if ! "${cmake_cmd[@]}"; then
    echo "execute command: ${cmake_cmd[*]} failed."
    return 1
  fi

  if [[ -n "$MAKE_PROFILER" ]]; then
    cmake_cmd=("$MAKE_PROFILER" -C "$BUILD_PATH" "-j${THREAD_NUM}")
    if [[ "$VERBOSE" == "1" ]]; then
      cmake_cmd+=("VERBOSE=1")
    fi
  else
    cmake_cmd=(cmake --build "$BUILD_PATH" "-j${THREAD_NUM}")
    if [[ "$VERBOSE" == "1" ]]; then
      cmake_cmd+=("--verbose")
    fi
  fi
  if ! "${cmake_cmd[@]}"; then
    echo "execute command: ${cmake_cmd[*]} failed."
    return 1
  fi

  cmake_cmd=(cpack -B "$BUILD_PATH" --config "$BUILD_PATH/CPackConfig.cmake")
  if ! "${cmake_cmd[@]}"; then
    echo "execute command: ${cmake_cmd[*]} failed."
    return 1
  fi
  echo "build success!"
}

function main() {
  checkopts "$@"

  # build start
  local start_time=$(date +%s)
  echo "---------------- build start $(date '+%Y-%m-%d %H:%M:%S') ----------------"

  build_project
  if [[ "$?" -ne 0 ]]; then
    echo "build failed.";
    exit 1;
  fi

  echo "---------------- build end  $(date '+%Y-%m-%d %H:%M:%S') ----------------"
  local end_time=$(date +%s)

  # 计算耗时（秒）
  local duration=$((end_time - start_time))

  # 格式化输出耗时（时:分:秒）
  local hours=$((duration / 3600))
  local minutes=$(( (duration % 3600) / 60 ))
  local seconds=$((duration % 60))
  echo "---------------- Total duration: ${hours} hour ${minutes} min ${seconds} sec ----------------"

}

main "$@"
