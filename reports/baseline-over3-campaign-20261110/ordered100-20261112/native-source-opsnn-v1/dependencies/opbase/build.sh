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

BASEPATH=$(cd "$(dirname $0)"; pwd)
BUILD_RELATIVE_PATH="build"
BUILD_OUT="build_out"
BUILD_PATH="${BASEPATH}/${BUILD_RELATIVE_PATH}/"
CORE_NUMS=$(cat /proc/cpuinfo| grep "processor"| wc -l)

# print usage message
dotted_line="----------------------------------------------------------------"
usage() {
    echo "Usage:"
    echo ""
    echo "    -h, --help     Print usage"
    echo "    -v, --verbose  Display build command"
    echo "Default Build Pkg Options:"
    echo $dotted_line
    echo "    -j<N>          Set the number of threads used for building ops_base, default is 8"
    echo "    -O<N>          Compile optimization options, support [O0 O1 O2 O3], default is O2"
    echo "    --make_clean "
    echo "                   Make clean and delete related file"
    echo "    --pkg          Build run package"
    echo "    --pkg-type=<TYPE>"
    echo "                   Specify package type (TYPE options: run/rpm/deb/all), Default: run"
    echo "    --build-type=<TYPE>"
    echo "                   Specify build type (TYPE options: Release/Debug), Default: Release"
    echo "    --cann_3rd_lib_path=<PATH>"
    echo "                   Set ascend third_party package install path, default ./third_party"
    echo "Test Options:"
    echo $dotted_line
    echo "    -u             Build and run all unit tests"
    echo "    --noexec       Only compile ut, do not execute"
    echo "    --cov          Enable code coverage for unit tests"
    echo "    --asan         Enable AddressSanitizer"
    echo ""
}

check_pkg_type() {
    local pkg_type="$1"
    if [[ "$pkg_type" != "run" && "$pkg_type" != "rpm" && "$pkg_type" != "deb" && "$pkg_type" != "all" ]]; then
        echo "[ERROR] --pkg-type only supports run/rpm/deb/all, got: $pkg_type"
        exit 1
    fi
}

parse_cmake_extra_args() {
    local args_str="$1"
    local arg permitted_args_str
    if [[ -z "$args_str" ]]; then
        echo "The parsed extra string is empty."
        return 0
    fi

    permitted_args_str="$(echo "$args_str" | tr "," "\n" | grep -v "^$" | grep -E "^ENABLE_BUILD_DEVICE=")"

    while read -r arg; do
        name="$(echo $arg | cut -d= -f1)"
        value="$(echo $arg | cut -d= -f2-)"
        echo "Set $name to $value."
    done <<< "$permitted_args_str"

    while read -r arg; do
        CMAKE_EXTRA_ARGS=(
            "${CMAKE_EXTRA_ARGS[@]}"
            "-D" "$arg"
        )
    done <<< "$permitted_args_str"
}

# parse and set options
checkopts() {
    VERBOSE=""
    THREAD_NUM=8
    ENABLE_UT="off"
    ENABLE_ST="off"
    MAKE_CLEAN_ALL="off"
    EXEC_TEST="off"
    BUILD_TYPE="Release"
    BUILD_MODE=""
    ENABLE_COVERAGE="off"
    ENABLE_PKG_ASAN="off"
    ENABLE_PKG="off"
    PACKAGE_TYPE="run"
    PACKAGE_TYPE_SET=FALSE
    CMAKE_EXTRA_ARGS=()
    if [[ -n "${ASCEND_HOME_PATH}" ]]; then
        echo "env exists ASCEND_HOME_PATH : ${ASCEND_HOME_PATH}"
    elif [ $UID -eq 0 ]; then
        export ASCEND_HOME_PATH=/usr/local/Ascend/cann
    else
        export ASCEND_HOME_PATH=~/Ascend/cann
    fi
    CANN_3RD_LIB_PATH="${BASEPATH}/third_party"

    # Process the options
    parsed_args=$(getopt -a -o j:hvusO: -l help,verbose,cov,make_clean,build-type:,noexec,pkg,asan,cann_3rd_lib_path:,extra-cmake-args:,pkg-type: -- "$@") || {
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
        VERBOSE="VERBOSE=1"
        shift
        ;;
        -u)
        ENABLE_UT="on"
        ENABLE_ST="off"
        EXEC_TEST="on"
        shift
        ;;
        -O)
        BUILD_MODE="-O$2"
        shift 2
        ;;
        --cov)
        ENABLE_COVERAGE="on"
        shift
        ;;
        --make_clean)
        MAKE_CLEAN_ALL="on"
        shift
        ;;
        --build-type)
        BUILD_TYPE=$2
        shift 2
        ;;
        --noexec)
        EXEC_TEST="off"
        shift
        ;;
        --cann_3rd_lib_path)
        CANN_3RD_LIB_PATH="$(realpath $2)"
        shift 2
        ;;
        --extra-cmake-args)
        parse_cmake_extra_args "$2"
        shift 2
        ;;
        --pkg)
        ENABLE_PKG="on"
        shift
        ;;
        --pkg-type)
        PACKAGE_TYPE="$2"
        check_pkg_type "${PACKAGE_TYPE}"
        PACKAGE_TYPE_SET=TRUE
        shift 2
        ;;
        --asan)
        ENABLE_PKG_ASAN="on"
        shift
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

    if [[ "$PACKAGE_TYPE_SET" == "TRUE" && "$ENABLE_PKG" != "on" ]]; then
        echo "[ERROR] --pkg-type can only be used with --pkg"
        exit 1
    fi
}

mk_dir() {
    local create_dir="$1"
    mkdir -pv "${create_dir}"
    echo "created ${create_dir}"
}

find_rpm_deb_package() {
    if [[ "$PACKAGE_TYPE" == "run" ]]; then
        return 0
    fi
    find "${BUILD_PATH}" -type f -name "cann-opbase*.${PACKAGE_TYPE}" | sort
}

clean_rpm_deb_package() {
    if [[ "$PACKAGE_TYPE" == "run" ]]; then
        return 0
    fi
    local package_files=()
    while IFS= read -r package_file; do
        package_files+=("${package_file}")
    done < <(find_rpm_deb_package)
    if [[ ${#package_files[@]} -eq 0 ]]; then
        return 0
    fi
    for package_file in "${package_files[@]}"; do
        rm -f "${package_file}"
        echo "[INFO] Removed stale package artifact: ${package_file}"
    done
}

collect_rpm_deb_package() {
    if [[ "$PACKAGE_TYPE" == "run" ]]; then
        return 0
    fi
    local package_files=()
    while IFS= read -r package_file; do
        package_files+=("${package_file}")
    done < <(find_rpm_deb_package)
    for package_file in "${package_files[@]}"; do
        cp -f "${package_file}" "${BUILD_OUT_PATH}/"
        echo "[INFO] Package artifact copied to ${BUILD_OUT_PATH}/$(basename "${package_file}")"
    done
}

# ops_base build start
cmake_generate_make() {
    local source_path="$1"
    local build_path="$2"
    local cmake_args="$3"
    if [[ "${MAKE_CLEAN_ALL}" == "on" ]];then
        echo "clear all files in build directory"
        [ -d "${build_path}" ] && rm -rf "${build_path}"
        [ -d "${BASEPATH}/third_party" ] && rm -rf "${BASEPATH}/third_party"
    fi
    mk_dir "${build_path}"
    cd "${build_path}"
    if [[ "${MAKE_CLEAN_ALL}" == "on" ]]; then
        [ -f CMakeCache.txt ] && rm CMakeCache.txt
        [ -f Makefile ] && rm Makefile
        [ -f cmake_install.cmake ] && rm cmake_install.cmake
        [ -d CMakeFiles ] && rm -rf CMakeFiles
    fi
    local cache_param_file=".cmake_build_params"
    local need_reconfigure=0
    if [ -f CMakeCache.txt ] && [ -f Makefile ]; then
        if [ -f "${cache_param_file}" ] && [ "$(cat "${cache_param_file}")" = "${cmake_args}" ]; then
            echo "CMake cache exists and parameters unchanged, skipping reconfiguration for incremental build"
        else
            echo "CMake parameters changed, forcing reconfiguration"
            rm -f CMakeCache.txt Makefile cmake_install.cmake
            rm -rf CMakeFiles
            need_reconfigure=1
        fi
    else
        need_reconfigure=1
    fi
    if [[ "${need_reconfigure}" -eq 1 ]]; then
        echo "${cmake_args} ${CMAKE_EXTRA_ARGS[@]}"
        cmake ${cmake_args} "${CMAKE_EXTRA_ARGS[@]}" ${source_path}
        if [ 0 -ne $? ]; then
            echo "execute command: cmake ${cmake_args} ${CMAKE_EXTRA_ARGS[@]} .. failed."
            exit 1
        fi
        echo "${cmake_args} ${CMAKE_EXTRA_ARGS[@]}" > "${cache_param_file}"
    fi
}

# create build path
build_ops_base() {
    echo "create build directory and build ops_base"
    cd "${BASEPATH}"

    BUILD_OUT_PATH="${BASEPATH}/${BUILD_OUT}/"
    if [ ! -d "${BUILD_OUT_PATH}" ]; then
        mkdir -p "${BUILD_OUT_PATH}"
    fi

    local cmake_pkg_type="${PACKAGE_TYPE}"
    [[ "${PACKAGE_TYPE}" == "all" ]] && cmake_pkg_type="run"

    CMAKE_ARGS="\
    -DENABLE_UT=${ENABLE_UT} \
    -DENABLE_ST=${ENABLE_ST} \
    -DBUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG=ON \
    -DCANN_3RD_LIB_PATH=${CANN_3RD_LIB_PATH} \
    -DDCMAKE_WGET_FLAGS='--no-check-certificate' \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DBUILD_MODE=${BUILD_MODE} \
    -DENABLE_PKG_ASAN=${ENABLE_PKG_ASAN} \
    -DENABLE_COVERAGE=${ENABLE_COVERAGE} \
    -DPACKAGE_TYPE=${cmake_pkg_type} \
    -DCMAKE_INSTALL_PREFIX=${BUILD_OUT_PATH}"

    cmake_generate_make "${BASEPATH}" "${BUILD_PATH}" "${CMAKE_ARGS}"

    if [[ "${PACKAGE_TYPE}" == "all" ]]; then
        local saved_pkg_type="${PACKAGE_TYPE}"
        for PACKAGE_TYPE in run rpm deb; do
            clean_rpm_deb_package
            cmake -DPACKAGE_TYPE="${PACKAGE_TYPE}" "${BUILD_PATH}" > /dev/null 2>&1
            make ${VERBOSE} -j${THREAD_NUM} package
            if [ $? -ne 0 ]; then
                echo "[ERROR] target:package (${PACKAGE_TYPE}) build failed!"
                exit 1
            fi
            collect_rpm_deb_package
        done
        PACKAGE_TYPE="${saved_pkg_type}"
    else
        clean_rpm_deb_package
        make ${VERBOSE} -j${THREAD_NUM} package
        if [ 0 -ne $? ]; then
            echo "execute command: make ${VERBOSE} -j${THREAD_NUM} package failed."
            return 1
        fi
        collect_rpm_deb_package
    fi

    # make package
    if [[ "${PACKAGE_TYPE}" == "run" || "${PACKAGE_TYPE}" == "all" ]]; then
        if [ ! -f ${BUILD_OUT_PATH}/cann*.run ];then
            echo "package ops_base run failed"
            return 1
        fi
    fi

    echo "ops_base build success!"
}

build_ops_base_llt() {
    echo "create build directory and build ops_base_llt"
    cd "${BASEPATH}"

    CMAKE_ARGS="\
    -DENABLE_UT=${ENABLE_UT} \
    -DENABLE_ST=${ENABLE_ST} \
    -DBUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG=ON \
    -DCANN_3RD_LIB_PATH=${CANN_3RD_LIB_PATH} \
    -DDCMAKE_WGET_FLAGS='--no-check-certificate' \
    -DENABLE_DEBUG=${ENABLE_DEBUG} \
    -DBUILD_MODE=${BUILD_MODE} \
    -DENABLE_COVERAGE=${ENABLE_COVERAGE}"

    cmake_generate_make "${BASEPATH}" "${BUILD_PATH}" "${CMAKE_ARGS}"

    make ${VERBOSE} -j${THREAD_NUM}
}

# generate coverage for ut
generate_llt_cov() {
    echo "start to generate ut coverage for ops-base"
    COVERAGE_SCRIPT="${BASEPATH}/scripts/util/generate_cpp_cov.sh"
    COVERAGE_REPORT_PATH="${BUILD_PATH}/cov"
    UT_COVERAGE_HTML_PATH="${COVERAGE_REPORT_PATH}/"
    UT_COVERAGE_DATA_FILE="${COVERAGE_REPORT_PATH}/coverage.info"
    mk_dir ${COVERAGE_REPORT_PATH}
    mk_dir ${UT_COVERAGE_HTML_PATH}
    source ${COVERAGE_SCRIPT} ${BUILD_PATH} ${UT_COVERAGE_DATA_FILE} ${UT_COVERAGE_HTML_PATH}
    echo "finish to generate llt coverage for ops-base"
}

main() {
    cd "${BASEPATH}"
    checkopts "$@"
    if [ "$THREAD_NUM" -gt "$CORE_NUMS" ];then
        echo "compile thread num:$THREAD_NUM over core num:$CORE_NUMS, adjust to core num"
        THREAD_NUM=$CORE_NUMS
    fi

    g++ -v

    if [[ "${ENABLE_UT}" == "on" || "${ENABLE_ST}" == "on" ]];then
        echo "---------------- ops_base_llt build start ----------------"
        build_ops_base_llt || { echo "ops_base_llt build failed."; exit 1; }
        echo "---------------- ops_base_llt build finished ----------------"
    else
        echo "---------------- ops_base build start ----------------"
        build_ops_base || { echo "ops_base build failed."; exit 1; }
        echo "---------------- ops_base build finished ----------------"
    fi
    
    if [[ "${ENABLE_UT}" == "on" && "${EXEC_TEST}" == "on" ]]; then
        if [ -f "${BASEPATH}"/"${BUILD_RELATIVE_PATH}"/tests/nnopbase/ut/nnopbase_utest ]; then
            source "${ASCEND_HOME_PATH}/bin/setenv.bash"
            export LD_LIBRARY_PATH="${BASEPATH}/${BUILD_RELATIVE_PATH}"/:$LD_LIBRARY_PATH
            cd "${BASEPATH}"/"${BUILD_RELATIVE_PATH}"/tests/nnopbase/ut/
            ./nnopbase_utest
            if [[ $? -ne 0 ]]; then
                echo "Execute nnopbase_utest failed."
                exit 1
            fi
            echo "Execute nnopbase_utest successful."
        else
            echo "nnopbase_utest does not generated"
            exit 1
        fi

        if [ -f "${BASEPATH}"/"${BUILD_RELATIVE_PATH}"/tests/op_common/op_common_utest ];then
            source "${ASCEND_HOME_PATH}/bin/setenv.bash"
            export LD_LIBRARY_PATH="${BASEPATH}/${BUILD_RELATIVE_PATH}"/:$LD_LIBRARY_PATH
            cd "${BASEPATH}"/"${BUILD_RELATIVE_PATH}"/tests/op_common/
            ./op_common_utest | tee op_common_utest.log
            if grep -q "\[  FAILED  \]" op_common_utest.log; then
                echo "Execute op_common_utest failed."
                exit 1
            fi
            echo "Execute op_common_utest successful."
        else
            echo "op_common_utest does not generated"
            exit 1
        fi

        echo "Execute ops_base_ut successful."
        
        if [[ "${ENABLE_COVERAGE}" == "on" ]];then
            generate_llt_cov
        fi
    fi

    if [[ "${ENABLE_ST}" == "on" && "${EXEC_TEST}" == "on" ]];then
        if [ -f "${BASEPATH}"/"${BUILD_RELATIVE_PATH}"/tests/nnopbase/st/nnopbase_stest ];then
            source "${ASCEND_HOME_PATH}/bin/setenv.bash"
            export LD_LIBRARY_PATH="${BASEPATH}/${BUILD_RELATIVE_PATH}"/:$LD_LIBRARY_PATH
            cd "${BASEPATH}"/"${BUILD_RELATIVE_PATH}"/tests/nnopbase/st/
            ./nnopbase_stest | tee nnopbase_stest.log
            if grep -q "\[  FAILED  \]" nnopbase_stest.log; then
                echo "Execute nnopbase_stest failed."
                exit 1
            fi
            echo "Execute nnopbase_stest successful."
        else
            echo "nnopbase_stest does not generated"
            exit 1
        fi

        echo "Execute ops_base_st successful."
        
        if [[ "${ENABLE_COVERAGE}" == "on" ]];then
            generate_llt_cov
        fi
    fi
}

main "$@"
