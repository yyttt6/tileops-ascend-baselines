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
    -l | --linker)
        linker=$2
        shift 2
        ;;
    -o | --output)
        output=$2
        shift 2
        ;;
    --aic-dir)
        aic_dir=$2
        shift 2
        ;;
    --aiv-dir)
        aiv_dir=$2
        shift 2
        ;;
    --build-type)
        build_type=$2
        shift 2
        ;;
    *)
        break
        ;;
    esac
done

current_dir=$(dirname $(readlink -f ${BASH_SOURCE[0]}))
mix_build_flag=mix_build.flag
aic_build_flag=aic_build.flag
aiv_build_flag=aiv_build.flag

if [ ! -d "${output}" ]; then
    mkdir -p ${output}
fi

if [ -n "${mix_build_flag}" ]; then
    rm -rf ${output}/${mix_build_flag}
fi

if [ -n "${aic_build_flag}" ]; then
    rm -rf ${output}/${aic_build_flag}
fi

if [ -n "${aiv_build_flag}" ]; then
    rm -rf ${output}/${aiv_build_flag}
fi

# mix mode
if [ -f "${aic_dir}/${mix_build_flag}" ] && [ -f "${aiv_dir}/${mix_build_flag}" ] ; then
    bash ${current_dir}/merge_obj.sh -l ${linker} -o ${output} -t ${build_type} -n device.o -m ${aic_dir}/device.o ${aiv_dir}/device.o

    touch ${output}/${mix_build_flag}
fi

# aic mode
if [ -f "${aic_dir}/${aic_build_flag}" ]; then
    bash ${current_dir}/merge_obj.sh -l ${linker} -o ${output} -t ${build_type} -n device_aic.o -m ${aic_dir}/device_aic.o

    touch ${output}/${aic_build_flag}
fi

# aiv mode
if [ -f "${aiv_dir}/${aiv_build_flag}" ]; then
    bash ${current_dir}/merge_obj.sh -l ${linker} -o ${output} -t ${build_type} -n device_aiv.o -m ${aiv_dir}/device_aiv.o

    touch ${output}/${aiv_build_flag}
fi
