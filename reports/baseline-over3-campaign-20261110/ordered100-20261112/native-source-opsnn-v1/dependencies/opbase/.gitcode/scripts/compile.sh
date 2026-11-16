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
echo $(grep -E "^VERSION_ID=" /etc/os-release | cut -d'"' -f2)
if [[ "${task_name}" =~ x86_compile_ubuntu24 ]] && [ "${TARGET_BRANCH}" == master ]; then
    echo "api-check=compile" >> "${ATOMGIT_OUTPUT}"
else
    echo "api-check=continue" >> "${ATOMGIT_OUTPUT}"
fi
if [[ "${task_name}" =~ Compile_Ascend_X86_ubuntu24 ]]; then
  sudo update-alternatives --set gcc /usr/bin/gcc-14
  sed -i "1i set(CMAKE_EXPORT_COMPILE_COMMANDS ON)" "CMakeLists.txt"
fi
rm -rf /home/jenkins/opensource/json
source /home/jenkins/Ascend/cann/bin/setenv.bash
bash build.sh --cann_3rd_lib_path=/home/jenkins/opensource -j16
echo "exec cmd: [bash build.sh --cann_3rd_lib_path=/home/jenkins/opensource -j16]"
compile_package_name=$(ls "${WORKSPACE}/build_out/" |grep -E "*.run$"|head -n1)
echo "compile package name is: ${compile_package_name}"
chmod +x ./build_out/${compile_package_name}
echo "y" | ./build_out/${compile_package_name} --full --install-path=${WORKSPACE}/tmp 2>&1 | tee ${WORKSPACE}/compile_log.txt
