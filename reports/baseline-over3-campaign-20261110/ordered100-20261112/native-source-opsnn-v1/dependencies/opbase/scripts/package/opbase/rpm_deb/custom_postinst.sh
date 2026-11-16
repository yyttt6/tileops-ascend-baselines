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

# create opbase stub lib symlinks in devlib/
arch="${PKG_ARCH_NAME}"
devlib_dir="${INSTALL_PATH}/${arch}-linux/devlib"
stub_src_rel="linux/${arch}"

STUB_LIBS="libnnopbase.so"

if [ -d "${devlib_dir}/${stub_src_rel}" ] && [ "$(id -u)" -eq 0 ]; then
    _writable=0
    [ -w "${devlib_dir}" ] && _writable=1
    [ $_writable -eq 0 ] && chmod u+w "${devlib_dir}" 2>/dev/null || true
    for lib in ${STUB_LIBS}; do
        if [ -f "${devlib_dir}/${stub_src_rel}/${lib}" ]; then
            ln -sf "./${stub_src_rel}/${lib}" "${devlib_dir}/${lib}"
        fi
    done
    [ $_writable -eq 0 ] && chmod u-w "${devlib_dir}" 2>/dev/null || true
fi

# create empty opp/vendors directory
opp_vendors_dir="${INSTALL_PATH}/opp/vendors"
if [ ! -d "${opp_vendors_dir}" ]; then
    mkdir -p "${opp_vendors_dir}"
    chmod 555 "${opp_vendors_dir}" 2>/dev/null || true
fi
