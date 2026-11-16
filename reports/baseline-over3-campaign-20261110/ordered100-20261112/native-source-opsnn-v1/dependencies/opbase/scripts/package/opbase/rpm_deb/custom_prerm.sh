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

# remove opbase stub lib symlinks in devlib/
arch="${PKG_ARCH_NAME}"
devlib_dir="${INSTALL_PATH}/${arch}-linux/devlib"

STUB_LIBS="libnnopbase.so"

for lib in ${STUB_LIBS}; do
    if [ -L "${devlib_dir}/${lib}" ]; then
        _writable=0
        [ -w "${devlib_dir}" ] && _writable=1
        [ $_writable -eq 0 ] && chmod u+w "${devlib_dir}" 2>/dev/null || true
        rm -f "${devlib_dir}/${lib}" 2>/dev/null || true
        [ $_writable -eq 0 ] && chmod u-w "${devlib_dir}" 2>/dev/null || true
    fi
done

# remove empty opp/vendors directory
opp_vendors_dir="${INSTALL_PATH}/opp/vendors"
if [ -d "${opp_vendors_dir}" ] && [ -z "$(ls -A "${opp_vendors_dir}")" ]; then
    rmdir "${opp_vendors_dir}" 2>/dev/null || true
fi
