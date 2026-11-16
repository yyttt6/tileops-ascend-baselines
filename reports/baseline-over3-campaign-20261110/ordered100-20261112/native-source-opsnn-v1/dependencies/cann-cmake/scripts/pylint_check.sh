#!/bin/sh
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# pylint pre-commit 包装脚本
# 当 pylint 未安装时（如 CI 门禁环境）自动跳过检查，避免阻断提交；
# pylint 已安装时正常执行检查。与 oat_check.sh 的缺失跳过策略一致。

if ! command -v pylint >/dev/null 2>&1; then
    echo "[pylint] pylint not installed, skipping check."
    echo "[pylint] To enable: pip install pylint"
    exit 0
fi

exec pylint "$@"
