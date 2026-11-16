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
# Script Name: fix_deb_control.sh
# Description: 修复 .deb 包中 control 文件末尾多余空行导致的打包错误
# Usage:       ./fix_deb_control.sh <your_package.deb>
# Example:     ./fix_deb_control.sh cann-npu-runtime_9.2.0_linux-x86_64.deb

set -e

START_TIME=$(date +%s)

DEB_FILE="$1"

if [ -z "$DEB_FILE" ] || [ ! -f "$DEB_FILE" ]; then
    echo "Error: Please provide a valid .deb file."
    echo "Usage: $0 <your_package.deb>"
    exit 1
fi

# 创建临时目录，失败时立即退出
WORK_DIR=$(mktemp -d) || { echo "Error: Failed to create temporary directory"; exit 1; }
trap 'rm -rf "$WORK_DIR"' EXIT
echo "Processing in working directory: $WORK_DIR"

# 保存原文件的绝对路径
ORIGINAL_DEB=$(realpath "$DEB_FILE")
ORIGINAL_DIR=$(dirname "$ORIGINAL_DEB")
ORIGINAL_FILENAME=$(basename "$ORIGINAL_DEB")

cp "$DEB_FILE" "$WORK_DIR/"
cd "$WORK_DIR"

ar x "$ORIGINAL_FILENAME"

# 解压 control
if [ -f "control.tar.xz" ]; then
    tar -xJf control.tar.xz
elif [ -f "control.tar.gz" ]; then
    tar -xzf control.tar.gz
else
    echo "Error: control.tar.xz or control.tar.gz not found"
    exit 1
fi

# 修复：删除 control 文件末尾的所有空行（保留中间空行）
sed -i -e :a -e '/^\n*$/{$d;N;ba' -e '}' control

echo "Cleaned trailing blank lines in control file."

# 重新打包 control
rm -f control.tar.xz control.tar.gz
tar -cJf control.tar.xz ./*

# 自动识别 data 文件
DATA_FILE=$(ls data.tar.* 2>/dev/null | head -1)
if [ -z "$DATA_FILE" ]; then
    echo "Error: data.tar.* file not found"
    exit 1
fi

# 重新打包成临时文件
TEMP_DEB="${ORIGINAL_FILENAME}.tmp"
ar rcs "$TEMP_DEB" debian-binary control.tar.xz "$DATA_FILE"

# 验证临时包是否完整可读
if ! ar t "$TEMP_DEB" >/dev/null 2>&1; then
    echo "Error: Generated package is corrupted: $TEMP_DEB"
    exit 1
fi

# 用临时文件替换原文件
mv "$TEMP_DEB" "$ORIGINAL_DIR/$ORIGINAL_FILENAME"

echo "Fix completed! Original package replaced: $ORIGINAL_DEB"

ELAPSED=$(($(date +%s) - START_TIME))
echo "Time: ${ELAPSED}s"
