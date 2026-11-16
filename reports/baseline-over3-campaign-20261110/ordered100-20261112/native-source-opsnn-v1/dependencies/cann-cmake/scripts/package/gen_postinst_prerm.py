#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import json
import argparse
import os
import platform
from typing import List

from utils.comm_log import CommLog

# 特殊模块列表，需要维护数据库
SPECIAL_MODULES = {"EngineeringCommon", "DevlibCommon", "SetEnvCommon"}


def read_append_file(filepath):
    """读取追加脚本文件内容，如果文件不存在或为空则返回空字符串"""
    if not filepath:
        return ""
    try:
        with open(filepath, 'r') as f:
            content = f.read().strip()
            if content:
                # 确保追加内容以换行开头
                return "\n" + content + "\n"
            return ""
    except FileNotFoundError:
        return ""
    except Exception as e:
        return ""


# 这里需要一个版本号，然后一个组件名
def generate_postinst(package_name, version, modules_data, source_dir, permission_lines=None):
    """
    功能描述：生成 postinst 脚本内容
    permission_lines 为可选的赋权（chmod）bash 代码行，由 gen_set_permission.generate_set_permission
    生成，内联在软链接创建之后，使安装后文件权限符合 XML install_mod 配置。
    """
    lines = [
        "#!/bin/bash",
        "set -e",
        "",
        f"INSTALL_PATH=\"/usr/local/Ascend/cann-{version}\"",
        f"PKG_ARCH_NAME=\"{platform.machine()}\"",
        f"DB_FILE=\"${{INSTALL_PATH}}/var/ascend_package_db.info\"",
        "mkdir -p $(dirname \"$DB_FILE\")",
        "touch \"$DB_FILE\"",
        "",
        f"PACKAGE_NAME=\"{package_name}\"",
        "",
        "echo \"Running post-install script...\"",
        ""
    ]

    for module, symlinks in modules_data.items():
        lines.append(f"# ========== Module: {module} ==========")
        # 先处理软链接创建 (无论是否特殊模块，都要创建)
        for sym in symlinks:
            target = sym['target']
            link = sym['link']
            link_dir = f"$(dirname \"{link}\")"
            lines.append(f"mkdir -p \"$INSTALL_PATH\"/{link_dir}")
            # 如果是EngineeringCommon模块，创建实体目录
            lines.append(f"if [ \"{module}\" = \"EngineeringCommon\" -o \"{module}\" = \"DevlibCommon\" ]; then")
            lines.append(f"    mkdir -p \"$INSTALL_PATH\"/{target}")
            lines.append("fi")
            lines.append(f"ln -sfnr \"$INSTALL_PATH\"/\"{target}\" \"$INSTALL_PATH\"/\"{link}\"")
            lines.append("")
        lines.pop()  # 移除最后一个空行，保持整洁
        lines.append("")

        # 对于特殊模块，处理数据库记录
        if module in SPECIAL_MODULES:
            lines.append(f"# Update database for module {module}")
            lines.append(f"if ! grep -q '^{module}|' \"$DB_FILE\"; then")
            lines.append(f"    echo \"{module}|\" >> \"$DB_FILE\"")
            lines.append("fi")
            lines.append(f"line=$(grep '^{module}|' \"$DB_FILE\")")
            lines.append("components=$(echo \"$line\" | cut -d'|' -f2)")
            lines.append(f"if ! echo \",$components,\" | grep -q \",$PACKAGE_NAME,\"; then")
            lines.append("    if [ -z \"$components\" ]; then")
            lines.append(f"        new_components=\"$PACKAGE_NAME\"")
            lines.append("    else")
            lines.append(f"        new_components=\"$components,$PACKAGE_NAME\"")
            lines.append("    fi")
            lines.append(f"    sed -i \"s/^{module}|.*$/{module}|$new_components/\" \"$DB_FILE\"")
            lines.append(f"    echo \"Added $PACKAGE_NAME to module {module}\"")
            lines.append("else")
            lines.append(f"    echo \"$PACKAGE_NAME already registered for module {module}\"")
            lines.append("fi")
            lines.append("")
        else:
            lines.append(f"# Module {module} does not require database tracking")
            lines.append("")

    if permission_lines:
        lines.append("")
        lines.extend(permission_lines)
        lines.append("")

    lines.append("echo \"Post-install completed.\"")
    append_content = read_append_file(os.path.join(source_dir, 'scripts', 'package', package_name, 'rpm_deb') + "/custom_postinst.sh")
    if append_content:
        lines.append(append_content)
    lines.append("exit 0")
    return "\n".join(lines)


def generate_prerm(package_name, version, modules_data, source_dir):
    """生成 prerm 脚本内容"""
    lines = [
        "#!/bin/bash",
        "set -e",
        "",
        f"INSTALL_PATH=\"/usr/local/Ascend/cann-{version}\"",
        f"PKG_ARCH_NAME=\"{platform.machine()}\"",
        f"DB_FILE=\"${{INSTALL_PATH}}/var/ascend_package_db.info\"",
        f"PACKAGE_NAME=\"{package_name}\"",
        "",
        "echo \"Running pre-uninstall script...\"",
        "# Function to remove empty parent directories up to a given root",
        "cleanup_parent_dirs() {",
        "    local link_path=\"$1\"",
        "    local stop_root=\"$2\"",
        "    local current_dir=$(dirname \"$link_path\")",
        "    while [ \"$current_dir\" != \"$stop_root\" ] && [ \"$current_dir\" != \"/\" ]; do",
        "        if [ -d \"$current_dir\" ]; then",
        "            rmdir \"$current_dir\" 2>/dev/null || break",
        "        else",
        "            break",
        "        fi",
        "        current_dir=$(dirname \"$current_dir\")",
        "    done",
        "}",
        "# Function to clean up database file and its parent directory if empty",
        "cleanup_db_and_var() {",
        "    if [ -f \"$DB_FILE\" ]; then",
        "        # Check if the file is empty (size 0) or contains only whitespace",
        "        if [ ! -s \"$DB_FILE\" ] || [ -z \"$(cat \"$DB_FILE\" | tr -d '[:space:]')\" ]; then",
        "            echo \"Database file is empty, removing it...\"",
        "            rm -f \"$DB_FILE\"",
        "            local db_dir=$(dirname \"$DB_FILE\")",
        "            # Try to remove the parent directory (var) if it is empty",
        "            if [ -d \"$db_dir\" ]; then",
        "                rmdir \"$db_dir\" 2>/dev/null && echo \"Removed empty directory: $db_dir\" || true",
        "            fi",
        "        fi",
        "    fi",
        "}",
        ""
    ]

    append_content = read_append_file(os.path.join(source_dir, 'scripts', 'package', package_name, 'rpm_deb') + "/custom_prerm.sh")
    if append_content:
        lines.append(append_content)
    
    for module, symlinks in modules_data.items():
        lines.append(f"# ========== Module: {module} ==========")
        if module in SPECIAL_MODULES:
            # 特殊模块：处理数据库，根据计数决定是否删除软链接
            lines.append(f"if [ -f \"$DB_FILE\" ] && grep -q '^{module}|' \"$DB_FILE\"; then")
            lines.append(f"   line=$(grep '^{module}|' \"$DB_FILE\")")
            lines.append("    components=$(echo \"$line\" | cut -d'|' -f2)")
            lines.append("    # Remove this package from the list")
            lines.append("    new_components=$(echo \"$components\" | tr ',' '\\n' | grep -v \"^$PACKAGE_NAME$\" | paste -sd ',' -)")
            lines.append("    if [ -z \"$new_components\" ]; then")
            # 没有其他组件：删除软链接并删除该模块的行
            for sym in symlinks:
                link = sym['link']
                target = sym['target']
                # 如果是EngineeringCommon模块，删除实体目录
                lines.append(f"        if [ \"{module}\" = \"EngineeringCommon\" -o \"{module}\" = \"DevlibCommon\" ]; then")
                lines.append(f"            if [ -d \"$INSTALL_PATH\"/\"{target}\" ]; then")
                lines.append(f"                rm -rf \"$INSTALL_PATH\"/\"{target}\"")
                lines.append(f"            fi")
                lines.append(f"        fi")
                lines.append(f"        if [ -L \"$INSTALL_PATH\"/\"{link}\" ]; then")
                lines.append(f"            rm -f \"$INSTALL_PATH\"/\"{link}\"")
                lines.append(f"            echo \"Removed symlink: {link}\"")
                lines.append(f"            cleanup_parent_dirs \"$INSTALL_PATH\"/\"{link}\" \"$INSTALL_PATH\"")
                lines.append(f"        fi")
            lines.append(f"        sed -i '/^{module}|/d' \"$DB_FILE\"")
            lines.append(f"        echo \"Removed module {module} from database\"")
            lines.append("         cleanup_db_and_var")
            lines.append("    else")
            lines.append(f'        sed -i "s/^{module}|.*$/{module}|$new_components/" "$DB_FILE"')
            lines.append(f"        echo \"Removed $PACKAGE_NAME from module {module}, remaining: $new_components\"")
            lines.append("    fi")
            lines.append("else")
            lines.append(f"    echo \"Module {module} not found in database, skipping\"")
            lines.append("fi")
            lines.append("")
        else:
            # 非特殊模块：直接删除软链接，不检查数据库
            lines.append("# Non-tracked module: directly remove symlinks")
            for sym in symlinks:
                link = sym['link']
                lines.append(f"if [ -L \"$INSTALL_PATH\"/\"{link}\" ]; then")
                lines.append(f"    rm -f \"$INSTALL_PATH\"/\"{link}\"")
                lines.append(f"    echo \"Removed symlink: {link}\"")
                lines.append(f"    cleanup_parent_dirs \"$INSTALL_PATH\"/\"{link}\" \"$INSTALL_PATH\"")
                lines.append(f"fi")
            lines.append("")

    lines.append("echo \"Pre-uninstall completed.\"")
    lines.append("exit 0")
    return "\n".join(lines)


def _resolve_target(path: str) -> str:
    """将安装路径解析为 postinst 中可用的目标串"""
    if path.startswith('/'):
        return f'"{path}"'
    return f'"$INSTALL_PATH/{path}"'


def generate_set_permission(items, version, script_dir=None) -> List[str]:
    """根据 file_install_list 生成安装后赋权（chmod）的 bash 代码行，供内联进 postinst"""
    lines = ["# ========== Set file permissions =========="]
    seen = set()
    for item in items:
        mod = getattr(item, 'permission', None)
        if not mod or mod == 'NA':
            continue
        path = getattr(item, 'relative_install_path', None)
        if not path or path == 'NA':
            continue
        key = (path, mod)
        if key in seen:
            continue
        seen.add(key)
        target = _resolve_target(path)
        lines.append(f'if [ -e {target} ]; then')
        # root 环境需要提权：550→555, 440→444, 750→755
        lines.append(f'    if [ "${{EUID:-0}}" -eq 0 ]; then')
        lines.append(f'        case "{mod}" in')
        lines.append(f'            550) chmod 555 {target} ;;')
        lines.append(f'            440) chmod 444 {target} ;;')
        lines.append(f'            750) chmod 755 {target} ;;')
        lines.append(f'            *) chmod {mod} {target} ;;')
        lines.append(f'        esac')
        lines.append(f'    else')
        lines.append(f'        chmod {mod} {target}')
        lines.append(f'    fi')
        lines.append('fi')

    # share/info/<name>/script 下的文件未在 xml 逐条配置，统一刷 550（非 root）/ 555（root）
    if script_dir:
        target = _resolve_target(script_dir)
        lines.append(f'if [ -d {target} ]; then')
        lines.append(f'    if [ "${{EUID:-0}}" -eq 0 ]; then')
        lines.append(f'        find {target} -type f -exec chmod 555 {{}} \\; 2>/dev/null || true')
        lines.append(f'    else')
        lines.append(f'        find {target} -type f -exec chmod 550 {{}} \\; 2>/dev/null || true')
        lines.append(f'    fi')
        lines.append('fi')
    return lines


def main():
    parser = argparse.ArgumentParser(description="Generate postinst and prerm from modules.txt")
    parser.add_argument("--package-name", required=True, help="Name of the package (e.g., cann-npu-runtime)")
    parser.add_argument("--modules-file", required=True, help="JSON file containing module symlinks (modules.txt)")
    parser.add_argument("--output-postinst", required=True, help="Output path for postinst script")
    parser.add_argument("--output-prerm", required=True, help="Output path for prerm script")
    args = parser.parse_args()

    with open(args.modules_file, 'r') as f:
        modules_data = json.load(f)   # 格式: {"ModuleName": [{"target": "...", "link": "..."}, ...]}

    postinst_content = generate_postinst(args.package_name, "9.0.0", modules_data)
    prerm_content = generate_prerm(args.package_name, "9.0.0", modules_data)

    with open(args.output_postinst, 'w') as f:
        f.write(postinst_content)
    os.chmod(args.output_postinst, 0o755)

    with open(args.output_prerm, 'w') as f:
        f.write(prerm_content)
    os.chmod(args.output_prerm, 0o755)

    CommLog.cilog_info("Generated: %s, %s", args.output_postinst, args.output_prerm)


if __name__ == "__main__":
    main()