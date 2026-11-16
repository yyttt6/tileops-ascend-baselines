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

import os
import json
import pytest

# 导入待测脚本模块
import gen_postinst_prerm


def _setup_main_env(tmp_path, monkeypatch, package_name="cann-test", modules_data=None):
    """设置 main() 测试环境: 创建 modules 文件、输出路径、sys.argv"""
    if modules_data is None:
        modules_data = {}
    modules_file = tmp_path / "modules.txt"
    modules_file.write_text(json.dumps(modules_data))
    output_postinst = str(tmp_path / "postinst")
    output_prerm = str(tmp_path / "prerm")
    monkeypatch.setattr("sys.argv", [
        "gen_postinst_prerm.py",
        "--package-name", package_name,
        "--modules-file", str(modules_file),
        "--output-postinst", output_postinst,
        "--output-prerm", output_prerm,
    ])
    return output_postinst, output_prerm


def _mock_generate_funcs(monkeypatch, postinst_ret="#!/bin/bash\necho test\n",
                         prerm_ret="#!/bin/bash\necho test\n"):
    """Mock generate_postinst 和 generate_prerm 为返回固定字符串的函数"""

    def fake_postinst(*args, **kwargs):
        return postinst_ret

    def fake_prerm(*args, **kwargs):
        return prerm_ret

    monkeypatch.setattr(gen_postinst_prerm, "generate_postinst", fake_postinst)
    monkeypatch.setattr(gen_postinst_prerm, "generate_prerm", fake_prerm)


class TestGenPostinstPrerm:
    """针对 gen_postinst_prerm 模块的全量单元测试（严格匹配 2 参数签名）"""

    # ==========================================
    # 1. 基础工具函数测试 (read_append_file & _resolve_target)
    # ==========================================

    @staticmethod
    def test_read_append_file_not_exists(tmp_path):
        """测试文件不存在时，应优雅返回空字符串"""
        result = gen_postinst_prerm.read_append_file(str(tmp_path / "nonexistent.sh"))
        assert result == ""

    @staticmethod
    def test_read_append_file_empty(tmp_path):
        """测试文件内容为空或仅有空白字符时，返回空字符串"""
        filepath = tmp_path / "empty.sh"
        filepath.write_text("   \n  ")
        result = gen_postinst_prerm.read_append_file(str(filepath))
        assert result == ""

    @staticmethod
    def test_read_append_file_with_content(tmp_path):
        """测试正常文件读取，确保追加内容的首尾包裹规范换行符 \\n"""
        filepath = tmp_path / "custom.sh"
        filepath.write_text("echo 'hello custom'")
        result = gen_postinst_prerm.read_append_file(str(filepath))
        assert result == "\necho 'hello custom'\n"

    @staticmethod
    def test_read_append_file_exception(monkeypatch, tmp_path):
        """测试文件打开抛出异常（如无权限）时，捕获并返回空""，不阻断流程"""

        def mock_open(*args, **kwargs):
            raise PermissionError("Mocked error")

        monkeypatch.setattr("builtins.open", mock_open)
        result = gen_postinst_prerm.read_append_file(str(tmp_path / "error.sh"))
        assert result == ""

    @staticmethod
    @pytest.mark.parametrize("path, expected", [
        ("/absolute/path/file.so", '"/absolute/path/file.so"'),
        ("relative/path/file.so", '"$INSTALL_PATH/relative/path/file.so"')
    ])
    def test_resolve_target(path, expected):
        """测试绝对路径与相对路径转换为 Bash 可用路径字符串的规则"""
        assert gen_postinst_prerm._resolve_target(path) == expected

    # ==========================================
    # 2. generate_postinst 核心生成器测试
    # ==========================================

    @staticmethod
    def test_generate_postinst_basic(tmp_path):
        """测试常规模块的 postinst 核心渲染与基本环境变量声明"""
        modules_data = {
            "NormalModule": [{"target": "lib/liba.so", "link": "lib64/liba.so"}]
        }
        permission_lines = ["chmod 755 $INSTALL_PATH/lib/liba.so"]

        result = gen_postinst_prerm.generate_postinst(
            package_name="cann-runtime",
            version="9.0.0",
            modules_data=modules_data,
            source_dir=str(tmp_path),
            permission_lines=permission_lines
        )

        assert "#!/bin/bash" in result
        assert 'INSTALL_PATH="/usr/local/Ascend/cann-9.0.0"' in result
        assert 'PACKAGE_NAME="cann-runtime"' in result
        assert "chmod 755 $INSTALL_PATH/lib/liba.so" in result
        assert "Post-install completed." in result

    @staticmethod
    def test_generate_postinst_engineering_common(tmp_path):
        """测试 EngineeringCommon 特殊模块，创建软链接前必须额外创建 target 实体目录"""
        modules_data = {
            "EngineeringCommon": [{"target": "ops/kernel", "link": "share/ops/kernel"}]
        }
        result = gen_postinst_prerm.generate_postinst("cann-ops", "9.0.0", modules_data, str(tmp_path))

        assert ('if [ "EngineeringCommon" = "EngineeringCommon" '
                '-o "EngineeringCommon" = "DevlibCommon" ]; then') in result
        assert 'mkdir -p "$INSTALL_PATH"/ops/kernel' in result

    @staticmethod
    def test_generate_postinst_special_modules_db(tmp_path):
        """测试特殊模块（如 DevlibCommon），需要通过 grep/sed 注册并维护组件计数数据库"""
        modules_data = {
            "DevlibCommon": [{"target": "include/dev.h", "link": "link/dev.h"}]
        }
        result = gen_postinst_prerm.generate_postinst("cann-dev", "9.0.0", modules_data, str(tmp_path))

        assert "# Update database for module DevlibCommon" in result
        assert 'grep -q \'^DevlibCommon|\' "$DB_FILE"' in result
        assert 'sed -i "s/^DevlibCommon|.*$/DevlibCommon|$new_components/" "$DB_FILE"' in result

    @staticmethod
    def test_generate_postinst_with_custom_hook(tmp_path):
        """测试外部自定义后置脚本存在时，能够正确插桩读取"""
        custom_dir = tmp_path / "scripts" / "package" / "cann-hook" / "rpm_deb"
        custom_dir.mkdir(parents=True)
        (custom_dir / "custom_postinst.sh").write_text("echo 'CUSTOM_POST_HOOK_TRIGGERED'")

        result = gen_postinst_prerm.generate_postinst("cann-hook", "9.0.0", {}, str(tmp_path))
        assert "echo 'CUSTOM_POST_HOOK_TRIGGERED'" in result
        assert result.endswith("exit 0")

    # ==========================================
    # 3. generate_prerm 核心生成器测试
    # ==========================================

    @staticmethod
    def test_generate_prerm_non_tracked(tmp_path):
        """测试非跟踪的常规模块卸载：直接删除软链接并回溯清理多级父目录"""
        modules_data = {
            "NormalModule": [{"target": "lib/liba.so", "link": "lib/link_a.so"}]
        }
        result = gen_postinst_prerm.generate_prerm("cann-normal", "9.0.0", modules_data, str(tmp_path))

        assert "# Non-tracked module: directly remove symlinks" in result
        assert 'if [ -L "$INSTALL_PATH"/"lib/link_a.so" ]; then' in result
        assert 'rm -f "$INSTALL_PATH"/"lib/link_a.so"' in result
        assert 'cleanup_parent_dirs "$INSTALL_PATH"/"lib/link_a.so" "$INSTALL_PATH"' in result
    
    @staticmethod
    def test_generate_prerm_special_module_lifecycle(tmp_path):
        """测试特殊模块卸载：联动组件清单数据库，计算剥离当前组件并按需执行 rm/sed"""
        modules_data = {
            "EngineeringCommon": [{"target": "ops/bin", "link": "share/bin"}]
        }
        result = gen_postinst_prerm.generate_prerm("cann-eng", "9.0.0", modules_data, str(tmp_path))

        assert 'if [ -f "$DB_FILE" ] && grep -q \'^EngineeringCommon|\' "$DB_FILE"; then' in result
        new_cmd = ("new_components=$(echo \"$components\" | tr ',' '\\n' "
                   "| grep -v \"^$PACKAGE_NAME$\" | paste -sd ',' -)")
        assert new_cmd in result
        assert 'rm -rf "$INSTALL_PATH"/"ops/bin"' in result
        assert 'sed -i \'/^EngineeringCommon|/d\' "$DB_FILE"' in result
        assert "cleanup_db_and_var" in result

    @staticmethod
    def test_generate_prerm_with_custom_hook(tmp_path):
        """测试外部自定义前置脚本存在时，能够正确插桩合并"""
        custom_dir = tmp_path / "scripts" / "package" / "cann-hook" / "rpm_deb"
        custom_dir.mkdir(parents=True)
        (custom_dir / "custom_prerm.sh").write_text("echo 'CUSTOM_PRE_HOOK_TRIGGERED'")

        result = gen_postinst_prerm.generate_prerm("cann-hook", "9.0.0", {}, str(tmp_path))
        assert "echo 'CUSTOM_PRE_HOOK_TRIGGERED'" in result


    # ==========================================
    # 4. generate_set_permission 赋权逻辑测试
    # ==========================================

    @staticmethod
    def test_generate_set_permission_skips_and_dedup():
        """测试权限列表解析：对 NA、缺失项进行安全拦截，并实现 (path, mod) 重复去重"""
        class MockItem:
            def __init__(self, path, perm):
                self.relative_install_path = path
                self.permission = perm

        items = [
            MockItem("NA", "755"),         # 路径无效
            MockItem("bin/app1", "NA"),    # 权限无效
            MockItem("bin/app2", None),    # 权限缺失
            MockItem("lib/liba.so", "644"), # 正常条目
            MockItem("lib/liba.so", "644"), # 重复条目（去重）
        ]

        result = gen_postinst_prerm.generate_set_permission(items, "9.0.0")
        result_str = "\n".join(result)

        assert "bin/app1" not in result_str
        assert "bin/app2" not in result_str
        assert result_str.count('chmod 644 "$INSTALL_PATH/lib/liba.so"') == 2

    @staticmethod
    def test_generate_set_permission_root_elevation():
        """安全基线测试：验证 root (EUID=0) 提权矩阵（550->555, 440->444, 750->755）的 case 转换"""
        class MockItem:
            def __init__(self, path, perm):
                self.relative_install_path = path
                self.permission = perm

        items = [
            MockItem("file_550", "550"),
            MockItem("file_440", "440"),
            MockItem("file_750", "750"),
            MockItem("file_700", "700")
        ]

        result = gen_postinst_prerm.generate_set_permission(items, "9.0.0")
        result_str = "\n".join(result)

        assert 'if [ "${EUID:-0}" -eq 0 ]; then' in result_str
        assert '550) chmod 555 "$INSTALL_PATH/file_550" ;;' in result_str
        assert '440) chmod 444 "$INSTALL_PATH/file_440" ;;' in result_str
        assert '750) chmod 755 "$INSTALL_PATH/file_750" ;;' in result_str
        assert 'chmod 550 "$INSTALL_PATH/file_550"' in result_str

    @staticmethod
    def test_generate_set_permission_script_path_matching():
        """测试内联路径判定：通过 items 中包含的 script 路径触发特定的脚本赋权生成"""
        class MockItem:
            def __init__(self, path, perm):
                self.relative_install_path = path
                self.permission = perm

        # 构造包含 script 关键字的特定相对路径条目
        items = [MockItem("share/info/hccl/script", "550")]

        result = gen_postinst_prerm.generate_set_permission(items, "9.0.0")
        result_str = "\n".join(result)

        assert 'share/info/hccl/script' in result_str
        assert 'if [ -e "$INSTALL_PATH/share/info/hccl/script" ]; then' in result_str

    @staticmethod
    def test_generate_set_permission_with_script_dir():
        """测试 script_dir 参数: 生成统一刷权限的 find 命令"""
        class MockItem:
            def __init__(self, path, perm):
                self.relative_install_path = path
                self.permission = perm

        items = [MockItem("bin/tool", "550")]

        result = gen_postinst_prerm.generate_set_permission(
            items, "9.0.0", script_dir="share/info/runtime/script"
        )
        result_str = "\n".join(result)

        assert 'if [ -d "$INSTALL_PATH/share/info/runtime/script" ]; then' in result_str
        find_cmd_555 = ('find "$INSTALL_PATH/share/info/runtime/script" '
                        '-type f -exec chmod 555 {} \\; 2>/dev/null || true')
        find_cmd_550 = ('find "$INSTALL_PATH/share/info/runtime/script" '
                        '-type f -exec chmod 550 {} \\; 2>/dev/null || true')
        assert find_cmd_555 in result_str
        assert find_cmd_550 in result_str

    @staticmethod
    def test_read_append_file_empty_path():
        """测试传入空路径时返回空字符串"""
        assert gen_postinst_prerm.read_append_file("") == ""

    @staticmethod
    def test_read_append_file_none_path():
        """测试传入 None 时返回空字符串"""
        assert gen_postinst_prerm.read_append_file(None) == ""


class TestGenPostinstPrermMain:
    """针对 gen_postinst_prerm.main() 入口函数的测试"""

    @staticmethod
    def test_main_missing_source_dir_raises(tmp_path, monkeypatch):
        """测试 main() 因 generate_postinst 缺少 source_dir 参数而抛 TypeError (已知问题)"""
        modules_data = {
            "NormalModule": [{"target": "lib/liba.so", "link": "lib64/liba.so"}]
        }
        _setup_main_env(
            tmp_path, monkeypatch, package_name="cann-runtime", modules_data=modules_data
        )
        with pytest.raises(TypeError, match="source_dir"):
            gen_postinst_prerm.main()

    @staticmethod
    def test_main_commlog_imported(tmp_path, monkeypatch):
        """测试 main() 中 CommLog 已正确导入, 不再 NameError (修复 #1)"""
        output_postinst, output_prerm = _setup_main_env(tmp_path, monkeypatch)
        _mock_generate_funcs(monkeypatch)
        gen_postinst_prerm.main()
        assert os.path.isfile(output_postinst)
        assert os.path.isfile(output_prerm)

    @staticmethod
    def test_main_hardcoded_version(tmp_path, monkeypatch):
        """测试 main() 使用硬编码版本号 9.0.0 (已知问题)"""
        _setup_main_env(tmp_path, monkeypatch, package_name="cann-ver")
        captured_args = {}

        def fake_generate_postinst(package_name, version, modules_data, **kwargs):
            captured_args['version'] = version
            return "#!/bin/bash\necho test\n"

        def fake_generate_prerm(package_name, version, modules_data, **kwargs):
            captured_args['prerm_version'] = version
            return "#!/bin/bash\necho test\n"

        monkeypatch.setattr(gen_postinst_prerm, "generate_postinst", fake_generate_postinst)
        monkeypatch.setattr(gen_postinst_prerm, "generate_prerm", fake_generate_prerm)
        gen_postinst_prerm.main()
        assert captured_args['version'] == "9.0.0"
        assert captured_args['prerm_version'] == "9.0.0"

    @staticmethod
    def test_main_writes_output_files(tmp_path, monkeypatch):
        """测试 main() 能正确写入 postinst 和 prerm 文件并设置权限"""
        output_postinst, output_prerm = _setup_main_env(tmp_path, monkeypatch)
        _mock_generate_funcs(
            monkeypatch,
            postinst_ret="#!/bin/bash\necho postinst\n",
            prerm_ret="#!/bin/bash\necho prerm\n",
        )
        gen_postinst_prerm.main()
        assert os.path.isfile(output_postinst)
        assert os.path.isfile(output_prerm)
        assert os.stat(output_postinst).st_mode & 0o777 == 0o755
        assert os.stat(output_prerm).st_mode & 0o777 == 0o755
        assert "echo postinst" in open(output_postinst).read()
        assert "echo prerm" in open(output_prerm).read()

    @staticmethod
    def test_main_reads_modules_json(tmp_path, monkeypatch):
        """测试 main() 正确读取 modules JSON 文件并传递给生成函数"""
        modules_data = {
            "EngineeringCommon": [{"target": "ops/kernel", "link": "share/ops/kernel"}]
        }
        _setup_main_env(
            tmp_path, monkeypatch, package_name="cann-ops", modules_data=modules_data
        )
        captured = {}

        def fake_generate_postinst(package_name, version, modules_data, **kwargs):
            captured['package_name'] = package_name
            captured['modules_data'] = modules_data
            return "#!/bin/bash\necho test\n"

        monkeypatch.setattr(gen_postinst_prerm, "generate_postinst", fake_generate_postinst)
        monkeypatch.setattr(gen_postinst_prerm, "generate_prerm",
                            lambda *a, **k: "#!/bin/bash\necho test\n")
        gen_postinst_prerm.main()
        assert captured['package_name'] == "cann-ops"
        assert captured['modules_data'] == modules_data

    @staticmethod
    def test_main_argparse_required_args(tmp_path, monkeypatch):
        """测试 main() 的 argparse 参数: 缺少 required 参数时报错"""
        monkeypatch.setattr("sys.argv", ["gen_postinst_prerm.py"])

        def fake_error(self_parser, message):
            raise ValueError(message)

        monkeypatch.setattr("argparse.ArgumentParser.error", fake_error)
        with pytest.raises(ValueError):
            gen_postinst_prerm.main()
