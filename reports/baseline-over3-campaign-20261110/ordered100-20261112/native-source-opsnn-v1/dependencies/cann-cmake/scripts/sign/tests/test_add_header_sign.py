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

"""add_header_sign.py 单元测试。

测试覆盖：
- AddHeaderConfig：dataclass 与 from_xml 工厂方法
- read_xml：XML 读取（正常/空文件/解析错误返回None）
- check_config_item：节点属性校验

- safe_run_cmd：命令执行（成功/失败/FileNotFoundError/list格式/shell=False/超时）
- query_sign_attr/query_sign_ext/query_certtype：产物类型查询（正常/失败回退/非法值/超时）
- get_item_set：配置解析（正常/缺文件/非法配置/XML解析失败）
- build_inifile：ini 生成（签名/不签名/cms/无cms/失败/共享目录/ET构造XML）
- build_sign：签名制作（成功/文件不存在/签名失败/命令不累积/无ls冗余/无print残留）
- add_bios_esbc_header：ESBC头（有/无nvcnt/工具缺失/命令失败/命令失败）
- convert_der_file：CRL转换（PEM成功/文件缺失/openssl失败/DER直拷/空文件/非法首字节/拷贝失败/同路径）
- is_pem_format：PEM格式探测（PEM头/无PEM头/空文件/不存在/前导空白）
- build_image_pack_cmd：命令构建（签名/不签名/position/additional/自定义ext和certtype）
- add_bios_header：编排流程（完整/各步失败/der失败处理/finally清理/查询传递）
- check_params：参数校验（全部存在/各项缺失/sign_file_dir校验）
- define_parser：参数解析（必选/默认值/choices约束）
- setenv：环境变量（未设置时用全路径/已设置时保留）
- main：入口（sign_flag=false/true成功/各步失败）
"""

import os
import sys
import time
import xml.etree.ElementTree as ET
from contextlib import ExitStack, contextmanager
from subprocess import CompletedProcess
from unittest import mock

import pytest

import add_header_sign
from add_header_sign import AddHeaderConfig


# ===================== 辅助函数 =====================


def make_conf(**kwargs):
    """构造 AddHeaderConfig，未传参数使用默认值。"""
    defaults = dict(
        input="test.bin",
        output="out.bin",
        version="1.0",
        fw_version="",
        type="cms",
        tag="test_tag",
        rootrsa="default_rsa_rootkey",
        subrsa="default_rsa_subkey",
        additional="",
        sign_alg="PKCSv1.5",
        encrypt_alg="",
        encrypt_type="",
        nvcnt="",
        rsatag="",
        position="",
        image_pack_version="1.0",
        bist_flag="",
    )
    defaults.update(kwargs)
    return AddHeaderConfig(**defaults)


IMAGE_PACK_SCRIPT = "/bios/image_pack.py"


def make_image_pack_param(conf, add_sign="true", **kwargs):
    """构造 ImagePackParam，未传参数使用默认值。"""
    defaults = dict(
        input_file="/sign/a.bin",
        sign_path="/tmp/sign" if add_sign == "true" else "/tmp",
        der_file="/crl.der",
        image_pack_script=IMAGE_PACK_SCRIPT,
    )
    defaults.update(kwargs)
    return add_header_sign.ImagePackParam(conf_item=conf, add_sign=add_sign, **defaults)


def make_bios_header_param(item_set, sign_file_dir, root_dir, **kwargs):
    """构造 BiosHeaderParam，未传参数使用默认值。"""
    defaults = dict(
        bios_tool_path="/bios",
        sign_tool_path="/sign",
        add_sign="true",
    )
    defaults.update(kwargs)
    return add_header_sign.BiosHeaderParam(
        item_size_set=item_set,
        sign_file_dir=str(sign_file_dir),
        root_dir=str(root_dir),
        **defaults,
    )


def make_bios_header_param_from_env(root_dir, bios_tool, sign_dir, item_set, **kwargs):
    """从 setup_bios_env 结果构造 BiosHeaderParam（默认 bios_tool_path/sign_tool_path）。"""
    return make_bios_header_param(
        item_set,
        sign_dir,
        root_dir,
        bios_tool_path=str(bios_tool),
        sign_tool_path="/fake/sign_tool",
        **kwargs,
    )


def setup_der_test_env(tmp_path, der_newer):
    """创建 der/crl 测试环境。der_newer=True 时 der 比 crl 新，否则 crl 比 der 新。"""
    sign_dir = tmp_path / "sign"
    sign_dir.mkdir()
    (sign_dir / "a.bin").touch()
    der_file = sign_dir / "SWSCRL.der"
    crl_file = sign_dir / "SWSCRL.crl"
    if der_newer:
        crl_file.touch()
        time.sleep(0.1)
        der_file.touch()
    else:
        der_file.touch()
        time.sleep(0.1)
        crl_file.touch()
    sign_tool_dir = tmp_path / "sign_tools"
    sign_tool_dir.mkdir()
    conf = make_conf(input="a.bin", type="cms", tag="ta", version="1.0", nvcnt="3")
    item_set = {"a.bin": conf}
    return sign_dir, sign_tool_dir, item_set


def run_bios_header_and_get_crl_path(mocks, item_set, sign_dir, tmp_path):
    """运行 add_bios_header 并返回 convert_der_file 收到的 crl_path。"""
    add_header_sign.add_bios_header(
        make_bios_header_param(
            item_set, sign_dir, tmp_path, sign_tool_path="/sign_tool.py"
        )
    )
    return mocks["convert_der_file"].call_args[0][0]


def make_xml_node(attribs):
    """构造带指定属性的 ET.Element 节点。"""
    node = ET.Element("item")
    for k, v in attribs.items():
        node.set(k, v)
    return node


def write_config_xml(path, items, inis=None):
    """在 path 写入 bios_check_cfg.xml 格式的配置文件。

    items: list of dict, 每个 dict 为 <item> 节点的属性
    inis:  list of dict, 每个 dict 为 <ini> 节点的属性
    """
    lines = ["<bios_check_cfg>"]
    for attrs in items:
        parts = " ".join(f'{k}="{v}"' for k, v in attrs.items())
        lines.append(f"  <item {parts}/>")
    for attrs in inis or []:
        parts = " ".join(f'{k}="{v}"' for k, v in attrs.items())
        lines.append(f"  <ini {parts}/>")
    lines.append("</bios_check_cfg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def setup_bios_env(
    tmp_path,
    with_esbc_script=True,
    with_ini_gen=False,
    file_content=None,
    **conf_kwargs,
):
    """创建 add_bios_header 测试环境（esbc_header + image_pack + sign 目录）。

    返回 (root_dir, bios_tool, sign_dir, item_set)。
    """
    root_dir = str(tmp_path)
    if with_esbc_script:
        esbc_dir = tmp_path / "scripts" / "signtool" / "esbc_header"
        esbc_dir.mkdir(parents=True)
        (esbc_dir / "esbc_header.py").touch()
    bios_tool = tmp_path / "scripts" / "signtool" / "image_pack"
    bios_tool.mkdir(parents=True)
    (bios_tool / "image_pack.py").touch()
    if with_ini_gen:
        (bios_tool / "ini_gen.py").touch()
    sign_dir = tmp_path / "sign"
    sign_dir.mkdir()
    if file_content is not None:
        (sign_dir / "a.bin").write_bytes(file_content)
    else:
        (sign_dir / "a.bin").touch()
    conf = make_conf(input="a.bin", **conf_kwargs)
    return root_dir, bios_tool, sign_dir, {"a.bin": conf}


def setup_and_run_inifile(tmp_path, add_sign="true", **conf_kwargs):
    """创建 build_inifile 测试环境并执行，返回 (result, sign_dir, tmp_dir)。"""
    sign_dir = tmp_path / "sign"
    sign_dir.mkdir()
    (sign_dir / "a.bin").touch()
    tmp_dir = tmp_path / "tmp"
    tmp_dir.mkdir()
    conf = make_conf(input="a.bin", **conf_kwargs)
    item_set = {"a.bin": conf}
    result = add_header_sign.build_inifile(
        item_set, str(sign_dir), "/fake/bios_tool", str(tmp_dir), add_sign
    )
    return result, sign_dir, tmp_dir


@contextmanager
def patched_bios_header_deps(**overrides):
    """Mock add_bios_header 的常用依赖，返回 mock 字典。

    默认 patch：safe_run_cmd=(True,""), build_image_pack_cmd=["python3","image_pack.py"],
    build_sign=True, build_inifile=True, add_bios_esbc_header=True, convert_der_file=True。
    通过 overrides 可覆盖返回值或新增 patch（值为 None 时不设 return_value）。
    """
    default_patches = {
        "safe_run_cmd": (True, ""),
        "build_image_pack_cmd": ["python3", "image_pack.py"],
        "build_sign": True,
        "build_inifile": True,
        "add_bios_esbc_header": True,
        "convert_der_file": True,
    }
    default_patches.update(overrides)
    with ExitStack() as stack:
        mocks = {}
        for name, rv in default_patches.items():
            m = stack.enter_context(mock.patch(f"add_header_sign.{name}"))
            if rv is not None:
                m.return_value = rv
            mocks[name] = m
        yield mocks


def run_main_with_mock_parser(sign_script="", sign_flag="true"):
    """用 mock parser 运行 main，返回结果。"""
    with mock.patch.object(add_header_sign, "define_parser") as mock_parser:
        args = mock.Mock()
        args.sign_file_dir = "/tmp/sign"
        args.sign_flag = sign_flag
        args.bios_check_cfg = "bios_check_cfg.xml"
        args.version = "1.0"
        args.sign_script = sign_script
        mock_parser.return_value.parse_args.return_value = args
        return add_header_sign.main()


@pytest.fixture(autouse=True)
def set_hi_python():
    """所有测试自动设置 HI_PYTHON 环境变量和模块常量。"""
    old_env = os.environ.get("HI_PYTHON")
    old_mod = add_header_sign.HI_PYTHON
    os.environ["HI_PYTHON"] = "python3"
    add_header_sign.HI_PYTHON = "python3"
    yield
    if old_env is not None:
        os.environ["HI_PYTHON"] = old_env
    else:
        os.environ.pop("HI_PYTHON", None)
    add_header_sign.HI_PYTHON = old_mod


# ===================== AddHeaderConfig =====================


class TestAddHeaderConfig:
    """AddHeaderConfig 数据类。"""

    @staticmethod
    def test_all_fields_stored():
        conf = AddHeaderConfig(
            input="in.bin",
            output="out.bin",
            version="2.0",
            fw_version="fw1",
            type="cms",
            tag=["t1"],
            rootrsa="root",
            subrsa="sub",
            additional="add",
            sign_alg="RSA",
            encrypt_alg="AES",
            encrypt_type="type1",
            nvcnt="5",
            rsatag="rsa1",
            position="before_header",
            image_pack_version="2.0",
            bist_flag="bist1",
        )
        assert conf.input == "in.bin"
        assert conf.output == "out.bin"
        assert conf.version == "2.0"
        assert conf.fw_version == "fw1"
        assert conf.type == "cms"
        assert conf.tag == ["t1"]
        assert conf.rootrsa == "root"
        assert conf.subrsa == "sub"
        assert conf.additional == "add"
        assert conf.sign_alg == "RSA"
        assert conf.encrypt_alg == "AES"
        assert conf.encrypt_type == "type1"
        assert conf.nvcnt == "5"
        assert conf.rsatag == "rsa1"
        assert conf.position == "before_header"
        assert conf.image_pack_version == "2.0"
        assert conf.bist_flag == "bist1"

    @staticmethod
    def test_defaults_via_make_conf():
        conf = make_conf()
        assert conf.input == "test.bin"
        assert conf.version == "1.0"
        assert conf.type == "cms"
        assert conf.tag == "test_tag"
        assert conf.sign_alg == "PKCSv1.5"


# ===================== read_xml =====================


class TestReadXml:
    """XML 读取。"""

    @staticmethod
    def test_valid_xml(tmp_path):
        xml_file = tmp_path / "test.xml"
        xml_file.write_text("<root><item/></root>")
        tree = add_header_sign.read_xml(str(xml_file))
        assert tree is not None
        assert tree.getroot().tag == "root"

    @staticmethod
    def test_empty_xml(tmp_path):
        xml_file = tmp_path / "empty.xml"
        xml_file.write_text("<root/>")
        tree = add_header_sign.read_xml(str(xml_file))
        assert tree is not None
        assert tree.getroot().tag == "root"

    @staticmethod
    def test_parse_error_returns_none(tmp_path):
        """D6: 解析失败返回 None 而非抛异常。"""
        xml_file = tmp_path / "bad.xml"
        xml_file.write_text("not xml <<<")
        tree = add_header_sign.read_xml(str(xml_file))
        assert tree is None


# ===================== check_config_item =====================


class TestCheckConfigItem:
    """节点属性校验。"""

    @staticmethod
    def test_valid_with_input_output():
        node = make_xml_node({"input": "a.bin", "output": "out"})
        assert add_header_sign.check_config_item(node) is True

    @staticmethod
    def test_missing_input():
        node = make_xml_node({"output": "out"})
        assert add_header_sign.check_config_item(node) is False

    @staticmethod
    def test_missing_output():
        node = make_xml_node({"input": "a.bin"})
        assert add_header_sign.check_config_item(node) is False

    @staticmethod
    def test_cms_type_without_tag():
        node = make_xml_node({"input": "a.bin", "output": "out", "type": "cms"})
        assert add_header_sign.check_config_item(node) is False

    @staticmethod
    def test_cms_type_with_tag():
        node = make_xml_node(
            {"input": "a.bin", "output": "out", "type": "cms", "tag": "t1"}
        )
        assert add_header_sign.check_config_item(node) is True

    @staticmethod
    def test_non_cms_type_without_tag():
        node = make_xml_node({"input": "a.bin", "output": "out", "type": "rsa"})
        assert add_header_sign.check_config_item(node) is True

    @staticmethod
    def test_no_type():
        node = make_xml_node({"input": "a.bin", "output": "out"})
        assert add_header_sign.check_config_item(node) is True

    @staticmethod
    def test_slash_split_cms():
        node = make_xml_node(
            {"input": "a.bin", "output": "out", "type": "rsa/cms", "tag": "t"}
        )
        assert add_header_sign.check_config_item(node) is True

    @staticmethod
    def test_slash_split_cms_no_tag():
        node = make_xml_node({"input": "a.bin", "output": "out", "type": "rsa/cms"})
        assert add_header_sign.check_config_item(node) is False


# ===================== AddHeaderConfig.from_xml =====================


class TestParseItem:
    """节点解析与默认值。"""

    @staticmethod
    def test_full_attributes():
        node = make_xml_node(
            {
                "input": "a.bin",
                "output": "out",
                "version": "2.0",
                "type": "cms",
                "tag": "t1",
                "sign_alg": "RSA_PSS",
                "encrypt_alg": "AES",
                "encrypt_type": "CBC",
                "additional": "-x 1",
                "nvcnt": "3",
                "rsatag": "r1",
                "position": "before_header",
                "image_pack": "2.0",
                "rootrsa": "root1",
                "subrsa": "sub1",
                "bist_flag": "b1",
                "fw_version": "fw1",
            }
        )
        conf = AddHeaderConfig.from_xml(node)
        assert conf.input == "a.bin"
        assert conf.output == "out"
        assert conf.version == "2.0"
        assert conf.type == "cms"
        assert conf.tag == "t1"
        assert conf.sign_alg == "RSA_PSS"
        assert conf.encrypt_alg == "AES"
        assert conf.encrypt_type == "CBC"
        assert conf.additional == "-x 1"
        assert conf.nvcnt == "3"
        assert conf.rsatag == "r1"
        assert conf.position == "before_header"
        assert conf.image_pack_version == "2.0"
        assert conf.rootrsa == "root1"
        assert conf.subrsa == "sub1"
        assert conf.bist_flag == "b1"
        assert conf.fw_version == "fw1"

    @staticmethod
    def test_minimal_attributes():
        node = make_xml_node({"input": "a.bin", "output": "out", "version": "1.0"})
        conf = AddHeaderConfig.from_xml(node)
        assert conf.input == "a.bin"
        assert conf.type == ""
        assert conf.tag == ""
        assert conf.sign_alg == "PKCSv1.5"
        assert conf.encrypt_alg == ""
        assert conf.encrypt_type == ""
        assert conf.additional == ""
        assert conf.nvcnt == ""
        assert conf.rsatag == ""
        assert conf.position == ""
        assert conf.image_pack_version == "1.0"
        assert conf.rootrsa == "default_rsa_rootkey"
        assert conf.subrsa == "default_rsa_subkey"
        assert conf.bist_flag == ""

    @staticmethod
    def test_fw_version_default_empty():
        node = make_xml_node({"input": "a.bin", "output": "out", "version": "1.0"})
        conf = AddHeaderConfig.from_xml(node)
        assert conf.fw_version == ""

    @staticmethod
    def test_fw_version_explicit():
        node = make_xml_node(
            {"input": "a.bin", "output": "out", "version": "1.0", "fw_version": "9.9"}
        )
        conf = AddHeaderConfig.from_xml(node)
        assert conf.fw_version == "9.9"


# ===================== safe_run_cmd =====================


class TestSafeRunCmd:
    """命令执行。"""

    @staticmethod
    @mock.patch("add_header_sign.run")
    def test_success(mock_run):
        mock_run.return_value = CompletedProcess(args=[], returncode=0, stdout="ok")
        success, output = add_header_sign.safe_run_cmd(["echo", "hello"])
        assert success is True
        assert output == "ok"

    @staticmethod
    @mock.patch("add_header_sign.run")
    def test_failure(mock_run):
        mock_run.return_value = CompletedProcess(args=[], returncode=1, stdout="err")
        success, output = add_header_sign.safe_run_cmd(["false"])
        assert success is False
        assert output == "err"

    @staticmethod
    @mock.patch("add_header_sign.run")
    def test_shell_false(mock_run):
        mock_run.return_value = CompletedProcess(args=[], returncode=0, stdout="")
        add_header_sign.safe_run_cmd(["echo", "hi"])
        _, kwargs = mock_run.call_args
        assert kwargs.get("shell") is False

    @staticmethod
    @mock.patch("add_header_sign.run")
    def test_takes_list_not_string(mock_run):
        """D1: 接收 list 而非 string。"""
        mock_run.return_value = CompletedProcess(args=[], returncode=0, stdout="")
        cmd = ["python3", "-c", "print(1)"]
        add_header_sign.safe_run_cmd(cmd)
        args, _ = mock_run.call_args
        assert args[0] == cmd

    @staticmethod
    @mock.patch("add_header_sign.run")
    def test_work_dir(mock_run):
        mock_run.return_value = CompletedProcess(args=[], returncode=0, stdout="")
        add_header_sign.safe_run_cmd(["ls"], work_dir="/tmp")
        _, kwargs = mock_run.call_args
        assert kwargs.get("cwd") == "/tmp"

    @staticmethod
    @mock.patch("add_header_sign.run")
    def test_file_not_found_returns_false(mock_run):
        """B5: 可执行文件不存在时返回 (False, 错误信息) 而非抛异常。"""
        mock_run.side_effect = FileNotFoundError("[Errno 2] No such file")
        success, output = add_header_sign.safe_run_cmd(["nonexistent_cmd", "arg"])
        assert success is False
        assert "No such file" in output

    @staticmethod
    @mock.patch("add_header_sign.run")
    def test_permission_error_returns_false(mock_run):
        """可执行文件存在但无执行权限时返回 (False, 错误信息) 而非抛异常。"""
        mock_run.side_effect = PermissionError("[Errno 13] Permission denied")
        success, output = add_header_sign.safe_run_cmd(["/bin/some_cmd", "arg"])
        assert success is False
        assert "Permission denied" in output

    @staticmethod
    @mock.patch("add_header_sign.run")
    def test_timeout_returns_false(mock_run):
        """超时返回 (False, 超时信息) 而非抛异常。"""
        mock_run.side_effect = add_header_sign.TIMEOUT_EXPIRED(
            cmd=["slow_cmd"], timeout=10
        )
        success, output = add_header_sign.safe_run_cmd(["slow_cmd"], timeout=10)
        assert success is False
        assert "timed out" in output.lower()

    @staticmethod
    @mock.patch("add_header_sign.run")
    def test_timeout_passed_to_subprocess(mock_run):
        """timeout 参数透传给 subprocess.run。"""
        mock_run.return_value = CompletedProcess(args=[], returncode=0, stdout="")
        add_header_sign.safe_run_cmd(["echo", "hi"], timeout=30)
        _, kwargs = mock_run.call_args
        assert kwargs.get("timeout") == 30


# ===================== get_item_set =====================


class TestGetItemSet:
    """配置文件解析。"""

    @staticmethod
    def test_valid_config_with_existing_files(tmp_path):
        cfg = tmp_path / "cfg.xml"
        (tmp_path / "a.bin").touch()
        (tmp_path / "b.bin").touch()
        write_config_xml(
            str(cfg),
            [
                {"input": "a.bin", "output": "out", "type": "cms", "tag": "ta"},
                {"input": "b.bin", "output": "out", "type": "cms", "tag": "tb"},
            ],
        )
        success, items = add_header_sign.get_item_set(str(cfg), str(tmp_path), "1.0")
        assert success is True
        assert "a.bin" in items
        assert "b.bin" in items
        assert items["a.bin"].tag == "ta"
        assert items["b.bin"].version == "1.0"

    @staticmethod
    def test_missing_files_skipped(tmp_path):
        cfg = tmp_path / "cfg.xml"
        (tmp_path / "a.bin").touch()
        write_config_xml(
            str(cfg),
            [
                {"input": "a.bin", "output": "out", "type": "cms", "tag": "ta"},
                {"input": "missing.bin", "output": "out", "type": "cms", "tag": "tb"},
            ],
        )
        success, items = add_header_sign.get_item_set(str(cfg), str(tmp_path), "1.0")
        assert success is True
        assert "a.bin" in items
        assert "missing.bin" not in items

    @staticmethod
    def test_invalid_config_missing_input(tmp_path):
        cfg = tmp_path / "cfg.xml"
        write_config_xml(
            str(cfg),
            [
                {"output": "out", "type": "cms", "tag": "ta"},
            ],
        )
        success, items = add_header_sign.get_item_set(str(cfg), str(tmp_path), "1.0")
        assert success is False

    @staticmethod
    def test_cms_without_tag_invalid(tmp_path):
        cfg = tmp_path / "cfg.xml"
        (tmp_path / "a.bin").touch()
        write_config_xml(
            str(cfg),
            [
                {"input": "a.bin", "output": "out", "type": "cms"},
            ],
        )
        success, items = add_header_sign.get_item_set(str(cfg), str(tmp_path), "1.0")
        assert success is False

    @staticmethod
    def test_default_version_when_not_in_config(tmp_path):
        cfg = tmp_path / "cfg.xml"
        (tmp_path / "a.bin").touch()
        write_config_xml(
            str(cfg),
            [
                {"input": "a.bin", "output": "out", "type": "cms", "tag": "ta"},
            ],
        )
        success, items = add_header_sign.get_item_set(str(cfg), str(tmp_path), "3.5")
        assert success is True
        assert items["a.bin"].version == "3.5"

    @staticmethod
    def test_explicit_version_in_config(tmp_path):
        cfg = tmp_path / "cfg.xml"
        (tmp_path / "a.bin").touch()
        write_config_xml(
            str(cfg),
            [
                {
                    "input": "a.bin",
                    "output": "out",
                    "version": "9.9",
                    "type": "cms",
                    "tag": "ta",
                },
            ],
        )
        success, items = add_header_sign.get_item_set(str(cfg), str(tmp_path), "3.5")
        assert success is True
        assert items["a.bin"].version == "9.9"

    @staticmethod
    def test_nvcnt_in_config(tmp_path):
        cfg = tmp_path / "cfg.xml"
        (tmp_path / "a.bin").touch()
        write_config_xml(
            str(cfg),
            [
                {
                    "input": "a.bin",
                    "output": "out",
                    "type": "cms",
                    "tag": "ta",
                    "nvcnt": "3",
                },
            ],
        )
        success, items = add_header_sign.get_item_set(str(cfg), str(tmp_path), "1.0")
        assert success is True
        assert items["a.bin"].nvcnt == "3"

    @staticmethod
    def test_empty_config(tmp_path):
        cfg = tmp_path / "cfg.xml"
        write_config_xml(str(cfg), [])
        success, items = add_header_sign.get_item_set(str(cfg), str(tmp_path), "1.0")
        assert success is True
        assert len(items) == 0

    @staticmethod
    def test_no_type_item(tmp_path):
        cfg = tmp_path / "cfg.xml"
        (tmp_path / "a.bin").touch()
        write_config_xml(
            str(cfg),
            [
                {"input": "a.bin", "output": "out", "version": "1.0"},
            ],
        )
        success, items = add_header_sign.get_item_set(str(cfg), str(tmp_path), "1.0")
        assert success is True
        assert items["a.bin"].type == ""

    @staticmethod
    def test_xml_parse_error(tmp_path):
        """D6: XML 解析失败时返回 (False, None)。"""
        cfg = tmp_path / "cfg.xml"
        cfg.write_text("not valid xml <<<")
        success, items = add_header_sign.get_item_set(str(cfg), str(tmp_path), "1.0")
        assert success is False
        assert items is None


# ===================== build_inifile =====================


class TestBuildInifile:
    """ini 文件生成。"""

    @staticmethod
    @mock.patch("add_header_sign.safe_run_cmd", return_value=(True, ""))
    def test_add_sign_true_with_cms(mock_run, tmp_path):
        result, _, _ = setup_and_run_inifile(tmp_path, type="cms", tag="ta")
        assert result is True
        mock_run.assert_called_once()

    @staticmethod
    @mock.patch("add_header_sign.safe_run_cmd", return_value=(True, ""))
    def test_add_sign_true_without_cms(mock_run, tmp_path):
        result, _, _ = setup_and_run_inifile(tmp_path, type="", tag="")
        assert result is True
        mock_run.assert_not_called()

    @staticmethod
    @mock.patch("add_header_sign.safe_run_cmd", return_value=(True, ""))
    def test_no_image_info_xml_when_no_cms(mock_run, tmp_path):
        """P-3/OPT-6: 无 cms 镜像时不生成 image_info.xml。"""
        _, _, tmp_dir = setup_and_run_inifile(tmp_path, type="", tag="")
        assert not (tmp_dir / "image_info.xml").exists()

    @staticmethod
    @mock.patch("add_header_sign.safe_run_cmd", return_value=(True, ""))
    def test_add_sign_false(mock_run, tmp_path):
        result, _, _ = setup_and_run_inifile(
            tmp_path, add_sign="false", type="cms", tag="ta"
        )
        assert result is True
        mock_run.assert_not_called()

    @staticmethod
    @mock.patch("add_header_sign.safe_run_cmd", return_value=(False, "error"))
    def test_inifile_cmd_failure(mock_run, tmp_path):
        result, _, _ = setup_and_run_inifile(tmp_path, type="cms", tag="ta")
        assert result is False

    @staticmethod
    @mock.patch("add_header_sign.safe_run_cmd", return_value=(True, ""))
    def test_shared_parent_dir_no_crash(mock_run, tmp_path):
        """B4: 两个文件共享同一父目录时 makedirs(exist_ok=True) 不应崩溃。"""
        sign_dir = tmp_path / "sign"
        sign_dir.mkdir()
        sub = sign_dir / "sub"
        sub.mkdir()
        (sub / "a.bin").touch()
        (sub / "b.bin").touch()
        tmp_dir = tmp_path / "tmp"
        tmp_dir.mkdir()
        conf_a = make_conf(input="sub/a.bin", type="cms", tag="ta")
        conf_b = make_conf(input="sub/b.bin", type="cms", tag="tb")
        item_set = {"sub/a.bin": conf_a, "sub/b.bin": conf_b}
        result = add_header_sign.build_inifile(
            item_set, str(sign_dir), "/fake/bios_tool", str(tmp_dir), "true"
        )
        assert result is True

    @staticmethod
    @mock.patch("add_header_sign.safe_run_cmd", return_value=(True, ""))
    def test_image_info_xml_generated_via_et(mock_run, tmp_path):
        """D3: XML 由 ET 构造，可被 ET 解析。"""
        _, _, tmp_dir = setup_and_run_inifile(tmp_path, type="cms", tag="ta")
        info_xml = tmp_dir / "image_info.xml"
        assert info_xml.exists()
        tree = ET.parse(str(info_xml))
        root = tree.getroot()
        assert root.tag == "image_info"
        images = root.findall("image")
        assert len(images) == 1
        assert images[0].get("tag") == "ta"
        assert "a.bin" in images[0].get("path")

    @staticmethod
    @mock.patch("add_header_sign.safe_run_cmd", return_value=(True, ""))
    def test_cmd_is_list(mock_run, tmp_path):
        """D1: 传给 safe_run_cmd 的是 list。"""
        setup_and_run_inifile(tmp_path, type="cms", tag="ta")
        args, _ = mock_run.call_args
        assert isinstance(args[0], list)


# ===================== build_sign =====================


class TestBuildSign:
    """签名制作。"""

    @staticmethod
    @mock.patch("add_header_sign.safe_run_cmd", return_value=(True, ""))
    def test_success_single_file(mock_run, tmp_path):
        sign_dir = tmp_path / "sign"
        sign_dir.mkdir()
        (sign_dir / "a.bin").write_bytes(b"data_a")
        conf = make_conf(input="a.bin", type="cms", tag="ta")
        item_set = {"a.bin": conf}
        result = add_header_sign.build_sign(
            item_set, str(sign_dir), "/fake/sign_tool", str(tmp_path / "tmp")
        )
        assert result is True

    @staticmethod
    @mock.patch("add_header_sign.safe_run_cmd", return_value=(True, ""))
    def test_no_repeated_signing(mock_run, tmp_path):
        """B1: 多文件签名时命令不累积，只调用一次签名工具。"""
        sign_dir = tmp_path / "sign"
        sign_dir.mkdir()
        (sign_dir / "a.bin").write_bytes(b"data_a")
        (sign_dir / "b.bin").write_bytes(b"data_b")
        conf_a = make_conf(input="a.bin", type="cms", tag="ta")
        conf_b = make_conf(input="b.bin", type="cms", tag="tb")
        item_set = {"a.bin": conf_a, "b.bin": conf_b}
        result = add_header_sign.build_sign(
            item_set, str(sign_dir), "/fake/sign_tool", str(tmp_path / "tmp")
        )
        assert result is True
        sign_calls = [c for c in mock_run.call_args_list if "sign_tool" in str(c)]
        assert len(sign_calls) == 1, f"expected 1 sign call, got {len(sign_calls)}"

    @staticmethod
    @mock.patch("add_header_sign.safe_run_cmd", return_value=(True, ""))
    def test_no_print_to_stdout(mock_run, tmp_path, capsys):
        """B6: build_sign 不应有残留 print 输出。"""
        sign_dir = tmp_path / "sign"
        sign_dir.mkdir()
        (sign_dir / "a.bin").write_bytes(b"data_a")
        conf = make_conf(input="a.bin", type="cms", tag="ta")
        item_set = {"a.bin": conf}
        add_header_sign.build_sign(
            item_set, str(sign_dir), "/fake/sign_tool", str(tmp_path / "tmp")
        )
        captured = capsys.readouterr()
        assert "a.ini" not in captured.out

    @staticmethod
    @mock.patch("add_header_sign.safe_run_cmd", return_value=(True, ""))
    def test_no_redundant_ls(mock_run, tmp_path):
        """N1: 不再有冗余 ls 命令调用。"""
        sign_dir = tmp_path / "sign"
        sign_dir.mkdir()
        (sign_dir / "a.bin").write_bytes(b"data_a")
        conf = make_conf(input="a.bin", type="cms", tag="ta")
        item_set = {"a.bin": conf}
        add_header_sign.build_sign(
            item_set, str(sign_dir), "/fake/sign_tool", str(tmp_path / "tmp")
        )
        ls_calls = [
            c for c in mock_run.call_args_list if c[0][0] and c[0][0][0] == "ls"
        ]
        assert len(ls_calls) == 0

    @staticmethod
    def test_file_not_exist_returns_false(tmp_path):
        sign_dir = tmp_path / "sign"
        sign_dir.mkdir()
        conf = make_conf(input="ghost.bin", type="cms", tag="ta")
        item_set = {"ghost.bin": conf}
        result = add_header_sign.build_sign(
            item_set, str(sign_dir), "/fake/sign_tool", str(tmp_path / "tmp")
        )
        assert result is False

    @staticmethod
    def test_infile_is_directory_returns_false(tmp_path):
        """路径存在但为目录时返回 False，不抛 IsADirectoryError。"""
        sign_dir = tmp_path / "sign"
        sign_dir.mkdir()
        (sign_dir / "a.bin").mkdir()
        conf = make_conf(input="a.bin", type="cms", tag="ta")
        item_set = {"a.bin": conf}
        result = add_header_sign.build_sign(
            item_set, str(sign_dir), "/fake/sign_tool", str(tmp_path / "tmp")
        )
        assert result is False

    @staticmethod
    @mock.patch("add_header_sign.safe_run_cmd")
    def test_sign_cmd_failure_returns_false(mock_run, tmp_path):
        sign_dir = tmp_path / "sign"
        sign_dir.mkdir()
        (sign_dir / "a.bin").write_bytes(b"data_a")
        mock_run.return_value = (False, "sign error")
        conf = make_conf(input="a.bin", type="cms", tag="ta")
        item_set = {"a.bin": conf}
        result = add_header_sign.build_sign(
            item_set, str(sign_dir), "/fake/sign_tool", str(tmp_path / "tmp")
        )
        assert result is False

    @staticmethod
    @mock.patch("add_header_sign.safe_run_cmd", return_value=(True, ""))
    def test_non_cms_type_skipped(mock_run, tmp_path):
        sign_dir = tmp_path / "sign"
        sign_dir.mkdir()
        (sign_dir / "a.bin").write_bytes(b"data_a")
        conf = make_conf(input="a.bin", type="rsa", tag="ta")
        item_set = {"a.bin": conf}
        result = add_header_sign.build_sign(
            item_set, str(sign_dir), "/fake/sign_tool", str(tmp_path / "tmp")
        )
        assert result is True
        sign_calls = [c for c in mock_run.call_args_list if "sign_tool" in str(c)]
        assert len(sign_calls) == 0

    @staticmethod
    @mock.patch("add_header_sign.safe_run_cmd", return_value=(True, ""))
    def test_empty_item_set(mock_run, tmp_path):
        result = add_header_sign.build_sign(
            {}, str(tmp_path), "/fake/sign_tool", str(tmp_path / "tmp")
        )
        assert result is True
        sign_calls = [c for c in mock_run.call_args_list if "sign_tool" in str(c)]
        assert len(sign_calls) == 0

    @staticmethod
    @mock.patch("add_header_sign.safe_run_cmd", return_value=(True, ""))
    def test_cmd_is_list(mock_run, tmp_path):
        """D1: 传给 safe_run_cmd 的是 list。"""
        sign_dir = tmp_path / "sign"
        sign_dir.mkdir()
        (sign_dir / "a.bin").write_bytes(b"data_a")
        conf = make_conf(input="a.bin", type="cms", tag="ta")
        item_set = {"a.bin": conf}
        add_header_sign.build_sign(
            item_set, str(sign_dir), "/fake/sign_tool", str(tmp_path / "tmp")
        )
        args, _ = mock_run.call_args
        assert isinstance(args[0], list)

    @staticmethod
    @mock.patch("add_header_sign.safe_run_cmd", return_value=(True, ""))
    def test_cmd_structure_no_root_dir(mock_run, tmp_path):
        """命令结构断言：[HI_PYTHON, sign_tool_path, --crl-dir, sign_file_dir] + ini_files，不含 root_dir。"""
        sign_dir = tmp_path / "sign"
        sign_dir.mkdir()
        (sign_dir / "a.bin").write_bytes(b"data_a")
        conf = make_conf(input="a.bin", type="cms", tag="ta")
        item_set = {"a.bin": conf}
        add_header_sign.build_sign(
            item_set, str(sign_dir), "/fake/sign_tool", str(tmp_path / "tmp")
        )
        args, _ = mock_run.call_args
        cmd = args[0]
        assert cmd[0] == "python3"
        assert cmd[1] == "/fake/sign_tool"
        # --crl-dir 传入 sign_file_dir，实现每目标 CRL 隔离
        assert "--crl-dir" in cmd
        idx = cmd.index("--crl-dir")
        assert cmd[idx + 1] == str(sign_dir)
        # 命令中不应包含 root_dir（已移除）
        assert "/root" not in cmd
        assert len(cmd) == 5


# ===================== add_bios_esbc_header =====================


class TestAddBiosEsbcHeader:
    """ESBC 头添加。"""

    @staticmethod
    @mock.patch("add_header_sign.safe_run_cmd", return_value=(True, ""))
    def test_with_nvcnt(mock_run, tmp_path):
        root_dir, _, sign_dir, item_set = setup_bios_env(
            tmp_path, nvcnt="3", tag="ta", version="1.0"
        )
        result = add_header_sign.add_bios_esbc_header(root_dir, item_set, str(sign_dir))
        assert result is True
        mock_run.assert_called_once()

    @staticmethod
    @mock.patch("add_header_sign.safe_run_cmd", return_value=(True, ""))
    def test_without_nvcnt_skips(mock_run, tmp_path):
        root_dir, _, sign_dir, item_set = setup_bios_env(tmp_path, nvcnt="", tag="ta")
        result = add_header_sign.add_bios_esbc_header(root_dir, item_set, str(sign_dir))
        assert result is True
        mock_run.assert_not_called()

    @staticmethod
    def test_tool_dir_missing(tmp_path):
        root_dir = str(tmp_path)
        sign_dir = tmp_path / "sign"
        sign_dir.mkdir()
        conf = make_conf(input="a.bin", nvcnt="3", tag="ta")
        item_set = {"a.bin": conf}
        result = add_header_sign.add_bios_esbc_header(root_dir, item_set, str(sign_dir))
        assert result is False

    @staticmethod
    @mock.patch("add_header_sign.safe_run_cmd", return_value=(False, "err"))
    def test_cmd_failure(mock_run, tmp_path):
        root_dir, _, sign_dir, item_set = setup_bios_env(
            tmp_path, nvcnt="3", tag="ta", version="1.0"
        )
        result = add_header_sign.add_bios_esbc_header(root_dir, item_set, str(sign_dir))
        assert result is False


# ===================== convert_der_file =====================

PEM_HEADER = b"-----BEGIN X509 CRL-----\n"


class TestConvertDerFile:
    """CRL → DER 转换。"""

    @staticmethod
    @mock.patch("add_header_sign.safe_run_cmd", return_value=(True, ""))
    def test_success(mock_run, tmp_path):
        crl = tmp_path / "test.crl"
        crl.write_bytes(PEM_HEADER)
        der = str(tmp_path / "test.der")
        result = add_header_sign.convert_der_file(str(crl), der)
        assert result is True
        mock_run.assert_called_once()

    @staticmethod
    def test_crl_file_not_exist(tmp_path):
        der = str(tmp_path / "test.der")
        result = add_header_sign.convert_der_file("/nonexistent.crl", der)
        assert result is False

    @staticmethod
    @mock.patch("add_header_sign.safe_run_cmd", return_value=(False, "openssl error"))
    def test_openssl_failure(mock_run, tmp_path):
        crl = tmp_path / "test.crl"
        crl.write_bytes(PEM_HEADER)
        der = str(tmp_path / "test.der")
        result = add_header_sign.convert_der_file(str(crl), der)
        assert result is False

    @staticmethod
    @mock.patch("add_header_sign.safe_run_cmd", return_value=(True, ""))
    def test_no_print_on_error(mock_run, tmp_path, capsys):
        """N3: 错误用 logging 而非 print，stdout 不含 [ERROR] 行。"""
        add_header_sign.convert_der_file("/nonexistent.crl", str(tmp_path / "x.der"))
        captured = capsys.readouterr()
        assert not any(
            line.strip().startswith("[ERROR]") for line in captured.out.splitlines()
        )

    @staticmethod
    @mock.patch("add_header_sign.safe_run_cmd", return_value=(True, ""))
    def test_cmd_is_list(mock_run, tmp_path):
        """D1: 传给 safe_run_cmd 的是 list。"""
        crl = tmp_path / "test.crl"
        crl.write_bytes(PEM_HEADER)
        add_header_sign.convert_der_file(str(crl), str(tmp_path / "test.der"))
        args, _ = mock_run.call_args
        assert isinstance(args[0], list)
        assert args[0][0] == "openssl"

    @staticmethod
    @mock.patch("add_header_sign.safe_run_cmd")
    def test_der_input_copied_directly(mock_run, tmp_path):
        """输入已是 DER 格式时直接拷贝，不调用 openssl。"""
        mock_run.return_value = (True, "")
        crl = tmp_path / "test.crl"
        crl_bytes = b"\x30\x82\x01\x00" + b"\x00" * 16
        crl.write_bytes(crl_bytes)
        der = str(tmp_path / "test.der")
        result = add_header_sign.convert_der_file(str(crl), der)
        assert result is True
        mock_run.assert_not_called()
        assert os.path.isfile(der)
        with open(der, "rb") as f:
            assert f.read() == crl_bytes

    @staticmethod
    @mock.patch("add_header_sign.safe_run_cmd", return_value=(True, ""))
    def test_empty_file_returns_false(mock_run, tmp_path):
        """空文件既非 PEM 也非有效 DER，返回 False。"""
        crl = tmp_path / "test.crl"
        crl.touch()
        der = str(tmp_path / "test.der")
        result = add_header_sign.convert_der_file(str(crl), der)
        assert result is False
        mock_run.assert_not_called()

    @staticmethod
    @mock.patch("add_header_sign.safe_run_cmd", return_value=(True, ""))
    def test_invalid_first_byte_returns_false(mock_run, tmp_path):
        """首字节非 0x30 的非 PEM 文件视为非法 DER，返回 False。"""
        crl = tmp_path / "test.crl"
        crl.write_bytes(b"\x00\x01\x02\x03")
        der = str(tmp_path / "test.der")
        result = add_header_sign.convert_der_file(str(crl), der)
        assert result is False
        mock_run.assert_not_called()

    @staticmethod
    @mock.patch("add_header_sign.safe_run_cmd", return_value=(True, ""))
    @mock.patch("add_header_sign.shutil.copy", side_effect=OSError("disk full"))
    def test_der_copy_failure_returns_false(mock_copy, mock_run, tmp_path):
        """DER 分支拷贝失败（OSError）时返回 False。"""
        crl = tmp_path / "test.crl"
        crl.write_bytes(b"\x30\x82\x01\x00" + b"\x00" * 16)
        der = str(tmp_path / "nonexistent_dir" / "test.der")
        result = add_header_sign.convert_der_file(str(crl), der)
        assert result is False
        mock_run.assert_not_called()
        mock_copy.assert_called_once()

    @staticmethod
    @mock.patch("add_header_sign.safe_run_cmd", return_value=(True, ""))
    def test_same_path_returns_true(mock_run, tmp_path):
        """源/目标同路径时跳过转换与拷贝，直接返回 True。"""
        crl = tmp_path / "test.crl"
        crl.write_bytes(b"\x30\x82\x01\x00" + b"\x00" * 16)
        result = add_header_sign.convert_der_file(str(crl), str(crl))
        assert result is True
        mock_run.assert_not_called()


# ===================== is_pem_format =====================


class TestIsPemFormat:
    """PEM 格式探测。"""

    @staticmethod
    def test_pem_header(tmp_path):
        crl = tmp_path / "test.crl"
        crl.write_bytes(PEM_HEADER)
        assert add_header_sign.is_pem_format(str(crl)) is True

    @staticmethod
    def test_no_pem_header(tmp_path):
        crl = tmp_path / "test.crl"
        crl.write_bytes(b"\x30\x82\x01\x00" + b"\x00" * 16)
        assert add_header_sign.is_pem_format(str(crl)) is False

    @staticmethod
    def test_empty_file(tmp_path):
        crl = tmp_path / "test.crl"
        crl.touch()
        assert add_header_sign.is_pem_format(str(crl)) is False

    @staticmethod
    def test_file_not_exist():
        assert add_header_sign.is_pem_format("/nonexistent.crl") is False

    @staticmethod
    def test_leading_whitespace(tmp_path):
        """PEM 头前有空白（换行/空格）时仍能正确识别。"""
        crl = tmp_path / "test.crl"
        crl.write_bytes(b"\n   " + PEM_HEADER)
        assert add_header_sign.is_pem_format(str(crl)) is True


# ===================== query_sign_attr / query_sign_ext / query_certtype =====================


class TestQuerySignType:
    """查询签名脚本的产物扩展名与证书类型。"""

    @staticmethod
    @mock.patch("add_header_sign.safe_run_cmd")
    def test_query_sign_attr_success(mock_run):
        """正常查询返回 stdout 最后一行非空行。"""
        mock_run.return_value = (True, ".p7s\n")
        result = add_header_sign.query_sign_attr("/fake/script.py", "--print-sign-ext")
        assert result == ".p7s"

    @staticmethod
    @mock.patch("add_header_sign.safe_run_cmd")
    def test_query_sign_attr_takes_last_nonempty_line(mock_run):
        """容忍前导输出（Python 启动警告等），取最后一行非空行。"""
        mock_run.return_value = (True, "warning line\n\n.p7s\n")
        result = add_header_sign.query_sign_attr("/fake/script.py", "--print-sign-ext")
        assert result == ".p7s"

    @staticmethod
    @mock.patch("add_header_sign.safe_run_cmd", return_value=(False, "error"))
    def test_query_sign_attr_failure_returns_none(mock_run):
        """查询失败（脚本不识别 flag）返回 None。"""
        result = add_header_sign.query_sign_attr("/fake/script.py", "--print-sign-ext")
        assert result is None

    @staticmethod
    @mock.patch("add_header_sign.safe_run_cmd", return_value=(True, ""))
    def test_query_sign_attr_empty_output_returns_none(mock_run):
        """stdout 为空返回 None。"""
        result = add_header_sign.query_sign_attr("/fake/script.py", "--print-sign-ext")
        assert result is None

    @staticmethod
    @mock.patch("add_header_sign.safe_run_cmd", return_value=(True, "   \n  \n"))
    def test_query_sign_attr_only_whitespace_returns_none(mock_run):
        """stdout 仅含空白返回 None。"""
        result = add_header_sign.query_sign_attr("/fake/script.py", "--print-certtype")
        assert result is None

    @staticmethod
    @mock.patch("add_header_sign.safe_run_cmd")
    def test_query_sign_ext_success(mock_run):
        """query_sign_ext 正常返回扩展名。"""
        mock_run.return_value = (True, ".cms\n")
        assert add_header_sign.query_sign_ext("/fake/script.py") == ".cms"

    @staticmethod
    @mock.patch("add_header_sign.safe_run_cmd", return_value=(False, "err"))
    def test_query_sign_ext_fallback_default(mock_run):
        """query_sign_ext 查询失败回退默认 .p7s。"""
        assert add_header_sign.query_sign_ext("/fake/script.py") == ".p7s"

    @staticmethod
    @mock.patch("add_header_sign.safe_run_cmd")
    def test_query_certtype_hex_to_decimal(mock_run):
        """query_certtype 十六进制转十进制。"""
        cases = [("0x1", "1"), ("0x2", "2"), ("0xFFFFFFFF", "4294967295")]
        for hex_val, dec_val in cases:
            mock_run.return_value = (True, hex_val + "\n")
            assert add_header_sign.query_certtype("/fake/script.py") == dec_val

    @staticmethod
    @mock.patch("add_header_sign.safe_run_cmd", return_value=(False, "err"))
    def test_query_certtype_fallback_default(mock_run):
        """query_certtype 查询失败回退默认 1。"""
        assert add_header_sign.query_certtype("/fake/script.py") == "1"

    @staticmethod
    @mock.patch("add_header_sign.safe_run_cmd")
    def test_query_sign_ext_calls_safe_run_cmd(mock_run):
        """query_sign_ext 通过 safe_run_cmd 调用 --print-sign-ext flag。"""
        mock_run.return_value = (True, ".p7s\n")
        add_header_sign.query_sign_ext("/fake/script.py")
        args, _ = mock_run.call_args
        assert args[0] == ["python3", "/fake/script.py", "--print-sign-ext"]

    @staticmethod
    @mock.patch("add_header_sign.safe_run_cmd")
    def test_query_certtype_calls_safe_run_cmd(mock_run):
        """query_certtype 通过 safe_run_cmd 调用 --print-certtype flag。"""
        mock_run.return_value = (True, "0x1\n")
        add_header_sign.query_certtype("/fake/script.py")
        args, _ = mock_run.call_args
        assert args[0] == ["python3", "/fake/script.py", "--print-certtype"]

    @staticmethod
    @mock.patch("add_header_sign.safe_run_cmd")
    def test_query_certtype_invalid_hex_fallback(mock_run):
        """query_certtype 非法十六进制字符串回退默认 1。"""
        for invalid_val in ["abc", "0xGG", "xyz"]:
            mock_run.return_value = (True, invalid_val + "\n")
            assert add_header_sign.query_certtype("/fake/script.py") == "1"

    @staticmethod
    @mock.patch("add_header_sign.safe_run_cmd")
    def test_query_sign_ext_no_dot_prefix_fallback(mock_run):
        """query_sign_ext 返回无点号前缀（如 'p7s'）时回退默认 .p7s。"""
        mock_run.return_value = (True, "p7s\n")
        assert add_header_sign.query_sign_ext("/fake/script.py") == ".p7s"

    @staticmethod
    @mock.patch("add_header_sign.safe_run_cmd")
    def test_query_sign_attr_passes_timeout(mock_run):
        """query_sign_attr 传入 QUERY_TIMEOUT 超时参数。"""
        mock_run.return_value = (True, ".p7s\n")
        add_header_sign.query_sign_attr("/fake/script.py", "--print-sign-ext")
        _, kwargs = mock_run.call_args
        assert kwargs.get("timeout") == add_header_sign.QUERY_TIMEOUT


# ===================== build_image_pack_cmd =====================


class TestBuildImagePackCmd:
    """image_pack 命令构建。"""

    SCRIPT = "/bios/image_pack.py"

    def test_no_sign_basic_cmd(self):
        """流程1：不签名时命令含基础参数，不含 cms 相关参数。"""
        conf = make_conf(input="a.bin", type="", tag="ta", version="1.0", nvcnt="3")
        cmd = add_header_sign.build_image_pack_cmd(
            make_image_pack_param(conf, add_sign="false")
        )
        assert cmd[0] == "python3"
        assert self.SCRIPT in cmd
        assert "-raw_img" in cmd
        assert "-out_img" in cmd
        assert "-version" in cmd
        assert "-nvcnt" in cmd
        assert "-tag" in cmd
        assert "--addcms" not in cmd
        assert "-cms" not in cmd

    def test_no_sign_with_position(self):
        """流程1：有 position 时命令含 -position。"""
        conf = make_conf(
            input="a.bin",
            type="",
            tag="ta",
            version="1.0",
            nvcnt="3",
            position="before_header",
        )
        cmd = add_header_sign.build_image_pack_cmd(
            make_image_pack_param(conf, add_sign="false")
        )
        assert "-position" in cmd
        idx = cmd.index("-position")
        assert cmd[idx + 1] == "before_header"

    def test_sign_cms_cmd(self):
        """流程2：签名模式 cms 类型命令含 p7s/ini/crl/certtype。"""
        conf = make_conf(input="a.bin", type="cms", tag="ta", version="1.0", nvcnt="3")
        cmd = add_header_sign.build_image_pack_cmd(make_image_pack_param(conf))
        assert "--addcms" in cmd
        assert "-cms" in cmd
        assert "-ini" in cmd
        assert "-crl" in cmd
        assert "-certtype" in cmd
        idx = cmd.index("-certtype")
        assert cmd[idx + 1] == "1"

    def test_sign_cms_with_custom_ext(self):
        """流程2：sign_ext='.cms' 时 -cms 路径以 .cms 结尾。"""
        conf = make_conf(input="a.bin", type="cms", tag="ta", version="1.0", nvcnt="3")
        cmd = add_header_sign.build_image_pack_cmd(
            make_image_pack_param(conf, sign_ext=".cms", sign_certtype="4294967295")
        )
        idx = cmd.index("-cms")
        assert cmd[idx + 1].endswith(".ini.cms")
        assert not cmd[idx + 1].endswith(".ini.p7s")
        cert_idx = cmd.index("-certtype")
        assert cmd[cert_idx + 1] == "4294967295"

    def test_sign_cms_default_ext_and_certtype(self):
        """流程2：默认 sign_ext/sign_certtype 与改动前一致（回归保护）。"""
        conf = make_conf(input="a.bin", type="cms", tag="ta", version="1.0", nvcnt="3")
        cmd = add_header_sign.build_image_pack_cmd(make_image_pack_param(conf))
        idx = cmd.index("-cms")
        assert cmd[idx + 1].endswith(".ini.p7s")
        cert_idx = cmd.index("-certtype")
        assert cmd[cert_idx + 1] == "1"

    def test_sign_with_additional(self):
        """流程2：additional 参数被 shlex 拆分后加入命令。"""
        conf = make_conf(
            input="a.bin",
            type="cms",
            tag="ta",
            version="1.0",
            nvcnt="3",
            additional="-x 1 -y 2",
        )
        cmd = add_header_sign.build_image_pack_cmd(make_image_pack_param(conf))
        assert "-x" in cmd
        idx = cmd.index("-x")
        assert cmd[idx + 1] == "1"
        assert "-y" in cmd
        idx = cmd.index("-y")
        assert cmd[idx + 1] == "2"

    def test_sign_with_additional_quoted_spaces(self):
        """OPT-12: additional 含引号和空格时 shlex 正确拆分。"""
        conf = make_conf(
            input="a.bin",
            type="cms",
            tag="ta",
            version="1.0",
            nvcnt="3",
            additional='-x "hello world" -y bar',
        )
        cmd = add_header_sign.build_image_pack_cmd(make_image_pack_param(conf))
        assert "-x" in cmd
        idx = cmd.index("-x")
        assert cmd[idx + 1] == "hello world"
        assert "-y" in cmd
        idx = cmd.index("-y")
        assert cmd[idx + 1] == "bar"

    def test_returns_list(self):
        """返回值为 list。"""
        conf = make_conf(input="a.bin", type="", tag="ta", version="1.0")
        cmd = add_header_sign.build_image_pack_cmd(
            make_image_pack_param(conf, add_sign="false")
        )
        assert isinstance(cmd, list)

    def test_slash_split_type_no_duplicate_raw_img(self):
        """P2: type='rsa/cms' 时 -raw_img 只出现一次。"""
        conf = make_conf(
            input="a.bin", type="rsa/cms", tag="ta", version="1.0", nvcnt="3"
        )
        cmd = add_header_sign.build_image_pack_cmd(make_image_pack_param(conf))
        raw_img_count = cmd.count("-raw_img")
        assert raw_img_count == 1, f"expected 1 -raw_img, got {raw_img_count}"

    def test_slash_split_type_cms_params_present(self):
        """P2: type='rsa/cms' 时 cms 特有参数仍然存在。"""
        conf = make_conf(
            input="a.bin", type="rsa/cms", tag="ta", version="1.0", nvcnt="3"
        )
        cmd = add_header_sign.build_image_pack_cmd(make_image_pack_param(conf))
        assert "--addcms" in cmd
        assert "-cms" in cmd
        assert "-ini" in cmd
        assert "-crl" in cmd


# ===================== add_bios_header =====================


class TestAddBiosHeader:
    """编排流程。"""

    @staticmethod
    @mock.patch("add_header_sign.convert_der_file", return_value=True)
    @mock.patch("add_header_sign.safe_run_cmd", return_value=(True, ""))
    def test_full_flow_no_sign(mock_run, mock_der, tmp_path):
        """add_sign != true: 只加头不签名。"""
        root_dir, bios_tool, sign_dir, item_set = setup_bios_env(
            tmp_path, type="", tag="ta", version="1.0", nvcnt="3"
        )
        result = add_header_sign.add_bios_header(
            make_bios_header_param_from_env(
                root_dir, bios_tool, sign_dir, item_set, add_sign="false"
            )
        )
        assert result is True

    @staticmethod
    @mock.patch("add_header_sign.convert_der_file", return_value=True)
    @mock.patch("add_header_sign.safe_run_cmd", return_value=(True, ""))
    def test_full_flow_with_sign(mock_run, mock_der, tmp_path):
        """add_sign == true: 完整签名+加头流程。"""
        root_dir, bios_tool, sign_dir, item_set = setup_bios_env(
            tmp_path,
            with_ini_gen=True,
            file_content=b"data_a",
            type="cms",
            tag="ta",
            version="1.0",
            nvcnt="3",
        )
        result = add_header_sign.add_bios_header(
            make_bios_header_param_from_env(root_dir, bios_tool, sign_dir, item_set)
        )
        assert result is True

    @staticmethod
    @mock.patch("add_header_sign.convert_der_file", return_value=True)
    @mock.patch("add_header_sign.safe_run_cmd")
    def test_query_ext_certtype_passed_to_image_pack(mock_run, mock_der, tmp_path):
        """add_sign=true 时查询 ext/certtype 并传给 image_pack 命令。

        safe_run_cmd 的 side_effect 按调用顺序区分：
        esbc_header → ini_gen → 查询 ext → 查询 certtype → sign → image_pack。
        """
        root_dir, bios_tool, sign_dir, item_set = setup_bios_env(
            tmp_path,
            with_ini_gen=True,
            file_content=b"data_a",
            type="cms",
            tag="ta",
            version="1.0",
            nvcnt="3",
        )

        # 按调用顺序：esbc_header, ini_gen, query_sign_ext, query_certtype, sign, image_pack
        mock_run.side_effect = [
            (True, ""),  # esbc_header
            (True, ""),  # ini_gen
            (True, ".cms\n"),  # query_sign_ext
            (True, "0xFFFFFFFF\n"),  # query_certtype
            (True, ""),  # build_sign
            (True, ""),  # image_pack
        ]
        result = add_header_sign.add_bios_header(
            make_bios_header_param_from_env(root_dir, bios_tool, sign_dir, item_set)
        )
        assert result is True

        # image_pack 命令（最后一次调用）应使用查询到的 .cms 扩展名和 0xFFFFFFFF 证书
        image_pack_cmd = mock_run.call_args_list[-1].args[0]
        cms_idx = image_pack_cmd.index("-cms")
        assert image_pack_cmd[cms_idx + 1].endswith(".ini.cms")
        cert_idx = image_pack_cmd.index("-certtype")
        assert image_pack_cmd[cert_idx + 1] == "4294967295"

    @staticmethod
    def test_sign_true_no_cms_skips_der_conversion(tmp_path):
        """P0: add_sign=true + type='' (无 cms) 时不执行 der 转换，流程成功。"""
        with patched_bios_header_deps() as mocks:
            sign_dir = tmp_path / "sign"
            sign_dir.mkdir()
            (sign_dir / "a.bin").touch()
            conf = make_conf(input="a.bin", type="", tag="ta", version="1.0")
            item_set = {"a.bin": conf}
            result = add_header_sign.add_bios_header(
                make_bios_header_param(item_set, sign_dir, tmp_path)
            )
            assert result is True
            mocks["safe_run_cmd"].assert_called()
            mocks["convert_der_file"].assert_not_called()

    @staticmethod
    def test_crl_file_path_custom_der_conversion(tmp_path):
        """CRL_FILE_PATH 自定义路径时 der 转换从该路径读取 CRL。"""
        custom_crl = tmp_path / "custom.crl"
        custom_crl.write_text("fake crl")
        sign_dir = tmp_path / "sign"
        sign_dir.mkdir()
        (sign_dir / "a.bin").touch()
        conf = make_conf(input="a.bin", type="cms", tag="ta", version="1.0", nvcnt="3")
        item_set = {"a.bin": conf}
        with patched_bios_header_deps() as mocks:
            old = os.environ.get("CRL_FILE_PATH")
            os.environ["CRL_FILE_PATH"] = str(custom_crl)
            try:
                result = add_header_sign.add_bios_header(
                    make_bios_header_param(
                        item_set,
                        sign_dir,
                        tmp_path,
                        sign_tool_path="/sign_tool.py",
                    )
                )
                assert result is True
                mocks["convert_der_file"].assert_called_once()
                call_args = mocks["convert_der_file"].call_args[0]
                assert str(custom_crl) in call_args[0]
            finally:
                if old is not None:
                    os.environ["CRL_FILE_PATH"] = old
                else:
                    os.environ.pop("CRL_FILE_PATH", None)

    @staticmethod
    def test_crl_file_path_nonexistent_falls_back(tmp_path):
        """CRL_FILE_PATH 指向不存在文件时回退到 sign_file_dir/SWSCRL.crl（每目标隔离）。"""
        sign_dir = tmp_path / "sign"
        sign_dir.mkdir()
        (sign_dir / "a.bin").touch()
        default_crl = sign_dir / "SWSCRL.crl"
        default_crl.write_text("default crl")
        sign_tool_dir = tmp_path / "sign_tools"
        sign_tool_dir.mkdir()
        conf = make_conf(input="a.bin", type="cms", tag="ta", version="1.0", nvcnt="3")
        item_set = {"a.bin": conf}
        with patched_bios_header_deps() as mocks:
            old = os.environ.get("CRL_FILE_PATH")
            os.environ["CRL_FILE_PATH"] = "/nonexistent/custom.crl"
            try:
                result = add_header_sign.add_bios_header(
                    make_bios_header_param(
                        item_set,
                        sign_dir,
                        tmp_path,
                        sign_tool_path=str(sign_tool_dir / "sign_tool.py"),
                    )
                )
                assert result is True
                mocks["convert_der_file"].assert_called_once()
                call_args = mocks["convert_der_file"].call_args[0]
                assert str(default_crl) in call_args[0]
            finally:
                if old is not None:
                    os.environ["CRL_FILE_PATH"] = old
                else:
                    os.environ.pop("CRL_FILE_PATH", None)

    @staticmethod
    @mock.patch("add_header_sign.add_bios_esbc_header", return_value=False)
    def test_esbc_header_failure(mock_esbc, tmp_path):
        result = add_header_sign.add_bios_header(
            make_bios_header_param({}, tmp_path, tmp_path)
        )
        assert result is False

    @staticmethod
    @mock.patch("add_header_sign.add_bios_esbc_header", return_value=True)
    @mock.patch("add_header_sign.build_inifile", return_value=False)
    def test_inifile_failure(mock_ini, mock_esbc, tmp_path):
        result = add_header_sign.add_bios_header(
            make_bios_header_param({}, tmp_path, tmp_path)
        )
        assert result is False

    @staticmethod
    def test_sign_failure(tmp_path):
        """build_sign 失败时返回 False（需 has_cms=True 才触发 build_sign）。"""
        with patched_bios_header_deps(
            build_sign=False, query_sign_ext=".p7s", query_certtype="1"
        ):
            conf = make_conf(input="a.bin", type="cms", tag="ta", version="1.0")
            item_set = {"a.bin": conf}
            result = add_header_sign.add_bios_header(
                make_bios_header_param(item_set, tmp_path, tmp_path)
            )
            assert result is False

    @staticmethod
    @mock.patch("add_header_sign.add_bios_esbc_header", return_value=True)
    @mock.patch("add_header_sign.build_inifile", return_value=True)
    @mock.patch("add_header_sign.safe_run_cmd", return_value=(False, "err"))
    @mock.patch("add_header_sign.convert_der_file", return_value=True)
    def test_image_pack_failure(mock_der, mock_run, mock_ini, mock_esbc, tmp_path):
        conf = make_conf(input="a.bin", type="", tag="ta", version="1.0")
        item_set = {"a.bin": conf}
        sign_dir = tmp_path / "sign"
        sign_dir.mkdir()
        (sign_dir / "a.bin").touch()
        result = add_header_sign.add_bios_header(
            make_bios_header_param(
                item_set,
                sign_dir,
                tmp_path,
                add_sign="false",
            )
        )
        assert result is False

    @staticmethod
    @mock.patch("add_header_sign.add_bios_esbc_header", return_value=True)
    @mock.patch("add_header_sign.build_inifile", return_value=True)
    @mock.patch("add_header_sign.convert_der_file", return_value=False)
    @mock.patch("add_header_sign.safe_run_cmd", return_value=(True, ""))
    def test_der_conversion_failure_handled(
        mock_run, mock_der, mock_ini, mock_esbc, tmp_path
    ):
        """B2: der 转换失败时返回 False。"""
        conf = make_conf(input="a.bin", type="cms", tag="ta", version="1.0")
        item_set = {"a.bin": conf}
        sign_dir = tmp_path / "sign"
        sign_dir.mkdir()
        (sign_dir / "a.bin").touch()
        result = add_header_sign.add_bios_header(
            make_bios_header_param(item_set, sign_dir, tmp_path)
        )
        assert result is False

    @staticmethod
    @mock.patch("add_header_sign.shutil.rmtree")
    @mock.patch("add_header_sign.add_bios_esbc_header", return_value=True)
    @mock.patch("add_header_sign.build_inifile", return_value=True)
    def test_sign_tmp_preserved_on_failure(mock_ini, mock_esbc, mock_rmtree, tmp_path):
        """D2: 失败时 sign_tmp 保留不删除，以便定位问题。"""
        conf = make_conf(input="a.bin", type="cms", tag="ta", version="1.0")
        item_set = {"a.bin": conf}
        with ExitStack() as stack:
            stack.enter_context(
                mock.patch("add_header_sign.query_sign_ext", return_value=".p7s")
            )
            stack.enter_context(
                mock.patch("add_header_sign.query_certtype", return_value="1")
            )
            stack.enter_context(
                mock.patch("add_header_sign.build_sign", return_value=False)
            )
            stack.enter_context(
                mock.patch("add_header_sign.convert_der_file", return_value=True)
            )
            add_header_sign.add_bios_header(
                make_bios_header_param(item_set, tmp_path, tmp_path)
            )
        # 失败后 sign_tmp 应保留以便定位问题
        assert (tmp_path / "sign_tmp").is_dir(), (
            "sign_tmp should be preserved for debugging"
        )
        # 开始前无残留，rmtree 不应被调用
        mock_rmtree.assert_not_called()

    @staticmethod
    def test_sign_tmp_preserved_on_success(tmp_path):
        """D2: 成功后 sign_tmp 保留不删除。"""
        with (
            patched_bios_header_deps(),
            mock.patch("add_header_sign.shutil.rmtree") as mock_rmtree,
        ):
            conf = make_conf(input="a.bin", type="", tag="ta", version="1.0")
            item_set = {"a.bin": conf}
            sign_dir = tmp_path / "sign"
            sign_dir.mkdir()
            (sign_dir / "a.bin").touch()
            add_header_sign.add_bios_header(
                make_bios_header_param(
                    item_set,
                    sign_dir,
                    tmp_path,
                    add_sign="false",
                )
            )
            assert (sign_dir / "sign_tmp").is_dir(), (
                "sign_tmp should be preserved for debugging"
            )
            mock_rmtree.assert_not_called()

    @staticmethod
    def test_der_regenerated_when_crl_newer(tmp_path):
        """der 过期：crl 比 der 更新时重新转换。der/crl 现位于 sign_file_dir 下。"""
        sign_dir, sign_tool_dir, item_set = setup_der_test_env(
            tmp_path, der_newer=False
        )
        with patched_bios_header_deps() as mocks:
            add_header_sign.add_bios_header(
                make_bios_header_param(
                    item_set,
                    sign_dir,
                    tmp_path,
                    sign_tool_path=str(sign_tool_dir / "sign_tool.py"),
                )
            )
            mocks["convert_der_file"].assert_called_once()

    @staticmethod
    def test_der_not_regenerated_when_der_newer(tmp_path):
        """der 未过期：der 比 crl 更新时不重新转换。der/crl 现位于 sign_file_dir 下。"""
        sign_dir, sign_tool_dir, item_set = setup_der_test_env(tmp_path, der_newer=True)
        with patched_bios_header_deps() as mocks:
            add_header_sign.add_bios_header(
                make_bios_header_param(
                    item_set,
                    sign_dir,
                    tmp_path,
                    sign_tool_path=str(sign_tool_dir / "sign_tool.py"),
                )
            )
            mocks["convert_der_file"].assert_not_called()


# ===================== check_params =====================


class TestCheckParams:
    """参数校验。"""

    @staticmethod
    def test_all_exist(tmp_path):
        cfg = tmp_path / "cfg.xml"
        cfg.touch()
        bios_tool = tmp_path / "bios_tool"
        bios_tool.mkdir()
        sign_tool = tmp_path / "sign_tool.py"
        sign_tool.touch()
        sign_dir = tmp_path / "sign"
        sign_dir.mkdir()
        result = add_header_sign.check_params(
            {
                "config_file": str(cfg),
                "bios_tool_path": str(bios_tool),
                "sign_file_dir": str(sign_dir),
                "sgn_tool_path": str(sign_tool),
                "add_sign": "true",
            }
        )
        assert result is True

    @staticmethod
    def test_config_file_missing(tmp_path):
        bios_tool = tmp_path / "bios_tool"
        bios_tool.mkdir()
        sign_tool = tmp_path / "sign_tool.py"
        sign_tool.touch()
        sign_dir = tmp_path / "sign"
        sign_dir.mkdir()
        result = add_header_sign.check_params(
            {
                "config_file": "/nonexistent/cfg.xml",
                "bios_tool_path": str(bios_tool),
                "sign_file_dir": str(sign_dir),
                "sgn_tool_path": str(sign_tool),
                "add_sign": "true",
            }
        )
        assert result is False

    @staticmethod
    def test_bios_tool_missing(tmp_path):
        cfg = tmp_path / "cfg.xml"
        cfg.touch()
        sign_tool = tmp_path / "sign_tool.py"
        sign_tool.touch()
        sign_dir = tmp_path / "sign"
        sign_dir.mkdir()
        result = add_header_sign.check_params(
            {
                "config_file": str(cfg),
                "bios_tool_path": "/nonexistent/tool",
                "sign_file_dir": str(sign_dir),
                "sgn_tool_path": str(sign_tool),
                "add_sign": "true",
            }
        )
        assert result is False

    @staticmethod
    def test_sign_tool_missing(tmp_path):
        cfg = tmp_path / "cfg.xml"
        cfg.touch()
        bios_tool = tmp_path / "bios_tool"
        bios_tool.mkdir()
        sign_dir = tmp_path / "sign"
        sign_dir.mkdir()
        result = add_header_sign.check_params(
            {
                "config_file": str(cfg),
                "bios_tool_path": str(bios_tool),
                "sign_file_dir": str(sign_dir),
                "sgn_tool_path": "/nonexistent/sign_tool.py",
                "add_sign": "true",
            }
        )
        assert result is False

    @staticmethod
    def test_sign_file_dir_missing(tmp_path):
        """D5: sign_file_dir 不存在时返回 False。"""
        cfg = tmp_path / "cfg.xml"
        cfg.touch()
        bios_tool = tmp_path / "bios_tool"
        bios_tool.mkdir()
        sign_tool = tmp_path / "sign_tool.py"
        sign_tool.touch()
        result = add_header_sign.check_params(
            {
                "config_file": str(cfg),
                "bios_tool_path": str(bios_tool),
                "sign_file_dir": "/nonexistent/sign_dir",
                "sgn_tool_path": str(sign_tool),
                "add_sign": "true",
            }
        )
        assert result is False

    @staticmethod
    def test_sign_file_dir_not_a_dir(tmp_path):
        """D5: sign_file_dir 是文件而非目录时返回 False。"""
        cfg = tmp_path / "cfg.xml"
        cfg.touch()
        bios_tool = tmp_path / "bios_tool"
        bios_tool.mkdir()
        sign_tool = tmp_path / "sign_tool.py"
        sign_tool.touch()
        not_a_dir = tmp_path / "not_a_dir"
        not_a_dir.touch()
        result = add_header_sign.check_params(
            {
                "config_file": str(cfg),
                "bios_tool_path": str(bios_tool),
                "sign_file_dir": str(not_a_dir),
                "sgn_tool_path": str(sign_tool),
                "add_sign": "true",
            }
        )
        assert result is False


# ===================== define_parser =====================


class TestDefineParser:
    """命令行参数解析。"""

    @staticmethod
    def test_required_sign_file_dir():
        parser = add_header_sign.define_parser()
        args = parser.parse_args(["/tmp/sign", "true"])
        assert args.sign_file_dir == "/tmp/sign"
        assert args.sign_flag == "true"

    @staticmethod
    def test_sign_flag_default():
        parser = add_header_sign.define_parser()
        args = parser.parse_args(["/tmp/sign"])
        assert args.sign_flag == "false"

    @staticmethod
    def test_bios_check_cfg_default():
        parser = add_header_sign.define_parser()
        args = parser.parse_args(["/tmp/sign", "true"])
        assert args.bios_check_cfg == "bios_check_cfg.xml"

    @staticmethod
    def test_bios_check_cfg_custom():
        parser = add_header_sign.define_parser()
        args = parser.parse_args(
            ["/tmp/sign", "true", "--bios_check_cfg", "custom.xml"]
        )
        assert args.bios_check_cfg == "custom.xml"

    @staticmethod
    def test_version():
        parser = add_header_sign.define_parser()
        args = parser.parse_args(["/tmp/sign", "true", "--version", "2.0"])
        assert args.version == "2.0"

    @staticmethod
    def test_sign_script_default_empty():
        parser = add_header_sign.define_parser()
        args = parser.parse_args(["/tmp/sign", "true"])
        assert args.sign_script == ""

    @staticmethod
    def test_sign_script_custom():
        parser = add_header_sign.define_parser()
        args = parser.parse_args(
            ["/tmp/sign", "true", "--sign_script", "/custom/sign.py"]
        )
        assert args.sign_script == "/custom/sign.py"

    @staticmethod
    def test_missing_sign_file_dir_exits():
        parser = add_header_sign.define_parser()
        with mock.patch.object(parser, "error") as mock_error:
            parser.parse_args([])
        mock_error.assert_called_once()

    @staticmethod
    def test_invalid_sign_flag_exits():
        """D8: sign_flag 只接受 true/false。"""
        parser = add_header_sign.define_parser()
        with mock.patch.object(parser, "error") as mock_error:
            parser.parse_args(["/tmp/sign", "yes"])
        mock_error.assert_called()


# ===================== setenv =====================


class TestSetenv:
    """环境变量设置。"""

    @staticmethod
    def test_hi_python_not_set_uses_full_path():
        """B8: HI_PYTHON 未设置时应使用 sys.executable 全路径。"""
        old_env = os.environ.pop("HI_PYTHON", None)
        old_mod = add_header_sign.HI_PYTHON
        try:
            add_header_sign.setenv()
            assert os.environ["HI_PYTHON"] == sys.executable
            assert add_header_sign.HI_PYTHON == sys.executable
        finally:
            if old_env is not None:
                os.environ["HI_PYTHON"] = old_env
            else:
                os.environ.pop("HI_PYTHON", None)
            add_header_sign.HI_PYTHON = old_mod

    @staticmethod
    def test_hi_python_already_set_preserved():
        old_env = os.environ.get("HI_PYTHON")
        old_mod = add_header_sign.HI_PYTHON
        os.environ["HI_PYTHON"] = "/custom/python3"
        try:
            add_header_sign.setenv()
            assert os.environ["HI_PYTHON"] == "/custom/python3"
            assert add_header_sign.HI_PYTHON == "/custom/python3"
        finally:
            if old_env is not None:
                os.environ["HI_PYTHON"] = old_env
            else:
                os.environ.pop("HI_PYTHON", None)
            add_header_sign.HI_PYTHON = old_mod


# ===================== main =====================


class TestMain:
    """入口函数。"""

    @staticmethod
    def test_sign_flag_false_returns_true():
        """sign_flag=false 时直接返回 True。"""
        assert run_main_with_mock_parser(sign_flag="false") is True

    @staticmethod
    @mock.patch("add_header_sign.add_bios_header", return_value=True)
    @mock.patch("add_header_sign.setenv")
    @mock.patch("add_header_sign.check_params", return_value=True)
    @mock.patch("add_header_sign.get_item_set")
    def test_sign_flag_true_success(mock_get, mock_check, mock_setenv, mock_header):
        mock_get.return_value = (True, {})
        assert run_main_with_mock_parser() is True

    @staticmethod
    @mock.patch("add_header_sign.check_params", return_value=False)
    def test_check_params_failure(mock_check):
        assert run_main_with_mock_parser() is False

    @staticmethod
    @mock.patch("add_header_sign.check_params", return_value=True)
    @mock.patch("add_header_sign.get_item_set")
    def test_get_item_set_failure(mock_get, mock_check):
        mock_get.return_value = (False, None)
        assert run_main_with_mock_parser() is False

    @staticmethod
    @mock.patch("add_header_sign.add_bios_header", return_value=False)
    @mock.patch("add_header_sign.setenv")
    @mock.patch("add_header_sign.check_params", return_value=True)
    @mock.patch("add_header_sign.get_item_set")
    def test_add_bios_header_failure(mock_get, mock_check, mock_setenv, mock_header):
        mock_get.return_value = (True, {})
        assert run_main_with_mock_parser() is False

    @staticmethod
    @mock.patch("add_header_sign.add_bios_header", return_value=True)
    @mock.patch("add_header_sign.setenv")
    @mock.patch("add_header_sign.check_params", return_value=True)
    @mock.patch("add_header_sign.get_item_set")
    def test_custom_sign_script_used(mock_get, mock_check, mock_setenv, mock_header):
        mock_get.return_value = (True, {})
        run_main_with_mock_parser(sign_script="/custom/sign.py")
        call_args = mock_header.call_args
        assert "/custom/sign.py" in str(call_args)

    @staticmethod
    @mock.patch("add_header_sign.add_bios_header", return_value=True)
    @mock.patch("add_header_sign.setenv")
    @mock.patch("add_header_sign.check_params", return_value=True)
    @mock.patch("add_header_sign.get_item_set")
    def test_real_parser_sign_flag_true(mock_get, mock_check, mock_setenv, mock_header):
        """端到端：用真实 define_parser 解析 argv，验证 main 与 parser 集成。"""
        mock_get.return_value = (True, {})
        result = add_header_sign.main(["add_header_sign.py", "/tmp/sign", "true"])
        assert result is True
        mock_header.assert_called_once()

    @staticmethod
    @mock.patch("add_header_sign.add_bios_header", return_value=True)
    @mock.patch("add_header_sign.setenv")
    @mock.patch("add_header_sign.check_params", return_value=True)
    @mock.patch("add_header_sign.get_item_set")
    def test_real_parser_sign_flag_false(
        mock_get, mock_check, mock_setenv, mock_header
    ):
        """sign_flag=false 时用真实 parser 也应直接返回 True，不调用 add_bios_header。"""
        result = add_header_sign.main(["add_header_sign.py", "/tmp/sign", "false"])
        assert result is True
        mock_header.assert_not_called()

    @staticmethod
    def test_real_parser_help_returns_true():
        """--help 用真实 parser 时 SystemExit(0) 被捕获，返回 True。"""
        result = add_header_sign.main(["add_header_sign.py", "--help"])
        assert result is True

    @staticmethod
    def test_real_parser_invalid_flag_returns_false():
        """非法参数用真实 parser 时 SystemExit(2) 被捕获，返回 False。"""
        result = add_header_sign.main(["add_header_sign.py", "--unknown-flag"])
        assert result is False


# ===================== 并发隔离（方案 C） =====================


class TestConcurrencyIsolation:
    """并发签名隔离验证。

    方案 C 要求所有中间产物路径基于 sign_file_dir 推导，每目标独立，
    无任何路径在 root_dir 或 SCRIPT_DIR 下共享。
    """

    @staticmethod
    @mock.patch("add_header_sign.convert_der_file", return_value=True)
    @mock.patch("add_header_sign.safe_run_cmd", return_value=(True, ""))
    def test_sign_tmp_under_sign_file_dir(mock_run, mock_der, tmp_path):
        """sign_tmp 位于 sign_file_dir 下而非 root_dir 下。"""
        root_dir, bios_tool, sign_dir, item_set = setup_bios_env(
            tmp_path, type="", tag="ta", version="1.0", nvcnt="3"
        )
        add_header_sign.add_bios_header(
            make_bios_header_param_from_env(
                root_dir, bios_tool, sign_dir, item_set, add_sign="false"
            )
        )
        sign_tmp = sign_dir / "sign_tmp"
        assert sign_tmp.is_dir(), "sign_tmp should be under sign_file_dir"
        assert not (tmp_path / "sign_tmp").exists(), (
            "sign_tmp must not be created under root_dir"
        )

    @staticmethod
    @mock.patch("add_header_sign.convert_der_file", return_value=True)
    @mock.patch("add_header_sign.safe_run_cmd", return_value=(True, ""))
    def test_sign_tmp_not_in_root_dir(mock_run, mock_der, tmp_path):
        """root_dir 下不存在 sign_tmp 目录。"""
        root_dir, bios_tool, sign_dir, item_set = setup_bios_env(
            tmp_path, type="", tag="ta", version="1.0"
        )
        add_header_sign.add_bios_header(
            make_bios_header_param_from_env(
                root_dir, bios_tool, sign_dir, item_set, add_sign="false"
            )
        )
        assert not (tmp_path / "sign_tmp").exists()

    @staticmethod
    def test_der_under_sign_file_dir(tmp_path):
        """DER 文件位于 sign_file_dir 下。"""
        sign_dir = tmp_path / "sign"
        sign_dir.mkdir()
        (sign_dir / "a.bin").touch()
        conf = make_conf(input="a.bin", type="cms", tag="ta", version="1.0", nvcnt="3")
        item_set = {"a.bin": conf}
        with patched_bios_header_deps() as mocks:
            add_header_sign.add_bios_header(
                make_bios_header_param(
                    item_set,
                    sign_dir,
                    tmp_path,
                    sign_tool_path="/sign_tool.py",
                )
            )
            der_call_args = mocks["convert_der_file"].call_args[0]
            der_path = der_call_args[1]
            assert der_path.startswith(str(sign_dir)), (
                "der_file should be under sign_file_dir, got: %s" % der_path
            )
            assert der_path.endswith("SWSCRL.der")

    @staticmethod
    def test_crl_under_sign_file_dir_when_no_env(tmp_path):
        """未设 CRL_FILE_PATH 时 CRL 回退路径位于 sign_file_dir 下。"""
        sign_dir = tmp_path / "sign"
        sign_dir.mkdir()
        (sign_dir / "a.bin").touch()
        conf = make_conf(input="a.bin", type="cms", tag="ta", version="1.0", nvcnt="3")
        item_set = {"a.bin": conf}
        with patched_bios_header_deps() as mocks:
            old = os.environ.pop("CRL_FILE_PATH", None)
            try:
                crl_path = run_bios_header_and_get_crl_path(
                    mocks, item_set, sign_dir, tmp_path
                )
                assert crl_path.startswith(str(sign_dir)), (
                    "crl_file should be under sign_file_dir, got: %s" % crl_path
                )
                assert crl_path.endswith("SWSCRL.crl")
            finally:
                if old is not None:
                    os.environ["CRL_FILE_PATH"] = old

    @staticmethod
    def test_crl_reuse_env_when_set(tmp_path):
        """设 CRL_FILE_PATH 时 CRL 路径为环境变量值，不使用 sign_file_dir。"""
        sign_dir = tmp_path / "sign"
        sign_dir.mkdir()
        (sign_dir / "a.bin").touch()
        custom_crl = tmp_path / "custom.crl"
        custom_crl.write_text("crl")
        conf = make_conf(input="a.bin", type="cms", tag="ta", version="1.0", nvcnt="3")
        item_set = {"a.bin": conf}
        with patched_bios_header_deps() as mocks:
            old = os.environ.get("CRL_FILE_PATH")
            os.environ["CRL_FILE_PATH"] = str(custom_crl)
            try:
                crl_path = run_bios_header_and_get_crl_path(
                    mocks, item_set, sign_dir, tmp_path
                )
                assert crl_path == str(custom_crl)
            finally:
                if old is not None:
                    os.environ["CRL_FILE_PATH"] = old
                else:
                    os.environ.pop("CRL_FILE_PATH", None)

    @staticmethod
    @mock.patch("add_header_sign.shutil.rmtree")
    @mock.patch("add_header_sign.convert_der_file", return_value=True)
    @mock.patch("add_header_sign.safe_run_cmd", return_value=(True, ""))
    def test_sign_tmp_unique_per_call(mock_run, mock_der, mock_rmtree, tmp_path):
        """不同 sign_file_dir 产生不同的 sign_tmp 目录。"""
        root_dir, bios_tool, _, _ = setup_bios_env(tmp_path)

        sign_dir_a = tmp_path / "sign_a"
        sign_dir_a.mkdir()
        (sign_dir_a / "a.bin").touch()
        sign_dir_b = tmp_path / "sign_b"
        sign_dir_b.mkdir()
        (sign_dir_b / "b.bin").touch()

        conf_a = make_conf(input="a.bin", type="", tag="ta", version="1.0")
        conf_b = make_conf(input="b.bin", type="", tag="tb", version="1.0")

        add_header_sign.add_bios_header(
            make_bios_header_param_from_env(
                root_dir, bios_tool, sign_dir_a, {"a.bin": conf_a}, add_sign="false"
            )
        )
        add_header_sign.add_bios_header(
            make_bios_header_param_from_env(
                root_dir, bios_tool, sign_dir_b, {"b.bin": conf_b}, add_sign="false"
            )
        )

        # 两个 sign_tmp 目录应分别存在且路径不同
        assert (sign_dir_a / "sign_tmp").is_dir()
        assert (sign_dir_b / "sign_tmp").is_dir()
        assert str(sign_dir_a / "sign_tmp") != str(sign_dir_b / "sign_tmp")
        # 开始前无残留，rmtree 不应被调用
        mock_rmtree.assert_not_called()

    @staticmethod
    @mock.patch("add_header_sign.convert_der_file", return_value=True)
    @mock.patch("add_header_sign.safe_run_cmd", return_value=(True, ""))
    def test_sign_tmp_leftover_cleaned_at_start(mock_run, mock_der, tmp_path):
        """开始前删除上次残留的 sign_tmp 目录及内部文件。

        注意：此处刻意不 mock shutil.rmtree，以验证真实的残留目录删除行为。
        """
        root_dir, bios_tool, sign_dir, item_set = setup_bios_env(
            tmp_path, type="", tag="ta", version="1.0"
        )

        # 预先创建残留的 sign_tmp 目录及内部文件
        leftover = sign_dir / "sign_tmp"
        leftover.mkdir()
        (leftover / "stale.ini").touch()

        add_header_sign.add_bios_header(
            make_bios_header_param_from_env(
                root_dir, bios_tool, sign_dir, item_set, add_sign="false"
            )
        )
        # 残留文件应已被删除，sign_tmp 被 makedirs 重新创建
        assert leftover.is_dir(), "sign_tmp should be recreated after cleanup"
        assert not (leftover / "stale.ini").exists(), "stale file should be removed"

    @staticmethod
    def test_no_shared_paths_in_root_or_script_dir(tmp_path):
        """整改后 root_dir 和 sign_tool 目录下均不产生 SWSCRL.crl/SWSCRL.der/sign_tmp。"""
        root_dir = tmp_path
        sign_dir = tmp_path / "sign"
        sign_dir.mkdir()
        (sign_dir / "a.bin").touch()
        sign_tool_dir = tmp_path / "sign_tools"
        sign_tool_dir.mkdir()
        conf = make_conf(input="a.bin", type="cms", tag="ta", version="1.0", nvcnt="3")
        item_set = {"a.bin": conf}
        with patched_bios_header_deps():
            old = os.environ.pop("CRL_FILE_PATH", None)
            try:
                add_header_sign.add_bios_header(
                    make_bios_header_param(
                        item_set,
                        sign_dir,
                        root_dir,
                        sign_tool_path=str(sign_tool_dir / "sign_tool.py"),
                    )
                )
            finally:
                if old is not None:
                    os.environ["CRL_FILE_PATH"] = old
        # root_dir 下不应有 sign_tmp / SWSCRL.crl / SWSCRL.der
        assert not (root_dir / "sign_tmp").exists()
        assert not (root_dir / "SWSCRL.crl").exists()
        assert not (root_dir / "SWSCRL.der").exists()
        # sign_tool 目录下也不应有共享文件
        assert not (sign_tool_dir / "SWSCRL.crl").exists()
        assert not (sign_tool_dir / "SWSCRL.der").exists()

    @staticmethod
    @mock.patch("add_header_sign.shutil.rmtree")
    @mock.patch("add_header_sign.convert_der_file", return_value=True)
    @mock.patch("add_header_sign.safe_run_cmd", return_value=(True, ""))
    def test_concurrent_add_bios_header(mock_run, mock_der, mock_rmtree, tmp_path):
        """T1: 两个 add_bios_header 并行执行不互相干扰。

        用 ThreadPoolExecutor 并发调用两个不同 sign_file_dir 的签名流程，
        验证无 Python 层共享状态冲突（如模块级变量被改写）。
        """
        from concurrent.futures import ThreadPoolExecutor

        root_dir, bios_tool, _, _ = setup_bios_env(tmp_path)

        sign_dir_a = tmp_path / "sign_a"
        sign_dir_a.mkdir()
        (sign_dir_a / "a.bin").touch()
        sign_dir_b = tmp_path / "sign_b"
        sign_dir_b.mkdir()
        (sign_dir_b / "b.bin").touch()

        conf_a = make_conf(input="a.bin", type="", tag="ta", version="1.0", nvcnt="3")
        conf_b = make_conf(input="b.bin", type="", tag="tb", version="1.0", nvcnt="3")

        def call_a():
            return add_header_sign.add_bios_header(
                make_bios_header_param_from_env(
                    root_dir, bios_tool, sign_dir_a, {"a.bin": conf_a}, add_sign="false"
                )
            )

        def call_b():
            return add_header_sign.add_bios_header(
                make_bios_header_param_from_env(
                    root_dir, bios_tool, sign_dir_b, {"b.bin": conf_b}, add_sign="false"
                )
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_a = executor.submit(call_a)
            future_b = executor.submit(call_b)
            result_a = future_a.result()
            result_b = future_b.result()

        assert result_a is True
        assert result_b is True
        # 两个调用各自的 sign_tmp 都应保留
        assert (sign_dir_a / "sign_tmp").is_dir()
        assert (sign_dir_b / "sign_tmp").is_dir()
        # 开始前无残留，rmtree 不应被调用
        mock_rmtree.assert_not_called()

    @staticmethod
    @mock.patch("add_header_sign.convert_der_file", return_value=True)
    @mock.patch("add_header_sign.add_bios_esbc_header", return_value=True)
    @mock.patch("add_header_sign.build_inifile", return_value=True)
    @mock.patch("add_header_sign.safe_run_cmd", return_value=(True, ""))
    def test_crl_dir_consistent_with_crl_file_path(
        mock_run, mock_ini, mock_esbc, mock_der, tmp_path
    ):
        """T2: build_sign 传给 community_sign_build 的 --crl-dir 值与
        add_bios_header 中 crl_file 的父目录一致（跨脚本 CRL 路径一致性）。

        mock safe_run_cmd 捕获 build_sign 发起的 community_sign_build 子进程命令，
        断言 --crl-dir 值等于 add_bios_header 中 crl_file 的父目录。
        """
        sign_dir = tmp_path / "sign"
        sign_dir.mkdir()
        (sign_dir / "a.bin").touch()
        sign_tool_dir = tmp_path / "sign_tools"
        sign_tool_dir.mkdir()
        conf = make_conf(input="a.bin", type="cms", tag="ta", version="1.0", nvcnt="3")
        item_set = {"a.bin": conf}
        old = os.environ.pop("CRL_FILE_PATH", None)
        try:
            add_header_sign.add_bios_header(
                make_bios_header_param(
                    item_set,
                    sign_dir,
                    tmp_path,
                    sign_tool_path=str(sign_tool_dir / "sign_tool.py"),
                )
            )
        finally:
            if old is not None:
                os.environ["CRL_FILE_PATH"] = old

        # 从 safe_run_cmd 的调用中找到 build_sign 发起的签名命令（含 --crl-dir，排除查询命令）
        sign_cmd_call = None
        for call in mock_run.call_args_list:
            cmd = call.args[0]
            if "--crl-dir" in str(cmd):
                sign_cmd_call = cmd
                break

        assert sign_cmd_call is not None, "build_sign 的命令未被捕获"
        # --crl-dir 的值应等于 sign_file_dir（即 crl_file 的父目录）
        crl_dir_idx = sign_cmd_call.index("--crl-dir")
        crl_dir_value = sign_cmd_call[crl_dir_idx + 1]
        assert crl_dir_value == str(sign_dir)
        # crl_file 回退路径应为 sign_dir/SWSCRL.crl，父目录即 sign_dir
        assert str(sign_dir) in crl_dir_value
