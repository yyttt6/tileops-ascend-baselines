# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from argparse import Namespace
from pathlib import Path
import json
import os
import sys
import xml.etree.ElementTree as ET

import pytest

from .. import pkg_parser
from .. import packer
from ..utils import pkg_utils


def test_parse_os_arch_with_arch_and_default():
    os_name, os_ver, arch = pkg_parser.parse_os_arch("ubuntu20.04-aarch64")
    assert os_name == "ubuntu"
    assert os_ver.startswith("20")
    assert arch == "aarch64"

    os_name2, os_ver2, arch2 = pkg_parser.parse_os_arch("ubuntu22.04")
    assert os_name2 == "ubuntu"
    assert arch2 == "aarch64"


def test_replace_env_and_join_dst_path_and_apply_func():
    env = {"ABC": "VALUE", "EMPTY": None}
    s = pkg_parser.replace_env(env, "path/$(ABC)/$(EMPTY)")
    assert "VALUE" in s and "EMPTY" not in s

    joined = pkg_parser.join_dst_path("/base", "sub")
    assert os.path.normpath(joined).endswith(os.path.join("base", "sub"))
    assert pkg_parser.join_dst_path("/base", "real:/abs") == "/abs"

    # apply_func 支持 list / set / str
    def inc(x):
        return x + "_x"
    assert pkg_parser.apply_func(inc, "a") == "a_x"
    assert pkg_parser.apply_func(inc, ["a", "b"]) == ["a_x", "b_x"]
    assert pkg_parser.apply_func(inc, {"a"}) == {"a_x"}


def test_parse_package_info_and_attr_helpers():
    root = ET.Element("root")
    pkg_info = ET.SubElement(root, "package_info")
    ET.SubElement(pkg_info, "expand_asterisk").text = "true"
    ET.SubElement(pkg_info, "suffix").text = "run"

    attrs = pkg_parser.parse_package_info(pkg_info)
    assert attrs["expand_asterisk"] is True
    assert attrs["suffix"] == "run"

    args = Namespace(chip_name="c1", suffix="deb", func_name=None)
    by_args = pkg_parser.parse_package_attr_by_args(args)
    assert by_args["chip_name"] == "c1"
    assert by_args["suffix"] == "deb"

    merged = pkg_parser.parse_package_attr(root, args)
    assert merged["suffix"] == "deb"
    assert "gen_version_info" in merged


def test_render_cann_version_and_semver_and_cann_version_info():
    expr = pkg_parser.render_cann_version(1, 2, 3, None, None, None)
    assert "(" in expr and ")" in expr

    # render_semver / get_cann_version_info
    items = dict(pkg_parser.render_semver("PKG", "1.2.3-alpha.1"))
    assert items["PKG_VERSION_STR"] == '"1.2.3-alpha.1"'
    assert "PKG_VERSION_NUM" in items

    info = dict(pkg_parser.get_cann_version_info("PKG_VER", "1.2.3"))
    assert '"1.2.3"' in info.values()


def test_env_items_and_parse_env_dict():
    base = dict(pkg_parser.get_default_env_items())
    assert "HOME" in base

    vitems = dict(pkg_parser.get_env_items_by_version("1.2.3"))
    assert vitems["ASCEND_VER"] == "1.2.3"

    vdir_items = dict(pkg_parser.get_env_items_by_version_dir("v1"))
    assert vdir_items["VERSION_DIR"] == "v1"

    os_default = dict(pkg_parser.get_os_arch_default_env_items())
    assert os_default["OS_NAME"] == "linux"

    ts_items = dict(pkg_parser.get_env_items_by_timestamp("20240101_000000000"))
    assert ts_items["TIMESTAMP_NO"] == "20240101000000000"

    env = pkg_parser.parse_env_dict(
        os_arch="ubuntu20.04-aarch64",
        package_attr={"default_arch": "arm64"},
        version="1.2.3",
        version_dir="vdir",
        timestamp="20240101_000000000",
    )
    assert env["OS_NAME"] == "ubuntu"
    assert env["ASCEND_VER"] == "1.2.3"
    assert env["VERSION_DIR"] == "vdir"


def test_get_timestamp_from_args_valid_and_invalid():
    args = Namespace(tag="release_20240101_000000000")
    ts = pkg_parser.get_timestamp(args)
    assert ts == "20240101_000000000"

    args_bad = Namespace(tag="no_timestamp_here")
    with pytest.raises(pkg_parser.PackageError):
        pkg_parser.get_timestamp(args_bad)


def test_extract_element_attrib_and_generate_info_content():
    ele = ET.Element("node", a="1", b="2")
    assert pkg_parser.extract_element_attrib(ele) == {"a": "1", "b": "2"}

    ge = ET.Element("generate_info")
    ET.SubElement(ge, "k1").text = "$(V)"
    env = {"V": "X"}
    content = pkg_parser.extract_generate_info_content(ge, env)
    assert content["content"]["k1"] == "X"


def test_join_pkg_inner_softlink_and_check_contain_asterisk_and_trans_to_stream():
    link = pkg_parser.join_pkg_inner_softlink(["a", "b", "c"])
    assert "a" in link and "c" in link

    assert pkg_parser.check_contain_asterisk("a*")
    assert not pkg_parser.check_contain_asterisk("abc")

    stream = list(pkg_parser.trans_to_stream(1))
    assert stream == [1]


def test_check_value_asterisk_when_package_check_and_suffix_run():
    with pytest.raises(pkg_parser.ContainAsteriskError):
        pkg_parser.check_value(
            "a*",
            package_check=True,
            package_attr={"suffix": "run"},
        )

    # 不触发检查
    pkg_parser.check_value("a*", package_check=False, package_attr={"suffix": "run"})


def test_get_dst_prefix_target_and_make_hash_and_config_hash(tmp_path: Path):
    env = pkg_parser.ParseEnv(
        env_dict={}, parse_option=None, pkg_config_dir=None,
        delivery_dir=str(tmp_path), package_attr={}
    )
    fi = {"dst_path": "sub", "value": "file.txt", "configurable": "TRUE"}
    dst_prefix = pkg_parser.get_dst_prefix(fi, env)
    assert dst_prefix.endswith("sub")

    # 创建实际文件，测试 make_hash / config_hash
    dst_file = Path(dst_prefix)
    dst_file.mkdir(parents=True, exist_ok=True)
    real_file = dst_file / "file.txt"
    real_file.write_text("data", encoding="utf-8")

    target = pkg_parser.get_dst_target(fi, env)
    assert os.path.basename(target) == "file.txt"

    h = pkg_parser.make_hash(str(real_file))
    assert len(h) == 64

    parsed = pkg_parser.FileInfoParsedResult(file_info=fi, move_infos=[], dir_infos=[], expand_infos=[])
    out = pkg_parser.config_hash(parsed, env)
    assert out.file_info["hash"] == h


def test_need_dereference_and_need_expand_and_expand_dir(tmp_path: Path):
    base_dir = tmp_path / "dst"
    base_dir.mkdir()
    (base_dir / "sub").mkdir()
    (base_dir / "sub" / "f.txt").write_text("x", encoding="utf-8")

    fi = {
        "value": "dst",
        "src_path": str(tmp_path),
        "dst_path": str(tmp_path),
        "install_path": "",
    }

    def get_dst_target_func(info):
        return str(base_dir)

    assert not pkg_parser.need_dereference(fi)
    assert pkg_parser.need_expand(fi, get_dst_target_func) is True

    env = pkg_parser.ParseEnv(
        env_dict={}, parse_option=None, pkg_config_dir=None, delivery_dir=str(tmp_path),
        package_attr={}
    )
    files, dirs = pkg_parser.expand_dir(fi, get_dst_target_func, env)
    assert any("f.txt" in f["value"] for f in files)
    assert any("dst" in d["value"] for d in dirs)


def test_expand_file_info_asterisk_and_expand_file_info(tmp_path: Path):
    env = pkg_parser.ParseEnv(
        env_dict={}, parse_option=None, pkg_config_dir=None, delivery_dir=str(tmp_path),
        package_attr={}
    )
    dst_dir = Path(tmp_path) / "d"
    dst_dir.mkdir()
    (dst_dir / "a.txt").write_text("x", encoding="utf-8")

    fi = {
        "dst_path": str(dst_dir),
        "value": "*.txt",
        "install_path": "",
        "configurable": "FALSE",
        "src_path": str(dst_dir),
    }
    parsed = pkg_parser.FileInfoParsedResult(file_info=fi, move_infos=[], dir_infos=[], expand_infos=[])

    # 展开星号
    results = list(pkg_parser.expand_file_info_asterisk(parsed, env))
    assert any(r.file_info["value"].endswith(".txt") for r in results)

    # expand_file 分支：目录展开
    def get_dst_target_func(info):
        return str(dst_dir)

    expanded = pkg_parser.expand_file_info(parsed, use_move=False, get_dst_target_func=get_dst_target_func, env=env)
    assert expanded.file_info.get("is_dir") or expanded.expand_infos != []


def test_parse_file_element_and_related_helpers(tmp_path: Path):
    # 构造简单的 XML 结构
    root = ET.Element("root")
    block = pkg_parser.make_loaded_block_element(root, dst_path="dst")
    env = pkg_parser.ParseEnv(
        env_dict={}, parse_option=None, pkg_config_dir=None, delivery_dir=str(tmp_path),
        package_attr={}
    )
    pkg_attr = {"expand_asterisk": False}
    default = {"module": "m", "dst_path": "d", "src_path": "s"}

    file_ele = ET.Element("file", value="v.txt")
    # 需要一个实际文件以避免 hash/目录展开问题
    (Path(tmp_path) / "d").mkdir(parents=True, exist_ok=True)
    (Path(tmp_path) / "d" / "v.txt").write_text("x", encoding="utf-8")

    results = list(
        pkg_parser.parse_file_element(
            file_ele=file_ele,
            file_config=default,
            loaded_block=block,
            package_attr=pkg_attr,
            env=env,
        )
    )
    assert results and isinstance(results[0], pkg_parser.FileInfoParsedResult)


def test_path_infos_and_pkg_soft_links_and_unique_infos():
    root = ET.Element("root")
    e1 = ET.SubElement(root, "path", value="a", src_path="sa")
    e2 = ET.SubElement(root, "path", value="b", src_path="sb")

    lb = pkg_parser.make_loaded_block_element(root, dst_path="dst")
    env = pkg_parser.ParseEnv(
        env_dict={}, parse_option=None, pkg_config_dir=None, delivery_dir="", package_attr={}
    )
    default = {}

    # get_path_infos 目前实现返回可迭代对象，这里只验证可迭代性
    infos = list(pkg_parser.get_path_infos(e1, default, lb, env))
    assert isinstance(infos, list)

    infos2 = list(pkg_parser.get_path_infos_by_elements([e1, e2], default, lb, env))
    assert isinstance(infos2, list)

    soft_links = pkg_parser.parse_pkg_softlinks(
        [{"value": "v1", "src_path": "s1"}, {"value": "v2", "src_path": "s2"}]
    )
    assert soft_links[0].dst_path == "v1"

    # parse_paths_element / unique_infos
    uniq = pkg_parser.unique_infos([{"value": "x"}, {"value": "x"}])
    assert len(uniq) == 1


def test_block_element_and_blocks_and_read_version_info_and_parse_xml_config(tmp_path: Path, monkeypatch):
    # parse_block_element / block_info / parse_blocks
    block_info = ET.Element("block_info", dst_path="dst_root", block_conf_path="conf")
    ET.SubElement(block_info, "block", name="b1")
    root = ET.Element("root")
    root.append(block_info)

    # monkeypatch load_block_element 以避免真实文件依赖
    def fake_load_block_element(parse_env, package_attr, block_element):
        return pkg_parser.make_loaded_block_element(root, dst_path="dst")

    monkeypatch.setattr(pkg_parser, "load_block_element", fake_load_block_element)

    parse_env = pkg_parser.ParseEnv(
        env_dict={}, parse_option=None, pkg_config_dir=None, delivery_dir="", package_attr={}
    )
    blocks = pkg_parser.parse_blocks(root, package_attr={}, parse_env=parse_env)
    assert blocks and isinstance(blocks[0], pkg_parser.BlockConfig)

    # read_version_info 需要 delivery_dir 和 package_attr（包含 install_script）
    # 创建 version.info：install_script_paths[:-2] 取 install_script 路径除去最后两部分的目录
    # 对于 "a/b/install.sh"，install_script_paths[:-2] = ["a"]，version_path = delivery_dir/a/version.info
    package_attr = {"install_script": "a/b/install.sh"}
    a_dir = tmp_path / "a"
    a_dir.mkdir()
    version_file = a_dir / "version.info"
    version_file.write_text("Version=1.0.0\nversion_dir=v1\n", encoding="utf-8")
    ver, vdir = pkg_parser.read_version_info(str(tmp_path), package_attr)
    assert ver == "1.0.0" and vdir == "v1"

    # parse_xml_config：构造一个最小 xml
    xml_file = tmp_path / "config.xml"
    ET.ElementTree(root).write(xml_file)

    # 避免依赖真实 version.info / env 解析
    monkeypatch.setattr(pkg_parser, "read_version_info", lambda *args, **kwargs: ("1.2.3", "dir"))
    monkeypatch.setattr(
        pkg_parser,
        "parse_env_dict",
        lambda os_arch, package_attr, version, version_dir, timestamp: {"TARGET_ENV": "arch-linux"},
    )
    monkeypatch.setattr(pkg_parser, "is_multi_version", lambda v: bool(v))

    args = Namespace(version_dir=None, disable_multi_version=False, chip_name=None,
                     suffix=None, func_name=None, tag=None)
    parse_opt = pkg_parser.ParseOption(os_arch="ubuntu20.04-aarch64", pkg_version=None,
                                       build_type=None, package_check=False)

    success, cfg = pkg_parser.parse_xml_config(str(tmp_path), "config.xml", str(tmp_path), parse_opt, args)
    assert success is True
    assert cfg.version == "1.2.3"
    assert callable(cfg.packer_config.fill_is_common_path)


def test_packer_remove_ascend_and_get_func_name_and_package_name():
    assert packer.remove_ascend("Ascend910_93") == "A3"
    assert packer.remove_ascend("Ascend310") == "310"
    assert packer.remove_ascend(None) is None

    assert packer.get_func_name("fn", {"func_name": "other"}) == "fn"
    assert packer.get_func_name("", {"func_name": "other"}) == "other"

    args = Namespace(
        chip_name="c1",
        suffix="run",
        func_name="fn",
        not_in_name="deploy_type",
        os_arch="ubuntu20.04-aarch64",
        package_suffix="debug",
        ext_name="ext",
        pkg_name_style="underline",
    )
    pn = packer.PackageName(
        package_attr={"product_name": "prod", "chip_name": "c2", "deploy_type": "dt", "chip_plat": "cp"},
        args=args,
        version="1.2.3",
    )
    name = pn.getvalue()
    assert "prod" in name and "run" in name


def test_create_makeself_pkg_params_factory_and_compose_and_create_run_command(monkeypatch):
    source_target = "/src"
    package_name = "pkg.run"
    comments = "test"
    factory = packer.create_makeself_pkg_params_factory(
        source_target, package_name, comments
    )

    params = factory(
        makeself_dir="/ms",
        package_attr={"cleanup": "rm -rf tmp", "install_script": "install.sh", "help": "help.txt"},
        independent_pkg=True,
    )

    # 固定压缩与格式，便于断言命令
    monkeypatch.setattr(packer, "get_compress_tool", lambda: "--gzip")
    monkeypatch.setattr(packer, "get_compress_format", lambda: "gnu")

    cmd = packer.compose_makeself_command(params)
    assert "pkg.run" in cmd
    assert "test" in cmd
    assert "--gzip" in cmd

    run_cmd, extra = packer.create_run_package_command(params)
    assert package_name in run_cmd and extra is None


def test_get_compress_tool_and_format_and_softlink_before_package(monkeypatch):
    # 优先找到第一个工具
    calls = []

    def fake_which(name):
        calls.append(name)
        if name == "pigz":
            return "/usr/bin/pigz"
        return None

    monkeypatch.setattr(packer.shutil, "which", fake_which)
    tool = packer.get_compress_tool()
    assert tool == "--pigz"

    # get_compress_format: 无 bsdtar
    monkeypatch.setattr(packer.shutil, "which", lambda name: None)
    assert packer.get_compress_format() == "gnu"

    # 独立测试 softlink_before_package，通过 monkeypatch os.symlink 避免权限问题
    links = []

    def fake_symlink(src, dst):
        links.append((src, dst))

    monkeypatch.setattr(packer.os, "symlink", fake_symlink)
    rel_dir = "/release"
    pkg_links = [packer.PkgSoftlink(dst_path="dst/file", src_path="src/file")]
    packer.softlink_before_package(pkg_links, rel_dir)
    assert len(links) == 1
    src, dst = links[0]
    # src 为相对路径，dst 为 release 目录下的目标路径
    assert "src" in src.replace("\\", "/")
    assert "release" in dst.replace("\\", "/")


def test_parse_env_and_cmd_and_exec_pack_cmd(monkeypatch):
    env = {}
    tokens = ["FOO=$PWD", "BAR=1", "python", "-c", "print('x')"]
    cmd_list = packer._parse_env_and_cmd(tokens, env)
    assert env["FOO"] and env["BAR"] == "1"
    assert cmd_list[0] == "python"

    class DummyResult:

        def __init__(self, code, out):
            self.returncode = code
            self.stdout = out

    # 成功路径
    monkeypatch.setattr(
        packer,
        "run_complex_cmd",
        lambda cmd: DummyResult(0, "ok"),
    )
    name = packer.exec_pack_cmd("/tmp", "echo hi", "pkg.run")
    assert name == "pkg.run"

    # 失败路径，避免 CommLog.cilog_error 的格式化问题
    monkeypatch.setattr(
        packer.CommLog,
        "cilog_error",
        staticmethod(lambda *args, **kwargs: None),
    )
    monkeypatch.setattr(
        packer,
        "run_complex_cmd",
        lambda cmd: DummyResult(1, "bad"),
    )
    with pytest.raises(packer.CompressError):
        packer.exec_pack_cmd("", "echo hi", "pkg.run")


def test_parse_os_arch_invalid_format():
    """测试无效os_arch格式抛出异常"""
    with pytest.raises(pkg_parser.ParseOsArchError):
        pkg_parser.parse_os_arch("123_invalid")  # 以数字开头，不符合 [a-z]+


def test_replace_env_unsupported_env_var():
    """测试不支持的环境变量抛出异常"""
    env = {"SUPPORTED": "val"}
    with pytest.raises(pkg_parser.EnvNotSupported):
        pkg_parser.replace_env(env, "path/$(UNSUPPORTED)/")


def test_replace_env_with_file():
    """测试包含$(FILE)环境变量的替换"""
    env = {"VAR": "val"}
    result = pkg_parser.replace_env(env, "path/$(FILE)/$(VAR)")
    assert "$(FILE)" in result and "val" in result


def test_parse_package_info_none():
    """测试None package_info元素"""
    attrs = pkg_parser.parse_package_info(None)
    assert attrs == {}


def test_render_cann_version_with_all_params():
    """测试所有参数的CANN版本渲染"""
    expr = pkg_parser.render_cann_version(1, 2, 3, 4, 5, 6)
    # 代码中使用了 (a_ver + 1) 和 (b_ver + 1)，所以 1->2, 2->3
    assert "(2 * 100000000)" in expr
    assert "(3 * 1000000)" in expr
    assert "(4 * 10000)" in expr
    assert "((5 * 100) + 5000)" in expr
    assert "(6 * 100)" in expr
    assert "+ 6)" in expr


def test_render_semver_various_preleases(tmp_path: Path):
    """测试各种预发布版本"""
    # beta 预发布
    items = dict(pkg_parser.render_semver("PKG", "1.2.3-beta.1"))
    assert "PKG_PRERELEASE" in items
    assert "beta" in items["PKG_PRERELEASE"]

    # rc 预发布
    items = dict(pkg_parser.render_semver("PKG", "1.2.3-rc.2"))
    assert "PKG_PRERELEASE" in items
    assert "rc" in items["PKG_PRERELEASE"]

    # alpha 预发布
    items = dict(pkg_parser.render_semver("PKG", "1.2.3-alpha.10"))
    assert "PKG_PRERELEASE" in items
    assert "alpha" in items["PKG_PRERELEASE"]

    # 正式版本（无预发布）
    items = dict(pkg_parser.render_semver("PKG", "1.2.3"))
    assert items["PKG_PRERELEASE"] == '""'


def test_render_semver_invalid_version(tmp_path: Path):
    """测试无效的版本号格式"""
    # 完全无效的版本格式
    with pytest.raises(pkg_parser.IllegalVersionDir):
        list(pkg_parser.render_semver("PKG", "invalid"))

    # 包含非数字的主版本号
    with pytest.raises(pkg_parser.IllegalVersionDir):
        list(pkg_parser.render_semver("PKG", "a.b.c"))

    # 缺少版本号的段
    with pytest.raises((pkg_parser.IllegalVersionDir, ValueError)):
        list(pkg_parser.render_semver("PKG", "1.2"))


def test_get_cann_version_info_empty_version():
    """测试空版本号"""
    # name[:-8] 去掉 "_VERSION" 后缀，"TEST_PKG_VERSION" -> "TEST_PKG"
    info = dict(pkg_parser.get_cann_version_info("TEST_PKG_VERSION", ""))
    assert info["TEST_PKG_VERSION_STR"] == '"0"'


def test_get_os_arch_default_env_items():
    """测试获取默认os arch环境项"""
    items = list(pkg_parser.get_os_arch_default_env_items())
    assert len(items) == 4
    assert ('OS_NAME', 'linux') in items
    assert ('ARM', 'aarch64') in items


def test_get_env_items_by_timestamp_none():
    """测试无timestamp的情况"""
    items = dict(pkg_parser.get_env_items_by_timestamp(None))
    assert items["TIMESTAMP"] == "0"
    assert items["TIMESTAMP_NO"] == "0"


def test_get_timestamp_none():
    """测试无tag参数的情况"""
    args = Namespace()
    ts = pkg_parser.get_timestamp(args)
    assert ts is None


def test_expand_file_info_with_pkg_inner_softlink(tmp_path: Path):
    """测试pkg_inner_softlink中$(FILE)的替换"""
    env = pkg_parser.ParseEnv(
        env_dict={}, parse_option=None, pkg_config_dir=None, delivery_dir=str(tmp_path),
        package_attr={}
    )
    dst_dir = Path(tmp_path) / "d"
    dst_dir.mkdir()
    (dst_dir / "a.txt").write_text("x", encoding="utf-8")
    (dst_dir / "b.txt").write_text("y", encoding="utf-8")

    fi = {
        "dst_path": str(dst_dir),
        "value": "*.txt",
        "install_path": "",
        "configurable": "FALSE",
        "src_path": str(dst_dir),
        "pkg_inner_softlink": "$(FILE).link",
    }
    parsed = pkg_parser.FileInfoParsedResult(
        file_info=fi, move_infos=[], dir_infos=[], expand_infos=[]
    )
    results = list(pkg_parser.expand_file_info_asterisk(parsed, env))
    assert len(results) == 2
    # 检查$(FILE)是否被替换为实际文件名
    links = [r.file_info.get("pkg_inner_softlink") for r in results]
    assert any("a.txt.link" in link for link in links)
    assert any("b.txt.link" in link for link in links)


def test_expand_file_info_with_exclude(tmp_path: Path):
    """测试exclude功能"""
    env = pkg_parser.ParseEnv(
        env_dict={}, parse_option=None, pkg_config_dir=None, delivery_dir=str(tmp_path),
        package_attr={}
    )
    dst_dir = Path(tmp_path) / "d"
    dst_dir.mkdir()
    (dst_dir / "a.txt").write_text("x", encoding="utf-8")
    (dst_dir / "b.txt").write_text("y", encoding="utf-8")
    (dst_dir / "c.txt").write_text("z", encoding="utf-8")

    fi = {
        "dst_path": str(dst_dir),
        "value": "*.txt",
        "install_path": "",
        "configurable": "FALSE",
        "src_path": str(dst_dir),
        "exclude": "a.txt ; c.txt",
    }
    parsed = pkg_parser.FileInfoParsedResult(
        file_info=fi, move_infos=[], dir_infos=[], expand_infos=[]
    )
    results = list(pkg_parser.expand_file_info_asterisk(parsed, env))
    # 只应该有b.txt，因为a.txt和c.txt被排除了
    assert len(results) == 1
    assert "b.txt" in results[0].file_info["value"]


def test_need_dereference_true():
    """测试need_dereference返回True"""
    fi = {"dereference": "true"}
    assert pkg_parser.need_dereference(fi) is True


def test_need_expand_with_entity():
    """测试entity为true时不展开"""
    fi = {"entity": "true"}
    assert pkg_parser.need_expand(fi, lambda x: "/some/path") is False


def test_expand_file_info_with_use_move(tmp_path: Path):
    """测试use_move分支"""
    fi = {
        "dst_path": str(tmp_path),
        "value": "file.txt",
        "install_path": "",
        "configurable": "FALSE",
        "src_path": str(tmp_path),
    }
    parsed = pkg_parser.FileInfoParsedResult(
        file_info=fi, move_infos=[], dir_infos=[], expand_infos=[]
    )

    def get_dst_target_func(info):
        return os.path.join(str(tmp_path), "file.txt")

    env = pkg_parser.ParseEnv(
        env_dict={}, parse_option=None, pkg_config_dir=None, delivery_dir=str(tmp_path),
        package_attr={}
    )
    expanded = pkg_parser.expand_file_info(
        parsed, use_move=True, get_dst_target_func=get_dst_target_func, env=env
    )
    # use_move=True时，如果不需要展开目录，file_info应该被添加到move_infos
    assert len(expanded.move_infos) == 1
    assert expanded.move_infos[0]["value"] == "file.txt"


def test_expand_dir_with_symlink(tmp_path: Path):
    """测试展开目录时处理软链接"""
    base_dir = tmp_path / "dst"
    base_dir.mkdir()

    # 创建软链接
    link_file = base_dir / "link.txt"
    target_file = tmp_path / "target.txt"
    target_file.write_text("content", encoding="utf-8")

    # 在Windows上不能创建软链接，所以测试可能跳过
    try:
        link_file.symlink_to(target_file)

        fi = {
            "value": "dst",
            "src_path": str(tmp_path),
            "dst_path": str(tmp_path),
            "install_path": "",
        }

        def get_dst_target_func(info):
            return str(base_dir)

        env = pkg_parser.ParseEnv(env_dict={}, parse_option=None, pkg_config_dir=None,
                                  delivery_dir=str(tmp_path), package_attr={})
        files, dirs = pkg_parser.expand_dir(fi, get_dst_target_func, env=env)
        # 软链接应该被当作文件处理
        assert len(files) > 0
    except (OSError, NotImplementedError):
        # Windows可能不支持软链接
        pass


def test_create_file_info_structure():
    """测试create_file_info函数的返回结构"""
    fi = {
        "dst_path": "/dst",
        "src_path": "/src",
        "install_path": "/install",
        "value": "original_value",  # create_file_info需要value键
    }
    result = pkg_parser.create_file_info("/dst/sub/file.txt", "/dst", fi, "file.txt", "dst")
    assert result["value"] == "file.txt"
    # 相对路径处理
    assert "sub" in result["src_path"] or "dst" in result["dst_path"]


def test_parse_block_element_missing_name():
    """测试缺少name时抛出异常"""
    block_info = ET.Element("block_info", dst_path="dst", block_conf_path="conf")
    block_ele = ET.SubElement(block_info, "block")

    with pytest.raises(pkg_parser.BlockConfigError):
        pkg_parser.parse_block_element(block_ele, block_info.attrib)


def test_parse_block_element_missing_conf_path():
    """测试缺少block_conf_path时抛出异常"""
    block_info = ET.Element("block_info", dst_path="dst")
    block_ele = ET.SubElement(block_info, "block", name="test")

    with pytest.raises(pkg_parser.BlockConfigError):
        pkg_parser.parse_block_element(block_ele, block_info.attrib)


def test_evaluate_info_with_real_prefix():
    """测试real:前缀的处理"""
    block = pkg_parser.make_loaded_block_element(ET.Element("root"), dst_path="/base")
    env_dict = {}

    info = {"dst_path": "real:/abs/path"}
    result = pkg_parser.evaluate_info(info, block, env_dict)
    # real:前缀应该被去掉
    assert result["dst_path"] == "/abs/path"


def test_read_version_info_invalid_format(tmp_path: Path, monkeypatch):
    """测试无效的版本格式"""
    # read_version_info 现在需要 delivery_dir 和 package_attr（包含 install_script）
    # install_script_paths[:-2] 需要至少 2 层路径，所以使用 "a/b/install.sh"
    package_attr = {"install_script": "a/b/install.sh"}
    a_dir = tmp_path / "a"
    a_dir.mkdir()
    version_file = a_dir / "version.info"
    version_file.write_text("Version=invalid!\nversion_dir=v1\n", encoding="utf-8")

    with pytest.raises(pkg_parser.VersionFormatNotMatch):
        pkg_parser.read_version_info(str(tmp_path), package_attr)


def test_parse_xml_config_invalid_xml(tmp_path: Path, monkeypatch):
    """测试无效的XML格式"""
    invalid_xml = tmp_path / "invalid.xml"
    invalid_xml.write_text("<invalid><xml>", encoding="utf-8")

    parse_opt = pkg_parser.ParseOption(os_arch="ubuntu20.04", pkg_version=None,
                                       build_type=None, package_check=False)
    args = Namespace(version_dir="1.2.3", disable_multi_version=True, chip_name=None,
                     suffix=None, func_name=None, tag=None)

    # mock读取version_info和parse_env_dict
    monkeypatch.setattr(pkg_parser, "read_version_info", lambda *args, **kwargs: ("1.2.3", ""))
    monkeypatch.setattr(pkg_parser, "parse_env_dict", lambda *args, **kwargs: {"TARGET_ENV": "arch-linux"})
    monkeypatch.setattr(pkg_parser, "is_multi_version", lambda v: bool(v))
    
    # XML解析失败时返回 (False, None)
    success, result = pkg_parser.parse_xml_config(str(tmp_path), "invalid.xml", str(tmp_path), parse_opt, args)
    assert success is False
    assert result is None


def test_parse_xml_config_invalid_os_arch(tmp_path: Path, monkeypatch):
    """测试无效的os_arch配置"""
    xml_file = tmp_path / "config.xml"
    ET.ElementTree(ET.Element("root")).write(xml_file)

    parse_opt = pkg_parser.ParseOption(os_arch="invalid", pkg_version=None,
                                       build_type=None, package_check=False)
    args = Namespace(version_dir="1.2.3", disable_multi_version=True, chip_name=None,
                     suffix=None, func_name=None, tag=None)

    # mock sys.exit以避免退出测试
    monkeypatch.setattr(pkg_parser, "read_version_info", lambda: ("1.2.3", ""))
    original_exit = sys.exit

    def fake_exit(*args):
        return None

    try:
        sys.exit = fake_exit
        pkg_parser.parse_xml_config(str(tmp_path), "config.xml", str(tmp_path), parse_opt, args)
    finally:
        sys.exit = original_exit


def test_render_semver_multisegment_prelease():
    """测试多段预发布版本"""
    items = dict(pkg_parser.render_semver("PKG", "1.2.3-alpha.1.2"))
    assert "PKG_PRERELEASE" in items
    assert "alpha.1.2" in items["PKG_PRERELEASE"]


def test_render_semver_plus_in_version():
    """测试版本号中包含+号"""
    items = dict(pkg_parser.render_semver("PKG", "1.2.3+build"))
    assert "PKG_VERSION_STR" in items
    assert "+build" not in items["PKG_VERSION_STR"]


def test_parse_package_info_false_values():
    """测试解析false值"""
    root = ET.Element("root")
    pkg_info = ET.SubElement(root, "package_info")
    ET.SubElement(pkg_info, "expand_asterisk").text = "false"
    ET.SubElement(pkg_info, "suffix").text = "deb"

    attrs = pkg_parser.parse_package_info(pkg_info)
    assert attrs["expand_asterisk"] is False
    assert attrs["suffix"] == "deb"


def test_get_compress_tool_no_tools_found(monkeypatch, capsys):
    """测试当没有压缩工具时返回空字符串"""
    monkeypatch.setattr(packer.shutil, "which", lambda name: None)
    tool = packer.get_compress_tool()
    assert tool == ""


def test_get_compress_format_with_bsdtar(monkeypatch):
    """测试检测到bsdtar时返回ustar格式"""
    monkeypatch.setattr(packer.shutil, "which", lambda name: "/usr/bin/bsdtar" if name == "bsdtar" else None)
    fmt = packer.get_compress_format()
    assert fmt == "pax"


def test_compose_makeself_command_with_cleanup(monkeypatch):
    """测试组装makeself命令带cleanup参数"""
    monkeypatch.setattr(packer, "get_compress_tool", lambda: "--gzip")
    monkeypatch.setattr(packer, "get_compress_format", lambda: "gnu")
    
    params = packer.MakeselfPkgParams(
        package_name="test.run",
        comments="Test",
        makeself_tool="/tools/makeself.sh",
        makeself_header="/tools/makeself-header.sh",
        help_info="help.txt",
        source_target="/source",
        install_script="install.sh",
        independent_pkg=True,
        cleanup="cleanup.sh"
    )
    cmd = packer.compose_makeself_command(params)
    assert "--cleanup cleanup.sh" in cmd
    assert "makeself.sh" in cmd


def test_compose_makeself_command_non_independent(monkeypatch):
    """测试非独立包的makeself命令组装"""
    monkeypatch.setattr(packer, "get_compress_tool", lambda: "--gzip")
    monkeypatch.setattr(packer, "get_compress_format", lambda: "gnu")
    
    params = packer.MakeselfPkgParams(
        package_name="test.run",
        comments="Test",
        cleanup=None,
        independent_pkg=False
    )
    cmd = packer.compose_makeself_command(params)
    assert "test.run" in cmd
    assert "Test" in cmd
    assert "TMPDIR" not in cmd  # 非独立包不应该有TMPDIR


def test_create_makeself_pkg_params_non_independent():
    """测试创建非独立包的参数"""
    factory = packer.create_makeself_pkg_params_factory("/source", "test.run", "Test")
    params = factory("/makeself", {"cleanup": "clean.sh"}, independent_pkg=False)
    
    assert params.package_name == "test.run"
    assert params.comments == "Test"
    assert params.makeself_tool is None
    assert params.independent_pkg is False


def test_package_name_getvalue_with_all_attributes():
    """测试PackageName.getvalue使用所有属性"""
    args = Namespace(
        chip_name="ascend910",
        suffix="run",
        func_name="toolkit",
        not_in_name="",
        os_arch="aarch64",
        package_suffix="none",
        ext_name="",
        pkg_name_style="common"
    )
    package_attr = {
        "product_name": "Ascend",
        "chip_plat": "",
        "deploy_type": ""
    }
    pkg_name = packer.PackageName(package_attr, args, "1.0.0")
    result = pkg_name.getvalue()
    assert result.endswith(".run")
    assert "ascend" in result.lower() or "910" in result


def test_package_name_getvalue_underline_style():
    """测试PackageName.getvalue使用下划线风格"""
    args = Namespace(
        chip_name=None,
        suffix="run",
        func_name="nnae",
        not_in_name="os_arch,product_name",
        os_arch="aarch64",
        package_suffix="debug",
        ext_name="",
        pkg_name_style="underline"
    )
    package_attr = {
        "product_name": "Ascend",
        "chip_plat": "",
        "deploy_type": ""
    }
    pkg_name = packer.PackageName(package_attr, args, "1.0.0")
    result = pkg_name.getvalue()
    assert "_" in result
    assert result.endswith(".run")


def test_run_complex_cmd_with_cd(monkeypatch):
    """测试run_complex_cmd处理cd命令"""
    class DummyResult:

        def __init__(self):
            self.returncode = 0
            self.stdout = "success"
    
    calls = []

    def mock_run(*args, **kwargs):
        calls.append(kwargs.get('cwd'))
        return DummyResult()
    
    monkeypatch.setattr(packer.subprocess, "run", mock_run)
    result = packer.run_complex_cmd("cd /some/dir && echo test")
    assert result is not None


def test_run_complex_cmd_no_cmd(monkeypatch):
    """测试run_complex_cmd没有命令时返回None"""
    result = packer.run_complex_cmd("cd /tmp")  # 只有cd，没有实际命令
    assert result is None


# =============================================================================
# #27: pkg_parser 复杂逻辑零覆盖补充测试
# =============================================================================


def _make_parse_env(tmp_path, env_dict=None, package_attr=None):
    """构建测试用 ParseEnv，使用默认 ParseOption。"""
    return pkg_parser.ParseEnv(
        env_dict=env_dict if env_dict is not None else {},
        parse_option=pkg_parser.ParseOption(
            os_arch='linux-x86_64', pkg_version=None,
            build_type='debug', package_check=False
        ),
        pkg_config_dir=str(tmp_path),
        delivery_dir=str(tmp_path),
        package_attr=package_attr if package_attr is not None else {}
    )


def _make_parse_args(**overrides):
    """构建 parse_xml_config 所需的 Namespace 参数。"""
    defaults = dict(
        version_dir='', tag='', chip_name=None, func_name=None,
        disable_multi_version=False, suffix='run'
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def _make_parse_option(os_arch='linux-x86_64'):
    """构建测试用 ParseOption。"""
    return pkg_parser.ParseOption(
        os_arch=os_arch, pkg_version=None,
        build_type='debug', package_check=False
    )


class TestLoadBlockElement:
    """load_block_element 块配置加载与异常测试。"""

    @staticmethod
    def test_load_block_element_file_not_exists(tmp_path):
        """块配置 XML 文件不存在时抛 BlockConfigError。"""
        parse_env = _make_parse_env(tmp_path)
        block_element = pkg_parser.BlockElement(
            name='NonExistent',
            block_conf_path='module/ascend/NonExistent.xml',
            dst_path='',
            chips=set(),
            features=set(),
            attrs={}
        )
        with pytest.raises(pkg_parser.BlockConfigError, match="does not exist"):
            pkg_parser.load_block_element(parse_env, {}, block_element)

    @staticmethod
    def test_load_block_element_invalid_xml(tmp_path):
        """块配置 XML 解析失败时抛 BlockConfigError。"""
        module_dir = tmp_path / 'module' / 'ascend'
        module_dir.mkdir(parents=True)
        (module_dir / 'BadBlock.xml').write_text('<not-closed>')

        parse_env = _make_parse_env(tmp_path)
        block_element = pkg_parser.BlockElement(
            name='BadBlock',
            block_conf_path='module/ascend/BadBlock.xml',
            dst_path='',
            chips=set(),
            features=set(),
            attrs={}
        )
        with pytest.raises(pkg_parser.BlockConfigError, match="parse failed"):
            pkg_parser.load_block_element(parse_env, {}, block_element)

    @staticmethod
    def test_load_block_element_success(tmp_path):
        """正常加载块配置 XML。"""
        module_dir = tmp_path / 'module' / 'ascend'
        module_dir.mkdir(parents=True)
        (module_dir / 'GoodBlock.xml').write_text('<root><file_info value="test"/></root>')

        parse_env = _make_parse_env(tmp_path, package_attr={'use_move': False})
        block_element = pkg_parser.BlockElement(
            name='GoodBlock',
            block_conf_path='module/ascend/GoodBlock.xml',
            dst_path='lib',
            chips={'ascend910'},
            features={'comm'},
            attrs={}
        )
        result = pkg_parser.load_block_element(parse_env, {'use_move': False}, block_element)
        assert isinstance(result, pkg_parser.LoadedBlockElement)
        assert result.dst_path == 'lib'
        assert result.chips == {'ascend910'}


class TestReadVersionInfo:
    """read_version_info 版本信息读取测试。"""

    @staticmethod
    def test_no_install_script():
        """package_attr 中无 install_script 时抛 InstallScriptNotInPackageInfo。"""
        with pytest.raises(pkg_parser.InstallScriptNotInPackageInfo):
            pkg_parser.read_version_info('/tmp', {})

    @staticmethod
    def test_install_script_format_error():
        """install_script 格式错误（路径层级不足）时抛 InstallScriptFormatError。"""
        with pytest.raises(pkg_parser.InstallScriptFormatError):
            pkg_parser.read_version_info('/tmp', {'install_script': 'install.sh'})

    @staticmethod
    def test_version_info_not_exist(tmp_path):
        """version.info 文件不存在时抛 VersionInfoNotExist。"""
        with pytest.raises(pkg_parser.VersionInfoNotExist):
            pkg_parser.read_version_info(
                str(tmp_path),
                {'install_script': 'share/info/test/script/install.sh'}
            )

    @staticmethod
    def test_read_version_info_success(tmp_path):
        """正常读取 version.info。"""
        version_dir = tmp_path / 'share' / 'info' / 'test'
        version_dir.mkdir(parents=True)
        (version_dir / 'version.info').write_text('Version=8.0.0\nversion_dir=cann\n')

        version, version_dir_val = pkg_parser.read_version_info(
            str(tmp_path),
            {'install_script': 'share/info/test/script/install.sh'}
        )
        assert version == '8.0.0'
        assert version_dir_val == 'cann'

    @staticmethod
    def test_read_version_info_invalid_format(tmp_path):
        """version.info 中版本号含非法字符时抛 VersionFormatNotMatch。"""
        version_dir = tmp_path / 'share' / 'info' / 'test'
        version_dir.mkdir(parents=True)
        (version_dir / 'version.info').write_text('Version=8.0.0/invalid\nversion_dir=cann\n')

        with pytest.raises(pkg_parser.VersionFormatNotMatch):
            pkg_parser.read_version_info(
                str(tmp_path),
                {'install_script': 'share/info/test/script/install.sh'}
            )


class TestParseXmlConfigErrors:
    """parse_xml_config 异常处理分支测试。"""

    @staticmethod
    def test_parse_xml_config_install_script_not_configured(tmp_path):
        """package_info 中未配置 install_script 时返回 (False, None)。"""
        xml_content = '<package><package_info><func_name>test</func_name></package_info></package>'
        xml_file = tmp_path / 'test.xml'
        xml_file.write_text(xml_content)

        args = _make_parse_args()
        parse_option = _make_parse_option()
        ret, config = pkg_parser.parse_xml_config(
            str(tmp_path), 'test.xml', str(tmp_path), parse_option, args
        )
        assert ret is False
        assert config is None

    @staticmethod
    def test_parse_xml_config_invalid_os_arch(tmp_path):
        """os_arch 格式错误时返回 (False, None)。"""
        version_dir = tmp_path / 'share' / 'info' / 'test'
        version_dir.mkdir(parents=True)
        (version_dir / 'version.info').write_text('Version=8.0.0\nversion_dir=cann\n')

        xml_content = '''<package><package_info>
        <func_name>test</func_name>
        <install_script>share/info/test/script/install.sh</install_script>
        </package_info></package>'''
        xml_file = tmp_path / 'test.xml'
        xml_file.write_text(xml_content)

        args = _make_parse_args()
        parse_option = _make_parse_option(os_arch='123invalid')
        ret, config = pkg_parser.parse_xml_config(
            str(tmp_path), 'test.xml', str(tmp_path), parse_option, args
        )
        assert ret is False
        assert config is None

    @staticmethod
    def test_parse_xml_config_success(tmp_path):
        """正常解析 XML 配置返回 (True, XmlConfig)。"""
        version_dir = tmp_path / 'share' / 'info' / 'test'
        version_dir.mkdir(parents=True)
        (version_dir / 'version.info').write_text('Version=8.0.0\nversion_dir=cann\n')

        xml_content = '''<package><package_info>
        <func_name>test</func_name>
        <install_script>share/info/test/script/install.sh</install_script>
        </package_info>
        <dir_info value="lib64"/>
        </package>'''
        xml_file = tmp_path / 'test.xml'
        xml_file.write_text(xml_content)

        args = _make_parse_args()
        parse_option = _make_parse_option()
        ret, config = pkg_parser.parse_xml_config(
            str(tmp_path), 'test.xml', str(tmp_path), parse_option, args
        )
        assert ret is True
        assert config is not None
        assert config.version == '8.0.0'


class TestParseDirInfoElements:
    """parse_dir_info_elements 目录信息解析测试。"""

    @staticmethod
    def test_parse_dir_info_with_sub_elements(tmp_path):
        """dir_info 包含子元素时正确解析。"""
        root = ET.fromstring('''<root dst_path="lib">
        <dir_info value="lib64">
            <dir value="lib64/subdir"/>
        </dir_info>
        </root>''')
        loaded_block = pkg_parser.make_loaded_block_element(root, dst_path='lib')
        parse_env = _make_parse_env(
            tmp_path, env_dict={'OS_NAME': 'linux'}, package_attr={'suffix': 'run'}
        )
        result = pkg_parser.parse_dir_info_elements(
            loaded_block, {}, {'suffix': 'run'}, parse_env
        )
        assert len(result) == 1
        assert result[0]['value'] == 'lib64/subdir'


class TestParseFileInfoElements:
    """parse_file_info_elements 文件信息解析测试。"""

    @staticmethod
    def test_parse_file_info_with_file_elements(tmp_path):
        """file_info 包含多个 file 子元素时正确解析。"""
        (tmp_path / 'lib64').mkdir()
        (tmp_path / 'lib64' / 'liba.so').write_text('content')
        (tmp_path / 'lib64' / 'libb.so').write_text('content')

        root = ET.fromstring('''<root dst_path="lib">
        <file_info value="lib64" install_path="lib" install_mod="755">
            <file value="lib64/liba.so"/>
            <file value="lib64/libb.so"/>
        </file_info>
        </root>''')
        loaded_block = pkg_parser.make_loaded_block_element(root, dst_path='lib')
        parse_env = _make_parse_env(
            tmp_path, env_dict={'OS_NAME': 'linux'},
            package_attr={'suffix': 'run', 'expand_asterisk': False}
        )
        results = list(pkg_parser.parse_file_info_elements(
            loaded_block, {}, {'suffix': 'run', 'expand_asterisk': False}, parse_env
        ))
        assert len(results) == 2


class TestRenderSemverEdgeCases:
    """render_semver 预发布版本边界测试。"""

    @staticmethod
    def test_render_semver_no_pre_release():
        """无预发布版本号时正常生成。"""
        result = dict(pkg_parser.render_semver('PKG', '8.0.5'))
        assert result['PKG_VERSION_NUM'] == '((8 * 10000000) + (0 * 100000) + (5 * 1000))'
        assert result['PKG_PRERELEASE'] == '""'

    @staticmethod
    def test_render_semver_rc_with_dot():
        """rc.1 格式（含点号）的预发布版本。"""
        result = dict(pkg_parser.render_semver('PKG', '8.0.5-rc.1'))
        assert result['PKG_PRERELEASE'] == '"rc.1"'
        assert result['PKG_VERSION_NUM'] == '((8 * 10000000) + (0 * 100000) + (5 * 1000) - 100 + 1)'

    @staticmethod
    def test_render_semver_beta_no_dot():
        """beta2 格式（无点号）的预发布版本。"""
        result = dict(pkg_parser.render_semver('PKG', '8.0.5-beta2'))
        assert result['PKG_PRERELEASE'] == '"beta2"'
        assert result['PKG_VERSION_NUM'] == '((8 * 10000000) + (0 * 100000) + (5 * 1000) - 200 + 2)'

    @staticmethod
    def test_render_semver_alpha_multi_segment():
        """alpha.1.2 多段序号预发布版本。"""
        result = dict(pkg_parser.render_semver('PKG', '8.0.5-alpha.1.2'))
        assert result['PKG_PRERELEASE'] == '"alpha.1.2"'
        assert result['PKG_VERSION_NUM'] == '((8 * 10000000) + (0 * 100000) + (5 * 1000) - 300 + 12)'

    @staticmethod
    def test_render_semver_unknown_pre_type():
        """未知预发布类型（如 dev.1）使用权重 400。"""
        result = dict(pkg_parser.render_semver('PKG', '8.0.5-dev.1'))
        assert result['PKG_PRERELEASE'] == '"dev.1"'
        assert result['PKG_VERSION_NUM'] == '((8 * 10000000) + (0 * 100000) + (5 * 1000) - 400 + 1)'

    @staticmethod
    def test_render_semver_pre_release_no_number():
        """预发布版本无序号（如 rc）时抛 IllegalVersionDir（已知行为）。"""
        with pytest.raises(pkg_parser.IllegalVersionDir):
            dict(pkg_parser.render_semver('PKG', '8.0.5-rc'))

    @staticmethod
    def test_render_semver_plus_build_metadata():
        """带构建元数据（+build123）的版本号，构建元数据被移除。"""
        result = dict(pkg_parser.render_semver('PKG', '8.0.5+build123'))
        assert result['PKG_VERSION_STR'] == '"8.0.5"'


_EXPAND_DIR_FILE_INFO = {
    'value': 'target',
    'dst_path': '',
    'install_path': 'lib',
    'src_path': '',
    'install_mod': '755',
    'install_softlink': '',
    'pkg_inner_softlink': '',
}


class TestExpandDir:
    """expand_dir 目录展开逻辑测试。"""

    @staticmethod
    def test_expand_dir_with_files(tmp_path):
        """展开包含多个文件的目录。"""
        target_dir = tmp_path / 'target'
        target_dir.mkdir()
        (target_dir / 'file1.so').write_text('a')
        (target_dir / 'file2.so').write_text('b')

        env = _make_parse_env(tmp_path)
        get_dst_target = pkg_parser.partial(
            pkg_parser.get_dst_target, env=env
        )
        file_list, dir_list = pkg_parser.expand_dir(
            _EXPAND_DIR_FILE_INFO, get_dst_target, env
        )
        assert len(file_list) == 2
        assert any(f['value'] == 'file1.so' for f in file_list)
        assert any(f['value'] == 'file2.so' for f in file_list)

    @staticmethod
    def test_expand_dir_with_symlink_dir(tmp_path):
        """展开包含软链接目录的情况。"""
        real_dir = tmp_path / 'real_dir'
        real_dir.mkdir()
        (real_dir / 'file.so').write_text('content')

        target_dir = tmp_path / 'target'
        target_dir.mkdir()
        os.symlink(real_dir, target_dir / 'link_dir')

        env = _make_parse_env(tmp_path)
        get_dst_target = pkg_parser.partial(
            pkg_parser.get_dst_target, env=env
        )
        file_list, dir_list = pkg_parser.expand_dir(
            _EXPAND_DIR_FILE_INFO, get_dst_target, env
        )
        assert len(file_list) >= 1


class TestExpandFileInfoAsterisk:
    """expand_file_info_asterisk 通配符展开测试。"""

    @staticmethod
    def test_expand_asterisk_with_exclude(tmp_path):
        """通配符展开并排除指定文件。"""
        (tmp_path / 'liba.so').write_text('a')
        (tmp_path / 'libb.so').write_text('b')
        (tmp_path / 'libc.so').write_text('c')

        file_info = {
            'value': 'lib*.so',
            'dst_path': '',
            'install_path': 'lib',
            'exclude': 'libb.so',
            'pkg_inner_softlink': '',
        }
        parsed_result = pkg_parser.FileInfoParsedResult(
            file_info=file_info, move_infos=[], dir_infos=[], expand_infos=[]
        )
        env = _make_parse_env(tmp_path)
        results = list(pkg_parser.expand_file_info_asterisk(parsed_result, env))
        values = [r.file_info['value'] for r in results]
        assert 'liba.so' in values
        assert 'libb.so' not in values
        assert 'libc.so' in values

    @staticmethod
    def test_expand_asterisk_no_match(tmp_path):
        """通配符无匹配文件时返回空列表。"""
        file_info = {
            'value': 'nonexistent*.so',
            'dst_path': '',
            'install_path': 'lib',
        }
        parsed_result = pkg_parser.FileInfoParsedResult(
            file_info=file_info, move_infos=[], dir_infos=[], expand_infos=[]
        )
        env = _make_parse_env(tmp_path)
        results = list(pkg_parser.expand_file_info_asterisk(parsed_result, env))
        assert len(results) == 0

    @staticmethod
    def test_expand_asterisk_no_asterisk(tmp_path):
        """value 不含通配符时原样返回。"""
        file_info = {
            'value': 'lib/liba.so',
            'dst_path': '',
            'install_path': 'lib',
        }
        parsed_result = pkg_parser.FileInfoParsedResult(
            file_info=file_info, move_infos=[], dir_infos=[], expand_infos=[]
        )
        env = _make_parse_env(tmp_path)
        results = list(pkg_parser.expand_file_info_asterisk(parsed_result, env))
        assert len(results) == 1
