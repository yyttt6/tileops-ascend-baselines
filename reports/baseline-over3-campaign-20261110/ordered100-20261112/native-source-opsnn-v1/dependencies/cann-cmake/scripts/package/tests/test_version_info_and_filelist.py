# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from pathlib import Path
import os

import itertools
import xml.etree.ElementTree as ET
import pytest

from .. import version_info
from .. import filelist
from ..utils import pkg_utils, funcbase


def _make_sample_file_item(
    module="mod",
    operation="copy",
    rel_pkg="a.txt",
    rel_install="path/in/pkg",
    softlink=None,
    feature=None,
    is_common="N",
    pkg_inner_softlink=None,
    chip=None,
    is_dir=False,
):
    return filelist.create_file_item(
        module,
        operation,
        rel_pkg,
        rel_install,
        "N",
        "644",
        "root:root",
        "all",
        softlink or [],
        feature or set(),
        is_common,
        "FALSE",
        "",
        "block",
        pkg_inner_softlink or [],
        chip or set(),
        is_dir,
    )


def test_version_compare_and_match():
    assert version_info.Version.match("1.2.3")
    v1 = version_info.Version.parse("1.2.3")
    v2 = version_info.Version.parse("1.10.0")
    assert v1 < v2
    assert str(v1) == "1.2.3"


def test_interval_parse_and_required_list():
    iv = version_info.Interval.parse("[1.0.0,2.0.0)")
    reqs = iv.to_required_list()
    assert ">=" in reqs[0] and "1.0.0" in reqs[0]
    assert "<" in reqs[1] and "2.0.0" in reqs[1]


def test_require_sort_and_required_full_str():
    v_low = version_info.Interval.parse("[1.0.0,2.0.0)")
    v_high = version_info.Version.parse("3.0.0")
    req = version_info.Require(pkg_name="testpkg", versions=[v_high, v_low])
    req.sort_versions()
    # 排序后区间在前，单版本在后
    assert isinstance(req.versions[0], version_info.Interval)
    full = req.to_required_full_str()
    assert "required_package_testpkg_version" in full


def test_versionxml_and_item_parsing_and_get_version_dir():
    assert version_info.VersionXml.match("a.xml")
    assert not version_info.VersionXml.match("a.txt")

    # ItemElement / CompatibleElement
    root = ET.Element("compatible")
    ET.SubElement(root, "item", name="pkgA", version="1.0.0")
    ET.SubElement(root, "item", name="pkgB", version="")
    comp = version_info.CompatibleElement.parse(root, "9.9.9")
    assert len(comp.items) == 1
    assert comp.items[0].name == "pkgA"

    vx = version_info.VersionXml(
        release_version="1.0.0",
        version_dir="ver_dir",
        packages={"mypkg": comp},
    )
    assert vx.get_release_version() == "1.0.0"
    assert vx.get_version_dir() == "ver_dir"

    # collect_requires 成功路径
    requires = vx.collect_requires("mypkg")
    assert len(requires) == 1
    assert requires[0].pkg_name == "pkgA"

    # collect_requires 失败路径：非法版本表达式
    bad_comp_root = ET.Element("compatible")
    ET.SubElement(bad_comp_root, "item", name="bad", version="not_a_version")
    bad_comp = version_info.CompatibleElement.parse(bad_comp_root, "x")
    bad_vx = version_info.VersionXml(
        release_version="1.0.0",
        version_dir="dir",
        packages={"badpkg": bad_comp},
    )
    with pytest.raises(version_info.CollectRequiresFailed):
        bad_vx.collect_requires("badpkg")

    # get_version_dir 辅助函数
    assert version_info.get_version_dir(vx, False, None) == "ver_dir"
    assert version_info.get_version_dir(None, False, "from_arg") == "from_arg"
    assert version_info.get_version_dir(None, True, "x") is None


def test_is_version_number_and_multi_version_helpers():
    assert version_info.is_version_number("1.2.3")
    assert not version_info.is_version_number("1.2")
    assert version_info.is_multi_version("ver_dir")
    assert not version_info.is_multi_version("")


def test_version_info_file_content_and_requires_section(tmp_path: Path):
    req = version_info.Require(
        pkg_name="pkg",
        versions=[version_info.Version("1.0.0")],
    )
    vf = version_info.VersionInfoFile(
        version="1.2.3",
        version_dir="ver_dir",
        timestamp="20250101000000000",
        itf_version_info="itf=1",
        requires=[req],
    )
    out = tmp_path / "version.info"
    vf.save(out)
    content = out.read_text(encoding="utf-8")
    assert "Version=1.2.3" in content
    assert "version_dir=ver_dir" in content
    assert "timestamp=" in content
    assert "itf=1" in content
    assert "required_package_pkg_version" in content


def test_fileitem_to_string_and_header_and_soft_links_to_string():
    item = _make_sample_file_item(softlink=["link"], feature={"f1"})
    s = filelist.file_item_to_string(item)
    assert "mod,copy,a.txt,path/in/pkg" in s
    header = filelist.get_filelist_header_string()
    assert "module" in header and "chip" in header

    # soft_links_to_string 空/非空
    assert filelist.soft_links_to_string([]) == "NA"
    assert filelist.soft_links_to_string(["a", "b"]) == "a;b"


def test_create_file_item_type_validation():
    with pytest.raises(TypeError):
        filelist.create_file_item(
            "m",
            "copy",
            "a",
            "p",
            "N",
            "644",
            "root:root",
            "all",
            ["link"],
            ["not_set"],  # feature 需为 set
            "N",
            "FALSE",
            "",
            "block",
            [],
            set(),
            False,
        )


def test_is_relative_install_path_and_install_type_helpers():
    assert filelist.is_relative_install_path("a/b")
    assert not filelist.is_relative_install_path("/abs")

    class Dummy:
        def __init__(self, op, install_type):
            self.operation = op
            self.install_type = install_type

    fi = Dummy("copy", "all")
    assert filelist.is_specific_operations(fi, ["copy"])
    assert filelist.is_specific_install_type(fi, {"run"})


def test_fill_is_common_path_and_get_soft_links_not_in_common_paths():
    target_env = "arch-linux"
    f1 = _make_sample_file_item(
        rel_install=f"{target_env}/a",
        softlink=[f"{target_env}/b", "other/c"],
    )
    f2 = _make_sample_file_item(rel_install="other/d", softlink=[])
    fl = [f1, f2]

    out = list(filelist.fill_is_common_path(fl, target_env))
    assert out[0].is_common_path == "Y"
    # 第二个条目有软链指向公共目录，标记为 YY
    assert out[1].is_common_path in ("N", "YY")

    soft_links = list(filelist.get_soft_links_not_in_common_paths(fl, target_env))
    assert "other/c" in soft_links


def test_get_install_path_dirs_and_missing_dir_set_and_check_filelist():
    fi = _make_sample_file_item(
        operation="copy",
        rel_install="aaa/bbb/file.txt",
    )
    # 缺少中间目录，应该被检测到
    missing = filelist.get_missing_dir_set([fi])
    assert "aaa" in missing or "aaa/bbb" in missing

    # check_filelist 在有缺失目录时抛异常
    with pytest.raises(filelist.FilelistError):
        filelist.check_filelist([fi], check_features=False, check_move=False)


def test_print_helpers_and_move_safe_and_transform_funcs(tmp_path: Path):
    fi1 = _make_sample_file_item(
        operation="copy",
        rel_install="dir",
        rel_pkg="pkg/dir",
    )
    fi2 = _make_sample_file_item(
        operation="copy",
        rel_install="dir/sub/file",
        rel_pkg="pkg/dir/sub/file",
    )

    # print_missing_dir_set / print_unsafe_paths 主要验证可调用返回值
    mset = {"a", "b"}
    assert filelist.print_missing_dir_set(mset) == mset
    unsafe = ("x", "y")
    assert filelist.print_unsafe_paths(unsafe) == unsafe

    # check_move_safe 返回 tuple
    res = filelist.check_move_safe([fi1, fi2])
    assert isinstance(res, tuple)

    # is_nested_file_item 关系判断
    assert filelist.is_nested_file_item(fi1, None) == filelist.FileItemRelation.NOT_NESTED
    assert filelist.is_nested_file_item(fi1, fi1) == filelist.FileItemRelation.SAME
    assert filelist.is_nested_file_item(fi2, fi1) == filelist.FileItemRelation.NESTED

    # found_nested_file_item 抛异常
    with pytest.raises(filelist.FilelistError):
        filelist.found_nested_file_item(fi2, fi1)

    # convert_nested_path_in_filelist 将嵌套元素改为 del
    conv = list(filelist.convert_nested_path_in_filelist([fi1, fi2]))
    assert conv[0].operation == "copy"
    assert conv[1].operation == "del"

    # check_nested_path_in_filelist 只做检查和日志打印，不一定抛异常
    filelist.check_nested_path_in_filelist([fi1, fi2])

    # transform_nested_path_in_filelist 调用通过
    transformed = filelist.transform_nested_path_in_filelist([fi1, fi2])
    assert any(it.operation == "del" for it in transformed)

    # generate_filelist 输出文件
    out_dir = tmp_path
    filelist.generate_filelist([fi1, fi2], "filelist.csv", str(out_dir))
    out_file = out_dir / "filelist.csv"
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "module,operation" in content.splitlines()[0]

    # get_transform_nested_path_func
    assert filelist.get_transform_nested_path_func(True) is filelist.transform_nested_path_in_filelist
    # Test identity function behavior instead of object identity
    identity_func = filelist.get_transform_nested_path_func(False)
    test_list = [fi1, fi2]
    assert identity_func(test_list) is test_list


def test_get_common_path_basic():
    paths = ["/a/b/c", "/a/b/d"]
    common = filelist.get_common_path(paths)
    assert common in ("/a/b", "a\\b", "\\a\\b", "")


def test_pkg_utils_basic_helpers_and_errors():
    assert list(pkg_utils.flatten([[1], [2, 3]])) == [1, 2, 3]
    merged = pkg_utils.merge_dict({"a": 1}, {"b": 2})
    assert merged == {"a": 1, "b": 2}

    add = lambda x, y: x + y
    sp = pkg_utils.star_pipe(lambda a, b: (a + 1, b + 1), add)
    assert sp(1, 2) == 5

    swapped = pkg_utils.swap_args(lambda x, y: x - y)
    assert swapped(1, 3) == 2  # 实际执行 3 - 1

    cf = pkg_utils.config_feature_to_set("f1;f2", "feature")
    assert cf == {"f1", "f2"}
    assert pkg_utils.config_feature_to_string(set()) == "all"
    assert pkg_utils.config_feature_to_string({"b", "a"}) == "a;b"

    # 错误分支：空字符串 / all
    with pytest.raises(pkg_utils.PackageConfigError):
        pkg_utils.config_feature_to_set("", "feature")
    with pytest.raises(pkg_utils.PackageConfigError):
        pkg_utils.config_feature_to_set("all", "feature")

    # conditional_apply / pairwise / path_join / yield_if
    fn = pkg_utils.conditional_apply(lambda x: x > 0, lambda x: x + 1)
    assert fn(1) == 2 and fn(0) == 0

    seq = [1, 2, 3]
    pairs = list(pkg_utils.pairwise(seq))
    assert pairs == [(1, 2), (2, 3)]

    assert pkg_utils.path_join(None, "a") is None
    assert pkg_utils.path_join("base", "a", "b").endswith("base" + os.sep + "a" + os.sep + "b")

    yielded = list(pkg_utils.yield_if(1, lambda x: x == 1))
    assert yielded == [1]


def test_funcbase_basic_helpers_and_high_order():
    const_f = funcbase.constant(10)
    assert const_f() == 10

    plus1 = lambda x: x + 1
    mul2 = lambda x: x * 2
    pipe_f = funcbase.pipe(plus1, mul2)
    assert pipe_f(3) == 8

    ident = funcbase.identity("x")
    assert ident == "x"

    # dispatch / invoke / side_effect / star_apply / any_ / not_
    d = funcbase.dispatch(lambda x: x + 1, lambda x: x + 2)
    assert list(d(1)) == [2, 3]

    assert funcbase.invoke(lambda x, y: x + y, 1, 2) == 3

    calls = []

    def rec(x):
        calls.append(x)

    se = funcbase.side_effect(rec)
    assert se(5) == 5 and calls == [5]

    sa = funcbase.star_apply(lambda a, b: a + b)
    assert sa((1, 2)) == 3

    any_f = funcbase.any_(lambda x: False, lambda x: x > 0)
    assert any_f(1) is True

    not_f = funcbase.not_(lambda x: x > 0)
    assert not_f(0) is True and not_f(1) is False
