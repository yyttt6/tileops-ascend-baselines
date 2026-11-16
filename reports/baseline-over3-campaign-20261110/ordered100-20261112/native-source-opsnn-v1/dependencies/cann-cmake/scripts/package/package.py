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
import sys
import subprocess
import argparse
import traceback
import csv
from argparse import Namespace
from collections import namedtuple
from collections import defaultdict
from functools import partial
from itertools import chain
from typing import Iterator, List, Set, Tuple
import shlex
import json

from utils import pkg_utils
from filelist import (
    FileItem,
    FileList,
    check_filelist,
    create_file_item,
    generate_filelist,
    generate_record_file,
    get_transform_nested_path_func,
)
from packer import (
    PackageName,
    create_makeself_pkg_params_factory,
    create_run_package_command,
    softlink_before_package,
    exec_pack_cmd,
)
from pkg_parser import (
    ParseOption,
    XmlConfig,
    parse_xml_config,
    get_cann_version_info,
    get_target_name,
)
from utils.pkg_utils import (
    CONFIG_SCRIPT_PATH,
    CompressError,
    ContainAsteriskError,
    FilelistError,
    GenerateFilelistError,
    PackageNameEmptyError,
    UnknownOperateTypeError,
    path_join,
)
from gen_postinst_prerm import (
    generate_postinst,
    generate_prerm,
    generate_set_permission,
)
from utils.funcbase import invoke, pipe
from utils.comm_log import CommLog


def get_comments(package_name: PackageName) -> str:
    """获取run包注释。"""
    if not package_name.chip_name:
        comments = "_".join(
            [
                package_name.product_name.upper(),
                package_name.func_name.upper(),
                "RUN_PACKAGE",
            ]
        )
    else:
        comments = "_".join(
            [
                package_name.chip_name.upper(),
                package_name.func_name.upper(),
                "RUN_PACKAGE",
            ]
        )
    return f'"{comments}"'


def get_compress_cmd(
    delivery_dir: str, build_dir: str, pkg_args: Namespace, xml_config: XmlConfig
) -> str:
    """获取makeself压缩命令"""
    suffix = xml_config.package_attr.get("suffix")
    package_name = PackageName(xml_config.package_attr, pkg_args, xml_config.version)
    if suffix == "run":
        factory = create_makeself_pkg_params_factory(
            pkg_args.pkg_output_dir, package_name.getvalue(), get_comments(package_name)
        )
        params = factory(
            pkg_args.makeself_dir, xml_config.package_attr, pkg_args.independent_pkg
        )
        pack_cmd, err_msg = create_run_package_command(params)
        if err_msg:
            CommLog.cilog_error(err_msg)
            CommLog.cilog_error("create_run_command failed!")
            raise CompressError(package_name.getvalue())
        if pkg_args.independent_pkg:
            exec_pack_cmd(delivery_dir, pack_cmd, package_name.getvalue())
        try:
            makeself_dir = os.path.join(build_dir, "makeself.txt")
            with open(makeself_dir, "w") as f:
                f.write(pack_cmd)
        except Exception as exception:
            CommLog.cilog_error(f"save makeself.txt failed!{str(exception)}")
            raise CompressError(package_name.getvalue()) from exception
    elif suffix == "rpm" or suffix == "deb":
        CommLog.cilog_info("Processing package type: %s", suffix)
    else:
        CommLog.cilog_error("the repack type '%s' is not support!", suffix)
        raise CompressError(package_name.getvalue())

    return package_name.getvalue()


def make_parse_option(args_: argparse.Namespace) -> ParseOption:
    """创建解析参数。"""

    return ParseOption(
        args_.os_arch,
        args_.pkg_version,
        args_.build_type,
        args_.package_check,
        args_.ext_name,
    )


PrivatePackageOption = namedtuple(
    "PrivatePackageOption",
    [
        "os_arch",
        "package_suffix",
        "not_in_name",
        "pkg_version",
        "ext_name",
        "chip_name",
        "func_name",
        "version_dir",
        "disable_multi_version",
        "suffix",
    ],
)


class PackageOption(PrivatePackageOption):
    """打包配置参数。"""

    __slots__ = ()  # 优化内存，避免创建 __dict__

    def __new__(cls, *package_option_args, **kwargs):
        return super().__new__(cls, *package_option_args, **kwargs)


def do_copy(target_conf=None, delivery_dir="", release_dir="", package_name=None):
    """
    功能描述：根据拷贝类型来执行文件或目录拷贝
    返回值：True(成功)/False(失败)
    """
    if target_conf is None:
        target_conf = {}
    target_name = get_target_name(target_conf)
    dst_path = os.path.join(release_dir, target_conf.get("dst_path", ""))
    dst_fullpath = os.path.join(dst_path, target_name)

    pkg_softlink = target_conf.get("pkg_softlink")
    if pkg_softlink:
        rets = [
            create_softlink(
                dst_fullpath,
                os.path.join(release_dir, link),
                target_conf.get("optional") == "true",
            )
            for link in pkg_softlink
        ]
        if not all(rets):
            return False
    return True


def do_chmod(target_conf=None, release_dir=""):
    """
    打包时设置权限。

    注意：do_chmod 当前未被任何代码路径调用，实际文件权限由 makeself.cmake 中的
    find -exec chmod 统一处理。保留此函数以兼容已有测试及潜在的外部调用方。
    """
    if target_conf is None:
        target_conf = {}
    target_name = get_target_name(target_conf)
    dst_path = os.path.join(release_dir, target_conf.get("dst_path", ""))
    dst_fullpath = os.path.join(dst_path, target_name)
    pkg_mod = target_conf.get("pkg_mod", "")
    if pkg_mod:
        # 将命令拆分为列表，不使用 shell=True
        cmd_list = ["chmod", "-R", pkg_mod, dst_fullpath]

        try:
            result = subprocess.run(
                cmd_list, capture_output=True, text=True, check=False
            )
            status = result.returncode
            output = result.stdout + result.stderr

            if status != 0:  # subprocess 返回码，0 表示成功
                CommLog.cilog_error("chmod failed! Command: %s", " ".join(cmd_list))
                CommLog.cilog_info("%s", output)
                return False
        except Exception as e:
            CommLog.cilog_error("Execute chmod exception: %s", str(e))
            return False

    return True


def create_softlink(source: str, target: str, optional: bool) -> bool:
    """
    功能描述：创建软连接
    参数：source, target
    返回值：True(成功)/False(失败)
    """
    source = os.path.abspath(source.strip())
    target = os.path.abspath(target.strip())

    if not os.path.exists(source):
        if optional:
            return True
        return False

    link_target_path = os.path.dirname(target)
    link_target_name = os.path.basename(target)
    relative_path = os.path.relpath(source, link_target_path)
    if os.path.isfile(target) or os.path.islink(target):
        cmd_list = ["rm", "-f", target]
        try:
            result = subprocess.run(
                cmd_list, capture_output=True, text=True, check=False
            )
            status = result.returncode
            output = result.stdout + result.stderr
            if status != 0:
                CommLog.cilog_error("Error: rm -f %s failed, %s", target, output)
                return False
        except Exception as e:
            CommLog.cilog_error("Execute rm exception: %s", str(e))
            return False
    if os.path.isdir(target):
        CommLog.cilog_warning(
            "Warning: %s is already a directory, skipping soft link creation.", target
        )
        return True
    if not os.path.exists(link_target_path):
        os.makedirs(link_target_path)
    tmp_dir = os.getcwd()
    os.chdir(link_target_path)
    os.symlink(relative_path, link_target_name)
    os.chdir(tmp_dir)
    return True


def generate_hash_list(target_conf, hash_cfg_str, release_dir):
    target_name = get_target_name(target_conf)
    dst_path = os.path.join(release_dir, target_conf.get("dst_path", ""))

    dst_full_path = os.path.join(dst_path, target_name)
    hash_cmd = "sha256sum " + dst_full_path
    cmd_list = shlex.split(hash_cmd)
    process = subprocess.run(
        cmd_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    stat, output = process.returncode, process.stdout
    if stat != 0:
        CommLog.cilog_error("get_image_hash command: %s", hash_cmd)
        CommLog.cilog_info("get_image_hash failed!(%s)", output)
        return False, None
    hash_value = output.split()[0]
    hash_cfg_str += f"{target_name}={hash_value}\n"
    return True, hash_cfg_str


def generate_hash_file(delivery_dir, hash_list):
    """
    功能描述：生成hash文件
    参数：delivery_dir打包的临时目录
          hash_list生成的cfg列表
    返回值：True(成功)/False(失败)
    """
    hash_path = os.path.join(delivery_dir, "bin_hash.cfg")
    if not os.path.exists(hash_path):
        cmd = "touch " + hash_path
        cmd_list = shlex.split(cmd)
        process = subprocess.run(
            cmd_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        stat, output = process.returncode, process.stdout
        if stat != 0:
            CommLog.cilog_error("%s failed!", cmd)
            CommLog.cilog_info("%s", output)
            return False
    with open(os.path.join(hash_path), "w") as fw:
        fw.write(hash_list)
    return True


def generate_info_content(target_conf, ext_name) -> List[str]:
    """生成info内容。"""

    def toolchain_llvm_config() -> Iterator[Tuple[str, str]]:
        if "llvm" in ext_name:
            yield "toolchain", "llvm"

    content_list = [
        f"{key}={value}"
        for key, value in chain(target_conf["content"].items(), toolchain_llvm_config())
    ]
    return content_list


def generate_version_header_content(target_conf) -> Iterator[str]:
    """生成version_header内容。"""
    guard_name = target_conf["value"].replace(".", "_").upper()
    yield f"#ifndef {guard_name}"
    yield f"#define {guard_name}"
    yield ""
    for name, value in target_conf["content"].items():
        if name.endswith("_VERSION"):
            version_infos = get_cann_version_info(name, value)
            for version_name, version_value in version_infos:
                yield f"#define {version_name} {version_value}"
        else:
            yield f"#define {name} {value}"
    yield ""
    yield f"#endif /* {guard_name} */"
    yield ""


def generate_customized_file(target_conf, ext_name, build_dir):
    filepath = os.path.join(
        build_dir, target_conf["dst_path"], target_conf.get("value")
    )
    dirpath = os.path.dirname(filepath)

    generator = target_conf.get("generator", "info")
    if generator == "version_header":
        content_list = generate_version_header_content(target_conf)
    else:
        content_list = generate_info_content(target_conf, ext_name)

    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    if os.path.exists(filepath):
        os.chmod(filepath, 0o700)

    file_content = "\n".join(content_list)
    try:
        with open(filepath, "w") as file:
            file.write(file_content)
    except Exception as ex:
        CommLog.cilog_error(f"generate customized file {filepath} failed: {ex}!")
        return False

    os.chmod(filepath, 0o440)

    return True


def get_module(target_config) -> str:
    """获取配置模块。"""
    module = target_config.get("module", "NA")
    return module if module else "NA"


def get_operation(operation, target_config) -> str:
    """获取操作类型。"""
    if operation in ("copy", "move") and target_config.get("entity") == "true":
        return "copy_entity"
    return operation


def get_permission(target_config) -> str:
    """获取配置权限。"""
    return target_config.get("install_mod", "NA")


def get_owner_group(target_config) -> str:
    """获取配置属主。"""
    # install_own的可能值为$username:$usergroup
    # 防止变量在install_common_parser.sh中，被eval展开，添加\转义$
    # 由于awk会消耗1个\，所以需要2个转义符
    return target_config.get("install_own", "NA").replace("$", "\\\\$")


def get_install_type(target_config) -> str:
    """获取安装类型。"""
    return target_config.get("install_type", "NA")


def get_softlink(target_config) -> List[str]:
    """获取配置软链。"""
    softlink_str = target_config.get("install_softlink")
    if not softlink_str:
        return []
    return softlink_str.split(";")


def get_feature(target_config) -> Set[str]:
    """获取配置特性。"""
    return target_config["feature"]


def get_chip(target_config) -> Set[str]:
    """获取配置芯片。"""
    return target_config["chip"]


def get_configurable(target_config) -> str:
    """获取配置是否为配置文件。"""
    return target_config.get("configurable", "FALSE")


def get_hash_value(target_config) -> str:
    """获取配置哈希值。"""
    return target_config.get("hash", "NA")


def get_block(target_config) -> str:
    """获取配置块信息。"""
    return target_config.get("name", "NA")


def get_pkg_inner_softlink(target_config) -> List[str]:
    """获取配置包内软链。"""
    softlink_str = target_config.get("pkg_inner_softlink")
    if not softlink_str:
        return []
    return softlink_str.split(";")


def validate_path_consistency(filelist: FileList):
    """验证 dst_path 和 install_path 一致性，不一致则抛出异常。"""
    succ = True
    for fileitem in filelist:
        if fileitem.operation not in ("copy", "move", "copy_entity"):
            continue
        if fileitem.relative_path_in_pkg != fileitem.relative_install_path:
            CommLog.cilog_error(
                f"Configuration Error: 'dst_path' MUST be equal to 'install_path'.\n"
                f"  - Current dst_path: {fileitem.relative_path_in_pkg}\n"
                f"  - Current install_path: {fileitem.relative_install_path}\n"
            )
            succ = False

    if not succ:
        raise GenerateFilelistError("dst_path does not match install_path!")


def parse_install_info(
    delivery_dir: str, infos: List, operate_type, filter_key
) -> Iterator[FileItem]:
    """根据配置解析生成安装信息。"""
    for target_config in infos:
        target_name = get_target_name(target_config)
        if target_config.get("optional") == "true" and operate_type in ("copy", "move"):
            path = os.path.join(delivery_dir, target_config.get("dst_path"))
            value = os.path.join(
                delivery_dir, target_config.get("dst_path"), target_name
            )
            if not os.path.exists(path):
                continue
            if not os.path.exists(value):
                continue
        if operate_type in ("copy", "move"):
            relative_path_in_pkg = os.path.join(
                target_config.get("dst_path"), target_name
            )
            relative_install_path = path_join(
                target_config.get("install_path"), target_name
            )
            is_dir = target_config.get("is_dir", False)
        elif operate_type == "mkdir":
            relative_path_in_pkg = "NA"
            relative_install_path = target_config.get("value")
            is_dir = False
        elif operate_type == "del":
            relative_path_in_pkg = "NA"
            relative_install_path = path_join(
                target_config.get("install_path"), target_name
            )
            is_dir = False
        else:
            raise UnknownOperateTypeError(f"unknown operate type {operate_type}")

        if relative_install_path is None:
            continue

        install_type = get_install_type(target_config)
        if any(key in install_type for key in filter_key):
            is_in_docker = "TRUE"
        else:
            is_in_docker = "FALSE"

        file_item = create_file_item(
            get_module(target_config),
            get_operation(operate_type, target_config),
            relative_path_in_pkg,
            relative_install_path,
            is_in_docker,
            get_permission(target_config),
            get_owner_group(target_config),
            install_type,
            get_softlink(target_config),
            get_feature(target_config),
            "N",
            get_configurable(target_config),
            get_hash_value(target_config),
            get_block(target_config),
            get_pkg_inner_softlink(target_config),
            get_chip(target_config),
            is_dir,
        )

        yield file_item


def execute_repack_process(
    xmlconfig: XmlConfig,
    delivery_dir: str,
    pkg_args: Namespace,
    package_name: PackageName = None,
    package_option: PackageOption = None,
):
    """
    功能描述: 执行打包流程(拷贝--->签名--->打包)
    返回值: True(成功)/False(失败)
    """
    release_dir = delivery_dir
    build_dir = delivery_dir.replace("_CPack_Packages/makeself_staging", "")
    # 生成自定义文件
    for item in xmlconfig.generate_infos:
        if not generate_customized_file(item, package_option.ext_name, delivery_dir):
            return False

    hash_cfg_str = ""
    for item in chain(xmlconfig.package_content_list, xmlconfig.move_content_list):
        if get_target_name(item) == "bin_hash.cfg":
            ret = generate_hash_file(delivery_dir, hash_cfg_str)
            if not ret:
                CommLog.cilog_error("generate hash file failed!")
                return False
        # 拷贝文件
        if not do_copy(item, delivery_dir, release_dir, package_name):
            continue
        if "is_hash" in item:
            ret, hash_cfg_str = generate_hash_list(item, hash_cfg_str, release_dir)
            if not ret:
                CommLog.cilog_error(
                    "generate hash command %s failed!", get_target_name(item)
                )
                return False

    softlink_before_package(xmlconfig.pkg_softlinks, release_dir)

    # 校验包中文件或目录大小
    if pkg_args.check_size == "True":
        limit_list, tag = processing_csv_file(
            release_dir,
            package_name.func_name,
            package_name.chip_name,
            pkg_args.build_type,
        )
        if not tag:
            return False
        if limit_list:
            abspath = os.path.abspath(release_dir)
            replace_path = abspath + "/"
            result = check_add_dir(replace_path, abspath, limit_list)
            if not result:
                return False
    try:
        package_name = get_compress_cmd(
            pkg_args.pkg_output_dir, build_dir, pkg_args, xmlconfig
        )
    except CompressError:
        return False

    CommLog.cilog_info(
        "package %s generate filelist.csv and makeself cmd successfully!", package_name
    )
    return True


def check_path_is_conflict(xml_config):
    """
    功能描述: 检查打包时安装路径与软连接路径是否冲突
    参数: xml_config
    返回值: True(成功)/False(失败)
    """
    install_path_list = set()
    pkg_softlink_list = set()
    for item in xml_config.package_content_list:
        value_list = item.get("value").split("/")
        target_name = value_list[-1] if value_list[-1] else value_list[-2]
        if item.get("install_path"):
            install_path_list.add(os.path.join(item["install_path"], target_name))
        if item.get("pkg_inner_softlink"):
            pkg_softlink = item.get("pkg_inner_softlink")
            pkg_softlink_list.add(pkg_softlink)
    if install_path_list & pkg_softlink_list:
        CommLog.cilog_info(
            "intersection:{}".format(install_path_list & pkg_softlink_list)
        )
        CommLog.cilog_info(
            "path conflicting: pkg_inner_softlink dir equals install_path!!"
        )
        return False
    return True


def checksum_value(limit_value, release_dir):
    """
    功能描叙: 校验传入的文件或目录大小是否合格
    参数:
    limit_value: limit.csv中的一行数据如[compiler/bin, 3976, 110%]
    返回值: True(成功)/False(失败)
    """
    path = os.path.join(release_dir, limit_value[1])
    if len(limit_value) >= 7:
        try:
            max_value = int(limit_value[4])
        except ValueError:
            CommLog.cilog_error(
                "{0} configuration is not standard., Please check limit.csv.".format(
                    path
                )
            )
            return True
    else:
        CommLog.cilog_error(
            "{0} configuration is less than four, Please check limit.csv.".format(path)
        )
        return True
    if not os.path.exists(path):
        CommLog.cilog_warning("{0} doesn't exist, Please check limit.csv.".format(path))
        return True
    size = 0
    for root, dirs, files in os.walk(path):
        size += os.path.getsize(root)
        for f in files:
            filepath = os.path.join(root, f)
            if os.path.islink(filepath):
                continue
            if not os.path.exists(filepath):
                continue
            size += os.path.getsize(os.path.join(root, f))
    if size == 0:
        size = os.path.getsize(path)
    if size > max_value * 1024:
        CommLog.cilog_error(
            f"\n{path} size {size} bytes exceeds maximum {max_value * 1024} bytes"
        )
        return False
    return True


def processing_csv_file(release_dir, package_name, chip_name, build_type):
    """
    功能描叙: 处理limit.csv文件数据
    返回值: [],True(成功)/[],False(失败)
    """
    ret = True
    limit_list = []
    product = os.path.basename(os.path.dirname(release_dir))
    limit_path = os.path.join(
        pkg_utils.TOP_SOURCE_DIR, CONFIG_SCRIPT_PATH, "common/limit.csv"
    )
    if not os.path.exists(limit_path):
        CommLog.cilog_warning("{0} doesn't exist.".format(limit_path))
        return limit_list, ret
    with open(limit_path, "r") as file:
        reader = csv.reader(file)
        next(reader)
        for data in reader:
            if not data:
                CommLog.cilog_warning("The limit.csv file contains empty lines.")
                continue
            if is_match_line(package_name, chip_name, product, build_type, data):
                if data[1][-1] == "/":
                    limit_list.append(data[1][:-1])
                else:
                    limit_list.append(data[1])
                res = checksum_value(data, release_dir)
                if not res:
                    ret = False
    return limit_list, ret


def is_match_line(package_name, chip_name, product, build_type, data):
    return (
        package_name == data[0]
        and chip_name == data[5]
        and product == data[6]
        and build_type == data[7].lower()
    )


def check_add_dir(package_path, dirs, limit_list, ret=True):
    """
    功能描述: 校验新增目录
    参数: path, limit_list
    返回值: True(成功)/False(失败)
    """
    for limit_path in limit_list:
        if dirs == os.path.join(os.path.split(dirs)[0], limit_path):
            return ret
    for dir_file in os.listdir(dirs):
        path = os.path.join(dirs, dir_file)
        relative_path = path.replace(package_path, "")
        if os.path.isfile(path) and relative_path not in limit_list:
            CommLog.cilog_error(
                "{0} is not in limit.csv file and is newly added.".format(path)
            )
            ret = False
        elif os.path.isdir(path) and relative_path not in limit_list:
            ret = check_add_dir(package_path, path, limit_list, ret)
    return ret


def gen_file_install_list(
    delivery_dir: str, xml_config: XmlConfig, filter_key
) -> Tuple[FileList, FileList]:
    """生成filelist列表。"""
    file_install_list = []

    dir_filelist = parse_install_info(
        delivery_dir, xml_config.dir_install_list, "mkdir", filter_key
    )
    move_filelist = parse_install_info(
        delivery_dir, xml_config.move_content_list, "move", filter_key
    )
    pkg_filelist = parse_install_info(
        delivery_dir, xml_config.package_content_list, "copy", filter_key
    )
    gen_filelist = parse_install_info(
        delivery_dir, xml_config.generate_infos, "copy", filter_key
    )
    # file_info中配置为文件夹，这里是被展开的文件,则需要单独删除
    del_filelist = parse_install_info(
        delivery_dir, xml_config.expand_content_list, "del", filter_key
    )
    collect_filelist = list(
        chain(dir_filelist, move_filelist, pkg_filelist, gen_filelist)
    )
    collect_filelist = list(
        xml_config.packer_config.fill_is_common_path(collect_filelist)
    )
    all_filelist = list(chain(collect_filelist, del_filelist))
    for file_item in all_filelist:
        file_install_list.append(file_item)

    return file_install_list, []


def get_share_info_name(package_attr: dict) -> str:
    """获取share/info下的目录名。"""
    if "share_info_name" in package_attr:
        return package_attr["share_info_name"]
    return package_attr["func_name"]


def generate_modules_yaml(
    items, module_name, version, source_dir, output_txt, script_dir=None
):
    modules = defaultdict(list)
    module_set = set()

    for item in items:
        block = item.block
        softlinks = item.softlink
        target = item.relative_install_path

        if not block or block.upper() == "NA":
            continue
        if not softlinks:
            continue
        if not target:
            continue
        module_set.add(block)
        for softlink in softlinks:
            symlink_entry = {"target": target, "link": softlink}
            if symlink_entry not in modules[block]:
                modules[block].append(symlink_entry)
    permission_lines = generate_set_permission(items, version, script_dir)
    postinst_content = generate_postinst(
        module_name, version, modules, source_dir, permission_lines
    )
    prerm_content = generate_prerm(module_name, version, modules, source_dir)
    output_postinst = "postinst"
    output_prerm = "prerm"
    with open(output_postinst, "w") as f:
        f.write(postinst_content)
    os.chmod(output_postinst, 0o755)

    with open(output_prerm, "w") as f:
        f.write(prerm_content)
    os.chmod(output_prerm, 0o755)

    CommLog.cilog_info("Generated: %s, %s", output_postinst, output_prerm)
    with open(output_txt, "w") as f:
        json.dump(modules, f, indent=2, ensure_ascii=False)
    CommLog.cilog_info("Generated %s with %d modules", output_txt, len(modules))


def generate_filelist_file_by_xml_config(
    xml_config: XmlConfig,
    filter_key: List[str],
    delivery_dir: str,
    package_check: bool,
    suffix: str,
    pkg_name: str,
    version: str,
    source_dir: str,
):
    """生成文件列表文件。"""
    check_move = xml_config.package_attr.get("use_move", False)
    transform_nested_path_func = get_transform_nested_path_func(
        xml_config.package_attr.get("parallel") or check_move
    )
    check_features = xml_config.package_attr.get("check_features", False)

    file_install_list, [] = invoke(
        pipe(
            partial(gen_file_install_list, delivery_dir),
            partial(map, transform_nested_path_func),
            tuple,
        ),
        xml_config,
        filter_key,
    )
    file_install_list = list(file_install_list)
    # 根据suffix，生成postin和prein
    if suffix == "rpm" or suffix == "deb":
        script_dir = f"share/info/{get_share_info_name(xml_config.package_attr)}/script"
        generate_modules_yaml(
            file_install_list, pkg_name, version, source_dir, "modules.txt", script_dir
        )
    # 生成RECORD文件(记录具体文件安装路径)，并将其作为copy条目加入filelist随包打包
    share_info_name = get_share_info_name(xml_config.package_attr)
    record_item = generate_record_file(file_install_list, delivery_dir, share_info_name)
    file_install_list.append(record_item)
    generate_filelist(
        file_install_list,
        "filelist.csv",
        os.path.join(delivery_dir, "share", "info", share_info_name, "script"),
    )
    # 先生成再检查，有利于问题定位
    if package_check:
        check_filelist(file_install_list, check_features, check_move)
    if xml_config.package_attr.get("copy_all"):
        # 验证dst_path和install_path的一致性
        validate_path_consistency(file_install_list)


def get_pkg_xml_relative_path(pkg_args: Namespace) -> str:
    """获取包配置文件相对路径。"""

    def parts():
        yield pkg_args.pkg_name
        if pkg_args.chip_scenes:
            yield pkg_args.chip_scenes
        # 可以通過build_rule指定xml_file，而且优先级高于默认值
        if pkg_args.xml_file:
            yield pkg_args.xml_file
        else:
            yield f"{pkg_args.pkg_name}.xml"

    return os.path.join(*parts())


def get_filter_key(pkg_name):
    if pkg_name in ["driver", "firmware"]:
        return ["all", "docker"]
    return (
        []
        if pkg_name in ["aicpu_kernels_device", "aicpu_kernels_host"]
        else ["all", "run"]
    )


def main(pkg_name="", xml_file="", main_args=None):
    """
    功能描述: 执行打包流程(解析配置--->生成文件列表--->执行拷贝/打包动作)
    参数: pkg_name, os_arch, type
    返回值: True(成功)/False(失败)
    """
    if not main_args.delivery_dir:
        CommLog.cilog_error("Delivery dir is empty.")
        return False

    if main_args.suffix == "rpm" or main_args.suffix == "deb":
        delivery_dir = main_args.delivery_dir
    else:
        if os.path.basename(main_args.delivery_dir) == "makeself_staging":
            delivery_dir = main_args.delivery_dir
        else:
            delivery_dir = os.path.join(
                main_args.delivery_dir, "_CPack_Packages/makeself_staging"
            )
    if not os.path.exists(delivery_dir):
        CommLog.cilog_error(f"Delivery dir does not exist: {delivery_dir}")
        return False

    config_relative_path = get_pkg_xml_relative_path(main_args)
    parse_option = make_parse_option(main_args)

    try:
        ret, xml_config = parse_xml_config(
            os.path.join(main_args.source_dir, "scripts", "package"),
            config_relative_path,
            delivery_dir,
            parse_option,
            main_args,
        )
    except ContainAsteriskError as ex:
        CommLog.cilog_error(
            f"Value contain '*' in {config_relative_path}. value is '{ex.value}'."
        )
        return False
    if not ret:
        return False

    # 生成filelist.csv安装列表文件
    try:
        generate_filelist_file_by_xml_config(
            xml_config,
            get_filter_key(pkg_name),
            delivery_dir,
            main_args.package_check or xml_config.package_attr.get("package_check"),
            main_args.suffix,
            main_args.pkg_name,
            main_args.version_dir,
            main_args.source_dir,
        )
    except PackageNameEmptyError:
        CommLog.cilog_error(f"package name is empty in {xml_file}, please check it")
        return False
    except GenerateFilelistError as ex:
        CommLog.cilog_error(
            f"generate filelist {ex.filename} failed!",
        )
        return False
    except FilelistError as ex:
        CommLog.cilog_error("check filelist error! %s", str(ex))
        return False

    package_option = PackageOption(
        main_args.os_arch,
        main_args.package_suffix,
        main_args.not_in_name,
        main_args.pkg_version,
        main_args.ext_name,
        chip_name=main_args.chip_name,
        func_name=main_args.func_name,
        version_dir=main_args.version_dir,
        disable_multi_version=main_args.disable_multi_version,
        suffix=main_args.suffix,
    )

    package_name = PackageName(xml_config.package_attr, main_args, xml_config.version)

    # 检查install_path与pkg_inner_softlink路径是否冲突，若冲突则报错
    if not check_path_is_conflict(xml_config):
        return False

    # 生成打包命令
    return execute_repack_process(
        xml_config,
        delivery_dir,
        main_args,
        package_name=package_name,
        package_option=package_option,
    )


def lower_arg(value):
    return value.lower()


def args_parse():
    """
    功能描述 : 脚本入参解析
    参数 : 调用脚本的传参
    返回值 : 解析后的参数值
    """
    parser = argparse.ArgumentParser(
        description="This script is for package repack processing."
    )
    parser.add_argument(
        "-c",
        "--chip_scenes",
        metavar="chip_scenes",
        required=False,
        dest="chip_scenes",
        nargs="?",
        const="",
        default="",
        help="This parameter define chip id for package.",
    )
    parser.add_argument(
        "-n",
        "--pkg_name",
        metavar="pkg_name",
        required=False,
        help="This parameter define pkg_name for config_xml.",
    )
    parser.add_argument(
        "-o",
        "--os_arch",
        metavar="os_arch",
        required=False,
        dest="os_arch",
        nargs="?",
        const="",
        default=None,
        help="This parameter define the package's os_arch",
    )
    parser.add_argument(
        "-t",
        "--type",
        metavar="type",
        required=False,
        dest="type",
        nargs="?",
        const="",
        default="repack",
        help="This parameter define this script's function",
    )
    parser.add_argument(
        "-i",
        "--not_in_name",
        metavar="not_in_name",
        required=False,
        dest="not_in_name",
        nargs="?",
        const="",
        default="",
        help="This parameter define the package's name not contain the element",
    )
    parser.add_argument(
        "-v",
        "--pkg_version",
        metavar="pkg_version",
        required=False,
        dest="pkg_version",
        nargs="?",
        const="",
        default="",
        help="This parameter define the version for package.",
    )
    parser.add_argument(
        "-e",
        "--ext_name",
        metavar="ext_name",
        required=False,
        dest="ext_name",
        nargs="?",
        const="",
        default="",
        help="This parameter define the package's ext_name",
    )
    parser.add_argument(
        "--package_suffix",
        nargs="?",
        const="none",
        default="none",
        help="This parameter define the package suffix, debug or none",
    )
    parser.add_argument(
        "--suffix",
        metavar="suffix",
        type=lower_arg,
        required=False,
        dest="suffix",
        nargs="?",
        const="",
        default=None,
        help="This parameter define the package suffix, for example such as tar.gz",
    )
    parser.add_argument(
        "-b",
        "--build_type",
        metavar="build_type",
        required=False,
        dest="build_type",
        nargs="?",
        const="",
        default="debug",
        help="This parameter define release type of package",
    )
    parser.add_argument(
        "-x",
        "--xml",
        metavar="xml_file",
        required=False,
        dest="xml_file",
        nargs="?",
        const="",
        default="",
        help="This parameter define xml file",
    )
    parser.add_argument(
        "--chip_name",
        metavar="chip_name",
        required=False,
        dest="chip_name",
        nargs="?",
        const=None,
        default=None,
        help="This parameter define package chip name, has higher priority than chip name in xml",
    )
    parser.add_argument(
        "--func_name",
        metavar="func_name",
        required=False,
        dest="func_name",
        nargs="?",
        const=None,
        default=None,
        help="This parameter define package func name, has higher priority than func name in xml",
    )
    parser.add_argument(
        "--source_root",
        metavar="source_root",
        required=False,
        dest="source_root",
        nargs="?",
        const="",
        help="source root dir.",
    )
    parser.add_argument(
        "--makeself_dir",
        metavar="makeself_dir",
        required=False,
        dest="makeself_dir",
        nargs="?",
        const="",
        help="makeself dir.",
    )
    parser.add_argument(
        "--delivery_dir", metavar="delivery_dir", required=True, help="delivery dir."
    )
    parser.add_argument(
        "--source_dir", metavar="source_dir", required=True, help="source dir."
    )
    parser.add_argument(
        "--independent_pkg", action="store_true", help="Independent pkg."
    )
    parser.add_argument("--pkg-output-dir", default="", help="Package dirpath.")
    parser.add_argument(
        "--version_dir", nargs="?", const="", default="", help="Set version dir."
    )
    parser.add_argument("--tag", metavar="tag", nargs="?", const="", default="")
    parser.add_argument(
        "--disable-multi-version", action="store_true", help="Disable multi version."
    )
    # 检查打包配置
    parser.add_argument(
        "--package-check", action="store_true", help="check package config."
    )
    parser.add_argument(
        "--check_size",
        nargs="?",
        const="",
        default="",
        help="Check the size of a file or directory.",
    )
    parser.add_argument(
        "--pkg-name-style",
        metavar="pkg_name_style",
        default="common",
        help="Package name style.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    CommLog.cilog_info("%s", " ".join(sys.argv))
    args = args_parse()
    try:
        if args.source_root:
            pkg_utils.TOP_SOURCE_DIR = args.source_root
        if args.build_type == "":
            args.build_type = "debug"
        else:
            args.build_type = args.build_type.lower()
        status = main(args.pkg_name, args.xml_file, main_args=args)
    except Exception as e:
        CommLog.cilog_error("exception is occurred (%s)!", e)
        CommLog.cilog_info("%s", traceback.format_exc())
        status = False
    sys.exit(0 if status else 1)
