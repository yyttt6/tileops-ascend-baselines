#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# **************************************************************
# 文件名    ：add_header_sign.py
# 版本号    ：初稿
# 生成日期  ：2025年11月25日
# 功能描述  ：根据bios_check_cfg.xml配置文件，对各镜像文件制作cms签名并绑定ESBC头
# 使用方法  ：python3 add_header_sign.py <sign_file_dir> <sign_flag>
#             [--bios_check_cfg=<cfg>] [--version=<ver>] [--sign_script=<path>]
# 输入参数  ：sign_file_dir：待签名文件的根目录
#            sign_flag：是否需要数字签名（true/false，默认false）
#            --bios_check_cfg：签名配置XML文件（默认bios_check_cfg.xml）
#            --version：版本号
#            --sign_script：签名插件脚本路径
#            签名步骤 1、加esbc头; 2、生成ini文件; 3、进行签名; 4、签名结果写入文件头
# 返回值    ：True:成功，False:失败
# 修改历史  ：
# 日期    ：2025年11月25日
# 修改内容 ：创建文件
#          重构：统一日志为标准库 logging，AddHeaderConfig 改为 dataclass，
#               提取 build_image_pack_cmd 降低圈复杂度
# **************************************************************

import os
import sys
import time
import shutil
import logging
import argparse
import shlex
from dataclasses import dataclass
from subprocess import run, PIPE, STDOUT, TimeoutExpired as TIMEOUT_EXPIRED
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

MY_PATH = os.path.dirname(os.path.realpath(__file__))

# 模块级常量，setenv() 会更新此值；未设置时回退到当前 Python 解释器
HI_PYTHON = os.environ.get("HI_PYTHON", sys.executable)


# AddHeaderConfig XML 属性名 → (字段名, 默认值) 映射表
_CONFIG_ATTR_MAP: List[Tuple[str, str, str]] = [
    ("input", "input", ""),
    ("output", "output", ""),
    ("version", "version", ""),
    ("fw_version", "fw_version", ""),
    ("type", "type", ""),
    ("tag", "tag", ""),
    ("sign_alg", "sign_alg", "PKCSv1.5"),
    ("encrypt_alg", "encrypt_alg", ""),
    ("encrypt_type", "encrypt_type", ""),
    ("additional", "additional", ""),
    ("nvcnt", "nvcnt", ""),
    ("rsatag", "rsatag", ""),
    ("position", "position", ""),
    ("image_pack", "image_pack_version", "1.0"),
    ("rootrsa", "rootrsa", "default_rsa_rootkey"),
    ("subrsa", "subrsa", "default_rsa_subkey"),
    ("bist_flag", "bist_flag", ""),
]


@dataclass
class AddHeaderConfig:
    """加头工具配置类，从 bios_check_cfg.xml 的 <item> 节点解析。"""

    input: str = ""
    output: str = ""
    version: str = ""
    fw_version: str = ""
    type: str = ""
    tag: str = ""
    sign_alg: str = "PKCSv1.5"
    encrypt_alg: str = ""
    encrypt_type: str = ""
    additional: str = ""
    nvcnt: str = ""
    rsatag: str = ""
    position: str = ""
    image_pack_version: str = "1.0"
    rootrsa: str = "default_rsa_rootkey"
    subrsa: str = "default_rsa_subkey"
    bist_flag: str = ""

    @classmethod
    def from_xml(cls, node) -> "AddHeaderConfig":
        """从 XML <item> 节点解析属性，未配置的属性使用默认值。"""
        kwargs = {}
        for xml_attr, field_name, default in _CONFIG_ATTR_MAP:
            kwargs[field_name] = node.attrib.get(xml_attr, default)
        return cls(**kwargs)


def read_xml(in_path) -> Optional[ET.ElementTree]:
    """
    功能：读取XML，解析失败返回None
    """
    try:
        tree = ET.ElementTree()
        tree.parse(in_path)
        return tree
    except ET.ParseError as e:
        logger.error("parse xml failed: %s\n\t%s", in_path, e)
        return None


def check_config_item(node) -> bool:
    """校验节点必需属性"""
    if "input" not in node.attrib or "output" not in node.attrib:
        logger.error("bios_check_cfg.xml config format is invalid")
        return False

    if "type" in node.attrib:
        if "cms" in node.attrib["type"].split("/") and "tag" not in node.attrib:
            logger.error(
                "when bios_check_cfg.xml has cms type, it must has 'tag' attribute"
            )
            return False

    return True


@dataclass
class ImagePackParam:
    """build_image_pack_cmd 的参数封装。"""

    conf_item: AddHeaderConfig
    input_file: str
    sign_path: str
    der_file: str
    add_sign: str
    image_pack_script: str
    sign_ext: str = ".p7s"
    sign_certtype: str = "1"


@dataclass
class BiosHeaderParam:
    """add_bios_header 的参数封装。"""

    item_size_set: Dict[str, AddHeaderConfig]
    sign_file_dir: str
    bios_tool_path: str
    sign_tool_path: str
    root_dir: str
    add_sign: str


def safe_run_cmd(cmd_list: List[str], work_dir=None, timeout=None) -> Tuple[bool, str]:
    """安全执行列表格式命令，返回(是否成功, 输出)。

    timeout 为可选超时（秒），超时后返回 (False, 超时信息)。
    """
    try:
        result = run(
            cmd_list,
            cwd=work_dir,
            shell=False,
            stdout=PIPE,
            stderr=STDOUT,
            text=True,
            encoding="utf-8",
            timeout=timeout,
        )
        return result.returncode == 0, result.stdout
    except OSError as e:
        logger.error(
            "command not found or not executable: %s\n\t%s", " ".join(cmd_list), e
        )
        return False, str(e)
    except TIMEOUT_EXPIRED as e:
        logger.error("command timeout after %ss: %s", timeout, " ".join(cmd_list))
        return False, str(e)


# 签名产物类型默认值，查询失败时回退，保持对未更新第三方脚本的向后兼容
DEFAULT_SIGN_EXT = ".p7s"
DEFAULT_CERTTYPE = "1"
# 查询子进程超时（秒），查询仅做 arg 解析 + print，10 秒足够
QUERY_TIMEOUT = 10


def query_sign_attr(sign_tool_path: str, flag: str) -> Optional[str]:
    """查询签名脚本的某个属性（扩展名或证书类型）。

    调用 `python3 <sign_script> <flag>`，解析 stdout 最后一行非空行。
    查询失败（脚本不识别 flag、执行错误、输出为空等）返回 None，
    由调用方决定回退策略。
    """
    cmd = [HI_PYTHON, sign_tool_path, flag]
    success, output = safe_run_cmd(cmd, timeout=QUERY_TIMEOUT)
    if not success:
        logger.warning("query %s failed", flag)
        return None
    lines = [line.strip() for line in output.strip().splitlines() if line.strip()]
    if not lines:
        logger.warning("empty %s output", flag)
        return None
    return lines[-1]


def query_sign_ext(sign_tool_path: str) -> str:
    """查询签名产物扩展名。失败时回退到默认值，保持向后兼容。

    校验扩展名非空且以点号开头（如 `.p7s`、`.cms`），否则回退默认值，
    避免脚本返回无点号值（如 `p7s`）导致路径拼接错误。
    """
    ext = query_sign_attr(sign_tool_path, "--print-sign-ext")
    if ext is None or not ext.startswith("."):
        logger.warning(
            "query sign ext failed or invalid: %r, fallback to default: %s",
            ext,
            DEFAULT_SIGN_EXT,
        )
        return DEFAULT_SIGN_EXT
    logger.info("sign ext: %s", ext)
    return ext


def query_certtype(sign_tool_path: str) -> str:
    """查询证书类型并转为十进制字符串。失败时回退到默认值，保持向后兼容。

    返回十进制字符串是因为 image_pack.py 的 -certtype 参数使用 type=int，
    int("0xFFFFFFFF") 会报错，必须传十进制。
    非法十六进制字符串（如 "abc"）会触发 ValueError，同样回退默认值。
    """
    certtype = query_sign_attr(sign_tool_path, "--print-certtype")
    if certtype is None:
        logger.warning(
            "query certtype failed, fallback to default: %s", DEFAULT_CERTTYPE
        )
        return DEFAULT_CERTTYPE
    try:
        certtype_decimal = str(int(certtype, 0))
    except ValueError:
        logger.warning(
            "invalid certtype %r, fallback to default: %s", certtype, DEFAULT_CERTTYPE
        )
        return DEFAULT_CERTTYPE
    logger.info("certtype: %s -> %s", certtype, certtype_decimal)
    return certtype_decimal


def get_item_set(config_file, sign_file_dir, version) -> Tuple[bool, Optional[Dict]]:
    """
    功能：解析 bios_check_cfg.xml 配置文件，返回每个 <item> 节点对应的 AddHeaderConfig
    返回：(是否成功, 配置字典)。失败时第二个元素为 None。
          配置字典 key 为 input 属性值，value 为 AddHeaderConfig 对象。
    """
    item_size_set = {}
    tree = read_xml(config_file)
    if tree is None:
        return False, None

    origin_nodes = tree.findall("item")

    for node in origin_nodes:
        if "version" not in node.attrib:
            node.attrib["version"] = version
        if not check_config_item(node):
            return False, None

    # 排除不存在的文件
    nodes = []
    for node in origin_nodes:
        input_file = os.path.join(sign_file_dir, node.attrib["input"])

        if os.path.exists(input_file):
            nodes.append(node)
        else:
            logger.warning("Image file:%s not exists!", input_file)
            continue

    for node in nodes:
        cur_conf = AddHeaderConfig.from_xml(node)
        item_size_set[cur_conf.input] = cur_conf

    return True, item_size_set


def build_inifile(
    item_size_set, sign_file_dir, bios_tool_path, sign_tmp_path, add_sign
) -> bool:
    """
    功能：根据从bios_check_cfg.xml读取的配置，生成ini工具(ini_gen.py)的配置文件image_info.xml，
    然后调用ini_gen.py读取该配置文件生成每个镜像对应的ini摘要文件（内含SHA256哈希值）
    输入：item_size_set：待制作ini镜像的配置清单
          sign_file_dir：镜像根路径
          bios_tool_path：ini_gen.py 所在目录
          sign_tmp_path：临时工作目录，image_info.xml 和 ini 文件均生成于此
          add_sign：是否签名，仅 "true" 时才执行 ini 生成
    返回：False:失败，True：成功
    """
    cms_flag = False
    cmd = []
    # 仅在签名模式下才需要生成 ini 摘要文件
    if add_sign == "true":
        if not os.path.isdir(sign_tmp_path):
            os.makedirs(sign_tmp_path, exist_ok=True)
        # 用 ET 构造 image_info.xml，列出所有需要 cms 签名的镜像
        root = ET.Element("image_info")
        for infile, conf_item in item_size_set.items():
            inputfile = os.path.join(sign_file_dir, infile)
            # ini 输出目录与镜像在 sign_tmp_path 下的目录结构保持一致
            output_path = os.path.dirname(os.path.join(sign_tmp_path, infile))
            output_path = os.path.realpath(output_path)
            if not os.path.isdir(output_path):
                os.makedirs(output_path, exist_ok=True)
            # 只对 type 含 cms 且有 tag 的镜像生成 ini 条目
            if (
                "cms" in conf_item.type.split("/")
                and conf_item.tag != ""
                and os.path.isfile(inputfile)
            ):
                cms_flag = True
                image_elem = ET.SubElement(root, "image")
                image_elem.set("path", inputfile)
                image_elem.set("out", output_path)
                image_elem.set("tag", conf_item.tag)
                image_elem.set("ini_name", os.path.basename(infile))
        # 仅当存在 cms 类型镜像时才写 image_info.xml 并构建命令，避免生成空文件
        if cms_flag:
            ET.ElementTree(root).write(
                os.path.join(sign_tmp_path, "image_info.xml"),
                encoding="utf-8",
                xml_declaration=True,
            )
            gen_tool = os.path.join(bios_tool_path, "ini_gen.py")
            cmd = [
                HI_PYTHON,
                gen_tool,
                "-in_xml",
                os.path.join(sign_tmp_path, "image_info.xml"),
            ]

    # 仅当存在 cms 类型镜像时才调用 ini_gen.py，避免无意义执行
    if add_sign == "true" and cms_flag:
        logger.info("execute:%s", " ".join(cmd))
        success, output = safe_run_cmd(cmd)
        if not success:
            logger.error("build inifile failed!\n\t%s", output)
            return False
    return True


def build_sign(item_size_set, sign_file_dir, sign_tool_path, sign_tmp_path) -> bool:
    """
    功能：对需要 cms 签名的镜像制作 CMS(p7s) 签名
    输入：item_size_set：待签名的镜像配置清单
          sign_file_dir：镜像根路径
          sign_tool_path：签名工具脚本路径（community_sign_build.py）
          sign_tmp_path：临时工作目录，镜像和 ini 文件拷贝到此目录后签名
    返回：False:失败，True：成功
    """
    # 收集所有需要 cms 签名的文件，校验源文件为普通文件（非目录/不存在）
    cms_files = []
    for infile, conf_item in item_size_set.items():
        input_path = os.path.join(sign_file_dir, infile)
        if not os.path.isfile(input_path):
            logger.error("infile is not exist or not a file:%s", input_path)
            return False
        if "cms" in conf_item.type.split("/"):
            cms_files.append(infile)

    # 第一阶段：将待签名文件拷贝到临时目录，并收集对应的 ini 文件路径
    ini_files = []
    for file in cms_files:
        file_with_path = os.path.join(sign_file_dir, file)
        # 临时目录下同层架子目录，包含文件名的完整路径
        file_sign_des = os.path.realpath(os.path.join(sign_tmp_path, file))
        # 临时目录的路径，不包含文件名
        sign_path = os.path.dirname(file_sign_des)

        if not os.path.isdir(sign_path):
            os.makedirs(sign_path, exist_ok=True)
        # 待签名文件拷贝到临时路径下
        shutil.copy(file_with_path, file_sign_des)
        if not os.path.isfile(file_sign_des):
            logger.error("copy %s --> %s fail", file_with_path, file_sign_des)
            return False
        # 临时目录下ini文件完整路径，实际前面 build_inifile 时已生成到对应目录下
        ini_file = "{}.ini".format(os.path.join(sign_path, os.path.basename(file)))
        ini_files.append(ini_file)

    # 第二阶段：一次性调用签名工具对所有文件进行 CMS 签名，避免逐文件调用导致重复签名
    if ini_files:
        # 通过 --crl-dir 传入每目标独立的 CRL 输出目录，避免并行签名时共享 SCRIPT_DIR/SWSCRL.crl
        cmd = [HI_PYTHON, sign_tool_path, "--crl-dir", sign_file_dir] + ini_files
        logger.info("execute:%s", " ".join(cmd))
        # 签名后会在 ini 文件同目录下生成 p7s 文件，比如 a.ini => a.ini.p7s
        success, output = safe_run_cmd(cmd)
        if not success:
            logger.error("make cms sign failed!\n\t%s", output)
            return False
        logger.info("%s", output)

    return True


def _base_header_args(input_file, conf_item) -> List[str]:
    """构建镜像加头/绑定的基础参数列表（raw_img/out_img/version/nvcnt/tag）。"""
    return [
        "-raw_img",
        input_file,
        "-out_img",
        input_file,
        "-version",
        conf_item.version,
        "-nvcnt",
        conf_item.nvcnt,
        "-tag",
        conf_item.tag,
    ]


def add_bios_esbc_header(root_dir, item_size_set, sign_file_dir) -> bool:
    """
    功能：对配置了 nvcnt 的镜像绑定 esbc 二级头（256字节），包含版本、tag、nvcnt 信息
    输入：root_dir：工程根路径，用于定位 esbc_header.py 工具
          item_size_set：待加头的镜像配置清单
          sign_file_dir：镜像根路径
    返回：False:失败，True：成功
    """
    bios_esbc_header_tool_path = os.path.join(
        root_dir, "scripts", "signtool", "esbc_header"
    )
    # 检查加头工具目录是否存在
    if not os.path.exists(bios_esbc_header_tool_path):
        logger.error("bios esbc tool dir not exists")
        return False

    for input_filename, conf_item in item_size_set.items():
        input_file = os.path.join(sign_file_dir, input_filename)

        if conf_item.nvcnt:
            cmd = [
                HI_PYTHON,
                os.path.join(bios_esbc_header_tool_path, "esbc_header.py"),
            ] + _base_header_args(input_file, conf_item)
            logger.info("execute:%s", " ".join(cmd))
            success, output = safe_run_cmd(cmd)
            if not success:
                logger.error("add %s esbc header failed!\n\t%s", input_file, output)
                return False
    return True


def is_pem_format(file_path: str) -> bool:
    """判断文件是否为 PEM 格式。

    读取文件头部字节，去除前导空白后判断是否以 PEM 起始标记
    ``-----BEGIN`` 开头。文件不存在或读取失败时返回 False。
    """
    try:
        with open(file_path, "rb") as f:
            head = f.read(64)
    except OSError:
        return False
    return head.lstrip().startswith(b"-----BEGIN")


def convert_der_file(crl_file: str, der_file: str) -> bool:
    """
    将 CRL 文件转换为 DER 格式，保存到 der_file 指定路径。
    输入为 PEM 格式时调用 openssl 转换；输入已是 DER 格式时直接拷贝，
    避免 openssl 默认按 PEM 解析导致的转换失败。
    返回值：
        True - 成功
        False - 失败（包括文件不存在、格式非法、OpenSSL 未安装、转换/拷贝失败等）
    """
    if not os.path.isfile(crl_file):
        logger.error("input CRL file not found: %s", crl_file)
        return False
    # 源/目标同路径：输入已是目标格式，无需转换或拷贝
    if os.path.abspath(crl_file) == os.path.abspath(der_file):
        return True
    if is_pem_format(crl_file):
        cmd = ["openssl", "crl", "-in", crl_file, "-outform", "DER", "-out", der_file]
        success, output = safe_run_cmd(cmd)
        if not success:
            logger.error("openssl conversion failed: %s", output)
            return False
        return True
    # 非 PEM：校验是否为有效 DER（首字节为 ASN.1 SEQUENCE 标签 0x30 且非空）
    try:
        with open(crl_file, "rb") as f:
            first_byte = f.read(1)
    except OSError as e:
        logger.error("read file for DER check failed: %s\n\t%s", crl_file, e)
        return False
    if first_byte != b"\x30":
        logger.error("CRL file is neither valid PEM nor valid DER: %s", crl_file)
        return False
    try:
        shutil.copy(crl_file, der_file)
    except OSError as e:
        logger.error("copy DER file failed: %s -> %s\n\t%s", crl_file, der_file, e)
        return False
    return True


def build_image_pack_cmd(param: ImagePackParam) -> List[str]:
    """为单个镜像构建 image_pack.py 绑定命令。

    流程1（不签名）：仅绑定基础头信息（version/nvcnt/tag）
    流程2（签名模式）：绑定 cms 签名信息（p7s/ini/crl）
    """
    conf_item = param.conf_item
    input_file = param.input_file
    sign_path = param.sign_path
    der_file = param.der_file
    add_sign = param.add_sign
    image_pack_script = param.image_pack_script
    sign_ext = param.sign_ext
    sign_certtype = param.sign_certtype
    cmd = [HI_PYTHON, image_pack_script]

    if add_sign != "true" or conf_item.type == "":
        # 流程1：不签名，仅绑定基础头信息
        cmd += _base_header_args(input_file, conf_item)
        if conf_item.position != "":
            cmd += ["-position", conf_item.position]
    elif add_sign == "true" and conf_item.type != "":
        # 流程2：签名模式，绑定 cms 签名信息
        # 基础参数只追加一次，避免 type 含多种签名方式时重复
        add_cmd_parts = (
            shlex.split(conf_item.additional) if conf_item.additional else []
        )
        cmd += _base_header_args(input_file, conf_item) + add_cmd_parts

        # 循环内只追加各签名类型特有的参数
        for sign in conf_item.type.split("/"):
            if sign == "cms":
                # 临时目录下的ini文件路径
                ini_file = os.path.join(sign_path, os.path.basename(input_file))
                # 签名扩展名与证书类型由签名脚本查询得到（--print-sign-ext/--print-certtype）
                cmd += [
                    "-cms",
                    "{}.ini{}".format(ini_file, sign_ext),
                    "-ini",
                    "{}.ini".format(ini_file),
                    "-crl",
                    der_file,
                    "-certtype",
                    sign_certtype,
                    "--addcms",
                ]
        if conf_item.position != "":
            cmd += ["-position", conf_item.position]
    return cmd


def resolve_der_file(sign_file_dir: str, has_cms: bool) -> Optional[str]:
    """解析 CRL 路径并按需转换为 DER 格式，返回 der_file 路径或失败时 None。

    CRL 路径优先使用 CRL_FILE_PATH 环境变量，否则回退到 sign_file_dir/SWSCRL.crl。
    DER 文件放在 sign_file_dir 下，每目标独立，消除跨进程 TOCTOU 竞争。
    """
    crl_file = os.environ.get("CRL_FILE_PATH", "")
    if not (crl_file and os.path.isfile(crl_file)):
        crl_file = os.path.join(sign_file_dir, "SWSCRL.crl")
    der_file = os.path.join(sign_file_dir, "SWSCRL.der")
    if has_cms:
        need_convert = not os.path.isfile(der_file)
        if not need_convert and os.path.isfile(crl_file):
            need_convert = os.path.getmtime(crl_file) > os.path.getmtime(der_file)
        if need_convert:
            if not convert_der_file(crl_file, der_file):
                logger.error("convert der file failed: %s -> %s", crl_file, der_file)
                return None
    return der_file


def add_bios_header(param: BiosHeaderParam) -> bool:
    """
    功能：编排完整的镜像加头+签名流程，对每个镜像绑定最终文件头
    输入：item_size_set：待处理的镜像配置清单
          sign_file_dir：镜像根路径
          bios_tool_path：image_pack.py 等工具所在目录
          sign_tool_path：签名工具脚本路径（community_sign_build.py）
          root_dir：工程根路径
          add_sign：是否签名（"true" 执行完整签名流程，任何非 "true" 值仅加头不签名）
    返回：False:失败，True：成功

    流程顺序：1.加esbc头 → 2.生成ini → 3.CMS签名 → 4.绑定文件头
    """
    item_size_set = param.item_size_set
    sign_file_dir = param.sign_file_dir
    bios_tool_path = param.bios_tool_path
    sign_tool_path = param.sign_tool_path
    root_dir = param.root_dir
    add_sign = param.add_sign

    total = len(item_size_set)
    nvcnt_count = sum(1 for c in item_size_set.values() if c.nvcnt)
    cms_count = sum(1 for c in item_size_set.values() if "cms" in c.type.split("/"))
    has_cms = cms_count > 0
    start_time = time.time()

    # 镜像绑定分2种流程：
    # 1、add_sign不为true时：不需要签名，仅绑定BIOS字节头（version/nvcnt/tag）
    # 2、add_sign为true且type含cms时：绑定镜像文件、ini摘要、cms签名及证书、CRL

    # sign_tmp 位于 sign_file_dir 下，复用 CMake 层每目标唯一的 staging 目录隔离，
    # 避免并行签名时多个进程共享 ${root_dir}/sign_tmp 导致冲突
    sign_tmp_path = os.path.join(sign_file_dir, "sign_tmp")
    # 开始前删除上次残留的临时目录，避免残留文件干扰本次签名；结束后保留以便定位问题
    if os.path.isdir(sign_tmp_path):
        shutil.rmtree(sign_tmp_path)
    os.makedirs(sign_tmp_path, exist_ok=True)

    image_pack_script = os.path.join(bios_tool_path, "image_pack.py")

    # 步骤1：对配置了 nvcnt 的镜像添加 esbc 二级头
    logger.info("step 1/4: add esbc header, %d image(s)", nvcnt_count)
    if not add_bios_esbc_header(root_dir, item_size_set, sign_file_dir):
        return False

    # 步骤2：生成 ini 摘要文件（build_inifile 内部按 add_sign 判断是否执行）
    logger.info("step 2/4: generate ini digest")
    if not build_inifile(
        item_size_set, sign_file_dir, bios_tool_path, sign_tmp_path, add_sign
    ):
        return False

    # 步骤3：CMS 签名（仅签名模式执行）
    # 签名扩展名与证书类型由签名脚本通过 --print-sign-ext/--print-certtype 查询得到，
    # 查询失败时回退默认值（DEFAULT_SIGN_EXT/DEFAULT_CERTTYPE），兼容未更新的第三方脚本。
    # 仅在存在 cms 类型镜像时才查询并签名，避免无 cms 场景下的多余子进程调用。
    sign_ext = DEFAULT_SIGN_EXT
    sign_certtype = DEFAULT_CERTTYPE
    if add_sign == "true" and has_cms:
        logger.info("step 3/4: cms sign, %d image(s)", cms_count)
        sign_ext = query_sign_ext(sign_tool_path)
        sign_certtype = query_certtype(sign_tool_path)
        if not build_sign(item_size_set, sign_file_dir, sign_tool_path, sign_tmp_path):
            return False

    # 步骤4前准备：将 CRL 文件转为 DER 格式，供 image_pack 绑定时使用
    der_file = resolve_der_file(sign_file_dir, has_cms)
    if der_file is None:
        return False

    # 步骤4：用 image_pack.py 对每个镜像绑定最终文件头
    logger.info("step 4/4: bind header, %d image(s)", total)
    for input_name, conf_item in item_size_set.items():
        input_file = os.path.join(sign_file_dir, input_name)
        # 签名文件及ini文件存放目录
        sign_file = os.path.realpath(os.path.join(sign_tmp_path, input_name))
        sign_path = os.path.dirname(sign_file)

        pack_param = ImagePackParam(
            conf_item=conf_item,
            input_file=input_file,
            sign_path=sign_path,
            der_file=der_file,
            add_sign=add_sign,
            image_pack_script=image_pack_script,
            sign_ext=sign_ext,
            sign_certtype=sign_certtype,
        )
        cmd = build_image_pack_cmd(pack_param)

        logger.info("execute:%s", " ".join(cmd))
        success, output = safe_run_cmd(cmd)
        if not success:
            logger.error("add %s header failed!\n\t%s", input_file, output)
            return False

    elapsed = time.time() - start_time
    logger.info(
        "add header and sign success: total=%d, signed=%d, elapsed=%.1fs",
        total,
        cms_count,
        elapsed,
    )
    return True


def check_params(params) -> bool:
    """检查参数。"""
    # 检查BIOS配置文件是否存在
    if not os.path.exists(params["config_file"]):
        logger.error(
            "bios image header config file not exists:%s", params["config_file"]
        )
        return False
    # 检查BIOS工具是否存在
    if not os.path.exists(params["bios_tool_path"]):
        logger.error("biostool dir not exists")
        return False
    # 检查签名脚本是否存在
    if not os.path.exists(params["sgn_tool_path"]):
        logger.error("sign tools script not exists")
        return False
    # 检查待签名文件目录是否存在
    if not os.path.isdir(params["sign_file_dir"]):
        logger.error("sign file dir not exists: %s", params["sign_file_dir"])
        return False
    return True


def define_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("sign_file_dir", help="device release dir")
    parser.add_argument(
        "sign_flag",
        help="skip both header and sign (true/false)",
        default="false",
        nargs="?",
        choices=["true", "false"],
    )
    parser.add_argument(
        "--bios_check_cfg",
        help="default bios_check_cfg.xml",
        default="bios_check_cfg.xml",
    )
    # 签名版本信息，编译传入
    parser.add_argument("--version", help="version")
    # 签名插件脚本
    parser.add_argument("--sign_script", help="sign script path", default="")
    return parser


def setenv():
    """设置环境变量。"""
    global HI_PYTHON
    if "HI_PYTHON" not in os.environ:
        os.environ["HI_PYTHON"] = sys.executable
    HI_PYTHON = os.environ["HI_PYTHON"]


def main(argv=None) -> bool:
    """
    主函数，检查输入参数及环境检查,并调用功能函数
    """
    if argv is None:
        argv = sys.argv
    try:
        parser = define_parser()
        args = parser.parse_args(argv[1:])
    except SystemExit as e:
        # argparse 对 --help 返回 code=0（成功），对非法参数返回 code=2（失败）
        return e.code == 0 or e.code is None
    sign_file_dir = args.sign_file_dir
    add_sign = args.sign_flag
    bios_check_cfg = (
        args.bios_check_cfg if args.bios_check_cfg else "bios_check_cfg.xml"
    )
    version = args.version
    # 如果add_sign是false，则直接返回，加头&签名均不执行
    if add_sign == "false":
        return True
    # 将MY_PATH的父目录赋值给root_dir
    root_dir = os.path.dirname(os.path.dirname(MY_PATH))

    # 签名配置文件路径，相对于工程根目录
    config_file = os.path.join(root_dir, bios_check_cfg)
    logger.info("config_file=%s", config_file)

    bios_tool_path = os.path.join(root_dir, "scripts", "signtool", "image_pack")

    sgn_tool_path = os.path.join(root_dir, "scripts", "sign", "community_sign_build.py")
    # 判断args.sign_script是否存在值，签名插件脚本路径
    if hasattr(args, "sign_script") and args.sign_script:
        sgn_tool_path = args.sign_script

    if not check_params(
        {
            "config_file": config_file,
            "bios_tool_path": bios_tool_path,
            "sign_file_dir": sign_file_dir,
            "sgn_tool_path": sgn_tool_path,
            "add_sign": add_sign,
        }
    ):
        return False

    setenv()

    # 读取并解析签名配置文件，将解析结果存储在 item_size_set 变量中
    success, item_size_set = get_item_set(config_file, sign_file_dir, version)
    if not success:
        return False

    # 调用签名插件对需要签名的镜像进行签名，并绑定镜像文件
    return add_bios_header(
        BiosHeaderParam(
            item_size_set=item_size_set,
            sign_file_dir=sign_file_dir,
            bios_tool_path=bios_tool_path,
            sign_tool_path=sgn_tool_path,
            root_dir=root_dir,
            add_sign=add_sign,
        )
    )


if __name__ == "__main__":
    logging.basicConfig(
        format="[CannSign] [%(asctime)s] [%(levelname)s] [%(pathname)s] [line:%(lineno)d] %(message)s",
        level=logging.INFO,
    )
    sys.exit(0 if main() else 1)
