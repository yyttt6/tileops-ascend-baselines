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
from pathlib import Path
import os

from .. import merge_binary_info_config as mbic


def test_update_config_simple_merge_dict_and_scalar():
    base = {
        "op1": {"binaryList": [1], "attr": "base"},
        "op2": 10,
    }
    update = {
        "op1": {"binaryList": [2], "extra": "u"},
        "op3": 20,
    }

    merged = mbic.update_config(base, update)

    # key 集合
    assert set(merged.keys()) == {"op1", "op2", "op3"}
    # binaryList 拼接，字段顺序保持 base 在前
    assert merged["op1"]["binaryList"] == [1, 2]
    # 其他字段 update 优先
    assert merged["op1"]["attr"] == "base"
    assert merged["op1"]["extra"] == "u"
    # 非 dict 的值，update 覆盖 base
    assert merged["op2"] == 10
    assert merged["op3"] == 20


def test_merge_binary_list_handles_missing_and_non_list():
    base = {"binaryList": [1, 2]}
    update = {}
    merged = mbic._merge_binary_list(base, update)
    assert merged == [1, 2]

    base = {}
    update = {"binaryList": [3]}
    merged = mbic._merge_binary_list(base, update)
    assert merged == [3]

    base = {"binaryList": "not_a_list"}
    update = {"binaryList": "also_not_list"}
    merged = mbic._merge_binary_list(base, update)
    assert merged == []


def test_main_full_flow(tmp_path: Path):
    base_content = {"op": {"binaryList": [1]}}
    update_content = {"op": {"binaryList": [2]}}
    base_file = tmp_path / "base.json"
    update_file = tmp_path / "update.json"
    out_file = tmp_path / "out.json"

    base_file.write_text(json.dumps(base_content), encoding="utf-8")
    update_file.write_text(json.dumps(update_content), encoding="utf-8")

    ok = mbic.main(
        [
            "--base-file",
            str(base_file),
            "--update-file",
            str(update_file),
            "--output-file",
            str(out_file),
        ]
    )

    assert ok is True
    merged = json.loads(out_file.read_text(encoding="utf-8"))
    assert merged["op"]["binaryList"] == [1, 2]


def test_merge_files_binary_list_combination(tmp_path: Path):
    """测试多个文件的 binaryList 拼接功能"""
    f1 = tmp_path / "f1.json"
    f2 = tmp_path / "f2.json"
    
    # f1 有一个 bin, f2 有一个 bin
    f1.write_text(json.dumps({"Add": {"binaryList": [{"id": 1}], "version": "1.0"}}), encoding="utf-8")
    f2.write_text(json.dumps({"Add": {"binaryList": [{"id": 2}], "version": "2.0"}}), encoding="utf-8")

    # 执行合并
    result = mbic.merge_files([str(f1), str(f2)])
    
    # 验证 binaryList 是否成功拼接 (1 + 2)
    assert len(result["Add"]["binaryList"]) == 2
    assert result["Add"]["binaryList"][0]["id"] == 1
    assert result["Add"]["binaryList"][1]["id"] == 2
    # 验证非 binaryList 字段是否按“后来者居上”覆盖 (v2 覆盖 v1)
    assert result["Add"]["version"] == "2.0"


def test_merge_files_priority_first(tmp_path: Path):
    """测试 priority='first' 分支，确保第一个文件优先级更高"""
    f1 = tmp_path / "f1.json"
    f2 = tmp_path / "f2.json"
    
    f1.write_text(json.dumps({"Sub": {"attr": "high"}}), encoding="utf-8")
    f2.write_text(json.dumps({"Sub": {"attr": "low"}}), encoding="utf-8")

    # 当 priority 为 first 时，内部执行 update_config(f2, f1)
    # 结果应该是 f1 的值覆盖 f2 的值
    result = mbic.merge_files([str(f1), str(f2)], priority="first")
    
    assert result["Sub"]["attr"] == "high"


def test_main_with_input_files(tmp_path: Path):
    """测试命令行 --input-files 参数，覆盖 main 函数分支"""
    f1 = tmp_path / "f1.json"
    out = tmp_path / "out.json"
    f1.write_text(json.dumps({"Mul": {"binaryList": []}}), encoding="utf-8")

    # 模拟批量输入
    ok = mbic.main(["--input-files", str(f1), "--output-file", str(out)])
    
    assert ok is True
    assert os.path.exists(out)