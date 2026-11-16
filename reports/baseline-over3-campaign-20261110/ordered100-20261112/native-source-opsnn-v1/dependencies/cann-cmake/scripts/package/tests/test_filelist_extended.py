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

"""Extended tests for filelist operations."""

from dataclasses import dataclass
from typing import List, Optional
from package import filelist


@dataclass
class ItemSpec:
    """Spec for building a FileItem in tests."""

    operation: str
    rel_install: str
    is_dir: bool = False
    block: str = "NA"
    softlink: Optional[List[str]] = None
    pkg_inner_softlink: Optional[List[str]] = None

    def build(self) -> filelist.FileItem:
        return filelist.create_file_item(
            "mod",
            self.operation,
            self.rel_install,
            self.rel_install,
            "FALSE",
            "644",
            "root:root",
            "all",
            self.softlink or [],
            set(),
            "N",
            "FALSE",
            "NA",
            self.block,
            self.pkg_inner_softlink or [],
            set(),
            self.is_dir,
        )


class TestCreateFileItemExtended:
    """Extended tests for create_file_item."""

    @staticmethod
    def test_create_file_item_valid():
        """Test create_file_item with valid parameters."""
        item = filelist.create_file_item(
            "module",
            "copy",
            "pkg/path",
            "install/path",
            "TRUE",
            "755",
            "user:group",
            "run",
            [],
            {"all"},
            "N",
            "FALSE",
            "hash_value",
            "block",
            [],
            {"ascend910"},
            False,
        )
        assert item.module == "module"
        assert item.operation == "copy"


class TestIsSpecificInstallTypeExtended:
    """Extended tests for is_specific_install_type."""

    @staticmethod
    def test_is_specific_install_type_with_all():
        """Test when install_type contains 'all'."""
        item = filelist.create_file_item(
            "m",
            "op",
            "pkg",
            "install",
            "TRUE",
            "755",
            "u:g",
            "all;run",
            [],
            {"f"},
            "N",
            "FALSE",
            "h",
            "b",
            [],
            {"c"},
            False,
        )
        result = filelist.is_specific_install_type(item, {"docker"})
        assert result is True

    @staticmethod
    def test_is_specific_install_type_intersection():
        """Test when install_type has intersection."""
        item = filelist.create_file_item(
            "m",
            "op",
            "pkg",
            "install",
            "TRUE",
            "755",
            "u:g",
            "docker;run",
            [],
            {"f"},
            "N",
            "FALSE",
            "h",
            "b",
            [],
            {"c"},
            False,
        )
        result = filelist.is_specific_install_type(item, {"docker"})
        assert result is True

    @staticmethod
    def test_is_specific_install_type_no_match():
        """Test when no match."""
        item = filelist.create_file_item(
            "m",
            "op",
            "pkg",
            "install",
            "TRUE",
            "755",
            "u:g",
            "run",
            [],
            {"f"},
            "N",
            "FALSE",
            "h",
            "b",
            [],
            {"c"},
            False,
        )
        result = filelist.is_specific_install_type(item, {"docker"})
        assert result is False


class TestIsSpecificOperationsExtended:
    """Extended tests for is_specific_operations."""

    @staticmethod
    def test_is_specific_operations_match():
        """Test when operation matches."""
        item = filelist.create_file_item(
            "m",
            "copy",
            "pkg",
            "install",
            "TRUE",
            "755",
            "u:g",
            "run",
            [],
            {"f"},
            "N",
            "FALSE",
            "h",
            "b",
            [],
            {"c"},
            False,
        )
        result = filelist.is_specific_operations(item, ["copy", "move"])
        assert result is True

    @staticmethod
    def test_is_specific_operations_no_match():
        """Test when operation doesn't match."""
        item = filelist.create_file_item(
            "m",
            "del",
            "pkg",
            "install",
            "TRUE",
            "755",
            "u:g",
            "run",
            [],
            {"f"},
            "N",
            "FALSE",
            "h",
            "b",
            [],
            {"c"},
            False,
        )
        result = filelist.is_specific_operations(item, ["copy", "move"])
        assert result is False


class TestGetMissingDirSet:
    """Test get_missing_dir_set function."""

    @staticmethod
    def test_get_missing_dir_set_empty():
        """Test get_missing_dir_set with empty list."""
        result = filelist.get_missing_dir_set([])
        assert result == set()

    @staticmethod
    def test_get_missing_dir_set_with_copy():
        """Test get_missing_dir_set with copy items."""
        item = filelist.create_file_item(
            "m",
            "copy",
            "pkg/path",
            "/install/path/file.txt",
            "TRUE",
            "755",
            "u:g",
            "run",
            [],
            {"f"},
            "N",
            "FALSE",
            "h",
            "b",
            [],
            {"c"},
            False,
        )
        result = filelist.get_missing_dir_set([item])
        # Should return parent directories of the install path
        assert isinstance(result, set)


def _make_item(
    operation,
    rel_install,
    is_dir=False,
    block="NA",
    softlink=None,
    pkg_inner_softlink=None,
):
    return filelist.create_file_item(
        "mod",
        operation,
        rel_install,
        rel_install,
        "FALSE",
        "644",
        "root:root",
        "all",
        softlink or [],
        set(),
        "N",
        "FALSE",
        "NA",
        block,
        pkg_inner_softlink or [],
        set(),
        is_dir,
    )


class TestRecordFileItemFormat:
    """Tests for RECORD file item format consistency."""

    @staticmethod
    def test_record_file_item_permission_not_na():
        """RECORD entry permission should be concrete value (not NA)."""
        item = filelist.create_record_file_item("ops_math")
        assert item.permission != "NA"

    @staticmethod
    def test_record_file_item_owner_group():
        """RECORD entry owner_group should be escaped username:usergroup."""
        item = filelist.create_record_file_item("ops_math")
        assert item.owner_group == "\\\\$username:\\\\$usergroup"

    @staticmethod
    def test_record_file_item_csv_format():
        """RECORD CSV line should have concrete permission and owner_group."""
        item = filelist.create_record_file_item("ops_math")
        line = filelist.file_item_to_string(item)
        fields = line.split(",")
        assert fields[5] != "NA"
        assert fields[6] == "\\\\$username:\\\\$usergroup"

    @staticmethod
    def test_record_file_item_operation_copy():
        """RECORD entry operation must be 'copy'."""
        item = filelist.create_record_file_item("test_func")
        assert item.operation == "copy"

    @staticmethod
    def test_record_file_item_relative_path():
        """RECORD entry paths should be share/info/{name}/RECORD."""
        item = filelist.create_record_file_item("my_pkg")
        expected = filelist.get_record_file_relative_path("my_pkg")
        assert item.relative_path_in_pkg == expected
        assert item.relative_install_path == expected


class TestRecordFileItemScope:
    """Tests for RECORD record scope (all packaged files, excluding mkdir)."""

    @staticmethod
    def test_copy_is_recorded():
        assert (
            filelist.is_record_file_item(_make_item("copy", "lib64/libfoo.so")) is True
        )

    @staticmethod
    def test_copy_entity_is_recorded():
        assert (
            filelist.is_record_file_item(_make_item("copy_entity", "include/foo.h"))
            is True
        )

    @staticmethod
    def test_move_is_recorded():
        assert filelist.is_record_file_item(_make_item("move", "bin/tool")) is True

    @staticmethod
    def test_del_is_recorded():
        assert (
            filelist.is_record_file_item(_make_item("del", "old/removed.json")) is True
        )

    @staticmethod
    def test_mkdir_not_recorded():
        assert filelist.is_record_file_item(_make_item("mkdir", "some/dir")) is False

    @staticmethod
    def test_record_operations_constant():
        assert "copy" in filelist.RECORD_FILE_OPERATIONS
        assert "copy_entity" in filelist.RECORD_FILE_OPERATIONS
        assert "move" in filelist.RECORD_FILE_OPERATIONS
        assert "del" in filelist.RECORD_FILE_OPERATIONS
        assert "mkdir" not in filelist.RECORD_FILE_OPERATIONS

    @staticmethod
    def test_get_record_install_paths_includes_move_and_del():
        items = [
            _make_item("copy", "lib64/liba.so"),
            _make_item("move", "bin/tool"),
            _make_item("del", "old/removed.json"),
            _make_item("mkdir", "some/dir"),
        ]
        paths = filelist.get_record_install_paths(items)
        assert "lib64/liba.so" in paths
        assert "bin/tool" in paths
        assert "old/removed.json" in paths
        assert "some/dir" not in paths


class TestRecordSoftlinks:
    """Tests for softlink info displayed in RECORD."""

    @staticmethod
    def test_softlinks_included_for_copy():
        """copy entry softlinks should be recorded in RECORD."""
        item = _make_item(
            "copy",
            "lib64/libfoo.so",
            softlink=["lib64/libfoo.so.1", "lib64/libfoo.so.2"],
        )
        paths = filelist.get_record_install_paths([item])
        assert "lib64/libfoo.so" in paths
        assert "lib64/libfoo.so.1" in paths
        assert "lib64/libfoo.so.2" in paths

    @staticmethod
    def test_pkg_inner_softlinks_included():
        """pkg_inner_softlink paths should be recorded in RECORD."""
        item = _make_item(
            "copy", "lib64/libfoo.so", pkg_inner_softlink=["lib64/libfoo_inner.so"]
        )
        paths = filelist.get_record_install_paths([item])
        assert "lib64/libfoo.so" in paths
        assert "lib64/libfoo_inner.so" in paths

    @staticmethod
    def test_mkdir_softlinks_included():
        """mkdir entry softlinks should be recorded in RECORD (the fix)."""
        item = _make_item("mkdir", "some/dir", softlink=["dir_link", "dir_link2"])
        paths = filelist.get_record_install_paths([item])
        assert "some/dir" not in paths
        assert "dir_link" in paths
        assert "dir_link2" in paths

    @staticmethod
    def test_mkdir_pkg_inner_softlinks_included():
        """mkdir entry pkg_inner_softlink should be recorded in RECORD."""
        item = _make_item("mkdir", "some/dir", pkg_inner_softlink=["inner_dir_link"])
        paths = filelist.get_record_install_paths([item])
        assert "some/dir" not in paths
        assert "inner_dir_link" in paths

    @staticmethod
    def test_softlinks_dedup():
        """softlink paths should be deduplicated across entries."""
        item1 = _make_item("copy", "lib64/liba.so", softlink=["lib64/liba.so.1"])
        item2 = _make_item("copy", "lib64/libb.so", softlink=["lib64/liba.so.1"])
        paths = filelist.get_record_install_paths([item1, item2])
        assert paths.count("lib64/liba.so.1") == 1

    @staticmethod
    def test_generate_record_file_includes_mkdir_softlinks(tmp_path):
        """generate_record_file should include mkdir softlink paths."""
        items = [
            _make_item("copy", "lib64/liba.so", softlink=["lib64/liba.so.1"]),
            _make_item("mkdir", "include", softlink=["include_link"]),
        ]
        filelist.generate_record_file(items, str(tmp_path), "test_func")
        record_file = (
            tmp_path / "share" / "info" / "test_func" / filelist.RECORD_FILE_NAME
        )
        content = record_file.read_text(encoding="utf-8").splitlines()
        assert "lib64/liba.so" in content
        assert "lib64/liba.so.1" in content
        assert "include_link" in content
        assert "include" not in content
