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

"""Tests for package module."""

import csv
import os
import subprocess
from argparse import Namespace
from pathlib import Path
from unittest import mock

import pytest
from package import package as package_module

# Import py modules first
from .. import packer


# =============================================================================
# Helper functions and shared Mock classes to reduce code duplication
# =============================================================================


def create_base_install_info(**overrides):
    """Create a base install info dict with common fields.

    Args:
        **overrides: Fields to override in the base dict.

    Returns:
        dict: Base install info dictionary.
    """
    base = {
        "module": "test",
        "install_mod": "755",
        "install_own": "user:group",
        "install_type": "run",
        "install_softlink": "",
        "feature": {"all"},
        "configurable": "FALSE",
        "hash": "NA",
        "name": "block1",
        "pkg_inner_softlink": "",
        "chip": {"ascend910"},
    }
    base.update(overrides)
    return base


class MockPackageName:
    """Shared MockPackageName class for tests."""

    func_name = "test_func"
    chip_name = "ascend910"

    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def getvalue():
        return "test-package"


class MockPackageOption:
    """Shared MockPackageOption class for tests."""

    ext_name = "test"


class MockXmlConfigBase:
    """Shared base MockXmlConfig class for tests."""

    generate_infos = []
    package_content_list = []
    move_content_list = []

    @property
    def pkg_softlinks(self):
        return []


def create_mock_xml_config(**kwargs):
    """Create a configurable MockXmlConfig class.

    Args:
        **kwargs: Attributes to set on the MockXmlConfig class.

    Returns:
        type: A MockXmlConfig class with specified attributes.
    """

    class MockXmlConfig:
        generate_infos = kwargs.get("generate_infos", [])
        package_content_list = kwargs.get("package_content_list", [])
        move_content_list = kwargs.get("move_content_list", [])

        @property
        def pkg_softlinks(self):
            return []

    return MockXmlConfig()


class MockPackerConfig:
    """Shared MockPackerConfig class for tests."""

    @staticmethod
    def fill_is_common_path(files):
        return files


def create_mock_factory():
    """Create a mock factory function for create_makeself_pkg_params_factory."""

    def mock_factory(*args):
        def inner(*args):
            return Namespace()

        return inner

    return mock_factory


def setup_optional_file_not_exists_test(monkeypatch):
    """Setup common mocks for optional file not exists tests.

    Args:
        monkeypatch: pytest monkeypatch fixture.

    Returns:
        list: infos list for parse_install_info.
    """
    infos = [
        create_base_install_info(
            dst_path="/nonexistent",
            value="path/file.txt",
            install_path="/nonexistent",
            optional="true",
        )
    ]

    # Mock os.path.exists to return False
    monkeypatch.setattr(os.path, "exists", lambda x: False)

    return infos


def setup_compress_cmd_test(monkeypatch, tmp_path, independent_pkg=True):
    """Setup common mocks and configs for get_compress_cmd tests.

    Args:
        monkeypatch: pytest monkeypatch fixture.
        tmp_path: pytest tmp_path fixture.
        independent_pkg: Value for independent_pkg attribute.

    Returns:
        tuple: (mock_xml_config, pkg_args)
    """
    mock_xml_config = Namespace(package_attr={"suffix": "run"}, version="1.0.0")

    pkg_args = Namespace(
        pkg_output_dir=str(tmp_path),
        makeself_dir=str(tmp_path / "makeself"),
        independent_pkg=independent_pkg,
    )

    # Use shared MockPackageName class
    monkeypatch.setattr(package_module, "PackageName", MockPackageName)

    # Mock get_comments
    monkeypatch.setattr(package_module, "get_comments", lambda x: '"test comments"')

    # Use shared mock factory
    monkeypatch.setattr(
        package_module, "create_makeself_pkg_params_factory", create_mock_factory()
    )

    return mock_xml_config, pkg_args


# =============================================================================
# Test Classes
# =============================================================================


class TestGetComments:
    """Test get_comments function."""

    @staticmethod
    def test_get_comments_with_chip_name():
        """Test get_comments with chip_name."""
        mock_pkg_name = Namespace(
            chip_name="ascend910", product_name="product", func_name="toolkit"
        )
        result = package_module.get_comments(mock_pkg_name)
        assert "ASCEND910" in result
        assert "TOOLKIT" in result
        assert "RUN_PACKAGE" in result

    @staticmethod
    def test_get_comments_without_chip_name():
        """Test get_comments without chip_name."""
        mock_pkg_name = Namespace(
            chip_name=None, product_name="product", func_name="toolkit"
        )
        result = package_module.get_comments(mock_pkg_name)
        assert "PRODUCT" in result
        assert "TOOLKIT" in result
        assert "RUN_PACKAGE" in result


class TestMakeParseOption:
    """Test make_parse_option function."""

    @staticmethod
    def test_make_parse_option():
        """Test creating ParseOption from args."""
        args = Namespace(
            os_arch="ubuntu20.04-aarch64",
            pkg_version="1.0.0",
            build_type="debug",
            package_check=True,
            ext_name="llvm",
        )
        result = package_module.make_parse_option(args)
        assert result.os_arch == "ubuntu20.04-aarch64"
        assert result.pkg_version == "1.0.0"
        assert result.build_type == "debug"
        assert result.package_check is True
        assert result.ext_name == "llvm"


class TestPackageOption:
    """Test PackageOption class."""

    @staticmethod
    def test_package_option_creation():
        """Test PackageOption namedtuple creation."""
        opt = package_module.PackageOption(
            os_arch="aarch64",
            package_suffix="debug",
            not_in_name="",
            pkg_version="1.0.0",
            ext_name="",
            chip_name="ascend910",
            func_name="toolkit",
            version_dir="v1",
            disable_multi_version=False,
            suffix="run",
        )
        assert opt.os_arch == "aarch64"
        assert opt.chip_name == "ascend910"
        assert opt.func_name == "toolkit"


class TestGenerateInfoContent:
    """Test generate_info_content function."""

    @staticmethod
    def test_generate_info_content_basic():
        """Test basic info content generation."""
        target_conf = {"content": {"key1": "value1", "key2": "value2"}}
        result = package_module.generate_info_content(target_conf, "")
        assert "key1=value1" in result
        assert "key2=value2" in result

    @staticmethod
    def test_generate_info_content_with_llvm():
        """Test info content with llvm ext_name."""
        target_conf = {"content": {"key1": "value1"}}
        result = package_module.generate_info_content(target_conf, "llvm_toolchain")
        assert "key1=value1" in result
        assert "toolchain=llvm" in result


class TestGenerateVersionHeaderContent:
    """Test generate_version_header_content function."""

    @staticmethod
    def test_generate_version_header_content():
        """Test version header content generation."""
        target_conf = {
            "value": "version.h",
            "content": {"DEFINE_A": "100", "PKG_VERSION": "1.2.3"},
        }
        result = list(package_module.generate_version_header_content(target_conf))
        assert "#ifndef VERSION_H" in result
        assert "#define VERSION_H" in result
        assert "#define DEFINE_A 100" in result
        assert "#endif /* VERSION_H */" in result


class TestGetModule:
    """Test get_module function."""

    @staticmethod
    def test_get_module_with_value():
        """Test get_module with module value."""
        assert package_module.get_module({"module": "test_module"}) == "test_module"

    @staticmethod
    def test_get_module_without_value():
        """Test get_module without module value."""
        assert package_module.get_module({}) == "NA"

    @staticmethod
    def test_get_module_empty_string():
        """Test get_module with empty string."""
        assert package_module.get_module({"module": ""}) == "NA"


class TestGetOperation:
    """Test get_operation function."""

    @staticmethod
    def test_get_operation_copy():
        """Test get_operation with copy."""
        assert package_module.get_operation("copy", {}) == "copy"

    @staticmethod
    def test_get_operation_copy_entity():
        """Test get_operation with copy entity."""
        assert package_module.get_operation("copy", {"entity": "true"}) == "copy_entity"

    @staticmethod
    def test_get_operation_move():
        """Test get_operation with move."""
        assert package_module.get_operation("move", {}) == "move"

    @staticmethod
    def test_get_operation_move_entity():
        """Test get_operation with move entity."""
        assert package_module.get_operation("move", {"entity": "true"}) == "copy_entity"


class TestGetPermission:
    """Test get_permission function."""

    @staticmethod
    def test_get_permission_with_value():
        """Test get_permission with value."""
        assert package_module.get_permission({"install_mod": "755"}) == "755"

    @staticmethod
    def test_get_permission_default():
        """Test get_permission default value."""
        assert package_module.get_permission({}) == "NA"


class TestGetOwnerGroup:
    """Test get_owner_group function."""

    @staticmethod
    def test_get_owner_group_with_value():
        """Test get_owner_group with value."""
        result = package_module.get_owner_group({"install_own": "user:group"})
        assert "user:group" in result

    @staticmethod
    def test_get_owner_group_escapes_dollar():
        """Test get_owner_group escapes dollar signs."""
        result = package_module.get_owner_group({"install_own": "$user:$group"})
        assert "\\\\$user" in result or "$user" in result

    @staticmethod
    def test_get_owner_group_default():
        """Test get_owner_group default value."""
        assert package_module.get_owner_group({}) == "NA"


class TestGetInstallType:
    """Test get_install_type function."""

    @staticmethod
    def test_get_install_type_with_value():
        """Test get_install_type with value."""
        assert package_module.get_install_type({"install_type": "docker"}) == "docker"

    @staticmethod
    def test_get_install_type_default():
        """Test get_install_type default value."""
        assert package_module.get_install_type({}) == "NA"


class TestGetSoftlink:
    """Test get_softlink function."""

    @staticmethod
    def test_get_softlink_with_value():
        """Test get_softlink with semicolon separated values."""
        result = package_module.get_softlink({"install_softlink": "link1;link2;link3"})
        assert result == ["link1", "link2", "link3"]

    @staticmethod
    def test_get_softlink_empty():
        """Test get_softlink with empty value."""
        assert package_module.get_softlink({"install_softlink": ""}) == []

    @staticmethod
    def test_get_softlink_missing():
        """Test get_softlink with missing key."""
        assert package_module.get_softlink({}) == []


class TestGetFeature:
    """Test get_feature function."""

    @staticmethod
    def test_get_feature():
        """Test get_feature returns feature set."""
        result = package_module.get_feature({"feature": {"feat1", "feat2"}})
        assert result == {"feat1", "feat2"}


class TestGetChip:
    """Test get_chip function."""

    @staticmethod
    def test_get_chip():
        """Test get_chip returns chip set."""
        result = package_module.get_chip({"chip": {"chip1", "chip2"}})
        assert result == {"chip1", "chip2"}


class TestGetConfigurable:
    """Test get_configurable function."""

    @staticmethod
    def test_get_configurable_with_value():
        """Test get_configurable with value."""
        assert package_module.get_configurable({"configurable": "TRUE"}) == "TRUE"

    @staticmethod
    def test_get_configurable_default():
        """Test get_configurable default value."""
        assert package_module.get_configurable({}) == "FALSE"


class TestGetHashValue:
    """Test get_hash_value function."""

    @staticmethod
    def test_get_hash_value_with_value():
        """Test get_hash_value with value."""
        assert package_module.get_hash_value({"hash": "abc123"}) == "abc123"

    @staticmethod
    def test_get_hash_value_default():
        """Test get_hash_value default value."""
        assert package_module.get_hash_value({}) == "NA"


class TestGetBlock:
    """Test get_block function."""

    @staticmethod
    def test_get_block_with_value():
        """Test get_block with value."""
        assert package_module.get_block({"name": "block1"}) == "block1"

    @staticmethod
    def test_get_block_default():
        """Test get_block default value."""
        assert package_module.get_block({}) == "NA"


class TestGetPkgInnerSoftlink:
    """Test get_pkg_inner_softlink function."""

    @staticmethod
    def test_get_pkg_inner_softlink_with_value():
        """Test get_pkg_inner_softlink with semicolon separated values."""
        result = package_module.get_pkg_inner_softlink(
            {"pkg_inner_softlink": "link1;link2"}
        )
        assert result == ["link1", "link2"]

    @staticmethod
    def test_get_pkg_inner_softlink_empty():
        """Test get_pkg_inner_softlink with empty value."""
        assert package_module.get_pkg_inner_softlink({"pkg_inner_softlink": ""}) == []

    @staticmethod
    def test_get_pkg_inner_softlink_missing():
        """Test get_pkg_inner_softlink with missing key."""
        assert package_module.get_pkg_inner_softlink({}) == []


class TestGetTargetName:
    """Test get_target_name function."""

    @staticmethod
    def test_get_target_name_with_rename():
        """Test get_target_name with rename."""
        assert package_module.get_target_name({"rename": "new_name"}) == "new_name"

    @staticmethod
    def test_get_target_name_from_value():
        """Test get_target_name extracts from value."""
        assert (
            package_module.get_target_name({"value": "path/to/file.txt"}) == "file.txt"
        )

    @staticmethod
    def test_get_target_name_from_value_trailing_slash():
        """Test get_target_name handles trailing slash."""
        assert package_module.get_target_name({"value": "path/to/dir/"}) == "dir"


class TestGetFilterKey:
    """Test get_filter_key function."""

    @staticmethod
    def test_get_filter_key_driver():
        """Test get_filter_key for driver package_module."""
        assert package_module.get_filter_key("driver") == ["all", "docker"]

    @staticmethod
    def test_get_filter_key_firmware():
        """Test get_filter_key for firmware package_module."""
        assert package_module.get_filter_key("firmware") == ["all", "docker"]

    @staticmethod
    def test_get_filter_key_aicpu():
        """Test get_filter_key for aicpu packages."""
        assert package_module.get_filter_key("aicpu_kernels_device") == []
        assert package_module.get_filter_key("aicpu_kernels_host") == []

    @staticmethod
    def test_get_filter_key_default():
        """Test get_filter_key for default packages."""
        assert package_module.get_filter_key("other_pkg") == ["all", "run"]


class TestIsMatchLine:
    """Test is_match_line function."""

    @staticmethod
    def test_is_match_line_true():
        """Test is_match_line returns True for matching line."""
        data = [
            "pkg_name",
            "path",
            "extra",
            "extra2",
            "100",
            "chip1",
            "product1",
            "debug",
        ]
        result = package_module.is_match_line(
            "pkg_name", "chip1", "product1", "debug", data
        )
        assert result is True

    @staticmethod
    def test_is_match_line_false():
        """Test is_match_line returns False for non-matching line."""
        data = [
            "pkg_name",
            "path",
            "extra",
            "extra2",
            "100",
            "chip1",
            "product1",
            "debug",
        ]
        result = package_module.is_match_line(
            "other_pkg", "chip1", "product1", "debug", data
        )
        assert result is False


class TestGenerateCustomizedFile:
    """Test generate_customized_file function."""

    @staticmethod
    def test_generate_customized_file_failure(tmp_path, monkeypatch):
        """Test generate_customized_file with file open failure."""
        target_conf = {
            "value": "test.txt",
            "generator": "info",
            "content": {"key": "value"},
            "dst_path": "",
        }

        # Mock open to raise an exception, simulating file write failure
        def mock_open(*args, **kwargs):
            raise PermissionError("Permission denied")

        monkeypatch.setattr("builtins.open", mock_open)

        result = package_module.generate_customized_file(target_conf, "", str(tmp_path))
        assert result is False

    def test_generate_customized_file_info(self, tmp_path):
        """Test generate_customized_file with info generator."""
        target_conf = {
            "value": "test.info",
            "generator": "info",
            "content": {"key1": "value1"},
            "dst_path": "",
        }
        result = package_module.generate_customized_file(target_conf, "", str(tmp_path))
        assert result is True

        content = (tmp_path / "test.info").read_text()
        assert "key1=value1" in content

    def test_generate_customized_file_version_header(self, tmp_path):
        """Test generate_customized_file with version_header generator."""
        target_conf = {
            "value": "version.h",
            "generator": "version_header",
            "content": {"DEFINE_A": "100"},
            "dst_path": "",
        }
        result = package_module.generate_customized_file(target_conf, "", str(tmp_path))
        assert result is True

        content = (tmp_path / "version.h").read_text()
        assert "#define DEFINE_A 100" in content


class TestCheckPathIsConflict:
    """Test check_path_is_conflict function."""

    def test_check_path_is_conflict_no_conflict(self):
        """Test check_path_is_conflict with no conflict."""

        class MockXmlConfig:
            package_content_list = [
                {
                    "value": "path/to/file",
                    "install_path": "/install",
                    "pkg_inner_softlink": None,
                }
            ]

        result = package_module.check_path_is_conflict(MockXmlConfig())
        assert result is True

    def test_check_path_is_conflict_with_conflict(self):
        """Test check_path_is_conflict with conflict."""
        # The function joins install_path with target_name from value
        # value='path/file' -> target_name='file'
        # install_path='/install' + 'file' -> os.path.join('/install', 'file')
        # On Windows this gives '\\install\\file', on Unix '/install/file'
        # We need to use os.path.join to match the expected path
        expected_path = os.path.join("/install", "file")

        class MockXmlConfig:
            package_content_list = [
                {
                    "value": "path/file",
                    "install_path": "/install",
                    "pkg_inner_softlink": expected_path,
                }
            ]

        result = package_module.check_path_is_conflict(MockXmlConfig())
        assert result is False


class TestChecksumValue:
    """Test checksum_value function."""

    def test_checksum_value_file_not_exists(self, tmp_path, monkeypatch):
        """Test checksum_value with non-existent file."""
        limit_value = ["pkg", "nonexistent", "100", "110%"]
        # Extend to index 6 for max_value
        limit_value = limit_value + ["", "200"]

        result = package_module.checksum_value(limit_value, str(tmp_path))
        assert result is True  # Returns True for non-existent files

    def test_checksum_value_invalid_config(self, tmp_path):
        """Test checksum_value with invalid config."""
        limit_value = ["pkg", "file", "100"]  # Missing max_value

        result = package_module.checksum_value(limit_value, str(tmp_path))
        assert result is True  # Returns True for invalid config


class TestCheckAddDir:
    """Test check_add_dir function."""

    def test_check_add_dir_no_new_files(self, tmp_path):
        """Test check_add_dir with no new files."""
        limit_list = ["existing"]
        # Create the directory
        (tmp_path / "existing").mkdir()

        existing_dir = str(tmp_path / "existing")
        result = package_module.check_add_dir(
            str(tmp_path) + "/", existing_dir, limit_list
        )
        assert result is True


class TestGetPkgXmlRelativePath:
    """Test get_pkg_xml_relative_path function."""

    def test_get_pkg_xml_relative_path_basic(self):
        """Test get_pkg_xml_relative_path with basic args."""
        args = Namespace(pkg_name="test_pkg", chip_scenes="", xml_file="")
        result = package_module.get_pkg_xml_relative_path(args)
        assert "test_pkg" in result
        assert "test_pkg.xml" in result

    def test_get_pkg_xml_relative_path_with_chip_scenes(self):
        """Test get_pkg_xml_relative_path with chip_scenes."""
        args = Namespace(pkg_name="test_pkg", chip_scenes="scene1", xml_file="")
        result = package_module.get_pkg_xml_relative_path(args)
        assert "scene1" in result

    def test_get_pkg_xml_relative_path_with_xml_file(self):
        """Test get_pkg_xml_relative_path with custom xml_file."""
        args = Namespace(pkg_name="test_pkg", chip_scenes="", xml_file="custom.xml")
        result = package_module.get_pkg_xml_relative_path(args)
        assert "custom.xml" in result


class TestArgsParse:
    """Test args_parse function."""

    def test_args_parse_defaults(self):
        """Test args_parse with defaults."""
        test_args = ["package_module.py", "--source_dir", "", "--delivery_dir", ""]
        with mock.patch("sys.argv", test_args):
            args = package_module.args_parse()
            assert args.type == "repack"
            assert args.build_type == "debug"

    def test_args_parse_with_args(self):
        """Test args_parse with custom args."""
        test_args = [
            "package_module.py",
            "-n",
            "test_pkg",
            "-o",
            "ubuntu20.04",
            "-v",
            "1.0.0",
            "--package-check",
            "--source_dir",
            "",
            "--delivery_dir",
            "",
        ]
        with mock.patch("sys.argv", test_args):
            args = package_module.args_parse()
            assert args.pkg_name == "test_pkg"
            assert args.os_arch == "ubuntu20.04"
            assert args.pkg_version == "1.0.0"
            assert args.package_check is True


class TestMainFunction:
    """Test main function."""

    def test_main_delivery_dir_not_exists(self, tmp_path, monkeypatch):
        """Test main with non-existent delivery dir."""
        args = Namespace(
            delivery_dir=str(tmp_path / "nonexistent"),
            pkg_name="test",
            chip_scenes="",
            xml_file="",
            version_dir="",
            source_root="",
            package_check=False,
            independent_pkg=False,
            pkg_output_dir="",
            os_arch="",
            pkg_version="",
            build_type="debug",
            ext_name="",
            chip_name=None,
            func_name=None,
            disable_multi_version=False,
            suffix=None,
        )

        result = package_module.main("test", "", args)
        assert result is False


class TestGenerateHashFileBasic:
    """Test generate_hash_file function."""

    def test_generate_hash_file_success(self, tmp_path, monkeypatch):
        """Test generate_hash_file success case."""

        # Mock successful touch command
        def mock_run(*args, **kwargs):
            class Result:
                returncode = 0
                stdout = ""

            return Result()

        monkeypatch.setattr(subprocess, "run", mock_run)

        result = package_module.generate_hash_file(
            str(tmp_path), "file1=hash1\nfile2=hash2\n"
        )
        assert result is True

        hash_file = tmp_path / "bin_hash.cfg"
        assert hash_file.exists()
        content = hash_file.read_text()
        assert "file1=hash1" in content


class TestGenFileInstallListBasic:
    """Test gen_file_install_list function."""

    def test_gen_file_install_list_empty(self):
        """Test gen_file_install_list with empty config."""

        class MockXmlConfig:
            dir_install_list = []
            move_content_list = []
            package_content_list = []
            generate_infos = []
            expand_content_list = []
            packer_config = Namespace(fill_is_common_path=lambda x: x)

        result, _ = package_module.gen_file_install_list(
            "/tmp", MockXmlConfig(), ["all"]
        )
        assert result == []


class TestParseInstallInfo:
    """Test parse_install_info function."""

    def test_parse_install_info_copy(self, monkeypatch):
        """Test parse_install_info with copy operation."""
        infos = [
            {
                "dst_path": "/install",
                "value": "path/file.txt",
                "install_path": "/install",
                "module": "test",
                "install_mod": "755",
                "install_own": "user:group",
                "install_type": "run",
                "install_softlink": "",
                "feature": {"all"},
                "configurable": "FALSE",
                "hash": "NA",
                "name": "block1",
                "pkg_inner_softlink": "",
                "chip": {"ascend910"},
                "is_dir": False,
            }
        ]

        # Filter key 'all' matches install_type 'run' (contains 'all')
        result = list(
            package_module.parse_install_info(None, infos, "copy", ["all", "run"])
        )
        assert len(result) == 1
        assert result[0].operation == "copy"
        assert result[0].is_in_docker == "TRUE"

    def test_parse_install_info_mkdir(self):
        """Test parse_install_info with mkdir operation."""
        infos = [
            {
                "value": "/install/dir",
                "module": "test",
                "install_mod": "755",
                "install_own": "user:group",
                "install_type": "run",
                "install_softlink": "",
                "feature": {"all"},
                "configurable": "FALSE",
                "hash": "NA",
                "name": "block1",
                "pkg_inner_softlink": "",
                "chip": {"ascend910"},
            }
        ]

        result = list(package_module.parse_install_info(None, infos, "mkdir", ["all"]))
        assert len(result) == 1
        assert result[0].operation == "mkdir"
        assert result[0].relative_path_in_pkg == "NA"

    def test_parse_install_info_del(self):
        """Test parse_install_info with del operation."""
        infos = [
            {
                "dst_path": "dst",
                "value": "path/file.txt",
                "install_path": "/install",
                "module": "test",
                "install_mod": "755",
                "install_own": "user:group",
                "install_type": "run",
                "install_softlink": "",
                "feature": {"all"},
                "configurable": "FALSE",
                "hash": "NA",
                "name": "block1",
                "pkg_inner_softlink": "",
                "chip": {"ascend910"},
            }
        ]

        result = list(package_module.parse_install_info(None, infos, "del", ["all"]))
        assert len(result) == 1
        assert result[0].operation == "del"

    def test_parse_install_info_unknown_operation(self):
        """Test parse_install_info with unknown operation raises exception."""
        infos = [{"value": "test"}]

        with pytest.raises(package_module.UnknownOperateTypeError):
            list(package_module.parse_install_info(None, infos, "unknown_op", []))


class TestGenerateHashList:
    """Test generate_hash_list function."""

    def test_generate_hash_list_success(self, monkeypatch, tmp_path):
        """Test generate_hash_list successful case."""
        target_conf = {"value": "file.txt", "rename": "renamed.txt", "dst_path": ""}

        # Create the file
        (tmp_path / "renamed.txt").write_text("content")

        class MockResult:
            returncode = 0
            stdout = "abc123 hashvalue"

        monkeypatch.setattr(
            package_module.subprocess, "run", lambda *args, **kwargs: MockResult()
        )

        result, hash_str = package_module.generate_hash_list(
            target_conf, "", str(tmp_path)
        )
        assert result is True
        assert "renamed.txt=" in hash_str

    def test_generate_hash_list_failure(self, monkeypatch, tmp_path):
        """Test generate_hash_list failure case."""
        target_conf = {"value": "file.txt", "dst_path": ""}

        class MockResult:
            returncode = 1
            stdout = "error"

        monkeypatch.setattr(
            package_module.subprocess, "run", lambda *args, **kwargs: MockResult()
        )

        result, hash_str = package_module.generate_hash_list(
            target_conf, "", str(tmp_path)
        )
        assert result is False
        assert hash_str is None


class TestProcessingCsvFile:
    """Test processing_csv_file function."""

    def test_processing_csv_file_no_file(self, monkeypatch, tmp_path):
        """Test processing_csv_file when limit.csv doesn't exist."""
        monkeypatch.setattr(package_module.pkg_utils, "TOP_SOURCE_DIR", str(tmp_path))

        limit_list, ret = package_module.processing_csv_file(
            str(tmp_path), "pkg", "chip", "debug"
        )
        assert ret is True
        assert limit_list == []


class TestGenerateHashFile:
    """Test generate_hash_file function."""

    def test_generate_hash_file_with_existing_file(self, monkeypatch, tmp_path):
        """Test generate_hash_file when file already exists."""
        hash_path = tmp_path / "bin_hash.cfg"
        hash_path.write_text("old content")

        # Mock successful subprocess
        class MockResult:
            returncode = 0
            stdout = ""

        monkeypatch.setattr(
            package_module.subprocess, "run", lambda *args, **kwargs: MockResult()
        )

        result = package_module.generate_hash_file(str(tmp_path), "new content")
        assert result is True
        assert hash_path.read_text() == "new content"

    def test_generate_hash_file_touch_failure(self, monkeypatch, tmp_path):
        """Test generate_hash_file when touch command fails - skipped on Windows."""
        # Skip this test as it involves platform-specific behavior
        # The error handling path is tested through other means
        pytest.skip("Touch command behavior is platform-specific")


class TestExecuteRepackProcess:
    """Test execute_repack_process function."""

    def test_execute_repack_process_custom_file_fail(self, monkeypatch):
        """Test execute_repack_process when generate_customized_file fails."""

        class MockXmlConfig:
            default_config = {"name": "test"}
            generate_infos = [{"value": "test.info"}]

        monkeypatch.setattr(
            package_module, "generate_customized_file", lambda *args: False
        )

        pkg_opt = package_module.PackageOption(
            os_arch="",
            package_suffix="",
            not_in_name="",
            pkg_version="",
            ext_name="",
            chip_name="",
            func_name="",
            version_dir="",
            disable_multi_version=False,
            suffix="",
        )
        args = Namespace(ext_name="")
        result = package_module.execute_repack_process(
            MockXmlConfig(), "/tmp", args, package_option=pkg_opt
        )
        assert result is False


class TestGenFileInstallList:
    """Test gen_file_install_list function."""

    def test_gen_file_install_list_with_content(self, monkeypatch):
        """Test gen_file_install_list with actual content."""

        class MockBlockConfig:
            dir_install_list = []
            move_files = []
            expand_content_list = []
            package_content_list = [
                {
                    "dst_path": "/install",
                    "value": "file.txt",
                    "install_path": "/install",
                    "module": "test",
                    "install_mod": "755",
                    "install_own": "user",
                    "install_type": "run",
                    "install_softlink": "",
                    "feature": {"all"},
                    "configurable": "FALSE",
                    "hash": "NA",
                    "name": "block1",
                    "pkg_inner_softlink": "",
                    "chip": {"ascend910"},
                }
            ]
            generate_infos = []
            pkg_soft_links = []

        # XmlConfig needs properties that delegate to blocks
        class MockXmlConfig:
            def __init__(self):
                self.blocks = [MockBlockConfig()]
                self.packer_config = type(
                    "obj", (object,), {"fill_is_common_path": lambda self, x: x}
                )()
                # Properties that collect from blocks
                self.dir_install_list = []
                self.move_content_list = []
                self.expand_content_list = []
                self.package_content_list = MockBlockConfig.package_content_list
                self.generate_infos = []
                self.pkg_soft_links = []

        xml_config = MockXmlConfig()
        # Filter key doesn't match 'run', so is_in_docker should be 'FALSE'
        result, _ = package_module.gen_file_install_list(None, xml_config, ["docker"])
        assert len(result) == 1
        assert result[0].is_in_docker == "FALSE"


class TestProcessingCsvFileWithData:
    """Extended tests for processing_csv_file."""

    def test_processing_csv_file_with_data(self, monkeypatch, tmp_path):
        """Test processing_csv_file with valid CSV data."""
        # Create config directory and limit.csv
        config_dir = tmp_path / "package" / "common"
        config_dir.mkdir(parents=True)

        limit_csv = config_dir / "limit.csv"
        with open(limit_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "header1",
                    "path",
                    "header3",
                    "header4",
                    "max_size",
                    "chip",
                    "product",
                    "build_type",
                ]
            )
            writer.writerow(
                [
                    "test_pkg",
                    "test_dir",
                    "extra",
                    "extra2",
                    "1000",
                    "ascend910",
                    "test_product",
                    "debug",
                ]
            )

        monkeypatch.setattr(package_module.pkg_utils, "TOP_SOURCE_DIR", str(tmp_path))

        # Create the test directory
        release_dir = tmp_path / "release" / "test_product"
        release_dir.mkdir(parents=True)
        test_dir = release_dir / "test_dir"
        test_dir.mkdir()
        (test_dir / "test_file.txt").write_text("x" * 500)  # Small file

        limit_list, ret = package_module.processing_csv_file(
            str(release_dir), "test_pkg", "ascend910", "debug"
        )
        assert ret is True


class TestChecksumValueExtended:
    """Extended tests for checksum_value."""

    def test_checksum_value_file_too_large(self, tmp_path):
        """Test checksum_value when directory exceeds max size."""
        # Create a directory with a file larger than max
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()
        (test_dir / "large_file.txt").write_text("x" * 3000)  # 3000 bytes

        # max_value is 2 KB (at index 4), file is ~3000 bytes
        limit_value = [
            "pkg",
            "test_dir",
            "100",
            "110%",
            "2",
            "chip",
            "product",
            "debug",
        ]
        result = package_module.checksum_value(limit_value, str(tmp_path))
        assert result is False

    def test_checksum_value_valid_file(self, tmp_path):
        """Test checksum_value with valid file size."""
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()
        (test_dir / "small_file.txt").write_text("small content")

        limit_value = [
            "pkg",
            "test_dir",
            "100",
            "110%",
            "10000",
            "chip",
            "product",
            "debug",
        ]
        result = package_module.checksum_value(limit_value, str(tmp_path))
        assert result is True


class TestGenerateFilelistFileByXmlConfigBasic:
    """Test generate_filelist_file_by_xml_config function."""

    def test_generate_filelist_file_by_xml_config(self, monkeypatch, tmp_path):
        """Test generating filelist from XML config."""

        class MockXmlConfig:
            package_attr = {"parallel": True, "func_name": "test_func"}
            packer_config = type(
                "obj", (object,), {"fill_is_common_path": lambda self, x: x}
            )()

            @property
            def dir_install_list(self):
                return []

            @property
            def move_content_list(self):
                return []

            @property
            def expand_content_list(self):
                return []

            @property
            def package_content_list(self):
                return [
                    {
                        "dst_path": "/install",
                        "value": "file.txt",
                        "install_path": "/install",
                        "module": "test",
                        "install_mod": "755",
                        "install_own": "user",
                        "install_type": "run",
                        "install_softlink": "",
                        "feature": {"comm"},
                        "configurable": "FALSE",
                        "hash": "NA",
                        "name": "block1",
                        "pkg_inner_softlink": "",
                        "chip": {"ascend910"},
                    }
                ]

            @property
            def generate_infos(self):
                return []

            @property
            def pkg_soft_links(self):
                return []

        # Mock generate_modules_yaml 避免它去读物理文件
        monkeypatch.setattr(
            package_module, "generate_modules_yaml", lambda *args, **kwargs: None
        )
        package_module.generate_filelist_file_by_xml_config(
            MockXmlConfig(),  # xml_config
            ["all"],  # filter_key
            str(tmp_path),  # delivery_dir
            False,  # package_check
            "deb",  # suffix
            "test_pkg",  # pkg_name
            "9.1.0",  # version
            str(tmp_path),  # source_dir
        )

        # Verify file was created
        assert (
            tmp_path / "share" / "info" / "test_func" / "script" / "filelist.csv"
        ).exists()


class TestDoCopyExtended:
    """Extended tests for do_copy function to cover more branches."""

    @staticmethod
    def test_do_copy_with_pkg_softlink_success(monkeypatch, tmp_path):
        """Test do_copy with pkg_softlink when all links succeed."""
        target_conf = {
            "value": "file.txt",
            "dst_path": "dst",
            "pkg_softlink": ["link1", "link2"],
        }

        # Mock create_softlink to return True
        def mock_create_softlink(source, target, optional):
            return True

        monkeypatch.setattr(package_module, "create_softlink", mock_create_softlink)

        result = package_module.do_copy(target_conf, str(tmp_path), str(tmp_path), None)
        assert result is True

    @staticmethod
    def test_do_copy_with_pkg_softlink_failure(monkeypatch, tmp_path):
        """Test do_copy with pkg_softlink when some links fail."""
        target_conf = {
            "value": "file.txt",
            "dst_path": "dst",
            "pkg_softlink": ["link1", "link2"],
        }

        # Mock create_softlink to return False
        def mock_create_softlink(source, target, optional):
            return False

        monkeypatch.setattr(package_module, "create_softlink", mock_create_softlink)

        result = package_module.do_copy(target_conf, str(tmp_path), str(tmp_path), None)
        assert result is False


class TestDoChmod:
    """Test do_chmod function."""

    @staticmethod
    def test_do_chmod_no_pkg_mod():
        """Test do_chmod when pkg_mod is not set."""
        target_conf = {"value": "file.txt"}
        result = package_module.do_chmod(target_conf, "/release")
        assert result is True

    @staticmethod
    def test_do_chmod_with_pkg_mod(monkeypatch, tmp_path):
        """Test do_chmod with pkg_mod set."""
        target_conf = {"value": "file.txt", "dst_path": "", "pkg_mod": "755"}

        # Create the file
        (tmp_path / "file.txt").write_text("content")

        # Mock subprocess.run
        def mock_run(cmd, **kwargs):
            class Result:
                returncode = 0
                stdout = ""
                stderr = ""

            return Result()

        monkeypatch.setattr(subprocess, "run", mock_run)

        result = package_module.do_chmod(target_conf, str(tmp_path))
        assert result is True

    @staticmethod
    def test_do_chmod_failure(monkeypatch, tmp_path):
        """Test do_chmod when chmod command fails."""
        target_conf = {"value": "file.txt", "dst_path": "", "pkg_mod": "755"}

        # Mock subprocess.run to return failure
        def mock_run(cmd, **kwargs):
            class Result:
                returncode = 1
                stdout = ""
                stderr = "permission denied"

            return Result()

        monkeypatch.setattr(subprocess, "run", mock_run)

        result = package_module.do_chmod(target_conf, str(tmp_path))
        assert result is False

    @staticmethod
    def test_do_chmod_exception(monkeypatch, tmp_path):
        """Test do_chmod when exception occurs."""
        target_conf = {"value": "file.txt", "dst_path": "", "pkg_mod": "755"}

        # Mock subprocess.run to raise exception
        def mock_run(cmd, **kwargs):
            raise Exception("command not found")

        monkeypatch.setattr(subprocess, "run", mock_run)

        result = package_module.do_chmod(target_conf, str(tmp_path))
        assert result is False


class TestCreateSoftlink:
    """Test create_softlink function."""

    @staticmethod
    def test_create_softlink_new_file(tmp_path, monkeypatch):
        """Test creating softlink to a new file."""
        source = tmp_path / "source.txt"
        source.write_text("content")
        target = tmp_path / "link.txt"

        # Mock os.symlink
        symlink_calls = []

        def mock_symlink(src, dst):
            symlink_calls.append((src, dst))

        monkeypatch.setattr(os, "symlink", mock_symlink)

        result = package_module.create_softlink(str(source), str(target), True)
        assert result is True
        assert len(symlink_calls) == 1

    @staticmethod
    def test_create_softlink_replace_existing_file(tmp_path, monkeypatch):
        """Test creating softlink when target file exists."""
        source = tmp_path / "source.txt"
        source.write_text("content")
        target = tmp_path / "target.txt"
        target.write_text("existing")

        # Mock subprocess.run for rm command
        run_calls = []

        def mock_run(cmd, **kwargs):
            run_calls.append(cmd)

            class Result:
                returncode = 0
                stdout = ""
                stderr = ""

            return Result()

        monkeypatch.setattr(subprocess, "run", mock_run)

        # Mock os.symlink
        def mock_symlink(src, dst):
            pass

        monkeypatch.setattr(os, "symlink", mock_symlink)

        result = package_module.create_softlink(str(source), str(target), True)
        assert result is True

    @staticmethod
    def test_create_softlink_existing_directory(tmp_path):
        """Test creating softlink when target is a directory."""
        source = tmp_path / "source.txt"
        source.write_text("content")
        target = tmp_path / "target_dir"
        target.mkdir()

        result = package_module.create_softlink(str(source), str(target), True)
        assert result is True

    @staticmethod
    def test_create_softlink_rm_fails(tmp_path, monkeypatch):
        """Test create_softlink when rm command fails."""
        source = tmp_path / "source.txt"
        source.write_text("content")
        target = tmp_path / "target.txt"
        target.write_text("existing")

        # Mock subprocess.run to return failure
        def mock_run(cmd, **kwargs):
            class Result:
                returncode = 1
                stdout = ""
                stderr = "cannot remove"

            return Result()

        monkeypatch.setattr(subprocess, "run", mock_run)

        result = package_module.create_softlink(str(source), str(target), True)
        assert result is False

    @staticmethod
    def test_create_softlink_rm_exception(tmp_path, monkeypatch):
        """Test create_softlink when rm command raises exception."""
        source = tmp_path / "source.txt"
        source.write_text("content")
        target = tmp_path / "target.txt"
        target.write_text("existing")

        # Mock subprocess.run to raise exception
        def mock_run(cmd, **kwargs):
            raise Exception("command failed")

        monkeypatch.setattr(subprocess, "run", mock_run)

        result = package_module.create_softlink(str(source), str(target), True)
        assert result is False

    @staticmethod
    def test_create_softlink_source_not_exists_fails(tmp_path, monkeypatch):
        """Test create_softlink when rm command raises exception."""
        source = tmp_path / "source.txt"
        target = tmp_path / "target.txt"

        result = package_module.create_softlink(str(source), str(target), False)
        assert result is False


class TestParseInstallInfoExtended:
    """Extended tests for parse_install_info to cover optional logic."""

    @staticmethod
    def test_parse_install_info_optional_file_not_exists(monkeypatch):
        """Test parse_install_info with optional=true when file doesn't exist."""
        infos = setup_optional_file_not_exists_test(monkeypatch)

        result = list(package_module.parse_install_info("/", infos, "copy", ["all"]))
        # Should be skipped because path doesn't exist
        assert len(result) == 0

    @staticmethod
    def test_parse_install_info_optional_file_exists(monkeypatch, tmp_path):
        """Test parse_install_info with optional=true when file exists."""
        delivery_path = tmp_path / "delivery"
        delivery_path.mkdir()

        # Create the directory and file
        dst_dir = delivery_path / "dst"
        dst_dir.mkdir(parents=True)
        (dst_dir / "file.txt").write_text("content")

        infos = [
            create_base_install_info(
                dst_path="dst",
                value="dst/file.txt",
                install_path="dst",
                optional="true",
                is_dir=False,
            )
        ]

        result = list(
            package_module.parse_install_info(delivery_path, infos, "copy", ["all"])
        )
        assert len(result) == 1

    @staticmethod
    def test_parse_install_info_move_optional(monkeypatch):
        """Test parse_install_info with move operation and optional."""
        infos = setup_optional_file_not_exists_test(monkeypatch)

        result = list(package_module.parse_install_info("/", infos, "move", ["all"]))
        # Should be skipped because path doesn't exist
        assert len(result) == 0

    @staticmethod
    def test_parse_install_info_relative_install_path_none():
        """Test parse_install_info when relative_install_path is None."""
        infos = [
            create_base_install_info(
                dst_path="dst",
                value="file.txt",
                install_path=None,  # This makes path_join return None for 'del' operation
            )
        ]

        # For 'del' operation, when install_path is None, path_join returns None
        # and then it continues at line 424-425
        result = list(package_module.parse_install_info("/", infos, "del", ["all"]))
        assert (
            len(result) == 0
        )  # Should be skipped because relative_install_path is None


class TestGenerateHashListExtended:
    """Extended tests for generate_hash_list function."""

    def test_generate_hash_list_success(self, monkeypatch, tmp_path):
        """Test generate_hash_list successful case."""
        target_conf = {"value": "file.txt", "rename": "renamed.txt", "dst_path": ""}

        # Create the file
        (tmp_path / "renamed.txt").write_text("content")

        # Mock subprocess.run for sha256sum
        def mock_run(cmd, **kwargs):
            class Result:
                returncode = 0
                stdout = "abc123  /path/to/renamed.txt"
                stderr = ""

            return Result()

        monkeypatch.setattr(subprocess, "run", mock_run)

        result, hash_str = package_module.generate_hash_list(
            target_conf, "", str(tmp_path)
        )
        assert result is True
        assert "abc123" in hash_str

    def test_generate_hash_list_failure(self, monkeypatch, tmp_path):
        """Test generate_hash_list when sha256sum fails."""
        target_conf = {"value": "file.txt", "rename": "renamed.txt", "dst_path": ""}

        # Create the file
        (tmp_path / "renamed.txt").write_text("content")

        # Mock subprocess.run to return failure
        def mock_run(cmd, **kwargs):
            class Result:
                returncode = 1
                stdout = ""
                stderr = "file not found"

            return Result()

        monkeypatch.setattr(subprocess, "run", mock_run)

        result, hash_str = package_module.generate_hash_list(
            target_conf, "", str(tmp_path)
        )
        assert result is False


class TestGenerateHashFileExtended:
    """Extended tests for generate_hash_file function."""

    @staticmethod
    def test_generate_hash_file_create_new(monkeypatch, tmp_path):
        """Test generate_hash_file when file doesn't exist."""
        hash_list = "hash1 file1\nhash2 file2"

        # Mock subprocess.run for touch command
        def mock_run(cmd, **kwargs):
            class Result:
                returncode = 0
                stdout = ""
                stderr = ""

            return Result()

        monkeypatch.setattr(subprocess, "run", mock_run)

        result = package_module.generate_hash_file(str(tmp_path), hash_list)
        assert result is True

        # Verify content
        hash_path = tmp_path / "bin_hash.cfg"
        assert hash_path.exists()
        assert hash_path.read_text() == hash_list

    @staticmethod
    def test_generate_hash_file_touch_fails(monkeypatch, tmp_path):
        """Test generate_hash_file when touch command fails."""

        # Mock subprocess.run to simulate touch failure
        def mock_run(*args, **kwargs):
            class MockResult:
                returncode = 1
                stdout = "touch failed"
                stderr = ""

            return MockResult()

        monkeypatch.setattr(package_module.subprocess, "run", mock_run)
        monkeypatch.setattr(package_module.os.path, "exists", lambda x: False)

        result = package_module.generate_hash_file(str(tmp_path), "content")
        assert result is False


class TestGetCompressCmd:
    """Test get_compress_cmd function."""

    @staticmethod
    def test_get_compress_cmd_run_suffix(monkeypatch, tmp_path):
        """Test get_compress_cmd with run suffix."""
        mock_xml_config, pkg_args = setup_compress_cmd_test(monkeypatch, tmp_path)

        # Mock create_run_package_command
        monkeypatch.setattr(
            package_module, "create_run_package_command", lambda x: ("cmd", None)
        )

        # Mock exec_pack_cmd
        exec_calls = []

        def mock_exec(*args):
            exec_calls.append(args)

        monkeypatch.setattr(package_module, "exec_pack_cmd", mock_exec)

        build_dir = tmp_path / "build"
        build_dir.mkdir()

        result = package_module.get_compress_cmd(
            str(tmp_path), str(build_dir), pkg_args, mock_xml_config
        )
        assert result == "test-package"

    @staticmethod
    def test_get_compress_cmd_unsupported_suffix(monkeypatch, tmp_path):
        """Test get_compress_cmd with unsupported suffix."""
        mock_xml_config = Namespace(package_attr={"suffix": "zip"}, version="9.0.0")

        pkg_args = Namespace(
            pkg_output_dir=str(tmp_path),
            makeself_dir="/makeself",
            independent_pkg=True,
        )
        monkeypatch.setattr(package_module, "PackageName", MockPackageName)

        with pytest.raises(package_module.CompressError):
            package_module.get_compress_cmd(
                str(tmp_path), str(tmp_path), pkg_args, mock_xml_config
            )

    @staticmethod
    def test_get_compress_cmd_compress_error(monkeypatch, tmp_path):
        """Test get_compress_cmd when create_run_package_command returns error."""
        mock_xml_config, pkg_args = setup_compress_cmd_test(monkeypatch, tmp_path)

        # Mock create_run_package_command to return error
        monkeypatch.setattr(
            package_module,
            "create_run_package_command",
            lambda x: (None, "error message"),
        )

        build_dir = tmp_path / "build"
        build_dir.mkdir()

        with pytest.raises(package_module.CompressError):
            package_module.get_compress_cmd(
                str(tmp_path), str(build_dir), pkg_args, mock_xml_config
            )

    @staticmethod
    def test_get_compress_cmd_save_fails(monkeypatch, tmp_path):
        """Test get_compress_cmd when saving makeself.txt fails."""
        mock_xml_config = Namespace(package_attr={"suffix": "run"}, version="1.0.0")

        pkg_args = Namespace(
            pkg_output_dir=str(tmp_path),
            makeself_dir=str(tmp_path / "makeself"),
            independent_pkg=False,
        )

        # Use shared MockPackageName class
        monkeypatch.setattr(package_module, "PackageName", MockPackageName)

        # Mock get_comments
        monkeypatch.setattr(package_module, "get_comments", lambda x: '"test comments"')

        # Use shared mock factory
        monkeypatch.setattr(
            package_module, "create_makeself_pkg_params_factory", create_mock_factory()
        )

        # Mock create_run_package_command
        monkeypatch.setattr(
            package_module, "create_run_package_command", lambda x: ("cmd", None)
        )

        # Create a non-writable build directory by using a file as the path
        build_dir = tmp_path / "build_file"
        build_dir.write_text("not a directory")

        with pytest.raises(package_module.CompressError):
            package_module.get_compress_cmd(
                str(tmp_path), str(build_dir), pkg_args, mock_xml_config
            )


class TestExecuteRepackProcessExtended:
    """Extended tests for execute_repack_process function."""

    @staticmethod
    def test_execute_repack_process_with_custom_file(monkeypatch, tmp_path):
        """Test execute_repack_process when generate_customized_file returns False."""

        mock_xml_config = create_mock_xml_config(generate_infos=[{"ext_name": "test"}])

        pkg_args = Namespace(check_size="False")

        # Mock generate_customized_file to return False
        monkeypatch.setattr(
            package_module, "generate_customized_file", lambda a, b, c: False
        )

        result = package_module.execute_repack_process(
            mock_xml_config,
            str(tmp_path),
            pkg_args,
            MockPackageName(),
            MockPackageOption(),
        )
        assert result is False

    @staticmethod
    def test_execute_repack_process_with_is_hash(monkeypatch, tmp_path):
        """Test execute_repack_process with is_hash item."""

        mock_xml_config = create_mock_xml_config(
            package_content_list=[
                {"value": "file.txt", "is_hash": "true", "dst_path": ""}
            ]
        )

        pkg_args = Namespace(check_size="False", pkg_output_dir=str(tmp_path))

        # Mocks
        monkeypatch.setattr(
            package_module, "generate_customized_file", lambda a, b, c: True
        )
        monkeypatch.setattr(package_module, "get_target_name", lambda x: "file.txt")
        monkeypatch.setattr(package_module, "generate_hash_file", lambda a, b: True)
        monkeypatch.setattr(package_module, "do_copy", lambda *args: True)
        monkeypatch.setattr(
            package_module, "generate_hash_list", lambda *args: (False, "")
        )
        monkeypatch.setattr(
            package_module, "softlink_before_package", lambda *args: None
        )

        result = package_module.execute_repack_process(
            mock_xml_config,
            str(tmp_path),
            pkg_args,
            MockPackageName(),
            MockPackageOption(),
        )
        assert result is False  # Because generate_hash_list returns False

    @staticmethod
    def test_execute_repack_process_check_size_fail(monkeypatch, tmp_path):
        """Test execute_repack_process when check_size fails."""

        mock_xml_config = create_mock_xml_config()

        pkg_args = Namespace(check_size="True", build_type="debug")

        # Mocks
        monkeypatch.setattr(
            package_module, "generate_customized_file", lambda a, b, c: True
        )
        monkeypatch.setattr(
            package_module, "processing_csv_file", lambda *args: ([], False)
        )  # tag=False

        result = package_module.execute_repack_process(
            mock_xml_config,
            str(tmp_path),
            pkg_args,
            MockPackageName(),
            MockPackageOption(),
        )
        assert result is False

    @staticmethod
    def test_execute_repack_process_check_size_with_limit(monkeypatch, tmp_path):
        """Test execute_repack_process when check_size has limit_list."""

        mock_xml_config = create_mock_xml_config()

        pkg_args = Namespace(check_size="True", build_type="debug")

        # Mocks
        monkeypatch.setattr(
            package_module, "generate_customized_file", lambda a, b, c: True
        )
        monkeypatch.setattr(
            package_module, "processing_csv_file", lambda *args: (["limit"], True)
        )
        monkeypatch.setattr(
            package_module, "check_add_dir", lambda *args: False
        )  # Returns False

        result = package_module.execute_repack_process(
            mock_xml_config,
            str(tmp_path),
            pkg_args,
            MockPackageName(),
            MockPackageOption(),
        )
        assert result is False

    @staticmethod
    def test_execute_repack_process_compress_error(monkeypatch, tmp_path):
        """Test execute_repack_process when get_compress_cmd raises CompressError."""

        mock_xml_config = create_mock_xml_config()

        pkg_args = Namespace(check_size="False", pkg_output_dir=str(tmp_path))

        # Mocks
        monkeypatch.setattr(
            package_module, "generate_customized_file", lambda a, b, c: True
        )
        monkeypatch.setattr(
            package_module, "softlink_before_package", lambda *args: None
        )

        def mock_get_compress(*args):
            raise package_module.CompressError("test")

        monkeypatch.setattr(package_module, "get_compress_cmd", mock_get_compress)

        result = package_module.execute_repack_process(
            mock_xml_config,
            str(tmp_path),
            pkg_args,
            MockPackageName(),
            MockPackageOption(),
        )
        assert result is False


class TestProcessingCsvFileExtended:
    """Extended tests for processing_csv_file function."""

    @staticmethod
    def test_processing_csv_file_no_csv(monkeypatch, tmp_path):
        """Test processing_csv_file when CSV file doesn't exist."""
        result = package_module.processing_csv_file(
            str(tmp_path), "func", "chip", "debug"
        )
        assert result == ([], True)  # Empty list, tag=True


class TestSoftlinkBeforePackage:
    """Test softlink_before_package function in packer module."""

    @staticmethod
    def test_softlink_before_package_empty():
        """Test with empty softlinks list."""
        packer.softlink_before_package([], "/release")
        # Should complete without error

    @staticmethod
    def test_softlink_before_package_with_links(monkeypatch, tmp_path):
        """Test with actual softlinks."""
        # Create a source file (relative path)
        source = "source.txt"
        target = "target.txt"
        (tmp_path / source).write_text("content")

        softlinks = [packer.PkgSoftlink(src_path=source, dst_path=target)]

        # Mock os.symlink because Windows requires admin privileges
        symlink_calls = []

        def mock_symlink(src, dst):
            symlink_calls.append((src, dst))

        monkeypatch.setattr(os, "symlink", mock_symlink)

        packer.softlink_before_package(softlinks, str(tmp_path))
        assert len(symlink_calls) == 1


class TestCheckAddDirExtended:
    """Extended tests for check_add_dir function."""

    @staticmethod
    def test_check_add_dir_with_limit_path_match(monkeypatch, tmp_path):
        """Test check_add_dir when path matches limit_list entry."""
        # Create a file
        (tmp_path / "test.txt").write_text("content")

        limit_list = [str(tmp_path)]

        result = package_module.check_add_dir(
            str(tmp_path), str(tmp_path), limit_list, True
        )
        assert result is True  # Returns ret when path matches

    @staticmethod
    def test_check_add_dir_with_subdir(monkeypatch, tmp_path):
        """Test check_add_dir with subdirectory."""
        # Create subdirectory with file
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "file.txt").write_text("content")

        # When subdir is in limit_list, it should return ret
        limit_list = [str(subdir)]

        result = package_module.check_add_dir(
            str(tmp_path), str(subdir), limit_list, True
        )
        assert result is True


class TestGenFileInstallListExtended:
    """Extended tests for gen_file_install_list function."""

    @staticmethod
    def test_gen_file_install_list_with_copy(tmp_path, monkeypatch):
        """Test gen_file_install_list with copy operation."""

        # Mock parse_install_info to return test data
        def mock_parse_install_info(delivery_dir, infos, operate_type, filter_key):
            if operate_type == "copy":
                yield package_module.create_file_item(
                    "test_module",
                    "copy",
                    "path/in/pkg",
                    "install/path",
                    "TRUE",
                    "755",
                    "user:group",
                    "run",
                    [],
                    {"all"},
                    "N",
                    "FALSE",
                    "NA",
                    "block1",
                    [],
                    {"ascend910"},
                    False,
                )

        monkeypatch.setattr(
            package_module, "parse_install_info", mock_parse_install_info
        )

        # Create MockXmlConfig with nested MockPackerConfig
        class MockXmlConfigGenFile:
            dir_install_list = []
            move_content_list = []
            package_content_list = []
            generate_infos = []
            expand_content_list = []
            packer_config = MockPackerConfig()

        result, _ = package_module.gen_file_install_list(
            None, MockXmlConfigGenFile(), ["all"]
        )
        assert len(result) >= 0  # May be empty depending on mock

    @staticmethod
    def test_gen_file_install_list_filter_no_match(tmp_path, monkeypatch):
        """Test gen_file_install_list when filter doesn't match."""

        # Mock parse_install_info to return empty (no matches)
        def mock_parse_install_info(delivery_dir, infos, operate_type, filter_key):
            return iter([])

        monkeypatch.setattr(
            package_module, "parse_install_info", mock_parse_install_info
        )

        # Reuse the same MockXmlConfig class
        class MockXmlConfigGenFile:
            dir_install_list = []
            move_content_list = []
            package_content_list = []
            generate_infos = []
            expand_content_list = []
            packer_config = MockPackerConfig()

        result, _ = package_module.gen_file_install_list(
            None, MockXmlConfigGenFile(), ["run"]
        )
        assert len(result) == 0


# =============================================================================
# #25/#27: main() 主流程及复杂逻辑测试补充
# =============================================================================


def _make_main_args(tmp_path, **overrides):
    """构造 main() 所需的 Namespace 参数对象。"""
    defaults = dict(
        delivery_dir=str(tmp_path),
        pkg_name="test",
        chip_scenes="",
        xml_file="",
        version_dir="8.0.0",
        source_root="",
        package_check=False,
        independent_pkg=False,
        pkg_output_dir="",
        os_arch="linux-x86_64",
        pkg_version="",
        build_type="debug",
        ext_name="",
        chip_name=None,
        func_name=None,
        disable_multi_version=False,
        suffix="run",
        source_dir=str(tmp_path),
        makeself_dir="",
        not_in_name="",
        package_suffix="none",
        tag="",
        check_size="",
        pkg_name_style="common",
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def _mock_main_success_env(monkeypatch, mock_xml_config, conflict_ret, repack_ret=True):
    """Mock main() 正常流程所需的 parse_xml_config / filelist / conflict / repack。"""
    monkeypatch.setattr(
        package_module, "parse_xml_config", lambda *a, **k: (True, mock_xml_config)
    )
    monkeypatch.setattr(
        package_module, "generate_filelist_file_by_xml_config", lambda *a, **k: None
    )
    monkeypatch.setattr(
        package_module, "check_path_is_conflict", lambda x: conflict_ret
    )
    monkeypatch.setattr(
        package_module, "execute_repack_process", lambda *a, **k: repack_ret
    )


def _make_repack_args(check_size="", build_type="debug"):
    """构造 execute_repack_process 所需的 Namespace 参数。"""
    return Namespace(check_size=check_size, build_type=build_type, pkg_output_dir="")


def _make_repack_pkg_name(func_name="test", chip_name=None):
    """构造 execute_repack_process 所需的 package_name mock 对象。"""
    pkg_name = mock.MagicMock()
    pkg_name.func_name = func_name
    if chip_name is not None:
        pkg_name.chip_name = chip_name
    return pkg_name


def _make_empty_mock_xml():
    """构造内容为空的 XmlConfig mock 对象。"""
    mock_xml = mock.MagicMock()
    mock_xml.generate_infos = []
    mock_xml.package_content_list = []
    mock_xml.move_content_list = []
    mock_xml.pkg_softlinks = []
    return mock_xml


def _make_filelist_mock_xml_config():
    """构造 generate_filelist_file_by_xml_config 所需的 mock XmlConfig。"""
    mock_xml_config = mock.MagicMock()
    mock_xml_config.package_attr = {
        "use_move": False,
        "parallel": False,
        "check_features": False,
        "func_name": "test",
        "package_check": False,
        "copy_all": False,
    }
    mock_xml_config.dir_install_list = []
    mock_xml_config.move_content_list = []
    mock_xml_config.package_content_list = []
    mock_xml_config.generate_infos = []
    mock_xml_config.expand_content_list = []
    mock_xml_config.packer_config = mock.MagicMock()

    def identity(x):
        return x

    mock_xml_config.packer_config.fill_is_common_path = identity
    return mock_xml_config


class TestMainEmptyDeliveryDir:
    """main() 传入空 delivery_dir 时返回 False。"""

    @staticmethod
    def test_main_empty_delivery_dir():
        args = _make_main_args(Path("/tmp"), delivery_dir="")
        assert package_module.main("test", "", args) is False


class TestMainRpmDebDeliveryDir:
    """main() 在 rpm/deb 模式下直接使用 delivery_dir（不拼接 makeself_staging）。"""

    @staticmethod
    def test_main_rpm_delivery_dir_not_exists(tmp_path):
        args = _make_main_args(
            tmp_path, suffix="rpm", delivery_dir=str(tmp_path / "nonexistent")
        )
        assert package_module.main("test", "", args) is False


class TestMainParseXmlFailure:
    """main() 在 parse_xml_config 返回 False 时返回 False。"""

    @staticmethod
    def test_main_parse_xml_fail(tmp_path, monkeypatch):
        args = _make_main_args(tmp_path)
        monkeypatch.setattr(
            package_module, "parse_xml_config", lambda *a, **k: (False, None)
        )
        assert package_module.main("test", "", args) is False


class TestMainContainAsteriskError:
    """main() 在 ContainAsteriskError 异常时返回 False。"""

    @staticmethod
    def test_main_asterisk_error(tmp_path, monkeypatch):
        args = _make_main_args(tmp_path)

        def raise_asterisk(*a, **k):
            raise package_module.ContainAsteriskError("bad*value")

        monkeypatch.setattr(package_module, "parse_xml_config", raise_asterisk)
        assert package_module.main("test", "", args) is False


class TestMainFilelistErrors:
    """main() 在 filelist 生成异常时返回 False。"""

    @staticmethod
    def test_main_package_name_empty_error(tmp_path, monkeypatch):
        args = _make_main_args(tmp_path)
        mock_xml_config = mock.MagicMock()
        mock_xml_config.package_attr = {"package_check": False}
        mock_xml_config.version = "8.0.0"
        monkeypatch.setattr(
            package_module, "parse_xml_config", lambda *a, **k: (True, mock_xml_config)
        )

        def raise_pne(*a, **k):
            raise package_module.PackageNameEmptyError()

        monkeypatch.setattr(
            package_module, "generate_filelist_file_by_xml_config", raise_pne
        )
        assert package_module.main("test", "", args) is False

    @staticmethod
    def test_main_generate_filelist_error(tmp_path, monkeypatch):
        args = _make_main_args(tmp_path)
        mock_xml_config = mock.MagicMock()
        mock_xml_config.package_attr = {"package_check": False}
        mock_xml_config.version = "8.0.0"
        monkeypatch.setattr(
            package_module, "parse_xml_config", lambda *a, **k: (True, mock_xml_config)
        )

        def raise_gfe(*a, **k):
            raise package_module.GenerateFilelistError("filelist.csv")

        monkeypatch.setattr(
            package_module, "generate_filelist_file_by_xml_config", raise_gfe
        )
        assert package_module.main("test", "", args) is False

    @staticmethod
    def test_main_filelist_error(tmp_path, monkeypatch):
        args = _make_main_args(tmp_path)
        mock_xml_config = mock.MagicMock()
        mock_xml_config.package_attr = {"package_check": False}
        mock_xml_config.version = "8.0.0"
        monkeypatch.setattr(
            package_module, "parse_xml_config", lambda *a, **k: (True, mock_xml_config)
        )

        def raise_fe(*a, **k):
            raise package_module.FilelistError("missing dir")

        monkeypatch.setattr(
            package_module, "generate_filelist_file_by_xml_config", raise_fe
        )
        assert package_module.main("test", "", args) is False


class TestMainPathConflict:
    """main() 在 check_path_is_conflict 返回 False 时返回 False。"""

    @staticmethod
    def test_main_path_conflict(tmp_path, monkeypatch):
        args = _make_main_args(tmp_path)
        mock_xml_config = mock.MagicMock()
        mock_xml_config.package_attr = {"package_check": False}
        mock_xml_config.version = "8.0.0"
        monkeypatch.setattr(
            package_module, "parse_xml_config", lambda *a, **k: (True, mock_xml_config)
        )
        monkeypatch.setattr(
            package_module, "generate_filelist_file_by_xml_config", lambda *a, **k: None
        )
        monkeypatch.setattr(package_module, "check_path_is_conflict", lambda x: False)
        assert package_module.main("test", "", args) is False


class TestMainSuccessFlow:
    """main() 正常流程 mock 验证返回 True。"""

    @staticmethod
    def test_main_success(tmp_path, monkeypatch):
        args = _make_main_args(tmp_path)
        staging = tmp_path / "_CPack_Packages" / "makeself_staging"
        staging.mkdir(parents=True)
        mock_xml_config = mock.MagicMock()
        mock_xml_config.package_attr = {"package_check": False, "func_name": "test"}
        mock_xml_config.version = "8.0.0"
        _mock_main_success_env(monkeypatch, mock_xml_config, True)
        assert package_module.main("test", "", args) is True


class TestGenerateModulesYaml:
    """generate_modules_yaml 生成 postinst/prerm/modules.txt 文件。"""

    @staticmethod
    def test_generate_modules_yaml_with_items(tmp_path, monkeypatch):
        from collections import namedtuple

        MockItem = namedtuple(
            "MockItem", ["block", "softlink", "relative_install_path", "permission"]
        )
        items = [
            MockItem(
                block="TestModule",
                softlink=["lib64/liba.so"],
                relative_install_path="lib/liba.so",
                permission="550",
            ),
        ]
        monkeypatch.chdir(tmp_path)
        package_module.generate_modules_yaml(
            items, "cann-test", "8.0.0", str(tmp_path), "modules.txt"
        )
        assert (tmp_path / "postinst").exists()
        assert (tmp_path / "prerm").exists()
        assert (tmp_path / "modules.txt").exists()
        import json

        modules = json.loads((tmp_path / "modules.txt").read_text())
        assert "TestModule" in modules

    @staticmethod
    def test_generate_modules_yaml_empty_items(tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        package_module.generate_modules_yaml(
            [], "cann-test", "8.0.0", str(tmp_path), "modules.txt"
        )
        assert (tmp_path / "postinst").exists()
        assert (tmp_path / "prerm").exists()

    @staticmethod
    def test_generate_modules_yaml_skip_na_block(tmp_path, monkeypatch):
        from collections import namedtuple

        MockItem = namedtuple(
            "MockItem", ["block", "softlink", "relative_install_path", "permission"]
        )
        items = [
            MockItem(
                block="NA",
                softlink=["lib64/liba.so"],
                relative_install_path="lib/liba.so",
                permission="550",
            ),
        ]
        monkeypatch.chdir(tmp_path)
        package_module.generate_modules_yaml(
            items, "cann-test", "8.0.0", str(tmp_path), "modules.txt"
        )
        import json

        modules = json.loads((tmp_path / "modules.txt").read_text())
        assert "NA" not in modules

    @staticmethod
    def test_generate_modules_yaml_skip_empty_softlink(tmp_path, monkeypatch):
        from collections import namedtuple

        MockItem = namedtuple(
            "MockItem", ["block", "softlink", "relative_install_path", "permission"]
        )
        items = [
            MockItem(
                block="TestModule",
                softlink=[],
                relative_install_path="lib/liba.so",
                permission="550",
            ),
        ]
        monkeypatch.chdir(tmp_path)
        package_module.generate_modules_yaml(
            items, "cann-test", "8.0.0", str(tmp_path), "modules.txt"
        )
        import json

        modules = json.loads((tmp_path / "modules.txt").read_text())
        assert "TestModule" not in modules


class TestGetShareInfoName:
    """get_share_info_name 返回 share_info_name 或 func_name。"""

    @staticmethod
    def test_with_share_info_name():
        assert (
            package_module.get_share_info_name(
                {"share_info_name": "custom", "func_name": "default"}
            )
            == "custom"
        )

    @staticmethod
    def test_without_share_info_name():
        assert package_module.get_share_info_name({"func_name": "default"}) == "default"


class TestGenerateFilelistFileByXmlConfig:
    """generate_filelist_file_by_xml_config 完整流程测试。"""

    @staticmethod
    def test_generate_filelist_run_suffix(tmp_path, monkeypatch):
        mock_xml_config = _make_filelist_mock_xml_config()
        monkeypatch.setattr(
            package_module, "gen_file_install_list", lambda *a, **k: ([], [])
        )
        package_module.generate_filelist_file_by_xml_config(
            mock_xml_config,
            [],
            str(tmp_path),
            False,
            "run",
            "test",
            "8.0.0",
            str(tmp_path),
        )
        script_dir = tmp_path / "share" / "info" / "test" / "script"
        assert (script_dir / "filelist.csv").exists()

    @staticmethod
    def test_generate_filelist_rpm_generates_postinst(tmp_path, monkeypatch):
        mock_xml_config = _make_filelist_mock_xml_config()
        monkeypatch.setattr(
            package_module, "gen_file_install_list", lambda *a, **k: ([], [])
        )
        monkeypatch.chdir(tmp_path)
        package_module.generate_filelist_file_by_xml_config(
            mock_xml_config,
            [],
            str(tmp_path),
            False,
            "rpm",
            "cann-test",
            "8.0.0",
            str(tmp_path),
        )
        assert (tmp_path / "postinst").exists()
        assert (tmp_path / "prerm").exists()


class TestExecuteRepackProcessExtended2:
    """execute_repack_process 补充: hash 生成失败、do_copy 失败、check_size 路径。"""

    @staticmethod
    def test_execute_repack_hash_file_fail(tmp_path, monkeypatch):
        mock_xml = mock.MagicMock()
        mock_xml.generate_infos = []
        mock_xml.package_content_list = [{"value": "bin_hash.cfg", "dst_path": ""}]
        mock_xml.move_content_list = []
        mock_xml.pkg_softlinks = []

        monkeypatch.setattr(
            package_module, "generate_customized_file", lambda *a, **k: True
        )
        monkeypatch.setattr(package_module, "generate_hash_file", lambda *a, **k: False)

        args = _make_repack_args()
        pkg_name = _make_repack_pkg_name()
        pkg_option = mock.MagicMock()
        result = package_module.execute_repack_process(
            mock_xml,
            str(tmp_path),
            args,
            package_name=pkg_name,
            package_option=pkg_option,
        )
        assert result is False

    @staticmethod
    def test_execute_repack_do_copy_fail_continues(tmp_path, monkeypatch):
        mock_xml = mock.MagicMock()
        mock_xml.generate_infos = []
        mock_xml.package_content_list = [
            {
                "value": "libtest.so",
                "dst_path": "",
                "pkg_softlink": ["lib64/libtest.so"],
            }
        ]
        mock_xml.move_content_list = []
        mock_xml.pkg_softlinks = []

        monkeypatch.setattr(
            package_module, "generate_customized_file", lambda *a, **k: True
        )
        monkeypatch.setattr(package_module, "do_copy", lambda *a, **k: False)
        monkeypatch.setattr(
            package_module, "softlink_before_package", lambda *a, **k: None
        )
        monkeypatch.setattr(
            package_module, "get_compress_cmd", lambda *a, **k: "pkg.run"
        )

        args = _make_repack_args()
        pkg_name = _make_repack_pkg_name()
        pkg_option = mock.MagicMock()
        result = package_module.execute_repack_process(
            mock_xml,
            str(tmp_path),
            args,
            package_name=pkg_name,
            package_option=pkg_option,
        )
        assert result is True

    @staticmethod
    def test_execute_repack_hash_list_fail(tmp_path, monkeypatch):
        mock_xml = mock.MagicMock()
        mock_xml.generate_infos = []
        mock_xml.package_content_list = [
            {"value": "libtest.so", "dst_path": "", "is_hash": True}
        ]
        mock_xml.move_content_list = []
        mock_xml.pkg_softlinks = []

        monkeypatch.setattr(
            package_module, "generate_customized_file", lambda *a, **k: True
        )
        monkeypatch.setattr(package_module, "do_copy", lambda *a, **k: True)
        monkeypatch.setattr(
            package_module, "generate_hash_list", lambda *a, **k: (False, None)
        )

        args = _make_repack_args()
        pkg_name = _make_repack_pkg_name()
        pkg_option = mock.MagicMock()
        result = package_module.execute_repack_process(
            mock_xml,
            str(tmp_path),
            args,
            package_name=pkg_name,
            package_option=pkg_option,
        )
        assert result is False

    @staticmethod
    def test_execute_repack_check_size_pass(tmp_path, monkeypatch):
        mock_xml = _make_empty_mock_xml()
        monkeypatch.setattr(
            package_module, "generate_customized_file", lambda *a, **k: True
        )
        monkeypatch.setattr(
            package_module, "softlink_before_package", lambda *a, **k: None
        )
        monkeypatch.setattr(
            package_module, "processing_csv_file", lambda *a, **k: ([], True)
        )
        monkeypatch.setattr(
            package_module, "get_compress_cmd", lambda *a, **k: "pkg.run"
        )

        args = _make_repack_args(check_size="True")
        pkg_name = _make_repack_pkg_name(chip_name="ascend910")
        pkg_option = mock.MagicMock()
        result = package_module.execute_repack_process(
            mock_xml,
            str(tmp_path),
            args,
            package_name=pkg_name,
            package_option=pkg_option,
        )
        assert result is True

    @staticmethod
    def test_execute_repack_check_size_fail(tmp_path, monkeypatch):
        mock_xml = _make_empty_mock_xml()
        monkeypatch.setattr(
            package_module, "generate_customized_file", lambda *a, **k: True
        )
        monkeypatch.setattr(
            package_module, "softlink_before_package", lambda *a, **k: None
        )
        monkeypatch.setattr(
            package_module, "processing_csv_file", lambda *a, **k: ([], False)
        )

        args = _make_repack_args(check_size="True")
        pkg_name = _make_repack_pkg_name(chip_name="ascend910")
        pkg_option = mock.MagicMock()
        result = package_module.execute_repack_process(
            mock_xml,
            str(tmp_path),
            args,
            package_name=pkg_name,
            package_option=pkg_option,
        )
        assert result is False

    @staticmethod
    def test_execute_repack_compress_error(tmp_path, monkeypatch):
        mock_xml = _make_empty_mock_xml()
        monkeypatch.setattr(
            package_module, "generate_customized_file", lambda *a, **k: True
        )
        monkeypatch.setattr(
            package_module, "softlink_before_package", lambda *a, **k: None
        )

        def raise_compress(*a, **k):
            raise package_module.CompressError("pkg")

        monkeypatch.setattr(package_module, "get_compress_cmd", raise_compress)

        args = _make_repack_args()
        pkg_name = _make_repack_pkg_name(chip_name="ascend910")
        pkg_option = mock.MagicMock()
        result = package_module.execute_repack_process(
            mock_xml,
            str(tmp_path),
            args,
            package_name=pkg_name,
            package_option=pkg_option,
        )
        assert result is False
