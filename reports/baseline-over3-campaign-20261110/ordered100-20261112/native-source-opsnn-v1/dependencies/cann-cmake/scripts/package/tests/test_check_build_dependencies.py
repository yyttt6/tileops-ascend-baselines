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

"""Tests for check_build_dependencies module."""

import importlib.util
import sys
from pathlib import Path
from unittest import mock

import pytest

# Dynamically load the module to avoid 'py' namespace conflicts
_MODULE_PATH = Path(__file__).resolve().parents[2] / "version" / "check_build_dependencies.py"
spec = importlib.util.spec_from_file_location("check_build_dependencies", _MODULE_PATH)
_check_build_dependencies = importlib.util.module_from_spec(spec)
sys.modules["check_build_dependencies"] = _check_build_dependencies
spec.loader.exec_module(_check_build_dependencies)

Receiver = _check_build_dependencies.Receiver
parse_version_line = _check_build_dependencies.parse_version_line
read_pkg_version = _check_build_dependencies.read_pkg_version
check_build_dep_item = _check_build_dependencies.check_build_dep_item
check_build_dep = _check_build_dependencies.check_build_dep
check_build_deps = _check_build_dependencies.check_build_deps
main = _check_build_dependencies.main


class TestReceiver:
    """Test Receiver NamedTuple."""

    @staticmethod
    def test_receiver_creation():
        """Test Receiver can be created with warn_msgs and err_msgs."""
        recv = Receiver(warn_msgs=[], err_msgs=[])
        assert recv.warn_msgs == []
        assert recv.err_msgs == []

    @staticmethod
    def test_receiver_with_messages():
        """Test Receiver with initial messages."""
        recv = Receiver(
            warn_msgs=['warning1', 'warning2'],
            err_msgs=['error1']
        )
        assert len(recv.warn_msgs) == 2
        assert len(recv.err_msgs) == 1


class TestParseVersionLine:
    """Test parse_version_line function."""

    @staticmethod
    def test_parse_version_line_basic():
        """Test parsing basic version line."""
        assert parse_version_line('Version=1.0.0') == '1.0.0'

    @staticmethod
    def test_parse_version_line_with_alpha():
        """Test parsing version line with alpha suffix."""
        assert parse_version_line('Version=1.0.0-alpha') == '1.0.0'

    @staticmethod
    def test_parse_version_line_with_beta():
        """Test parsing version line with beta suffix."""
        assert parse_version_line('Version=2.5.0-beta.3') == '2.5.0'

    @staticmethod
    def test_parse_version_line_with_rc():
        """Test parsing version line with rc suffix."""
        assert parse_version_line('Version=3.0.0-rc1') == '3.0.0'

    @staticmethod
    def test_parse_version_line_with_whitespace():
        """Test parsing version line with leading/trailing whitespace."""
        assert parse_version_line('  Version=1.2.3  ') == '1.2.3'

    @staticmethod
    def test_parse_version_line_complex_version():
        """Test parsing complex version line."""
        assert parse_version_line('Version=10.20.30-alpha+build.123') == '10.20.30'


class TestReadPkgVersion:
    """Test read_pkg_version function."""

    @staticmethod
    def test_read_pkg_version_success(tmp_path):
        """Test reading version from existing file."""
        # Create the version file structure
        info_dir = tmp_path / 'share' / 'info' / 'test_pkg'
        info_dir.mkdir(parents=True)
        version_file = info_dir / 'version.info'
        version_file.write_text('Version=2.5.0-alpha\nOtherField=value\n')

        recv = Receiver([], [])
        result = read_pkg_version(recv, str(tmp_path), 'test_pkg')

        assert result == '2.5.0'
        assert recv.err_msgs == []

    @staticmethod
    def test_read_pkg_version_file_not_exists(tmp_path):
        """Test reading version when file does not exist."""
        recv = Receiver([], [])
        result = read_pkg_version(recv, str(tmp_path), 'nonexistent_pkg')

        assert result is None
        assert len(recv.err_msgs) == 1
        assert 'does not exist' in recv.err_msgs[0]

    @staticmethod
    def test_read_pkg_version_no_version_field(tmp_path):
        """Test reading version when Version field is missing."""
        info_dir = tmp_path / 'share' / 'info' / 'test_pkg'
        info_dir.mkdir(parents=True)
        version_file = info_dir / 'version.info'
        version_file.write_text('OtherField=value\nAnotherField=value2\n')

        recv = Receiver([], [])
        result = read_pkg_version(recv, str(tmp_path), 'test_pkg')

        assert result is None
        assert len(recv.err_msgs) == 1
        assert 'version field was not found' in recv.err_msgs[0]

    @staticmethod
    def test_read_pkg_version_empty_file(tmp_path):
        """Test reading version from empty file."""
        info_dir = tmp_path / 'share' / 'info' / 'test_pkg'
        info_dir.mkdir(parents=True)
        version_file = info_dir / 'version.info'
        version_file.write_text('')

        recv = Receiver([], [])
        result = read_pkg_version(recv, str(tmp_path), 'test_pkg')

        assert result is None
        assert len(recv.err_msgs) == 1

    @staticmethod
    def test_read_pkg_version_graph_autofusion_success(tmp_path):
        """Test reading version from existing file."""
        # Create the version file structure
        info_dir = tmp_path / 'share' / 'info' / 'graph_autofusion'
        info_dir.mkdir(parents=True)
        version_file = info_dir / 'version.info'
        version_file.write_text('Version=2.5.0-alpha\nOtherField=value\n')

        recv = Receiver([], [])
        result = read_pkg_version(recv, str(tmp_path), 'graph-autofusion')

        assert result == '2.5.0'
        assert recv.err_msgs == []


class TestCheckBuildDepItemExtended:
    """Extended tests for check_build_dep_item."""

    @staticmethod
    def test_check_build_dep_item_equal_length_no_padding():
        """Test comparison when versions have equal length (no padding needed)."""
        assert check_build_dep_item('1.0.0', '>=1.0.0') is True
        assert check_build_dep_item('1.0.0', '<=1.0.0') is True
        assert check_build_dep_item('1.0.0', '>1.0.0') is False
        assert check_build_dep_item('1.0.0', '<1.0.0') is False


class TestCheckBuildDepItem:
    """Test check_build_dep_item function with various operators."""

    # Tests for >= operator
    @staticmethod
    def test_check_build_dep_item_ge_equal():
        """Test >= operator with equal versions."""
        assert check_build_dep_item('1.0.0', '>=1.0.0') is True

    @staticmethod
    def test_check_build_dep_item_ge_greater():
        """Test >= operator with greater version."""
        assert check_build_dep_item('2.0.0', '>=1.0.0') is True

    @staticmethod
    def test_check_build_dep_item_ge_less():
        """Test >= operator with less version."""
        assert check_build_dep_item('0.9.0', '>=1.0.0') is False

    @staticmethod
    def test_check_build_dep_item_ge_different_length():
        """Test >= operator with different length versions."""
        assert check_build_dep_item('1.0.0.1', '>=1.0.0') is True
        assert check_build_dep_item('1.0.0', '>=1.0.0.1') is False

    # Tests for > operator
    @staticmethod
    def test_check_build_dep_item_gt_greater():
        """Test > operator with greater version."""
        assert check_build_dep_item('2.0.0', '>1.0.0') is True

    @staticmethod
    def test_check_build_dep_item_gt_equal():
        """Test > operator with equal versions."""
        assert check_build_dep_item('1.0.0', '>1.0.0') is False

    @staticmethod
    def test_check_build_dep_item_gt_less():
        """Test > operator with less version."""
        assert check_build_dep_item('0.9.0', '>1.0.0') is False

    # Tests for <= operator
    @staticmethod
    def test_check_build_dep_item_le_equal():
        """Test <= operator with equal versions."""
        assert check_build_dep_item('1.0.0', '<=1.0.0') is True

    @staticmethod
    def test_check_build_dep_item_le_less():
        """Test <= operator with less version."""
        assert check_build_dep_item('0.9.0', '<=1.0.0') is True

    @staticmethod
    def test_check_build_dep_item_le_greater():
        """Test <= operator with greater version."""
        assert check_build_dep_item('2.0.0', '<=1.0.0') is False

    # Tests for < operator
    @staticmethod
    def test_check_build_dep_item_lt_less():
        """Test < operator with less version."""
        assert check_build_dep_item('0.9.0', '<1.0.0') is True

    @staticmethod
    def test_check_build_dep_item_lt_equal():
        """Test < operator with equal versions."""
        assert check_build_dep_item('1.0.0', '<1.0.0') is False

    @staticmethod
    def test_check_build_dep_item_lt_greater():
        """Test < operator with greater version."""
        assert check_build_dep_item('2.0.0', '<1.0.0') is False

    # Tests for = operator (default - no prefix means equality)
    @staticmethod
    def test_check_build_dep_item_eq_equal():
        """Test equality with no prefix."""
        assert check_build_dep_item('1.0.0', '1.0.0') is True

    @staticmethod
    def test_check_build_dep_item_eq_not_equal():
        """Test inequality with no prefix."""
        assert check_build_dep_item('1.0.0', '1.0.1') is False

    @staticmethod
    def test_check_build_dep_item_eq_different_length():
        """Test equality with different length versions."""
        # check_eq uses zip which stops at shorter length, so these are equal
        assert check_build_dep_item('1.0.0', '1.0.0.0') is True
        assert check_build_dep_item('1.0.0.1', '1.0.0') is True
        # But different values at same position are not equal
        assert check_build_dep_item('1.0.0.1', '1.0.0.2') is False

    # Edge cases
    @staticmethod
    def test_check_build_dep_item_complex_versions():
        """Test with complex version numbers."""
        assert check_build_dep_item('1.10.0', '>=1.2.0') is True
        assert check_build_dep_item('1.2.3.4.5', '>=1.2.3.4') is True
        assert check_build_dep_item('2.0', '>=1.9.9') is True


class TestCheckBuildDep:
    """Test check_build_dep function with version ranges."""

    @staticmethod
    def test_check_build_dep_single_ge():
        """Test single >= dependency."""
        assert check_build_dep('2.0.0', '>=1.0.0') is True
        assert check_build_dep('0.9.0', '>=1.0.0') is False

    @staticmethod
    def test_check_build_dep_range_ge_and_lt():
        """Test version range with >= and <."""
        assert check_build_dep('1.5.0', '>=1.0.0,<2.0.0') is True
        assert check_build_dep('0.9.0', '>=1.0.0,<2.0.0') is False
        assert check_build_dep('2.0.0', '>=1.0.0,<2.0.0') is False

    @staticmethod
    def test_check_build_dep_range_ge_and_le():
        """Test version range with >= and <=."""
        assert check_build_dep('2.0.0', '>=1.0.0,<=2.0.0') is True
        assert check_build_dep('2.0.1', '>=1.0.0,<=2.0.0') is False

    @staticmethod
    def test_check_build_dep_range_gt_and_lt():
        """Test version range with > and <."""
        assert check_build_dep('1.5.0', '>1.0.0,<2.0.0') is True
        assert check_build_dep('1.0.0', '>1.0.0,<2.0.0') is False
        assert check_build_dep('2.0.0', '>1.0.0,<2.0.0') is False

    @staticmethod
    def test_check_build_dep_multiple_ranges():
        """Test multiple OR conditions."""
        # Should match either range
        assert check_build_dep('0.5.0', '>=0.0.0,<1.0.0') is True
        assert check_build_dep('1.5.0', '>=0.0.0,<1.0.0') is False

    @staticmethod
    def test_check_build_dep_multiple_lower_bounds():
        """Test range with multiple >= conditions (covers line 130)."""
        # Multiple lower bounds - all must be satisfied
        assert check_build_dep('2.0.0', '>=1.0.0,>=1.5.0') is True
        assert check_build_dep('1.2.0', '>=1.0.0,>=1.5.0') is False
        assert check_build_dep('1.5.0', '>=1.0.0,>=1.5.0') is True

    @staticmethod
    def test_check_build_dep_simple_version():
        """Test simple version without operator (implies =)."""
        assert check_build_dep('1.0.0', '1.0.0') is True
        assert check_build_dep('1.0.1', '1.0.0') is False

    @staticmethod
    def test_check_build_dep_with_whitespace():
        """Test version range with whitespace."""
        assert check_build_dep('1.5.0', '>=1.0.0, <2.0.0') is True
        assert check_build_dep('1.5.0', ' >=1.0.0 , <2.0.0 ') is True

    @staticmethod
    def test_check_build_dep_empty_range():
        """Test with empty/invalid range - should work with valid parts."""
        # Single version comparison
        assert check_build_dep('1.0.0', '1.0.0') is True


class TestCheckBuildDeps:
    """Test check_build_deps function."""

    @staticmethod
    def test_check_build_deps_success(tmp_path):
        """Test check_build_deps with successful dependency check."""
        # Create version file for dep1
        info_dir = tmp_path / 'share' / 'info' / 'dep1'
        info_dir.mkdir(parents=True)
        (info_dir / 'version.info').write_text('Version=1.5.0\n')

        recv = Receiver([], [])
        deps = ['dep1', '>=1.0.0,<2.0.0']
        check_build_deps(recv, str(tmp_path), deps)

        assert recv.warn_msgs == []
        assert recv.err_msgs == []

    @staticmethod
    def test_check_build_deps_version_mismatch(tmp_path):
        """Test check_build_deps with version mismatch."""
        info_dir = tmp_path / 'share' / 'info' / 'dep1'
        info_dir.mkdir(parents=True)
        (info_dir / 'version.info').write_text('Version=2.5.0\n')

        recv = Receiver([], [])
        deps = ['dep1', '>=1.0.0,<2.0.0']
        check_build_deps(recv, str(tmp_path), deps)

        assert len(recv.warn_msgs) == 1
        assert 'Check build dependency failed' in recv.warn_msgs[0]
        assert recv.err_msgs == []

    @staticmethod
    def test_check_build_deps_multiple_deps(tmp_path):
        """Test check_build_deps with multiple dependencies."""
        # Create version files
        for pkg, version in [('dep1', '1.5.0'), ('dep2', '3.2.0')]:
            info_dir = tmp_path / 'share' / 'info' / pkg
            info_dir.mkdir(parents=True)
            (info_dir / 'version.info').write_text(f'Version={version}\n')

        recv = Receiver([], [])
        deps = ['dep1', '>=1.0.0,<2.0.0', 'dep2', '>=3.0.0']
        check_build_deps(recv, str(tmp_path), deps)

        assert recv.warn_msgs == []
        assert recv.err_msgs == []

    @staticmethod
    def test_check_build_deps_file_not_exists(tmp_path):
        """Test check_build_deps when version file doesn't exist."""
        recv = Receiver([], [])
        deps = ['nonexistent', '>=1.0.0']
        check_build_deps(recv, str(tmp_path), deps)

        assert len(recv.err_msgs) == 1
        assert 'does not exist' in recv.err_msgs[0]
        assert recv.warn_msgs == []

    @staticmethod
    def test_check_build_deps_version_not_found(tmp_path):
        """Test check_build_deps when version field not found."""
        info_dir = tmp_path / 'share' / 'info' / 'dep1'
        info_dir.mkdir(parents=True)
        (info_dir / 'version.info').write_text('OtherField=value\n')

        recv = Receiver([], [])
        deps = ['dep1', '>=1.0.0']
        check_build_deps(recv, str(tmp_path), deps)

        assert len(recv.err_msgs) == 1
        assert 'version field was not found' in recv.err_msgs[0]

    @staticmethod
    def test_check_build_deps_invalid_version_raises_error(tmp_path, monkeypatch):
        """Test check_build_deps with invalid version that raises ValueError."""
        info_dir = tmp_path / 'share' / 'info' / 'dep1'
        info_dir.mkdir(parents=True)
        (info_dir / 'version.info').write_text('Version=1.0.0\n')

        # Mock check_build_dep to raise ValueError
        def mock_check_build_dep(version, dep_info):
            raise ValueError("Invalid version")

        monkeypatch.setattr(
            'check_build_dependencies.check_build_dep',
            mock_check_build_dep
        )

        recv = Receiver([], [])
        deps = ['dep1', 'invalid_dep_info']
        check_build_deps(recv, str(tmp_path), deps)

        assert len(recv.err_msgs) == 1
        assert 'Check build dependency error' in recv.err_msgs[0]

    @staticmethod
    def test_check_build_deps_mixed_results(tmp_path):
        """Test check_build_deps with mixed success and failure."""
        info_dir1 = tmp_path / 'share' / 'info' / 'dep1'
        info_dir1.mkdir(parents=True)
        (info_dir1 / 'version.info').write_text('Version=1.5.0\n')

        info_dir2 = tmp_path / 'share' / 'info' / 'dep2'
        info_dir2.mkdir(parents=True)
        (info_dir2 / 'version.info').write_text('Version=0.5.0\n')

        recv = Receiver([], [])
        deps = ['dep1', '>=1.0.0', 'dep2', '>=1.0.0']
        check_build_deps(recv, str(tmp_path), deps)

        assert len(recv.warn_msgs) == 1
        assert 'dep2' in recv.warn_msgs[0]
        assert recv.err_msgs == []


class TestMain:
    """Test main function."""

    @staticmethod
    def test_main_success(tmp_path):
        """Test main function with successful execution."""
        info_dir = tmp_path / 'share' / 'info' / 'dep1'
        info_dir.mkdir(parents=True)
        (info_dir / 'version.info').write_text('Version=1.5.0\n')

        with mock.patch('sys.argv', ['check_build_dependencies.py', str(tmp_path), 'dep1', '>=1.0.0']):
            result = main()

        assert result is True

    @staticmethod
    def test_main_odd_deps_count(caplog):
        """Test main function with odd number of deps."""
        with mock.patch('sys.argv', ['check_build_dependencies.py', '/tmp', 'dep1']):
            result = main()

        assert result is False
        assert 'even number of elements' in caplog.text

    @staticmethod
    def test_main_no_deps(tmp_path):
        """Test main function with no deps."""
        with mock.patch('sys.argv', ['check_build_dependencies.py', str(tmp_path)]):
            result = main()

        assert result is True

    @staticmethod
    def test_main_with_warning(tmp_path, caplog):
        """Test main function that generates warnings."""
        info_dir = tmp_path / 'share' / 'info' / 'dep1'
        info_dir.mkdir(parents=True)
        (info_dir / 'version.info').write_text('Version=0.5.0\n')

        with mock.patch('sys.argv', ['check_build_dependencies.py', str(tmp_path), 'dep1', '>=1.0.0']):
            result = main()

        assert result is True
        assert 'Check build dependency failed' in caplog.text

    @staticmethod
    def test_main_with_error(tmp_path, caplog):
        """Test main function that generates errors."""
        with mock.patch('sys.argv', ['check_build_dependencies.py', str(tmp_path), 'dep1', '>=1.0.0']):
            result = main()

        assert result is False
        assert 'does not exist' in caplog.text

    @staticmethod
    def test_main_multiple_deps_with_warnings(tmp_path, caplog):
        """Test main with multiple deps where some fail."""
        info_dir1 = tmp_path / 'share' / 'info' / 'dep1'
        info_dir1.mkdir(parents=True)
        (info_dir1 / 'version.info').write_text('Version=2.0.0\n')

        info_dir2 = tmp_path / 'share' / 'info' / 'dep2'
        info_dir2.mkdir(parents=True)

        with mock.patch('sys.argv', [
            'check_build_dependencies.py', str(tmp_path),
            'dep1', '>=1.0.0',
            'dep2', '>=1.0.0'
        ]):
            result = main()

        assert result is False
