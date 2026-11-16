#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for generate_version_info.py."""

import subprocess
import sys
from pathlib import Path


import generate_version_info


class TestGenVersionInfoContent:
    """Tests for gen_version_info_content function."""

    @staticmethod
    def test_basic_content_no_deps(monkeypatch):
        """Test generating content without dependencies and without tagInfo env var."""
        monkeypatch.delenv("tagInfo", raising=False)

        content = list(generate_version_info.gen_version_info_content("1.0.0", []))

        assert content[0] == "Version=1.0.0"
        assert content[1] == "version_dir=cann"
        assert content[-2].startswith("timestamp=")
        assert content[-1] == ""

    @staticmethod
    def test_content_with_single_dep(monkeypatch):
        """Test generating content with a single dependency pair."""
        monkeypatch.delenv("tagInfo", raising=False)

        content = list(
            generate_version_info.gen_version_info_content("2.0.0", ["pkg1", "1.1.0"])
        )

        assert content[0] == "Version=2.0.0"
        assert content[1] == "version_dir=cann"
        assert content[2] == 'required_package_pkg1_version="1.1.0"'
        assert content[-2].startswith("timestamp=")
        assert content[-1] == ""

    @staticmethod
    def test_content_with_multiple_deps(monkeypatch):
        """Test generating content with multiple dependency pairs."""
        monkeypatch.delenv("tagInfo", raising=False)

        deps = ["pkg1", "1.0.0", "pkg2", "2.0.0", "pkg3", "3.0.0"]
        content = list(generate_version_info.gen_version_info_content("3.0.0", deps))

        assert content[0] == "Version=3.0.0"
        assert content[1] == "version_dir=cann"
        assert content[2] == 'required_package_pkg1_version="1.0.0"'
        assert content[3] == 'required_package_pkg2_version="2.0.0"'
        assert content[4] == 'required_package_pkg3_version="3.0.0"'
        assert content[-2].startswith("timestamp=")
        assert content[-1] == ""

    @staticmethod
    def test_timestamp_from_taginfo_env(monkeypatch):
        """Test generating timestamp from tagInfo environment variable."""
        monkeypatch.setenv("tagInfo", "release_20250115_120000_abc123")

        content = list(generate_version_info.gen_version_info_content("1.0.0", []))

        # Extract timestamp line
        timestamp_line = [line for line in content if line.startswith("timestamp=")][0]
        assert timestamp_line == "timestamp=20250115_120000"

    @staticmethod
    def test_timestamp_from_taginfo_with_multiple_underscores(monkeypatch):
        """Test extracting timestamp from tagInfo with multiple underscore segments."""
        monkeypatch.setenv("tagInfo", "my_release_package_20250304_101030_xyz789")

        content = list(generate_version_info.gen_version_info_content("1.0.0", []))

        timestamp_line = [line for line in content if line.startswith("timestamp=")][0]
        # Should take 3rd and 2nd from last segments
        assert timestamp_line == "timestamp=20250304_101030"

    @staticmethod
    def test_timestamp_from_current_time_when_no_taginfo(monkeypatch):
        """Test generating timestamp from current time when tagInfo is not set."""
        monkeypatch.delenv("tagInfo", raising=False)

        content = list(generate_version_info.gen_version_info_content("1.0.0", []))

        timestamp_line = [line for line in content if line.startswith("timestamp=")][0]
        # Verify timestamp format: YYYYMMDD_HHMMSSmmm (18 chars)
        timestamp_value = timestamp_line.replace("timestamp=", "")
        assert len(timestamp_value) == 18  # YYYYMMDD_HHMMSSmmm
        assert timestamp_value[8] == "_"

    @staticmethod
    def test_empty_deps_list(monkeypatch):
        """Test with empty deps list - should not generate any required_package lines."""
        monkeypatch.delenv("tagInfo", raising=False)

        content = list(generate_version_info.gen_version_info_content("1.0.0", []))

        # Should have: Version, version_dir, timestamp, and empty line
        assert len(content) == 4
        assert content[0] == "Version=1.0.0"
        assert content[1] == "version_dir=cann"
        assert content[-2].startswith("timestamp=")
        assert content[-1] == ""


class TestMain:
    """Tests for main function."""

    @staticmethod
    def test_main_success_with_deps(tmp_path: Path, monkeypatch):
        """Test main function success with dependencies."""
        output_file = TestMain._setup_main_test(
            monkeypatch, tmp_path, "1.2.3", deps=["dep1", "1.0.0", "dep2", "2.0.0"]
        )

        result = generate_version_info.main()

        assert result is True
        assert output_file.exists()
        content = output_file.read_text(encoding="utf-8")
        assert "Version=1.2.3" in content
        assert "version_dir=cann" in content
        assert 'required_package_dep1_version="1.0.0"' in content
        assert 'required_package_dep2_version="2.0.0"' in content
        assert "timestamp=" in content

    @staticmethod
    def test_main_success_no_deps(tmp_path: Path, monkeypatch):
        """Test main function success without dependencies."""
        output_file = TestMain._setup_main_test(monkeypatch, tmp_path, "3.0.0")

        result = generate_version_info.main()

        assert result is True
        assert output_file.exists()
        content = output_file.read_text(encoding="utf-8")
        assert "Version=3.0.0" in content
        assert "version_dir=cann" in content
        assert "required_package_" not in content
        assert "timestamp=" in content

    @staticmethod
    def test_main_error_odd_number_of_deps(tmp_path: Path, monkeypatch):
        """Test main function returns False when deps has odd number of elements."""
        output_file = TestMain._setup_main_test(
            monkeypatch,
            tmp_path,
            "1.0.0",
            deps=["dep1", "1.0.0", "dep2"],  # 3 elements - odd number
        )

        result = generate_version_info.main()

        assert result is False
        # File should not be created when there's an error
        assert not output_file.exists()

    @staticmethod
    def test_main_with_single_dep_pair(tmp_path: Path, monkeypatch):
        """Test main with exactly one dependency pair."""
        output_file = TestMain._setup_main_test(
            monkeypatch, tmp_path, "2.0.0", deps=["single_dep", "1.5.0"]
        )

        result = generate_version_info.main()

        assert result is True
        content = output_file.read_text(encoding="utf-8")
        assert 'required_package_single_dep_version="1.5.0"' in content

    @staticmethod
    def test_main_output_file_ends_with_newline(tmp_path: Path, monkeypatch):
        """Test that output file ends with a newline."""
        output_file = TestMain._setup_main_test(monkeypatch, tmp_path, "1.0.0")

        generate_version_info.main()

        content = output_file.read_bytes()
        # Content should end with newline
        assert content.endswith(b"\n")

    @staticmethod
    def _setup_main_test(
        monkeypatch, tmp_path, version, deps=None, output_file_name="version.info"
    ):
        """Helper method to setup main function test."""
        monkeypatch.delenv("tagInfo", raising=False)
        output_file = tmp_path / output_file_name

        args = ["generate_version_info.py", version]
        if deps:
            args.extend(deps)
        args.extend(["--output", str(output_file)])

        monkeypatch.setattr("sys.argv", args)
        return output_file


class TestMainEntryPoint:
    """Tests for main entry point."""

    @staticmethod
    def test_main_entry_point_success(tmp_path: Path, monkeypatch):
        """Test __main__ entry point with successful execution."""
        TestMainEntryPoint._run_main_entry_point(monkeypatch, tmp_path, "1.0.0")

        # main() returns True and sys.exit(True) will exit with 0
        # but we can just verify main returns True
        result = generate_version_info.main()
        assert result is True

    @staticmethod
    def test_main_entry_point_failure(tmp_path: Path, monkeypatch):
        """Test __main__ entry point with failed execution (odd deps)."""
        TestMainEntryPoint._run_main_entry_point(
            monkeypatch,
            tmp_path,
            "1.0.0",
            deps=["dep1", "1.0.0", "dep2"],  # odd number of deps
        )

        result = generate_version_info.main()
        assert result is False

    @staticmethod
    def _run_main_entry_point(monkeypatch, tmp_path, version, deps=None):
        """Helper to run main entry point test."""
        monkeypatch.delenv("tagInfo", raising=False)
        output_file = tmp_path / "version.info"

        args = ["generate_version_info.py", version]
        if deps:
            args.extend(deps)
        args.extend(["--output", str(output_file)])

        monkeypatch.setattr("sys.argv", args)
        return output_file


class TestScriptAsMain:
    """Tests running the script as __main__ using subprocess."""

    @staticmethod
    def test_script_success_exit_code(tmp_path: Path):
        """Test script exits with code 0 on success."""
        output_file = tmp_path / "version.info"
        script_path = Path(generate_version_info.__file__)

        result = subprocess.run(
            [sys.executable, str(script_path), "1.0.0", "--output", str(output_file)],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert output_file.exists()

    @staticmethod
    def test_script_failure_exit_code(tmp_path: Path):
        """Test script exits with code 1 on failure (odd deps)."""
        output_file = tmp_path / "version.info"
        script_path = Path(generate_version_info.__file__)

        result = subprocess.run(
            [
                sys.executable,
                str(script_path),
                "1.0.0",
                "dep1",
                "1.0.0",
                "dep2",  # odd number of deps
                "--output",
                str(output_file),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1
        assert "error" in result.stderr.lower()
