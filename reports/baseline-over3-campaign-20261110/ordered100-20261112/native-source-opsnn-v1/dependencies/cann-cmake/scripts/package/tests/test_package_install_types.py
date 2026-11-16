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

"""Tests for package install types and matching functions."""

import os
import sys
from argparse import Namespace

import pytest

from package import package as package_module
from package import filelist


class TestInstallTypeHelpers:
    """Test get_install_type function variations."""

    @staticmethod
    def test_get_install_type_with_value():
        """Test get_install_type when install_type is set."""
        target_config = {'install_type': 'docker'}
        result = package_module.get_install_type(target_config)
        assert result == 'docker'

    @staticmethod
    def test_get_install_type_default():
        """Test get_install_type default value."""
        target_config = {}
        result = package_module.get_install_type(target_config)
        assert result == 'NA'


class TestIsMatchLineExtended:
    """Extended tests for is_match_line function."""

    @staticmethod
    def test_is_match_line_all_match():
        """Test is_match_line when all fields match."""
        data = ['test_pkg', 'path', '100', 'M', 'desc', 'ascend910', 'product', 'release']
        result = package_module.is_match_line('test_pkg', 'ascend910', 'product', 'release', data)
        assert result is True

    @staticmethod
    def test_is_match_line_no_match():
        """Test is_match_line when fields don't match."""
        data = ['other_pkg', 'path', '100', 'M', 'desc', 'ascend310', 'other', 'debug']
        result = package_module.is_match_line('test_pkg', 'ascend910', 'product', 'release', data)
        assert result is False

    @staticmethod
    def test_is_match_line_partial_match():
        """Test is_match_line when only some fields match."""
        data = ['test_pkg', 'path', '100', 'M', 'desc', 'ascend310', 'product', 'release']
        result = package_module.is_match_line('test_pkg', 'ascend910', 'product', 'release', data)
        assert result is False  # chip_name doesn't match


class TestFileListHelpersExtended:
    """Extended tests for filelist helpers."""

    @staticmethod
    def test_is_relative_install_path_absolute():
        """Test is_relative_install_path with absolute path."""
        result = filelist.is_relative_install_path('/absolute/path')
        assert result is False

    @staticmethod
    def test_is_relative_install_path_relative():
        """Test is_relative_install_path with relative path."""
        result = filelist.is_relative_install_path('relative/path')
        assert result is True

    @staticmethod
    def test_soft_links_to_string_empty():
        """Test soft_links_to_string with empty list."""
        result = filelist.soft_links_to_string([])
        assert result == 'NA'

    @staticmethod
    def test_soft_links_to_string_with_links():
        """Test soft_links_to_string with links."""
        result = filelist.soft_links_to_string(['link1', 'link2'])
        assert result == 'link1;link2'


class TestArgsParseExtended:
    """Extended tests for args_parse function."""

    @staticmethod
    def test_args_parse_help_flag(monkeypatch):
        """Test args_parse with --help flag."""
        monkeypatch.setattr(
            sys, 'argv', ['script', '--help', '--source_dir', '', '--delivery_dir', '']
        )

        # Use a mock to capture exit behavior without raising SystemExit
        exit_code = [None]

        def mock_exit(code=0):
            exit_code[0] = code

        monkeypatch.setattr(sys, 'exit', mock_exit)
        package_module.args_parse()
        assert exit_code[0] == 0


class TestFileItemToStringExtended:
    """Extended tests for file_item_to_string."""

    @staticmethod
    def test_file_item_to_string():
        """Test file_item_to_string function."""
        item = filelist.FileItem(
            module='test', operation='copy',
            relative_path_in_pkg='pkg/path',
            relative_install_path='install/path',
            is_in_docker='TRUE', permission='755',
            owner_group='user:group', install_type='run',
            softlink=[], feature={'all'}, is_common_path='N',
            configurable='FALSE', hash_value='NA', block='block1',
            pkg_inner_softlink=[], chip={'ascend910'}, is_dir=False
        )
        result = filelist.file_item_to_string(item)
        assert 'test' in result
        assert 'copy' in result
