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

"""Tests for filelist type validation and package module getters."""

import os
import sys
import tempfile
import subprocess
import shutil
from argparse import Namespace
from pathlib import Path
from unittest.mock import Mock

import pytest

from package import filelist
from package import package as package_module


class TestFileListTypeValidation:
    """Test create_file_item type validation."""

    @staticmethod
    def test_create_file_item_invalid_feature_type():
        """Test create_file_item with invalid feature type."""
        with pytest.raises(TypeError, match='feature parameter should be a set'):
            filelist.create_file_item(
                'module', 'copy', 'path', 'install', 'TRUE', '755',
                'user:group', 'run', [], 'all', 'N', 'FALSE', 'NA',
                'block', [], {'chip'}, False  # feature='all' is not a set
            )

    @staticmethod
    def test_create_file_item_invalid_chip_type():
        """Test create_file_item with invalid chip type."""
        with pytest.raises(TypeError, match='chip parameter should be a set'):
            filelist.create_file_item(
                'module', 'copy', 'path', 'install', 'TRUE', '755',
                'user:group', 'run', [], {'all'}, 'N', 'FALSE', 'NA',
                'block', [], 'ascend910', False  # chip='ascend910' is not a set
            )

    @staticmethod
    def test_create_file_item_invalid_softlink_type():
        """Test create_file_item with invalid softlink type."""
        with pytest.raises(TypeError, match='softlink parameter should be a list'):
            filelist.create_file_item(
                'module', 'copy', 'path', 'install', 'TRUE', '755',
                'user:group', 'run', 'invalid', {'all'}, 'N', 'FALSE', 'NA',
                'block', [], {'ascend910'}, False  # softlink='invalid' is not a list
            )

    @staticmethod
    def test_create_file_item_invalid_pkg_inner_softlink_type():
        """Test create_file_item with invalid pkg_inner_softlink type."""
        with pytest.raises(TypeError, match='pkg_inner_softlink parameter should be a list'):
            filelist.create_file_item(
                'module', 'copy', 'path', 'install', 'TRUE', '755',
                'user:group', 'run', [], {'all'}, 'N', 'FALSE', 'NA',
                'block', 'invalid', {'ascend910'}, False  # pkg_inner_softlink='invalid' is not a list
            )


class TestPackageModuleExtended:
    """Extended tests for package module."""

    @staticmethod
    def test_get_comments_with_empty_chip():
        """Test get_comments with empty chip_name."""
        mock_pkg_name = Namespace(
            chip_name='',
            product_name='product',
            func_name='toolkit'
        )
        result = package_module.get_comments(mock_pkg_name)
        assert 'PRODUCT' in result and 'TOOLKIT' in result

    @staticmethod
    def test_get_comments_without_chip():
        """Test get_comments without chip_name."""
        mock_pkg_name = Namespace(
            chip_name=None,
            product_name='product',
            func_name='toolkit'
        )
        result = package_module.get_comments(mock_pkg_name)
        assert 'PRODUCT' in result and 'TOOLKIT' in result

    @staticmethod
    def test_get_operation_with_entity():
        """Test get_operation when entity is 'true'."""
        target_config = {'entity': 'true'}
        result = package_module.get_operation('copy', target_config)
        assert result == 'copy_entity'

    @staticmethod
    def test_get_operation_with_entity_move():
        """Test get_operation with move and entity."""
        target_config = {'entity': 'true'}
        result = package_module.get_operation('move', target_config)
        assert result == 'copy_entity'

    @staticmethod
    def test_get_operation_copy():
        """Test get_operation with copy type."""
        target_config = {}
        result = package_module.get_operation('copy', target_config)
        assert result == 'copy'

    @staticmethod
    def test_get_operation_move():
        """Test get_operation with move type."""
        target_config = {}
        result = package_module.get_operation('move', target_config)
        assert result == 'move'

    @staticmethod
    def test_get_operation_del():
        """Test get_operation with del type."""
        target_config = {}
        result = package_module.get_operation('del', target_config)
        assert result == 'del'

    @staticmethod
    def test_get_operation_symlink():
        """Test get_operation with symlink type."""
        target_config = {}
        result = package_module.get_operation('symlink', target_config)
        assert result == 'symlink'

    @staticmethod
    def test_get_operation_mkdir():
        """Test get_operation with mkdir type."""
        target_config = {}
        result = package_module.get_operation('mkdir', target_config)
        assert result == 'mkdir'

    @staticmethod
    def test_get_permission_with_install_mod():
        """Test get_permission when install_mod is set."""
        target_config = {'install_mod': '755'}
        result = package_module.get_permission(target_config)
        assert result == '755'

    @staticmethod
    def test_get_permission_default():
        """Test get_permission default value."""
        target_config = {}
        result = package_module.get_permission(target_config)
        assert result == 'NA'

    @staticmethod
    def test_get_owner_group_with_install_own():
        """Test get_owner_group when install_own is set."""
        target_config = {'install_own': 'user:group'}
        result = package_module.get_owner_group(target_config)
        assert 'user:group' in result

    @staticmethod
    def test_get_owner_group_default():
        """Test get_owner_group default value."""
        target_config = {}
        result = package_module.get_owner_group(target_config)
        assert result == 'NA'

    @staticmethod
    def test_get_softlink_empty():
        """Test get_softlink when install_softlink is empty."""
        target_config = {'install_softlink': ''}
        result = package_module.get_softlink(target_config)
        assert result == []

    @staticmethod
    def test_get_softlink_with_value():
        """Test get_softlink when install_softlink has value."""
        target_config = {'install_softlink': 'link1;link2'}
        result = package_module.get_softlink(target_config)
        assert result == ['link1', 'link2']

    @staticmethod
    def test_get_feature_with_value():
        """Test get_feature with value."""
        target_config = {'feature': {'test'}}
        result = package_module.get_feature(target_config)
        assert result == {'test'}

    @staticmethod
    def test_get_chip_with_value():
        """Test get_chip with value."""
        target_config = {'chip': {'ascend910'}}
        result = package_module.get_chip(target_config)
        assert result == {'ascend910'}

    @staticmethod
    def test_get_configurable_none():
        """Test get_configurable when configurable is None."""
        target_config = {}
        result = package_module.get_configurable(target_config)
        assert result == 'FALSE'

    @staticmethod
    def test_get_configurable_with_value():
        """Test get_configurable with value."""
        target_config = {'configurable': 'TRUE'}
        result = package_module.get_configurable(target_config)
        assert result == 'TRUE'

    @staticmethod
    def test_get_hash_value_with_hash():
        """Test get_hash_value when hash is set."""
        target_config = {'hash': 'abc123'}
        result = package_module.get_hash_value(target_config)
        assert result == 'abc123'

    @staticmethod
    def test_get_hash_value_none():
        """Test get_hash_value when hash is None."""
        target_config = {}
        result = package_module.get_hash_value(target_config)
        assert result == 'NA'

    @staticmethod
    def test_get_block_with_name():
        """Test get_block when name is set."""
        target_config = {'name': 'block1'}
        result = package_module.get_block(target_config)
        assert result == 'block1'

    @staticmethod
    def test_get_block_default():
        """Test get_block default value."""
        target_config = {}
        result = package_module.get_block(target_config)
        assert result == 'NA'

    @staticmethod
    def test_get_pkg_inner_softlink_empty():
        """Test get_pkg_inner_softlink when empty."""
        target_config = {}
        result = package_module.get_pkg_inner_softlink(target_config)
        assert result == []

    @staticmethod
    def test_get_pkg_inner_softlink_with_value():
        """Test get_pkg_inner_softlink with value."""
        target_config = {'pkg_inner_softlink': 'link1;link2'}
        result = package_module.get_pkg_inner_softlink(target_config)
        assert result == ['link1', 'link2']


class TestGetTargetNameExtended:
    """Extended tests for get_target_name function."""

    @staticmethod
    def test_get_target_name_with_rename():
        """Test get_target_name when rename is set."""
        target_conf = {
            'value': 'original.txt',
            'rename': 'renamed.txt'
        }
        result = package_module.get_target_name(target_conf)
        assert result == 'renamed.txt'

    @staticmethod
    def test_get_target_name_basename():
        """Test get_target_name gets basename of value."""
        target_conf = {
            'value': 'path/to/file.txt'
        }
        result = package_module.get_target_name(target_conf)
        assert result == 'file.txt'


class TestValidatePathConsistency:
    """Test validate_path_consistency function."""

    @staticmethod
    def test_validate_path_consistency_equal():
        """Test when dst_path equals install_path."""
        fileitems = [
            Mock(
                operation='copy',
                relative_path_in_pkg='x86_64-linux/include/version.h',
                relative_install_path='x86_64-linux/include/version.h',
            )
        ]
        # Should not raise
        package_module.validate_path_consistency(fileitems)

    @staticmethod
    def test_validate_path_consistency_different():
        """Test when dst_path differs from install_path."""
        fileitems = [
            Mock(
                operation='copy',
                relative_path_in_pkg='include/version.h',
                relative_install_path='x86_64-linux/include/version.h',
            )
        ]
        with pytest.raises(package_module.GenerateFilelistError):
            package_module.validate_path_consistency(fileitems)

    @staticmethod
    def test_validate_path_consistency_do_nothing():
        """Test do nothing."""
        fileitems = [
            Mock(
                operation='mkdir',
            )
        ]
        package_module.validate_path_consistency(fileitems)


class TestModuleFunction:
    """Test get_module function."""

    @staticmethod
    def test_get_module_with_value():
        """Test get_module with module value."""
        target_config = {'module': 'test_module'}
        result = package_module.get_module(target_config)
        assert result == 'test_module'

    @staticmethod
    def test_get_module_default():
        """Test get_module default value."""
        target_config = {}
        result = package_module.get_module(target_config)
        assert result == 'NA'

    @staticmethod
    def test_get_module_empty():
        """Test get_module with empty module."""
        target_config = {'module': ''}
        result = package_module.get_module(target_config)
        assert result == 'NA'
