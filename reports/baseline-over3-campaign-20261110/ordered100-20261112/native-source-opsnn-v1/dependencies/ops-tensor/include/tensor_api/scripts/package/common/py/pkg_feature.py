#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.	
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of	
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""打包特性。"""

from enum import IntEnum
from itertools import groupby
from pathlib import Path
from typing import Dict, Iterable, Optional, Set, Tuple, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass

from common.py.utils.pkg_utils import PackageConfigError


def pkg_feature_to_set(feature: str) -> Set[str]:
    """打包feature转换为集合。"""
    if not feature:
        return set()

    return set(feature.strip().split(','))


def config_feature_to_set(feature_str: str, feature_type: str = 'feature') -> Set[str]:
    """配置feature转换为集合。"""
    if feature_str is None:
        return set()

    if isinstance(feature_str, set):
        return feature_str

    if feature_str == '':
        raise PackageConfigError(f"Not allow to config {feature_type} empty.")

    features = set(feature_str.split(';'))
    if 'all' in features:
        raise PackageConfigError(f"Not allow to config {feature_type} all.")
    return features


def config_feature_to_string(features: Set[str]) -> str:
    """配置feature集合转换为字符串。"""
    if not features:
        return 'all'
    return ';'.join(sorted(features))


@dataclass
class PkgFeature(ABC):
    """打包特性。"""
    excludes: Set[str]
    exclude_all: bool
    includes: Set[str]

    @abstractmethod
    def _matched(self, config_features: Set[str]) -> bool:
        """是否匹配。"""

    def matched(self, config_features: Set[str]) -> bool:
        """
        配置文件配置的config_feature和打包选项的pkg_feature是否匹配。

        匹配则返回Ture，表示需要打包该文件。
        不匹配返回False，表示不需要打包该文件。
        """
        if bool(config_features & self.excludes):
            return False

        if self._matched(config_features):
            return True

        # 配置的feature为空，说明是公共文件
        # 并且没有通过exclude_all排除公共文件
        return bool(not config_features and not self.exclude_all)


@dataclass
class NormalPkgFeature(PkgFeature):
    """普通打包特性。"""

    def _matched(self, config_features: Set[str]) -> bool:
        return bool(
            config_features & self.includes
        )


@dataclass
class AllPkgFeature(PkgFeature):
    """
    all打包特性。

    如果打包选项为特性为all，那么所有特性文件都打包。
    """

    def _matched(self, config_features: Set[str]) -> bool:
        return True


def make_pkg_feature(features: Set[str], exclude_all: bool = False) -> PkgFeature:
    """创建PkgFeature。"""

    class FeatureType(IntEnum):
        """特性类型。"""
        INCLUDE = 1
        EXCLUDE = 2

    def feature_keyfunc(feature: str) -> int:
        """分组函数。"""
        if feature.startswith('-'):
            return FeatureType.EXCLUDE
        return FeatureType.INCLUDE

    def group_features_to_set(key: FeatureType,
                              group_features: Iterable[str]) -> Set[str]:
        """分组特性转换为集合。"""
        if key == FeatureType.INCLUDE:
            return set(group_features)

        return {feature[1:] for feature in group_features}

    def get_features_dict(features: Set[str]) -> Dict[int, Set[str]]:
        """分组include的feature和exclude的feature。"""
        return {
            key: group_features_to_set(key, group_features)
            for key, group_features in
            groupby(
                sorted(features, key=feature_keyfunc),
                key=feature_keyfunc
            )
        }

    def classify_features(features: Set[str]) -> Tuple[Set[str], Set[str]]:
        """分类include和exclude两种feature。"""
        feature_dict = get_features_dict(features)
        return (
            feature_dict.get(FeatureType.INCLUDE, set()),
            feature_dict.get(FeatureType.EXCLUDE, set())
        )

    def with_features(includes: Set[str], excludes: Set[str]) -> PkgFeature:
        # 如果输入feature中，没有包含性的feature，或存在all，那么配置AllPkgFeature流程
        # 也就是说，允许打包时输入feature为all，feature.list文件中配置all
        # 输入feature为all，统一在这里处理，转换为AllPkgFeature
        if not includes or 'all' in includes:
            return AllPkgFeature(excludes, exclude_all, set())

        return NormalPkgFeature(excludes, exclude_all, includes)

    return with_features(
        *classify_features(features)
    )


def feature_compatible(left: PkgFeature, right: PkgFeature) -> bool:
    """feature是否相容。场景说明见测试。"""
    # 首先处理excludes
    if bool(left.includes & right.excludes) or bool(left.excludes & right.includes):
        # 两侧的includes与excludes互斥，为假。
        return False

    # exclude_all不接受对方为空集
    if (left.exclude_all and not right.includes) or (right.exclude_all and not left.includes):
        return False
    # exclude处理完成后，就不再关心排除场景

    # 定义：空集与任何集合相容（相交）
    if not left.includes or not right.includes:
        return True
    # 否则，两侧都是有集。
    # 取决于两侧集合的相交情况。
    return bool(left.includes & right.includes)


def load_feature_list(filepath: Optional[Union[Path, str]]) -> Set[str]:
    """加载feature.list文件。"""
    if not filepath:
        return set()

    with Path(filepath).open(encoding='utf-8') as file:
        return {
            line.strip() for line in file
            if line.strip() and not line.startswith('#')
        }


def combine_feature_and_feature_list(feature: str, feature_list_path: Optional[str]) -> Set[str]:
    """合并feature和feature_list参数。"""
    return pkg_feature_to_set(feature) | load_feature_list(feature_list_path)
