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
import urllib.request
import os
import sys
import logging

logging.basicConfig(stream=sys.stdout,
                    format='[%(asctime)s] [%(lineno)s] %(levelname)s: %(message)s',
                    level=logging.INFO)


def down_files_native(url_list):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    failed_urls = []

    for url in url_list:

        file_name = url.split('/')[-1]

        if not file_name:
            file_name = "downloaded_file"

        # 将下载的文件保存到脚本所在目录
        file_path = os.path.join(current_dir, file_name)

        try:
            urllib.request.urlretrieve(url, file_path)
            logging.info("Successfully downloaded %s", url)
        except OSError as ex:
            logging.error("Failed to download %s, error: %s", url, ex)
            failed_urls.append(url)
            if os.path.exists(file_path):
                os.remove(file_path)

    return failed_urls

if __name__ == "__main__":
    my_urls = [
        "https://cann-3rd.obs.cn-north-4.myhuaweicloud.com/json/json-3.11.3.tar.gz",
        ("https://gitcode.com/cann-src-third-party/makeself/releases/download/"
        "release-2.5.0-patch1.0/makeself-release-2.5.0-patch1.tar.gz"),
        "https://gitcode.com/cann-src-third-party/eigen/releases/download/5.0.0-h0.trunk/eigen-5.0.0.tar.gz",
        "https://gitcode.com/cann-src-third-party/protobuf/releases/download/v25.1/protobuf-25.1.tar.gz",
        ("https://cann-3rd.obs.cn-north-4.myhuaweicloud.com/abseil-cpp/"
        "abseil-cpp-20230802.1.tar.gz"),
        "https://cann-3rd.obs.cn-north-4.myhuaweicloud.com/cmake/cmake-master-049.tar.gz"
    ]

    failed_downloads = down_files_native(my_urls)

    if failed_downloads:
        logging.error("Third-party library download failed, failed urls: %s",
                      failed_downloads)
        sys.exit(1)