# 快速入门

快速体验项目前，请参考本项目README完成环境准备和源码下载，此处不再赘述。

注意，全量命令参数通过`bash build.sh --help`查看，参数含义详见[build参数说明](./zh/appendix/build.md#参数说明)。

## 源码构建

### 安装构建工具

> **说明**：对于CANNLab或Docker环境，默认已安装源码编译所需的构建工具，无需手动安装，可跳过本章操作。

请先确保编译环境的基础库依赖已安装，注意满足版本号要求。

- python >= 3.7.0
- gcc >= 7.3.0
- g++ >= 7.3.0
- cmake >= 3.16.0
- pigz（可选，安装后可提升打包速度，建议版本 >= 2.4）
- patch >= 2.7（用于给第三方源码打补丁，如protobuf）
- dos2unix
- git

上述依赖可通过项目根目录install\_deps.sh一键安装，命令如下，若遇到不支持系统，请参考该文件自行适配。

```bash
bash install_deps.sh
```

### 安装第三方依赖

> **说明**：对于CANNLab或Docker环境，默认联网，编译过程中会自动下载第三方依赖，无需手动安装，可跳过本章操作。

本项目编译过程依赖的第三方开源软件列表如下：

| 开源软件   | 版本       | 下载地址                                                     |
| ---------- | ---------- | ------------------------------------------------------------ |
| json       | 3.11.3     | [json-3.11.3.tar.gz](https://cann-3rd.obs.cn-north-4.myhuaweicloud.com/json/json-3.11.3.tar.gz) |
| makeself   | 2.5.0      | [makeself-release-2.5.0-patch1.tar.gz](https://gitcode.com/cann-src-third-party/makeself/releases/download/release-2.5.0-patch1.0/makeself-release-2.5.0-patch1.tar.gz) |
| eigen      | 5.0.0      | [eigen-5.0.0.tar.gz](https://gitcode.com/cann-src-third-party/eigen/releases/download/5.0.0-h0.trunk/eigen-5.0.0.tar.gz) |
| protobuf   | 25.1       | [protobuf-25.1.tar.gz](https://cann-3rd.obs.cn-north-4.myhuaweicloud.com/protobuf/protobuf-25.1.tar.gz) |
| abseil-cpp | 20230802.1 | [abseil-cpp-20230802.1.tar.gz](https://cann-3rd.obs.cn-north-4.myhuaweicloud.com/abseil-cpp/abseil-cpp-20230802.1.tar.gz) |
| cann-cmake | master-049 | [cmake-master-049.tar.gz](https://cann-3rd.obs.cn-north-4.myhuaweicloud.com/cmake/cmake-master-049.tar.gz) |

若您的编译环境可以访问网络，请参考[联网编译](#联网编译)，编译脚本会自动联网下载第三方软件。否则，请参考[未联网编译](#未联网编译)手动下载第三方软件。

### 联网编译

若在有互联网的环境下编译，编译过程中会自动安装第三方依赖，无需手动安装。具体过程如下：

1. **编译opbase包**

    进入项目根目录，执行如下编译命令：

    ```bash
    bash build.sh --pkg [-j${n}]
    # 如：bash build.sh --pkg -j16
    ```

    - -j（可选）：指定编译线程数，默认值为8，加快编译速度；若超过CPU核心数会自动调整为CPU核心数。

    若提示如下信息，则说明编译成功。

    ```bash
    Self-extractable archive "cann-opbase_${cann_version}_linux-${arch}.run" successfully created.
    ```

    编译成功后，run包存储于build_out目录下。

2. **安装opbase包**

    ```bash
    ./cann-opbase_${cann_version}_linux-${arch}.run --full --install-path=${install_path}
    ```

    \$\{install\_path\}表示指定安装路径，若不指定，默认安装路径为：`/usr/local/Ascend`；若指定，一般安装在\$\{install\_path\}目录下。

3. **（可选）卸载opbase包**

    ```bash
    # 卸载命令
    ./${install_path}/cann/share/info/opbase/script/uninstall.sh
    ```

### 未联网编译

若在没有连接互联网的环境下编译，需要提前准备好依赖的第三方软件，再进行源码编译。具体过程如下：

1. **下载第三方依赖**

    在联网环境中提前下载第三方软件，目前有如下方式，请按需选择：

    - 方式1：根据[第三方软件依赖](#安装第三方依赖)提供的表格手动下载，若从其他地址下载，注意版本号一致。

    - 方式2：通过[third_lib_download.py](../scripts/tools/third_lib_download.py)脚本一键下载，该脚本在本项目`scripts/tools/`目录，下载该脚本并执行如下命令：

        ```bash
        python ${scripts_dir}/third_lib_download.py
        ```

        \$\{scripts\_dir\}表示脚本存放路径，下载的第三方软件包默认存放在当前脚本所在目录。

2. **编译算子包**

   将下载好的第三方软件上传至离线环境，可存放在`third_party`目录或自定义目录下。**推荐前者，其编译命令与联网编译场景下的命令一致。**

    - **third\_party目录**（推荐）

      请在本项目根目录创建`third_party`目录（若有则无需创建），将第三方软件拷贝到该指定目录。此时编译命令与联网编译命令一致：

      ```bash
      bash build.sh --pkg [-j${n}]
      # 如：bash build.sh --pkg -j16
      ```

    - **自定义目录**

      在离线环境的任意位置新建`${cann_3rd_lib_path}`目录，将第三方软件拷贝到该目录，请确保该目录有权限访问。

        ```bash
      mkdir -p ${cann_3rd_lib_path}
        ```

      此时在联网编译命令基础上额外增加`--cann_3rd_lib_path=${cann_3rd_lib_path}`用于指定第三方软件路径。假设路径为`/path/cann_3rd_lib_path`，编译命令如下：

      ```bash
      bash build.sh --pkg --cann_3rd_lib_path=${cann_3rd_lib_path} [-j${n}]
      # bash build.sh --pkg --cann_3rd_lib_path=/path/cann_3rd_lib_path -j16
      # bash build.sh --pkg --cann_3rd_lib_path=/path/cann_3rd_lib_path
      ```

3. **安装/卸载算子包**

    未联网和联网场景下编译得到算子包结果一样，默认存放于项目根目录build_out目录下，并且安装和卸载的操作命令也一样，具体参见[联网编译](#联网编译)。

## 本地验证

源码包部署后，可通过项目根目录build.sh执行UT/ST用例，验证项目功能是否正常。

> **说明**：执行UT用例依赖googletest单元测试框架，详细介绍参见[googletest官网](https://google.github.io/googletest/advanced.html#running-a-subset-of-the-tests)。

### 执行UT

#### 编译并执行UT

```bash
# 编译并执行所有的UT测试用例
bash build.sh -u
# 编译所有的UT测试用例但不执行
bash build.sh -u --noexec
# 执行UT并查看覆盖率
bash build.sh -u --cov
```

执行完成后出现如下内容，表示执行成功。

```bash
Global Global test environment tear-down
[==========] ${n} tests from ${m} test suites ran. (${x} ms total)
[  PASSED  ] ${n} tests.
Execute ops_base_ut successful.
```

#### 单独运行已编译的UT测试

若已完成UT编译（通过 `bash build.sh -u` 或 `bash build.sh -u --noexec`），可直接运行生成的测试二进制文件，无需重新编译。

```bash
# 设置环境变量
source ${ASCEND_HOME_PATH}/bin/setenv.bash
export LD_LIBRARY_PATH=build/:$LD_LIBRARY_PATH

# 运行nnopbase UT测试
cd build/tests/nnopbase/ut/
./nnopbase_utest

# 运行op_common UT测试
cd build/tests/op_common/
./op_common_utest
```

其中 `${ASCEND_HOME_PATH}` 为CANN安装路径，默认为 `/usr/local/Ascend/cann`。

> **说明**：googletest支持通过命令行参数过滤指定测试用例，例如 `./nnopbase_utest --gtest_filter=TestSuiteName.TestCaseName`，更多用法参见[googletest文档](https://google.github.io/googletest/advanced.html#running-a-subset-of-the-tests)。
