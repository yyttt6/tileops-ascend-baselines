# 安全声明

## 运行用户建议

基于安全性角度考虑，不建议使用root等管理员类型账户执行任何命令，遵循权限最小化原则。

## 文件权限控制

- 建议用户在主机（包括宿主机）及容器中设置运行系统umask值为0027及以上，保障新增文件夹默认最高权限为750，新增文件默认最高权限为640。
- 建议用户对个人隐私数据、商业资产、源文件和Runtime开发过程中保存的各类文件等敏感内容做好权限控制等安全措施。例如涉及本项目安装目录权限管控、输入公共数据文件权限管控，设定的权限建议参考[A-文件（夹）各场景权限管控推荐最大值](#a-文件夹各场景权限管控推荐最大值)。
- 用户安装和使用过程需要做好权限控制，建议参考[A-文件（夹）各场景权限管控推荐最大值](#a-文件夹各场景权限管控推荐最大值)文件权限参考进行设置。

## 公网地址声明

|      类型      |                                           开源代码地址                                           |                            文件名                             |             公网IP地址/公网URL地址/域名/邮箱地址/压缩文件地址             |                   用途说明                    |
| :------------: |:------------------------------------------------------------------------------------------:|:----------------------------------------------------------| :---------------------------------------------------------- |:-----------------------------------------|
|  依赖  | 不涉及  | third_party/abseil-cpp.cmake | https://gitcode.com/cann-src-third-party/abseil-cpp/releases/download/20230802.1/abseil-cpp-20230802.1.tar.gz | 从gitcode下载abseil-cpp源码，作用编译依赖 |
|  依赖  | 不涉及  | third_party/acl_compat.cmake | https://mirrors.huaweicloud.com/artifactory/cann-run/8.5.0/inner/${TARGET_ARCH}/acl-compat_8.5.0_linux-${TARGET_ARCH}.tar.gz | 从huaweicloud下载二进制依赖 |
|  依赖  | 不涉及  | third_party/benchmark.cmake | https://gitcode.com/cann-src-third-party/benchmark/releases/download/v1.8.3/benchmark-1.8.3.tar.gz | 从gitcode下载benchmark源码，作用编译依赖 |
|  依赖  | 不涉及  | third_party/boost.cmake | https://gitcode.com/cann-src-third-party/boost/releases/download/v1.87.0/boost_1_87_0.tar.gz | 从gitcode下载boost源码，作用编译依赖 |
|  依赖  | 不涉及  | third_party/cesc.cmake | https://gitcode.com/cann-src-third-party/libboundscheck/releases/download/v1.1.16/libboundscheck-v1.1.16.tar.gz | 从gitcode下载libboundscheck源码，作用编译依赖 |
|  依赖  | 不涉及  | third_party/eigen.cmake | https://gitcode.com/cann-src-third-party/eigen/releases/download/5.0.0-h0.trunk/eigen-5.0.0.tar.gz | 从gitcode下载eigen源码，作用编译依赖 |
|  依赖  | 不涉及  | third_party/gtest.cmake | https://gitcode.com/cann-src-third-party/googletest/releases/download/v1.14.0/googletest-1.14.0.tar.gz | 从gitcode下载googletest源码，作用编译依赖 |
|  依赖  | 不涉及  | third_party/gtest_shared.cmake | https://gitcode.com/cann-src-third-party/googletest/releases/download/v1.14.0/googletest-1.14.0.tar.gz | 从gitcode下载googletest源码，作用编译依赖 |
|  依赖  | 不涉及  | third_party/json.cmake | https://gitcode.com/cann-src-third-party/json/releases/download/v3.12.0/json-3.12.0.tar.gz | 从gitcode下载json源码，作用编译依赖 |
|  依赖  | 不涉及  | third_party/libboundscheck.cmake | https://gitcode.com/cann-src-third-party/libboundscheck/releases/download/v1.1.16/libboundscheck-v1.1.16.tar.gz | 从gitcode下载libboundscheck源码，作用编译依赖 |
|  依赖  | 不涉及  | third_party/makeself-fetch.cmake | https://gitcode.com/cann-src-third-party/makeself/releases/download/release-2.5.0-patch1.0/makeself-release-2.5.0-patch1.tar.gz | 从gitcode下载makeself源码，作用编译依赖 |
|  依赖  | 不涉及  | third_party/openssl.cmake | https://gitcode.com/cann-src-third-party/openssl/releases/download/openssl-3.0.9/openssl-openssl-3.0.9.tar.gz | 从gitcode下载openssl源码，作用编译依赖 |
|  依赖  | 不涉及  | third_party/pybind11.cmake | https://gitcode.com/cann-src-third-party/pybind11/releases/download/v2.13.6/pybind11-2.13.6.tar.gz | 从gitcode下载pybind11源码，作用编译依赖 |
|  依赖  | 不涉及  | third_party/rdma-core.cmake | https://gitcode.com/cann-src-third-party/rdma-core/releases/download/v42.7-h1/ | 从gitcode下载rdma-core源码和patch，作用编译依赖 |
|  依赖  | 不涉及  | third_party/seccomp.cmake | https://gitcode.com/cann-src-third-party/libseccomp/releases/download/v2.5.4/libseccomp-2.5.4.tar.gz | 从gitcode下载seccomp源码，作用编译依赖 |
|  依赖  | 不涉及  | third_party/secure_c.cmake | https://gitee.com/openeuler/libboundscheck/repository/archive/v1.1.16.tar.gz | 从openeuler下载libboundscheck源码，作用编译依赖 |
---

## 漏洞机制说明

[漏洞管理](https://gitcode.com/cann/community/blob/master/security/security.md)

## 附录

### A-文件（夹）各场景权限管控推荐最大值

| 类型           | Linux权限参考最大值 |
| -------------- | ---------------  |
| 用户主目录                        |   750（rwxr-x---）            |
| 程序文件(含脚本文件、库文件等)       |   550（r-xr-x---）             |
| 程序文件目录                      |   550（r-xr-x---）            |
| 配置文件                          |  640（rw-r-----）             |
| 配置文件目录                      |   750（rwxr-x---）            |
| 日志文件(记录完毕或者已经归档)        |  440（r--r-----）             |
| 日志文件(正在记录)                |    640（rw-r-----）           |
| 日志文件目录                      |   750（rwxr-x---）            |
| Debug文件                         |  640（rw-r-----）         |
| Debug文件目录                     |   750（rwxr-x---）  |
| 临时文件目录                      |   750（rwxr-x---）   |
| 维护升级文件目录                  |   770（rwxrwx---）    |
| 业务数据文件                      |   640（rw-r-----）    |
| 业务数据文件目录                  |   750（rwxr-x---）      |
| 密钥组件、私钥、证书、密文文件目录    |  700（rwx—----）      |
| 密钥组件、私钥、证书、加密密文        | 600（rw-------）      |
| 加解密接口、加解密脚本            |   500（r-x------）        |