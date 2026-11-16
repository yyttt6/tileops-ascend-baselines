# 项目目录

详细目录层级介绍如下：

```text
├── aicpu_common                                        # aicpu公共函数实现
├── cmake                                               # 项目工程编译目录
│   ├── third_party                                     # 项目依赖的第三方相关编译配置目录
│   ├── aarch64-hcc-toolchain.cmake                     # 项目工程编译工具链配置文件
│   ├── config.cmake                                    # 项目工程编译选项配置文件    
│   ├── func.cmake                                      # 项目工程编译protobuf文件
│   ├── makeself.cmake                                  # 项目工程自定义makeself打包文件
│   ├── package.cmake                                   # 项目工程编译打包安装相关配置文件
│   ├── Third_Party_Open_Source_Software_List.yaml      # 项目依赖的第三方软件库清单
│   └── variables.cmake                                 # 项目工程编译参数配置文件
├── docs                                                # 项目相关文档目录（zh为中文，en为英文）
├── include                                             # 项目公共头文件
│   └── nnopbase                                        # nnopbase模块头文件
│        ├── aclnn                                      # aclnn接口依赖头文件
│        └── opdev                                      # 算子开发依赖头文件
│            ├── aicpu                                  # aicpu算子开发相关头文件
│            └── op_common                              # 算子通用接口头文件
├── pkg_inc                                             # 项目包间接口头文件目录
│   └── op_common                                       # op_common模块头文件
│       ├── atvoss                                      # atvoss接口头文件，包括broadcast、elewise等
│       ├── log                                         # log相关接口头文件
│       ├── op_host                                     # host侧接口头文件
│       ├── aicpu_common                                # aicpu公共函数头文件
│       └── op_kernel                                   # kernel侧接口头文件
├── scripts                                             # 项目脚本文件存放目录
├── src
|   └── nnopbase                                        # nnopbase代码目录
│       ├── aicpu                                       # aicpu框架代码
│       ├── common                                      # nnopbase公共文件
│       ├── composite_op                                # 组合多算子框架代码
│       ├── individual_op                               # 单一算子框架代码
│       ├── stub                                        # 交叉编译场景打包脚本
│       ├── tls_guardian                                # 解决glibc问题补丁
│       └── CMakeLists.txt                              # nnopbase模块编译配置文件
│   └── op_common                                       # op_common源码实现
│       ├── atvoss                                      # atvoss接口源码实现
│       ├── log                                         # log相关接口源码实现
│       └── op_host                                     # host侧接口源码实现
├── tests                                               # 测试工程目录
│   ├── CMakeLists.txt
│   └── ut                                              # UT用例工程
│       ├── CMakeLists.txt                              # UT工程的cmakelist脚本
│       └── op_common                                   # op_common测试工程
├── build.sh                                            # 项目工程编译脚本
├── CMakeLists.txt                                      # 项目工程cmakelist入口
├── CONTRIBUTING.md                                     # 项目贡献指南文件
├── install_deps.sh                                     # 项目安装依赖包脚本
├── LICENSE                                             # 项目开源许可证信息
├── OAT.xml                                             # 配置脚本，代码仓工具使用，用于检查License是否规范
├── README.md                                           # 项目工程总介绍文档
├── SECURITY.md                                         # 项目安全声明文件
└── version.info                                        # 项目版本信息
```
