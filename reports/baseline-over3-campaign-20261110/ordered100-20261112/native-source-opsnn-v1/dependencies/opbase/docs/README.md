# 文档中心

## 目录结构

Docs目录结构说明如下：

```text
├── zh
  ├── api                                # API类文档
  ├── appendix                           # 附录类文档，如基本概念、目录结构等
  ├── figures                            # 图片目录
  ├── ...
  ├── install                            # 安装部署类文档
├── CONTRIBUTING_DOCS.md                 # 文档贡献说明
├── QUICKSTART.md                        # 快速入门
└── README.md
```

## 进阶教程

### 指南类文档

算子调用/开发过程中会使用本项目提供的基础框架能力和公共依赖项，与opbase框架相关的指南如下：

| 文档                                                         | 说明                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [算子调用指南](https://gitcode.com/cann/ops-math/blob/master/docs/README.md) | 提供算子库的调用方法和调用流程介绍，包括aclnn API、图模式调用等。<br>各算子库math、nn、cv、transformer调用指南基本一致，以math仓为例。 |
| [算子开发指南](https://gitcode.com/cann/ops-math/blob/master/docs/README.md) | 提供AI Core/AI CPU算子的开发流程和交付件介绍。<br>各算子库math、nn、cv、transformer开发指南基本一致，以math仓为例。 |
| aclnn API开发指南                                            | **正在建设中**，欢迎您关注和提问！                           |

### API类文档

opbase库里的API是所有算子库依赖的框架API，其提供各类算子基础调度框架和公共能力（例如日志获取、Tiling开发相关接口等）。

| 分类 | 说明 | 头文件 |
| ------------- |-------|-------|
| [aclnn公共接口](zh/api/nnopbase/aclnn/0_aclnn_meta_api.md) |调用aclnn API时，依赖框架基础API、结构体等Meta API，例如aclTensor/aclScalar等对象的创建/释放、各类属性信息的获取和设置等。|`${INSTALL_DIR}/include/aclnn/acl_meta.h`|
| [nnopbase接口](zh/api/nnopbase/opdev/0_opdev_api_list.md) |开发aclnn API时，依赖框架公共调度类和管理类接口，例如aclnn缓存、workspace复用等场景。|`${INSTALL_DIR}/include/aclnn/opdev/*.h`|
| [op_common接口](zh/api/op_common/0_op_common_list.md) |调用或开发算子时，提供op_common框架公共日志等能力，如Metadef、Log日志等。|`${INSTALL_DIR}/include/op_common/*.h`|

调用opbase API时，需include依赖的头文件，引用路径如上表。其中`${INSTALL_DIR}`请替换为CANN软件安装后存储路径，以root用户为例，安装后默认路径为`/usr/local/Ascend/cann`。

### 更多文档

| 文档                                                         | 说明                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [生态算子精度标准](zh/ops_precision_standard/experimental_standard.md) | 面向基于简易工程开发的算子，提供符合**生态开源贡献准则**的算子精度规范。 |

## 附录

### 基本概念

- **Level0层接口**

  简称L0接口，表示调用单Kernel的Host侧API，提供了细颗粒API（单Kernel下发）和算子API开发的基础结构体（如Tensor定义等）和公共基础能力（如workspace复用、引擎调度等），上层应用或者L2接口可通过L0接口的快速组装实现高性能计算。

  L0接口返回值类型是Tensor的类型结构，如aclTensor\*、std::tuple\<aclTensor\*, aclTensor\*\>、aclTensorList\*，最后一个参数固定为aclOpExecutor \*executor，类型与名称均不可变，示例如下：

  ```Cpp
  aclTensor* AddNd(aclTensor *x1, aclTensor *x2, aclOpExecutor *executor)
  ```

  L0接口命名空间为“namespace l0op”，接口名为“\$\{op\_type\}\$\{format\}\$\{dtype\}”，其中\$\{op\_type\}为算子名，\$\{format\}为算子输入/输出数据格式，\$\{dtype\}为算子输入/输出数据类型（对于非常规的输入/输出数据类型，需带上数据类型匹配关系）。调用示例如下：

  ```Cpp
  l0op::AddNd                          //Add算子输入均按ND计算
  l0op::MatMulNdFp162Fp32              //MatMul算子输入输出均按ND格式计算，并且2代表“To”，表示输入fp16、输出fp32
  l0op::MatMulNzFp162Fp16              //MatMul算子输入输出均按NZ格式计算，并且2代表“To”，表示输入与输出都是fp16
  ```

- **Level2层接口**

  简称L2接口，是对L0接口的高层级封装（内部通过调用单个或多个L0接口实现更灵活功能），表示更上层的Host侧API（**aclnn API通常被称为L2接口**）。该类接口提供单算子直调方式，屏蔽了算子内部实现逻辑，用户直接调用L2接口即可实现调用算子。

  L2接口返回值类型是aclnnStatus，一般包括获取workspaceSize和算子执行“两段式接口”：

  ```Cpp
  aclnnStatus aclnnXxxGetWorkspaceSize(const aclTensor *src, ..., aclTensor *out, ..., uint64_t *workspaceSize, aclOpExecutor **executor);
  aclnnStatus aclnnXxx(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
  ```

  - aclnnXxxGetWorkspaceSize最后两个参数固定为\(uint64\_t \*workspaceSize, aclOpExecutor \*\*executor\)，名称和类型均不可变。
  - aclnnXxx接口参数固定为\(void \*workspace, uint64\_t workspaceSize, aclOpExecutor \*executor, aclrtStream stream\)。

  其中aclnnXxxGetWorkspaceSize为第一段接口，主要用于计算本次API调用过程中需要多少workspace内存，获取到本次计算所需的workspaceSize后，按照workspaceSize申请NPU内存，然后调用第二段接口aclnnXxx执行计算。“Xxx”表示对应的算子类型，如Add算子。

  > **说明**：
  >
  > - workspace是指除输入/输出外，API在AI处理器上完成计算所需要的临时内存。
  >
  > - 二阶段接口aclnnXxx\(...\)不能重复调用，如下调用方式会出现异常：
  >
  >   ```text
  >   aclnnXxxGetWorkspaceSize(...)
  >   aclnnXxx(...)
  >   aclnnXxx(...)
  >   ```

- **原地算子接口**

  表示在原地址进行更新操作的算子接口，其计算过程中输入和输出为同一地址，以减少不必要的内存占用。aclnn类原地算子接口名一般定义为：aclnnInplaceXxxGetWorkspaceSize（一阶段接口）、aclnnInplaceXxx（二阶段接口）。

### build说明

[build参数说明](zh/appendix/build.md)
