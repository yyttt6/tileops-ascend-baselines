# nnopbase接口列表

无论是CANN内置算子或自定义算子（例如Transformer、NN类），均可通过aclnn API（也称为[Level2层接口](#l2)）实现底层算子调用。

**针对开发aclnn API场景**，本章提供开发过程中依赖的底层**框架能力接口**和**基础张量操作接口**（也称为[Level0层接口](#l0)）。

- **框架能力接口**：

  提供实现aclnn API的基础能力接口，比如算子执行器（opExecutor）处理、数据类型/格式/shape等操作，具体如[表1](#table1)所示；还包括常用类和宏，具体参见[表2](#table2)、[表3](#table3)。

- **基础张量操作接口**：

  提供实现aclnn API的基础张量操作接口，比如Tensor数据类型转换、shape重构等，具体如[表4](#table4)所示。

## 框架能力接口

**头文件说明**：调用本章接口时，请按实际情况include依赖的头文件，一般定义在`${INSTALL_DIR}/include`目录。其中\$\{INSTALL\_DIR\}表示为CANN软件安装后的路径，以root安装举例，安装后路径为`/usr/local/Ascend/cann`。

**表 1**  接口列表 <a id="table1"></a>

| 接口分类 | 说明 | 所属头文件 |
| --- | --- | --- |
| [bfloat16](bfloat16/bfloat16.md) | 详细介绍了bfloat16数据类型在CPU侧的实现类。 | aclnn/opdev/bfloat16.h |
| [common_types](common_types/common_types.md) | 详细介绍了aclTensor、aclScalar等基础的aclnn数据结构。 | aclnn/opdev/common_types.h |
| [data_type_utils](data_type_utils/data_type_utils.md) | 提供了DataType相关的基础接口，例如获取指定DataType的size等。 | aclnn/opdev/data_type_utils.h |
| fast_vector | 详细介绍了FastVector数据类型，该类型为aclnn中实现的高效vector数据结构。<br>**说明：该头文件定义的接口均为预留接口，开发者无需关注。** | aclnn/opdev/fast_vector.h |
| [format_utils](format_utils/format_utils.md) | 提供了Format相关的基础接口。 | aclnn/opdev/format_utils.h |
| [fp16_t](fp16_t/fp16_t.md) | 详细介绍了float16数据类型在CPU侧的实现类。 | aclnn/opdev/fp16_t.h |
| [float4_e1m2](float4_e1m2/float4_e1m2.md) | 详细介绍了Float4E1M2数据类型在CPU侧的实现类。 | aclnn/opdev/float4_e1m2.h |
| [float4_e2m1](float4_e2m1/float4_e2m1.md) | 详细介绍了Float4E2M1数据类型在CPU侧的实现类。 | aclnn/opdev/float4_e2m1.h |
| [float6_e2m3](float6_e2m3/float6_e2m3.md) | 详细介绍了Float6E2M3数据类型在CPU侧的实现类。 | aclnn/opdev/float6_e2m3.h |
| [float6_e3m2](float6_e3m2/float6_e3m2.md) | 详细介绍了Float6E3M2数据类型在CPU侧的实现类。 | aclnn/opdev/float6_e3m2.h |
| [float8_e4m3fn](float8_e4m3fn/float8_e4m3fn.md) | 详细介绍了Float8E4M3FN数据类型在CPU侧的实现类。 | aclnn/opdev/float8_e4m3fn.h |
| [float8_e5m2](float8_e5m2/float8_e5m2.md) | 详细介绍了Float8E5M2数据类型在CPU侧的实现类。 | aclnn/opdev/float8_e5m2.h |
| [float8_e8m0](float8_e8m0/float8_e8m0.md) | 详细介绍了Float8E8M0数据类型在CPU侧的实现类。 | aclnn/opdev/float8_e8m0.h |
| [hifloat4](hifloat4/hifloat4.md) | 详细介绍了HiFloat4数据类型在CPU侧的实现类。 | aclnn/opdev/hifloat4.h |
| [hifloat8](hifloat8/hifloat8.md) | 详细介绍了HiFloat8数据类型在CPU侧的实现类。 | aclnn/opdev/hifloat8.h |
| [framework_op](framework_op/framework_op.md) | 详细介绍了框架对外提供的从host侧到device侧拷贝能力。 | aclnn/opdev/framework_op.h |
| make_op_executor | 对外提供初始化aclOpExecutor的宏声明。<br>**说明：该头文件定义的接口均为预留接口，开发者无需关注。** | aclnn/opdev/make_op_executor.h |
| [object](object/object.md) | 详细介绍了aclnn中aclTensor等基础数据结构的基类Object类，用于重载实现new、delete方法。 | aclnn/opdev/object.h |
| [op_arg_def](op_arg_def/op_arg_def.md) | 详细介绍了OpArgContext类，并对外提供OP_INPUT等宏声明。 | aclnn/opdev/op_arg_def.h |
| [op_cache](op_cache/op_cache.md) | 详细介绍了OpExecCache及相关类，用于完成aclnn缓存，提升运行性能。 | aclnn/opdev/op_cache.h |
| [op_cache_container](op_cache_container/op_cache_container.md) | 详细介绍了带LRU淘汰机制的aclnn缓存容器。 | aclnn/opdev/op_cache_container.h |
| [op_config](op_config/op_config.md) | 提供了算子运行时相关的配置信息，如确定性计算开关等。 | aclnn/opdev/op_config.h |
| [op_def](op_def/op_def.md) | 定义基础枚举及常量，例如精度模式OpImplMode等。 | aclnn/opdev/op_def.h |
| [op_dfx](op_dfx/op_dfx.md) | 详细介绍了DfxGuard类，用于接口打印及上报profiling。 | aclnn/opdev/op_dfx.h |
| [aclnn返回码](aclnn_return_code/aclnn_return_code.md) | 定义了aclnn错误码。 | aclnn/opdev/op_errno.h |
| [op_executor](op_executor/op_executor.md) | 详细介绍了aclOpExecutor类。 | aclnn/opdev/op_executor.h |
| [op_log](op_log/op_log.md) | 定义aclnn中日志打印宏。 | aclnn/opdev/op_log.h |
| [platform](platform/platform.md) | 详细介绍了PlatformInfo类，用于存放SOC平台信息。 | aclnn/opdev/platform.h |
| [pool_allocator](pool_allocator/pool_allocator.md) | 详细介绍了PoolAllocator类，用于实现aclnn内部的CPU内存池。 | aclnn/opdev/pool_allocator.h |
| [shape_utils](shape_utils/shape_utils.md) | 提供了shape相关的基础操作，例如shape打印等。 | aclnn/opdev/shape_utils.h |
| [small_vector](small_vector/small_vector.md) | 详细介绍了SmallVector类，该类为aclnn中实现的高效vector数据结构，主要针对已知数据量较小的场景。 | aclnn/opdev/small_vector.h |
| [tensor_view_utils](tensor_view_utils/tensor_view_utils.md) | 提供了对于View类的基础操作，例如判断aclTensor是否连续等。 | aclnn/opdev/tensor_view_utils.h |
| [data_type_utils](op_common_related_interface/data_type_utils-6.md) | 提供了DataType相关的基础接口，例如判断指定DataType是否为整数类型等。 | aclnn/opdev/op_common/data_type_utils.h<br>**须知：该头文件后续版本会废弃，请使用aclnn/opdev/data_type_utils.h定义的接口替换** |
| [aicpu_args_handler](aicpu_related_interface/aicpu_args_handler.md) | 提供了AI CPU相关的组合计算任务的处理逻辑，例如拼接计算任务相关的参数等。 | aclnn/opdev/aicpu/aicpu_args_handler.h |
| [aicpu_ext_info_handle](aicpu_related_interface/aicpu_ext_info_handle.md) | 提供了AI CPU相关的计算任务拓展参数的处理逻辑，例如拼接解析拓展参数的接口。 | aclnn/opdev/aicpu/aicpu_ext_info_handle.h |
| [aicpu_task](aicpu_related_interface/aicpu_task.md) | 提供了AI CPU任务设置、下发等逻辑，例如设置调用哪个AI CPU算子，设置算子输入、输出等接口。 | aclnn/opdev/aicpu/aicpu_task.h |
| [aicpu_utils](aicpu_related_interface/aicpu_utils.md) | AI CPU任务需要的一些公共接口。 | aclnn/opdev/aicpu/aicpu_utils.h |

**表 2**  常用宏表  <a id="table2"></a>

| 宏名称 | 说明 | 所属头文件 |
| --- | --- | --- |
| [DFX_IN](common_macros_and_classes/DFX_IN.md) | 在L2_DFX_PHASE_1中，用于打包所有的host侧API输入参数。 | aclnn/opdev/op_dfx.h |
| [DFX_OUT](common_macros_and_classes/DFX_OUT.md) | 在L2_DFX_PHASE_1中，用于打包所有的host侧API输出参数。 | aclnn/opdev/op_dfx.h |
| [L0_DFX](common_macros_and_classes/L0_DFX.md) | 必须在host侧API L0接口中使用，用于接口及L0接口入参打印。 | aclnn/opdev/op_dfx.h |
| [L2_DFX_PHASE_1](common_macros_and_classes/L2_DFX_PHASE_1.md) | 必须在一阶段接口最前方调用，用于接口及一阶段入参打印。 | aclnn/opdev/op_dfx.h |
| [L2_DFX_PHASE_2](common_macros_and_classes/L2_DFX_PHASE_2.md) | 必须在二阶段接口最前方调用，用于接口打印。 | aclnn/opdev/op_dfx.h |
| [OP_TYPE_REGISTER](common_macros_and_classes/OP_TYPE_REGISTER.md) | 必须在L0接口最开始处使用，用于注册L0算子。 | aclnn/opdev/op_dfx.h |
| [OP_ATTR](common_macros_and_classes/OP_ATTR.md) | ADD_TO_LAUNCHER_LIST_AICORE中，打包算子属性参数。 | aclnn/opdev/op_arg_def.h |
| [OP_EMPTY_ARG](common_macros_and_classes/OP_EMPTY_ARG.md) | ADD_TO_LAUNCHER_LIST_AICORE中，用于占位一个空的输入或输出。 | aclnn/opdev/op_arg_def.h |
| [OP_INPUT](common_macros_and_classes/OP_INPUT.md) | ADD_TO_LAUNCHER_LIST_AICORE中，打包算子输入aclTensor。 | aclnn/opdev/op_arg_def.h |
| [OP_MODE](common_macros_and_classes/OP_MODE.md) | ADD_TO_LAUNCHER_LIST_AICORE中，打包算子运行选项，例如是否开启HF32。 | aclnn/opdev/op_arg_def.h |
| [OP_OUTPUT](common_macros_and_classes/OP_OUTPUT.md) | ADD_TO_LAUNCHER_LIST_AICORE中，打包算子输出aclTensor。 | aclnn/opdev/op_arg_def.h |
| [OP_OUTSHAPE](common_macros_and_classes/OP_OUTSHAPE.md) | ADD_TO_LAUNCHER_LIST_AICORE中，针对第三类算子，设置存放输出shape的aclTensor。 | aclnn/opdev/op_arg_def.h |
| [OP_OPTION](common_macros_and_classes/OP_OPTION.md) | ADD_TO_LAUNCHER_LIST_AICORE中，打包算子指定的精度模式。 | aclnn/opdev/op_arg_def.h |
| [OP_WORKSPACE](common_macros_and_classes/OP_WORKSPACE.md) | ADD_TO_LAUNCHER_LIST_AICORE中，打包算子显式指定的workspace参数。 | aclnn/opdev/make_op_executor.h |
| [CREATE_EXECUTOR](common_macros_and_classes/CREATE_EXECUTOR.md) | 创建一个UniqueExecutor对象，该对象为aclOpExecutor的生成工厂类。 | aclnn/opdev/make_op_executor.h |
| [INFER_SHAPE](common_macros_and_classes/INFER_SHAPE.md) | 针对指定算子，运行其infershape函数，推导输出shape。 | aclnn/opdev/make_op_executor.h |
| [ADD_TO_LAUNCHER_LIST_AICORE](common_macros_and_classes/ADD_TO_LAUNCHER_LIST_AICORE.md) | 创建某个AICore算子的执行任务，并置入aclOpExecutor的执行队列，在二阶段时执行。 | aclnn/opdev/make_op_executor.h |
| [OP_ATTR_NAMES](common_macros_and_classes/OP_ATTR_NAMES.md) | String类型的vector，打包AI CPU算子的字符类型属性。 | aclnn/opdev/aicpu/aicpu_task.h |
| [ADD_TO_LAUNCHER_LIST_AICPU](common_macros_and_classes/ADD_TO_LAUNCHER_LIST_AICPU.md) | 创建某个AI CPU算子的执行任务，并置入aclOpExecutor的执行队列，在二阶段时执行。 | aclnn/opdev/aicpu/aicpu_task.h |

**表 3**  常用class和struct表  <a id="table3"></a>

| class/struct名称 | 说明 | 所属头文件 |
| --- | --- | --- |
| aclOpExecutor | 用于表示算子执行器，记录整个host侧API运行信息的上下文结构，如L2接口执行过程中的计算图、L0算子launch子任务、workspace地址和大小等信息。<br>该类定义的成员变量为私有类型，开发者无需关注，定义的成员函数参见[op_executor](op_executor/op_executor.md)。 | aclnn/opdev/op_executor.h |
| aclTensor | 用于表示一个张量对象，包括描述张量的shape、dtype、format、address等信息，数据可以放在host侧或device侧。<br>该类定义的成员变量为私有类型，开发者无需关注，定义的成员函数参见[common_types](common_types/common_types.md)。 | aclnn/opdev/common_types.h |
| aclScalar | 用于表示一个标量对象，数据一般放在host侧。<br>该类定义的成员变量为私有类型，开发者无需关注，定义的成员函数参见[common_types](common_types/common_types.md)。 | aclnn/opdev/common_types.h |
| aclTensorList | 用于表示一组aclTensor类型组成的列表对象。<br>该类定义的成员变量为私有类型，开发者无需关注，定义的成员函数参见[common_types](common_types/common_types.md)。 | aclnn/opdev/common_types.h |
| aclScalarList | 用于表示一组aclScalar类型组成的列表对象。<br>该类定义的成员变量为私有类型，开发者无需关注，定义的成员函数参见[common_types](common_types/common_types.md)。 | aclnn/opdev/common_types.h |
| aclBoolArray | 用于表示一个布尔类型的数组对象，数据一般放在host侧。<br>该类定义的成员变量为私有类型，开发者无需关注，定义的成员函数参见[common_types](common_types/common_types.md)。 | aclnn/opdev/common_types.h |
| aclIntArray | 用于表示一个int64_t类型的数组对象，数据一般放在host侧。<br>该类定义的成员变量为私有类型，开发者无需关注，定义的成员函数参见[common_types](common_types/common_types.md)。 | aclnn/opdev/common_types.h |
| aclFloatArray | 用于表示一个fp32类型的数组对象，数据一般放在host侧。<br>该类定义的成员变量为私有类型，开发者无需关注，定义的成员函数参见[common_types](common_types/common_types.md)。 | aclnn/opdev/common_types.h |
| aclFp16Array | 用于表示一个fp16类型的数组对象，数据一般放在host侧。<br>该类定义的成员变量为私有类型，开发者无需关注，定义的成员函数参见[common_types](common_types/common_types.md)。 | aclnn/opdev/common_types.h |
| aclBf16Array | 用于表示一个bf16类型的数组对象，数据一般放在host侧。<br>该类定义的成员变量为私有类型，开发者无需关注，定义的成员函数参见[common_types](common_types/common_types.md)。 | aclnn/opdev/common_types.h |
| SmallVector | 该类使用内部内存池实现vector容器，基础功能与C++标准库中std::vector容器相同，无需每次扩容都申请内存，避免影响性能。其定义的成员变量为私有类型，开发者无需关注，定义的成员函数参见[small_vector](small_vector/small_vector.md)。 | aclnn/opdev/small_vector.h |
| OpExecMode | 用于表示算子运行模式的枚举类，定义参见[OpExecMode](common_macros_and_classes/OpExecMode.md)。 | aclnn/opdev/op_def.h |
| OpImplMode | 用于表示算子精度模式的枚举类，定义参见[OpImplMode](common_macros_and_classes/OpImplMode.md)。 | aclnn/opdev/op_def.h |

> **SmallVector类使用说明**：
>
> - op::FVector：本质是存储容量长度为8的SmallVector。
>
>   ```cpp
>   namespace op {
>   template<typename T, size_t N = 8>
>   using FVector = op::internal::SmallVector<T, N, op::internal::PoolAllocator<T>>;
>   }
>   ```
>
> - op::Strides：本质是存储容量长度为25的FVector，元素类型int64_t，存储stride信息。
>
>   ```cpp
>   namespace op {
>   constexpr uint64_t MAX_DIM_NUM = 25;
>   using Strides = FVector<int64_t, MAX_DIM_NUM>;
>   }
>   ```
>
> - op::ShapeVector：本质是存储容量长度为25的FVector，元素类型int64_t，存储shape信息。
>
>   ```cpp
>   namespace op {
>   constexpr uint64_t MAX_DIM_NUM = 25;
>   using ShapeVector = FVector<int64_t, MAX_DIM_NUM>;
>   }
>   ```

## 基础张量操作接口  <a id="table4"></a>

**表 4**  接口列表

| 接口名 | 说明 | 所属头文件 |
| --- | --- | --- |
| [Cast](L0/Cast.md) | 将输入tensor转换为指定的数据类型。 | aclnn_kernels/cast.h |
| [Contiguous](L0/Contiguous.md) | 将非连续tensor转换为连续tensor。 | aclnn_kernels/contiguous.h |
| [ViewCopy](L0/ViewCopy.md) | 将连续tensor搬运到连续或非连续tensor上。 | aclnn_kernels/contiguous.h |
| [Pad](L0/Pad.md) | 将输入tensor按照paddings的大小对各个维度进行填充，填充值为0。 | aclnn_kernels/pad.h |
| [Reshape](L0/Reshape.md) | 将输入tensor x的shape转换成该函数的第二个参数shape。 | aclnn_kernels/reshape.h |
| [Slice](L0/Slice.md) | 从输入tensor中提取所需的切片。 | aclnn_kernels/slice.h |
| [Transpose](L0/Transpose.md) | 将输入tensor x的shape按指定维度的排列顺序perm进行转置并输出。 | aclnn_kernels/transpose.h |
| [TransData](L0/TransData.md) | j将输入tensor的format转换为指定的dstPrimaryFormat。 | aclnn_kernels/transdata.h |
| [TransDataSpecial](L0/TransDataSpecial.md) | 将输入tensor的format转换为指定的dstPrimaryFormat，与TransData类似。 | aclnn_kernels/transdata.h |
| [ReFormat](L0/ReFormat.md) | 在指定format和输入x的维度相同时，将输入数据格式设置为目标format。 | aclnn_kernels/transdata.h |
| [IsNullptr](L0/IsNullptr.md) | 判断输入的指针是否为空。| aclnn_kernels/op_error_check.h |

## 基本概念

- **Level0层接口**  <a id="l0"></a>

    简称L0层接口，表示调用单Kernel的Host侧API，提供了细颗粒API（单Kernel下发）和算子API开发的基础结构体（如Tensor定义等）和公共基础能力（如workspace复用、引擎调度等），上层应用或者L2层接口可通过L0接口的快速组装实现高性能计算。

    L0接口返回值类型是Tensor的类型结构，如aclTensor\*、std::tuple<aclTensor\*, aclTensor\*\>、aclTensorList\*，最后一个参数固定为aclOpExecutor \*executor，类型与名称均不可变，示例如下：

    ```cpp
    aclTensor* AddNd(aclTensor *x1, aclTensor *x2, aclOpExecutor *executor)
    ```

    L0接口命名空间为“namespace l0op”，接口名为“$\{op\_type\}$\{format\}$\{dtype\}”，其中$\{op\_type\}为算子名，$\{format\}为算子输入/输出数据格式，$\{dtype\}为算子输入/输出数据类型（对于非常规的输入/输出数据类型，需带上数据类型匹配关系）。调用示例如下：

    ```cpp
    l0op::AddNd                               //Add算子输入均按ND计算
    l0op::MatMulNdFp162Fp32                   //MatMul算子输入输出均按ND格式计算，并且2代表“To”，表示输入fp16、输出fp32
    l0op::MatMulNzFp162Fp16                   //MatMul算子输入输出均按NZ格式计算，并且2代表“To”，表示输入与输出都是fp16
    ```

- **Level2层接口**  <a id="l2"></a>

    简称L2层接口，是对L0层接口的高层级封装（内部通过调用单个或多个L0接口实现更灵活功能），表示更上层的Host侧API。该类接口提供单算子直调方式，屏蔽了算子内部实现逻辑，用户直接调用L2接口即可实现调用算子。

    L2接口返回值类型是aclnnStatus，一般包括获取workspaceSize和算子执行“两段式接口”：

    ```cpp
    aclnnStatus aclnnXxxGetWorkspaceSize(const aclTensor *src, ..., aclTensor *out, ..., uint64_t *workspaceSize, aclOpExecutor **executor);
    aclnnStatus aclnnXxx(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
    ```

  - aclnnXxxGetWorkspaceSize最后两个参数固定为\(uint64\_t \*workspaceSize, aclOpExecutor \*\*executor\)，名称和类型均不可变。
  - aclnnXxx接口参数固定为\(void \*workspace, uint64\_t workspaceSize, aclOpExecutor \*executor, aclrtStream stream\)。

    其中aclnnXxxGetWorkspaceSize为第一段接口，主要用于计算本次API调用过程中需要多少workspace内存，获取到本次计算所需的workspaceSize后，按照workspaceSize申请NPU内存，然后调用第二段接口aclnnXxx执行计算_。_“Xxx”表示对应的算子类型，如Add算子_。_

    > **说明：**
    > - workspace是指除输入/输出外，API在AI处理器上完成计算所需要的临时内存，workspaceSize是指workspace内存大小。
    > - 二阶段接口aclnnXxx\(...\)不能重复调用，如下调用方式会出现异常：
    >
    >   ```cpp
    >   aclnnXxxGetWorkspaceSize(...)
    >   aclnnXxx(...)
    >   aclnnXxx(...)
    >   ```

- **原地算子接口**

    表示在原地址进行更新操作的算子接口，其计算过程中输入和输出为同一地址，以减少不必要的内存占用。aclnn类原地算子接口名一般定义为：aclnnInplaceXxxGetWorkspaceSize（一阶段接口）、aclnnInplaceXxx（二阶段接口）。
