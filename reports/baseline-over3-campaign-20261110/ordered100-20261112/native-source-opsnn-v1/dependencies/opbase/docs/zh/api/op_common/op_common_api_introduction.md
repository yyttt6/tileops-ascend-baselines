# op\_common接口列表

针对CANN算子开发或调用场景（无论是内置或自定义算子），其开发和调用过程会依赖opbase库提供的公共能力，包括日志获取、算子Tiling/InferShape开发依赖的接口等。

- **关键接口分类**：

  - log：算子日志打印相关宏。
  - math：数学计算相关接口。
  - platform：算子执行平台信息相关接口。
  - infershape：算子shape处理相关接口。

- **接口说明**：

  - 接口定位：提供详细参数说明的接口为正式接口，支持版本向后兼容；未提供详细参数说明的接口为预留接口，您可以暂时不关注。
  - 头文件路径：调用本章接口时，请按实际情况include依赖的头文件，一般定义在`${INSTALL_DIR}/include`目录。其中\$\{INSTALL\_DIR\}表示为CANN软件安装后的路径，以root安装举例，安装后路径为`/usr/local/Ascend/cann`。

**表 1**  op\_common接口列表

<table><thead>
  <tr>
    <th>分类</th>
    <th>接口</th>
    <th>说明</th>
    <th>所属头文件</th>
  </tr></thead>
<tbody>
  <tr>
    <td rowspan="44">log</td>
    <td><a href='log/OP_LOGE.md'>OP_LOGE</a></td>
    <td>打印算子ERROR级别日志。</td>
    <td rowspan="44">op_common/log/log.h</td>
  </tr>
  <tr>
    <td><a href='log/OP_LOGD.md'>OP_LOGD</a></td>
    <td>打印算子DEBUG级别日志。</td>
  </tr>
  <tr>
    <td><a href='log/OP_LOGI.md'>OP_LOGI</a></td>
    <td>打印算子INFO级别日志。</td>
  </tr>
  <tr>
    <td><a href='log/OP_LOGW.md'>OP_LOGW</a></td>
    <td>打印算子WARNING级别日志。</td>
  </tr>
  <tr>
    <td><a href='log/OP_CHECK_IF.md'>OP_CHECK_IF</a></td>
    <td>当condition条件成立时，输出日志，并执行return表达式。</td>
  </tr>
  <tr>
    <td><a href='log/OP_CHECK_NULL_WITH_CONTEXT.md'>OP_CHECK_NULL_WITH_CONTEXT</a></td>
    <td>根据传入的context上下文，校验传入的指针是否为nullptr。</td>
  </tr>
  <tr>
    <td><a href='log/OP_LOGE_WITH_INVALID_INPUT_SHAPE.md'>OP_LOGE_WITH_INVALID_INPUT_SHAPE</a></td>
    <td>记录并上报算子输入形状校验错误，上报EZ0001错误码。</td>
  </tr>
  <tr>
    <td><a href='log/OP_LOGE_WITH_INVALID_ATTR.md'>OP_LOGE_WITH_INVALID_ATTR</a></td>
    <td>记录并上报算子属性值校验错误，上报EZ0002错误码。</td>
  </tr>
  <tr>
    <td><a href='log/OP_LOGE_WITH_INVALID_ATTR_SIZE.md'>OP_LOGE_WITH_INVALID_ATTR_SIZE</a></td>
    <td>记录并上报算子属性大小校验错误，上报EZ0003错误码。</td>
  </tr>
  <tr>
    <td><a href='log/OP_LOGE_WITH_INVALID_INPUT.md'>OP_LOGE_WITH_INVALID_INPUT</a></td>
    <td>记录并上报算子输入参数为空校验错误，上报EZ0004错误码。</td>
  </tr>
  <tr>
    <td><a href='log/OP_LOGE_WITH_INVALID_INPUT_SHAPESIZE.md'>OP_LOGE_WITH_INVALID_INPUT_SHAPESIZE</a></td>
    <td>记录并上报算子输入形状大小校验错误，上报EZ0005错误码。</td>
  </tr>
  <tr>
    <td><a href='log/OP_LOGE_WITH_INVALID_INPUT_FORMAT.md'>OP_LOGE_WITH_INVALID_INPUT_FORMAT</a></td>
    <td>记录并上报算子输入格式校验错误，上报EZ0006错误码。</td>
  </tr>
  <tr>
    <td><a href='log/OP_LOGE_WITH_INVALID_INPUT_DTYPE.md'>OP_LOGE_WITH_INVALID_INPUT_DTYPE</a></td>
    <td>记录并上报算子输入数据类型校验错误，上报EZ0007错误码。</td>
  </tr>
  <tr>
    <td><a href='log/OP_LOGE_FOR_INVALID_SHAPE.md'>OP_LOGE_FOR_INVALID_SHAPE</a></td>
    <td>记录并上报参数形状校验错误，上报EZ0008错误码。</td>
  </tr>
  <tr>
    <td><a href='log/OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON.md'>OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON</a></td>
    <td>记录并上报参数形状校验错误（带原因），上报EZ0009错误码。</td>
  </tr>
  <tr>
    <td><a href='log/OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON.md'>OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON</a></td>
    <td>记录并上报多参数形状校验错误（带原因），上报EZ0010错误码。</td>
  </tr>
  <tr>
    <td><a href='log/OP_LOGE_FOR_INVALID_SHAPEDIM.md'>OP_LOGE_FOR_INVALID_SHAPEDIM</a></td>
    <td>记录并上报参数形状维度校验错误，上报EZ0011错误码。</td>
  </tr>
  <tr>
    <td><a href='log/OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON.md'>OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON</a></td>
    <td>记录并上报参数形状维度校验错误（带原因），上报EZ0012错误码。</td>
  </tr>
  <tr>
    <td><a href='log/OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON.md'>OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON</a></td>
    <td>记录并上报多参数形状维度校验错误（带原因），上报EZ0013错误码。</td>
  </tr>
  <tr>
    <td><a href='log/OP_LOGE_FOR_INVALID_SHAPESIZE.md'>OP_LOGE_FOR_INVALID_SHAPESIZE</a></td>
    <td>记录并上报参数形状大小校验错误，上报EZ0014错误码。</td>
  </tr>
  <tr>
    <td><a href='log/OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON.md'>OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON</a></td>
    <td>记录并上报参数形状大小校验错误（带原因），上报EZ0015错误码。</td>
  </tr>
  <tr>
    <td><a href='log/OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON.md'>OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON</a></td>
    <td>记录并上报多参数形状大小校验错误（带原因），上报EZ0016错误码。</td>
  </tr>
  <tr>
    <td><a href='log/OP_LOGE_FOR_INVALID_FORMAT.md'>OP_LOGE_FOR_INVALID_FORMAT</a></td>
    <td>记录并上报参数格式校验错误，上报EZ0017错误码。</td>
  </tr>
  <tr>
    <td><a href='log/OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON.md'>OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON</a></td>
    <td>记录并上报多参数格式校验错误（带原因），上报EZ0018错误码。</td>
  </tr>
  <tr>
    <td><a href='log/OP_LOGE_FOR_INVALID_DTYPE.md'>OP_LOGE_FOR_INVALID_DTYPE</a></td>
    <td>记录并上报参数数据类型校验错误，上报EZ0019错误码。</td>
  </tr>
  <tr>
    <td><a href='log/OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON.md'>OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON</a></td>
    <td>记录并上报参数数据类型校验错误（带原因），上报EZ0020错误码。</td>
  </tr>
  <tr>
    <td><a href='log/OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON.md'>OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON</a></td>
    <td>记录并上报多参数数据类型校验错误（带原因），上报EZ0021错误码。</td>
  </tr>
  <tr>
    <td><a href='log/OP_LOGE_FOR_INVALID_TENSORNUM.md'>OP_LOGE_FOR_INVALID_TENSORNUM</a></td>
    <td>记录并上报参数张量数量校验错误，上报EZ0022错误码。</td>
  </tr>
  <tr>
    <td><a href='log/OP_LOGE_FOR_INVALID_TENSORNUMS_WITH_REASON.md'>OP_LOGE_FOR_INVALID_TENSORNUMS_WITH_REASON</a></td>
    <td>记录并上报多参数张量数量校验错误（带原因），上报EZ0023错误码。</td>
  </tr>
  <tr>
    <td><a href='log/OP_LOGE_FOR_INVALID_VALUE.md'>OP_LOGE_FOR_INVALID_VALUE</a></td>
    <td>记录并上报参数值校验错误，上报EZ0024错误码。</td>
  </tr>
  <tr>
    <td><a href='log/OP_LOGE_FOR_INVALID_LISTSIZE.md'>OP_LOGE_FOR_INVALID_LISTSIZE</a></td>
    <td>记录并上报参数列表大小校验错误，上报EZ0025错误码。</td>
  </tr>
  <tr>
    <td><a href='log/OP_LOGE_FOR_INVALID_VALUE_WITH_REASON.md'>OP_LOGE_FOR_INVALID_VALUE_WITH_REASON</a></td>
    <td>记录并上报参数值校验错误（带原因），上报EZ0026错误码。</td>
  </tr>
  <tr>
    <td><a href='log/OP_LOGE_FOR_INVALID_VALUES_WITH_REASON.md'>OP_LOGE_FOR_INVALID_VALUES_WITH_REASON</a></td>
    <td>记录并上报多参数值校验错误（带原因），上报EZ0027错误码。</td>
  </tr>
  <tr>
    <td><a href='log/OP_LOGE_FOR_INVALID_STRIDE.md'>OP_LOGE_FOR_INVALID_STRIDE</a></td>
    <td>记录并上报参数stride校验错误，上报EZ0028错误码。</td>
  </tr>
  <tr>
    <td><a href='log/OP_LOGE_FOR_FILE_PATH.md'>OP_LOGE_FOR_FILE_PATH</a></td>
    <td>记录并上报文件路径无效错误，上报EZ0029错误码。</td>
  </tr>
  <tr>
    <td><a href='log/OP_LOGE_FOR_FILE_OPEN.md'>OP_LOGE_FOR_FILE_OPEN</a></td>
    <td>记录并上报文件打开失败错误，上报EZ0030错误码。</td>
  </tr>
  <tr>
    <td><a href='log/OP_LOGE_FOR_FILE_PARSE.md'>OP_LOGE_FOR_FILE_PARSE</a></td>
    <td>记录并上报文件解析失败错误，上报EZ0031错误码。</td>
  </tr>
  <tr>
    <td><a href='log/OP_LOGE_FOR_INVALID_CONFIG.md'>OP_LOGE_FOR_INVALID_CONFIG</a></td>
    <td>记录并上报配置项值无效错误，上报EZ0032错误码。</td>
  </tr>
  <tr>
    <td><a href='log/OP_LOGE_FOR_INVALID_CONFIG_WITH_REASON.md'>OP_LOGE_FOR_INVALID_CONFIG_WITH_REASON</a></td>
    <td>记录并上报配置项值无效错误（带原因），上报EZ0033错误码。</td>
  </tr>
  <tr>
    <td><a href='log/OP_LOGE_FOR_INVALID_CONFIGS_WITH_REASON.md'>OP_LOGE_FOR_INVALID_CONFIGS_WITH_REASON</a></td>
    <td>记录并上报多配置项值无效错误（带原因），上报EZ0034错误码。</td>
  </tr>
  <tr>
    <td><a href='log/OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON.md'>OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON</a></td>
    <td>记录并上报参数格式校验错误（带原因），上报EZ0035错误码。</td>
  </tr>
  <tr>
    <td><a href='log/OP_LOGE_FOR_INVALID_GRAPH_NODE.md'>OP_LOGE_FOR_INVALID_GRAPH_NODE</a></td>
    <td>记录并上报图节点无效错误，上报EZ0036错误码。</td>
  </tr>
  <tr>
    <td><a href='log/OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON.md'>OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON</a></td>
    <td>记录并上报参数无效错误（带原因），上报EZ0037错误码。</td>
  </tr>
  <tr>
    <td><a href='log/OP_LOGE_FOR_INVALID_LISTSIZE_WITH_REASON.md'>OP_LOGE_FOR_INVALID_LISTSIZE_WITH_REASON</a></td>
    <td>记录并上报参数列表大小校验错误（带原因），上报EZ0038错误码。</td>
  </tr>
  <tr>
    <td rowspan="5">math</td>
    <td><a href='math/FloorDiv.md'>FloorDiv</a></td>
    <td>向下取整的除法。</td>
    <td rowspan="5">op_common/op_host/util/math_util.h</td>
  </tr>
  <tr>
    <td><a href='math/FloorAlign.md'>FloorAlign</a></td>
    <td>以align为单元，向下对齐。</td>
  </tr>
  <tr>
    <td><a href='math/CeilDiv.md'>CeilDiv</a></td>
    <td>向上取整的除法。</td>
  </tr>
  <tr>
    <td><a href='math/CeilAlign.md'>CeilAlign</a></td>
    <td>以align为单元，向上对齐。</td>
  </tr>
  <tr>
    <td><a href='math/IsFloatEqual.md'>IsFloatEqual</a></td>
    <td>判断两个float类型或double类型的数值是否相等。</td>
  </tr>
  <tr>
    <td rowspan="11">platform</td>
    <td><a href='platform/GetAivCoreNum.md'>GetAivCoreNum</a></td>
    <td>获取平台AI Vector的核数。</td>
    <td rowspan="11">op_common/op_host/util/platform_util.h</td>
  </tr>
  <tr>
    <td><a href='platform/GetAicCoreNum.md'>GetAicCoreNum</a></td>
    <td>获取平台AI Cube的核数。</td>
  </tr>
  <tr>
    <td><a href='platform/GetUbSize.md'>GetUbSize</a></td>
    <td>获取平台UB（Unified Buffer）空间大小，单位bytes。</td>
  </tr>
  <tr>
     <td><a href='platform/GetUbBlockSize.md'>GetUbBlockSize</a></td>
    <td>获取平台UB（Unified Buffer）的block单元大小，单位bytes。</td>
  </tr>
  <tr>
     <td><a href='platform/GetVRegSize.md'>GetVRegSize</a></td>
    <td>获取平台向量寄存器（Vector Register）的大小，单位bytes。</td>
  </tr>
  <tr>
     <td><a href='platform/GetSimtMaxThreadNum.md'>GetSimtMaxThreadNum</a></td>
    <td>获取平台SIMT最大线程数。</td>
  </tr>
  <tr>
     <td><a href='platform/GetSimtMaxDCacheSize.md'>GetSimtMaxDCacheSize</a></td>
    <td>获取平台SIMT最大DCache大小，单位bytes。</td>
  </tr>
  <tr>
     <td><a href='platform/GetCacheLineSize.md'>GetCacheLineSize</a></td>
    <td>获取平台Cache Line的大小，单位bytes。</td>
  </tr>
  <tr>
     <td><a href='platform/GetSectorCacheLineSize.md'>GetSectorCacheLineSize</a></td>
    <td>获取平台Sector Cache Line的大小，单位bytes。</td>
  </tr>
  <tr>
     <td><a href='platform/GetNddmaDcacheSize.md'>GetNddmaDcacheSize</a></td>
    <td>获取平台NDDMA DCache的大小，单位bytes。</td>
  </tr>
  <tr>
     <td><a href='platform/GetWorkspaceSize.md'>GetWorkspaceSize</a></td>
    <td>获取平台LibApi Workspace的大小，单位bytes。</td>
  </tr>
  <tr>
    <td rowspan="8">infershape</td>
    <td><a href='infershape/SetUnknownRank.md'>SetUnknownRank</a></td>
    <td>设置输入shape为维度不确定的动态shape。</td>
    <td rowspan="4">op_common/op_host/util/shape_util.h</td>
  </tr>
  <tr>
    <td><a href='infershape/IsUnknownRank.md'>IsUnknownRank</a></td>
    <td>检查输入shape是否为维度不确定的shape。</td>
  </tr>
  <tr>
    <td><a href='infershape/SetUnknownShape.md'>SetUnknownShape</a></td>
    <td>设置输入shape的维度为rank，且每根轴长度都为不确定值。</td>
  </tr>
  <tr>
    <td><a href='infershape/IsUnknownShape.md'>IsUnknownShape</a></td>
    <td>检查输入shape的每一根轴长度是否都为不确定值。</td>
  </tr>
  <tr>
     <td><a href='infershape/InferShape4Broadcast.md'>InferShape4Broadcast</a></td>
    <td>broadcast类算子的infershape方法。</td>
    <td rowspan="2">op_common/op_host/infershape_broadcast_util.h</td>
  </tr>
  <tr>
     <td><a href='infershape/BroadcastShape.md'>BroadcastShape</a></td>
    <td>根据输入张量的shape推导broadcast后的输出shape。</td>
  </tr>
  <tr>
     <td><a href='infershape/InferShape4Elewise.md'>InferShape4Elewise</a></td>
    <td>elewise类算子的infershape方法。</td>
    <td rowspan="1">op_common/op_host/infershape_elewise_util.h</td>
  </tr>
  <tr>
     <td><a href='infershape/InferShape4Reduce.md'>InferShape4Reduce</a></td>
    <td>reduce类算子的infershape方法。</td>
    <td rowspan="1">op_common/op_host/infershape_reduce_util.h</td>
  </tr>
  <tr>
    <td><a href='reserved_interface_op_common.md'>预留接口</a></td>
    <td>-</td>
    <td>预留接口，暂无需关注。</td>
    <td>
      <ul>
        <li>op_common/op_host/*.h</li>
        <li>op_common/op_kernel/*.h</li>
        <li>op_common/op_graph/*.h</li>
      </ul></td>
  </tr>
</tbody>
</table>
