# L2\_DFX\_PHASE\_2

## 宏功能

用于L2二阶段接口aclnn_Xxx_时延统计，必须在二阶段接口最前方调用。

## 宏原型

```cpp
L2_DFX_PHASE_2(APIName)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| APIName | 输入 | Host侧L2接口名，如aclnnXxx。 |

## 约束说明

必须在L2二阶段接口的入口处调用，否则可能导致时延统计出现误差。

## 调用示例

```cpp
// abs算子的L2接口二阶段时延统计
L2_DFX_PHASE_2(aclnnAbs);
```
