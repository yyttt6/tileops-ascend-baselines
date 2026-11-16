# DFX\_OUT

## 宏功能

用于封装L2一阶段接口aclnn_Xxx_GetWorkspaceSize的输出参数。

## 宏原型

```cpp
DFX_OUT(...)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| ... | 输入 | Host侧L2一阶段接口的输出参数，可变长参数。 |

## 约束说明

无

## 调用示例

```cpp
// add算子的输出参数：out
DFX_OUT(out);
```
