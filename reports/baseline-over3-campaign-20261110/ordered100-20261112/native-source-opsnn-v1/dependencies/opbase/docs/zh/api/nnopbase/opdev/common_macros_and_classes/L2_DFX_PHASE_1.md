# L2\_DFX\_PHASE\_1

## 宏功能

用于L2一阶段接口aclnn_Xxx_GetWorkspaceSize时延统计及入参打印，必须在一阶段接口最前方调用。

## 宏原型

```cpp
L2_DFX_PHASE_1(APIName, IN, OUT)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| APIName | 输入 | Host侧L2接口名，如aclnnXxx。 |
| IN | 输入 | 算子的输入参数，由[DFX_IN](DFX_IN.md)封装。 |
| OUT | 输入 | 算子的输出参数，由[DFX_OUT](DFX_OUT.md)封装。 |

## 约束说明

- 必须在L2一阶段接口的入口处调用，否则可能导致时延统计出现误差。传入参数必须与L2接口的参数列表严格一致。
- 如下接口是上述宏定义会调用到的关联接口。

    ```text
    #define DFX_IN(...) std::make_tuple(__VA_ARGS__)
    #define DFX_OUT(...) std::make_tuple(__VA_ARGS__)
    ```

## 调用示例

```cpp
// abs算子的L2接口一阶段时延统计及参数打印，self为abs算子的输入，out为abs算子的输出
L2_DFX_PHASE_1(aclnnAbs, DFX_IN(self), DFX_OUT(out));
```
