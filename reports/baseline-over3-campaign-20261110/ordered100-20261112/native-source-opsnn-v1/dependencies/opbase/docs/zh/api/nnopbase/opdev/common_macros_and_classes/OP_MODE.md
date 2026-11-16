# OP\_MODE

## 宏功能

用于封装算子的运行模式，算子计算过程中是否开启HF32数据类型。

## 宏原型

```cpp
OP_MODE(x...)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| x... | 输入 | 指定算子运行模式，具体取值参见[OpExecMode](OpExecMode.md)。默认为0，即Default模式。 |

## 约束说明

无

## 调用示例

```cpp
// 封装算子的运行模式
OP_MODE(OP_EXEC_MODE_DEFAULT);
```
