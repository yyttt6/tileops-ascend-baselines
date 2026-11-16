# L0\_DFX

## 宏功能

用于L0接口时延统计（Profiling）及入参打印。

## 宏原型

```cpp
L0_DFX(profilingName, ...)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| profilingName | 输入 | Host侧L0接口名，例如AddAiCore。 |
| ... | 输入 | Host侧L0接口的入参，可变长参数。 |

## 约束说明

1. L0_DFX宏必须严格放置于L0函数的首行位置，以确保时延统计的准确性，若位置不当可能导致统计偏差或未定义错误。
2. 在L0函数内部使用该宏时，不得使用花括号对宏进行包围，否则会导致时延统计偏差或未定义错误。
3. 采用L0_DFX宏的L0接口禁止在函数体内进行递归调用或嵌套调用，即L0接口内不可再调用其他L0接口，以防止时延统计逻辑冲突。

## 调用示例

```cpp
// 统计L0接口AddAiCore的时延及参数打印，AddAiCore是L0接口的名字，self，other和addOut是L0接口参数
L0_DFX(AddAiCore, self, other, addOut);
```
