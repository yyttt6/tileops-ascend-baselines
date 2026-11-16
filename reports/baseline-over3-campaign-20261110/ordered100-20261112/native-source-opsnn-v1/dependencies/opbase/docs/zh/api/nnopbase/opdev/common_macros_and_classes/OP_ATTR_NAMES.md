# OP\_ATTR\_NAMES

## 宏功能

用于封装AI CPU算子的属性名字。

## 宏原型

```cpp
OP_ATTR_NAMES::op::FVector<std::string>
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| / | 输入 | AI CPU算子的属性名，类型为std::string的FVector。默认值为空的FVector，即OP_ATTR_NAMES()。 |

## 约束说明

无

## 调用示例

```cpp
// 封装AI CPU算子的3个属性名为Tindices，T和use_locking
OP_ATTR_NAMES({"Tindices", "T", "use_locking"});
```
