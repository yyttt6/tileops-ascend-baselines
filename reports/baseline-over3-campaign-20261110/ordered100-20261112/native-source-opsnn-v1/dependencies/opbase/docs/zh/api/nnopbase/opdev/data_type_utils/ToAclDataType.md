# ToAclDataType

## 功能说明

将op::DataType转换为对应的aclDataType。

## 函数原型

```cpp
aclDataType ToAclDataType(DataType type)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| type | 输入 | 待转换的原始数据类型op::DataType。 |

## 返回值说明

返回aclDataType类型。

## 约束说明

无

## 调用示例

```cpp
// 获取DT_FLOAT对应的aclDataType枚举
void Func() {
    aclDataType type = ToAclDataType(DT_FLOAT);
}
```
