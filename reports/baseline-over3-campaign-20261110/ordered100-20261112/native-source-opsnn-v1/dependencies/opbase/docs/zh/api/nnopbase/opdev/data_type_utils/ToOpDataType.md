# ToOpDataType

## 功能说明

将aclDataType转换为对应的op::DataType。

## 函数原型

```cpp
DataType ToOpDataType(aclDataType type)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| type | 输入 | 待转换的原始数据类型aclDataType。 |

## 返回值说明

返回op::DataType类型。

## 约束说明

无

## 调用示例

```cpp
// 获取ACL_FLOAT对应的DataType枚举
void Func() {
    DataType type = ToOpDataType(ACL_FLOAT);
}
```
