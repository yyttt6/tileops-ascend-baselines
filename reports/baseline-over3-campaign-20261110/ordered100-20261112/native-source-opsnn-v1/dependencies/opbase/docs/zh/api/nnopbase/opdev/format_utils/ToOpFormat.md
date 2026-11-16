# ToOpFormat

## 功能说明

将aclFormat转为对应的op::Format。

## 函数原型

```cpp
Format ToOpFormat(aclFormat format)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| format | 输入 | 待转换的原始数据格式aclFormat。 |

## 返回值说明

返回op::Format类型。

## 约束说明

无

## 调用示例

```cpp
// 获取ACL_FORMAT_ND对应的Format枚举
void Func() {
    Format format = ToOpFormat(ACL_FORMAT_ND);
}
```
