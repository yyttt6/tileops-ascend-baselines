# ToAclFormat

## 功能说明

将op::Format转为对应的aclFormat。

## 函数原型

```cpp
aclFormat ToAclFormat(Format format)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| format | 输入 | 待转换的原始数据格式op::Format。 |

## 返回值说明

返回aclFormat类型。

## 约束说明

无

## 调用示例

```cpp
// 获取FORMAT_ND对应的aclFormat枚举
void Func() {
    aclFormat format = ToAclFormat(FORMAT_ND);
}
```
