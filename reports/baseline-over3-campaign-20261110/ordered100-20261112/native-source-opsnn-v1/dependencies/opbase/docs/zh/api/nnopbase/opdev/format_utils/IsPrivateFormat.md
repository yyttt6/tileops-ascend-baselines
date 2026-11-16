# IsPrivateFormat

## 功能说明

判断输入的format是否为私有格式。

## 函数原型

```cpp
bool IsPrivateFormat(Format format)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| format | 输入 | 待校验的数据格式，数据类型为op::Format（即ge::Format）。 |

## 返回值说明

format为私有格式返回true，否则返回false。

## 约束说明

无

## 调用示例

```cpp
// 当input的storage format为私有格式时，返回true
void Func(const aclTensor *input) {
    if (IsPrivateFormat(input->GetStorageFormat())) {
        return;
    }
}
```
