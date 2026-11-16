# IsEmpty

## 功能说明

判断aclTensor是否为空tensor。

## 函数原型

```cpp
bool IsEmpty()
```

## 参数说明

无

## 返回值说明

如果aclTensor为空tensor，返回true，否则为false。

## 约束说明

无

## 调用示例

```cpp
// 判断input为空tensor则return，否则获取他的数据类型
void Func(const aclTensor *input) {
    if (input->IsEmpty()) {
        return;
    }
    op::DataType dataType = input->GetDataType();
}
```
