# IsContiguous

## 功能说明

判断输入tensor是否为一个连续tensor。

## 函数原型

```cpp
bool IsContiguous(const aclTensor *tensor)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| tensor | 输入 | 目标tensor。 |

## 返回值说明

若为连续tensor返回true，否则返回false。

## 约束说明

- 应传入非空tensor。
- 对于空指针场景，将返回true，并输出ERROR提示："check tensor != nullptr failed"。

## 调用示例

```cpp
// 判断当tensor非连续时，返回。
void Func(aclTensor *tensor) {
    if (!IsContiguous(tensor)) {
        return;
    }
}
```
