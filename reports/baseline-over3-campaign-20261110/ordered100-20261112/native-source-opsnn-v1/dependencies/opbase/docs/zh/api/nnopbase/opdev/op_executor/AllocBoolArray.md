# AllocBoolArray

## 功能说明

申请一个aclBoolArray，并使用指定内存中的值对其进行初始化。

## 函数原型

```cpp
aclBoolArray *AllocBoolArray(const bool *value, uint64_t size)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| value | 输入 | 源数据，用于初始化aclBoolArray。 |
| size | 输入 | 源数据的元素个数。 |

## 返回值说明

返回申请到的aclBoolArray对象，申请失败返回nullptr。

## 约束说明

入参指针不能为空。

## 调用示例

```cpp
// 申请一个长度为10的aclBoolArray
void Func(aclOpExecutor *executor) {
    bool myArray[10];
    aclBoolArray *array = executor->AllocBoolArray(myArray, 10);
}
```
