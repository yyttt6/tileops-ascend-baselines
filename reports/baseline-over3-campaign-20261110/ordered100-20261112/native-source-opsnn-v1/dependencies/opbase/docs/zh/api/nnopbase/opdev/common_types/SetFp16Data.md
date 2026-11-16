# SetFp16Data

## 功能说明

针对通过AllocHostTensor申请得到的host侧tensor，用一块float16类型的内存初始化tensor数据。

## 函数原型

```cpp
void SetFp16Data(const op::fp16_t *value, uint64_t size, op::DataType dataType)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| value | 输入 | 指向需要写入aclTensor的数据内存指针。 |
| size | 输入 | 需要写入的元素个数。 |
| dataType | 输入 | 数据类型为op::DataType（即ge::DataType），将数据转为指定的dataType后，再写入aclTensor。 |

## 返回值说明

无

## 约束说明

入参指针不能为空。

## 调用示例

```cpp
// 初始化一块fp16内存，赋值给input的前10个元素
void Func(const aclTensor *input) {
    fp16_t myArray[10];
    input->SetFp16Data(myArray, 10, DT_FLOAT16);
}
```
