# SetData

## 功能说明

针对通过AllocHostTensor申请得到的host侧tensor，设置指定位置的数据。

## 函数原型

- 针对tensor，设置指定索引处的值

    ```cpp
    void SetData(int64_t index, const T value, op::DataType dataType)
    ```

- 针对tensor，用一块已有内存初始化tensor数据

    ```cpp
    void SetData(const T *value, uint64_t size, op::DataType dataType)
    ```

## 参数说明

> **说明**：ge::DataType介绍参见[《基础数据结构和接口参考》](https://www.hiascend.com/document/redirect/CannCommunitybasicopapi)中”ge命名空间 > DataType“。

- 设置指定索引处的值的接口说明

    | 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| index | 输入 | 需要修改aclTensor的第几个元素。 |
| value | 输入 | 将aclTensor的指定元素修改为value的值。 |
| dataType | 输入 | 数据类型为op::DataType（即ge::DataType）。将value转为指定的dataType后，再写入aclTensor。 |

- 用一块已有内存初始化tensor数据的接口说明

    | 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| value | 输入 | 指向需要写入aclTensor的数据内存指针。 |
| size | 输入 | 需要写入的元素个数。 |
| dataType | 输入 | 数据类型为op::DataType（即ge::DataType）。将数据转为指定的dataType后，再写入aclTensor。 |

## 返回值说明

无

## 约束说明

入参指针不能为空。

## 调用示例

```cpp
// 初始化一块int64_t内存，分别将input的前10个数字，置为该内存的内容。并将input的第11个数字置为myArray的第一个数字。
void Func(const aclTensor *input) {
    int64_t myArray[10];
    input->SetData(myArray, 10, DT_INT64);
    input->SetData(10, myArray[0], DT_INT64);
}
```
