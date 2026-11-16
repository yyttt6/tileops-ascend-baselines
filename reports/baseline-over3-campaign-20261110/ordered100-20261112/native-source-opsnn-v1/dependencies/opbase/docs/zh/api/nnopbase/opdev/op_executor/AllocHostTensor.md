# AllocHostTensor

## 功能说明

申请一个host侧tensor，提供多个重载函数，可以指定不同的输入属性，如不同的数据类型。

## 函数原型

- **根据不同的输入信息组合，申请一个host侧tensor**

    ```cpp
    aclTensor *AllocHostTensor(const op::Shape &shape, op::DataType dataType, op::Format format = op::Format::FORMAT_ND)
    ```

    ```cpp
    aclTensor *AllocHostTensor(const op::Shape &storageShape, const op::Shape &originShape, op::DataType dataType, op::Format storageFormat, op::Format originFormat)
    ```

- **申请一个host侧tensor，将指定数据类型的内存作为该tensor内容**

    ```cpp
    aclTensor *AllocHostTensor(const int64_t *value, uint64_t size, op::DataType dataType)
    ```

    ```cpp
    aclTensor *AllocHostTensor(const uint64_t *value, uint64_t size, op::DataType dataType)
    ```

    ```cpp
    aclTensor *AllocHostTensor(const bool *value, uint64_t size, op::DataType dataType)
    ```

    ```cpp
    aclTensor *AllocHostTensor(const char *value, uint64_t size, op::DataType dataType)
    ```

    ```cpp
    aclTensor *AllocHostTensor(const int32_t *value, uint64_t size, op::DataType dataType)
    ```

    ```cpp
    aclTensor *AllocHostTensor(const uint32_t *value, uint64_t size, op::DataType dataType)
    ```

    ```cpp
    aclTensor *AllocHostTensor(const int16_t *value, uint64_t size, op::DataType dataType)
    ```

    ```cpp
    aclTensor *AllocHostTensor(const uint16_t *value, uint64_t size, op::DataType dataType)
    ```

    ```cpp
    aclTensor *AllocHostTensor(const int8_t *value, uint64_t size, op::DataType dataType)
    ```

    ```cpp
    aclTensor *AllocHostTensor(const uint8_t *value, uint64_t size, op::DataType dataType)
    ```

    ```cpp
    aclTensor *AllocHostTensor(const double *value, uint64_t size, op::DataType dataType)
    ```

    ```cpp
    aclTensor *AllocHostTensor(const float *value, uint64_t size, op::DataType dataType)
    ```

    ```cpp
    aclTensor *AllocHostTensor(const op::fp16_t *value, uint64_t size, op::DataType dataType)
    ```

    ```cpp
    aclTensor *AllocHostTensor(const op::bfloat16 *value, uint64_t size, op::DataType dataType)
    ```

## 参数说明

- 根据不同的输入信息组合，申请一个host侧tensor。

    | 参数 | 输入/输出 | 说明 |
    | --- | --- | --- |
    | shape | 输入 | 将aclTensor的[StorageShape](../common_types/GetStorageShape.md)和[OriginShape](../common_types/GetOriginalShape.md)都设置为指定的shape。 |
    | dataType | 输入 | 指定aclTensor的数据类型。 |
    | format | 输入 | 将aclTensor的[StorageFormat](../common_types/GetStorageFormat.md)和[OriginFormat](../common_types/GetOriginalFormat.md)都设置为指定的format。 |
    | storageShape | 输入 | 将aclTensor的[StorageShape](../common_types/GetStorageShape.md)设置为指定shape。 |
    | originShape | 输入 | 将aclTensor的[OriginShape](../common_types/GetOriginalShape.md)设置为指定shape。 |
    | storageFormat | 输入 | 将aclTensor的[StorageFormat](../common_types/GetStorageFormat.md)设置为指定format。 |
    | originFormat | 输入 | 将aclTensor的[OriginFormat](../common_types/GetOriginalFormat.md)设置为指定format。 |

- 申请一个host侧tensor，将指定数据类型的内存作为该tensor内容。

    | 参数 | 输入/输出 | 说明 |
    | --- | --- | --- |
    | value | 输入 | 指向不同数据类型的源数据。 |
    | size | 输入 | 源数据的元素个数。 |
    | dataType | 输入 | 将源数据转为dataType指定的数据类型后，写入tensor。 |

## 返回值说明

返回申请得到的aclTensor，申请失败则返回nullptr。

## 约束说明

入参指针不能为空。

## 调用示例

```cpp
// 申请一个host侧tensor，并将myArray中的数据拷贝到tensor中
void Func(aclOpExecutor *executor) {
    int64_t myArray[10];
    aclTensor *tensor = executor->AllocHostTensor(myArray, 10, DT_INT64);
}
```
