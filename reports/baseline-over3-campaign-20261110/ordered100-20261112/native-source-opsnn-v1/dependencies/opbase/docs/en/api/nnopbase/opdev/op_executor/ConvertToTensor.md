# ConvertToTensor

## Function

Converts host-side data of various data types to host-side `aclTensor` objects.

## Prototype

```cpp
const aclTensor *ConvertToTensor(const aclIntArray *value, DataType dataType)
```

```cpp
const aclTensor *ConvertToTensor(const aclBoolArray *value, DataType dataType)
```

```cpp
const aclTensor *ConvertToTensor(const aclFloatArray *value, DataType dataType)
```

```cpp
const aclTensor *ConvertToTensor(const aclFp16Array *value, DataType dataType)
```

```cpp
const aclTensor *ConvertToTensor(const aclBf16Array *value, DataType dataType)
```

```cpp
const aclTensor *ConvertToTensor(const aclScalar *value, DataType dataType)
```

```cpp
template<typename T> 
const aclTensor *ConvertToTensor(const T *value, uint64_t size, DataType dataType)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| value | Input| Source data on the host side.|
| size | Input| Number of elements in the source data.|
| dataType | Input| Target data format encapsulated in `aclTensor`.|

## Returns

Converted tensors on the host side.

## Restrictions

The input pointer cannot be null.

### Data type restrictions on parameters of each API

#### API 1: `ConvertToTensor(const aclIntArray *value, DataType dataType)`

| Parameter| Data Type|
| --- | --- |
| value | `aclIntArray`, `int64_t` data type|
| dataType | DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, and DT_DOUBLE|

#### API 2: `ConvertToTensor(const aclBoolArray *value, DataType dataType)`

| Parameter| Data Type|
| --- | --- |
| value | `aclBoolArray`, `bool` data type|
| dataType | DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, and DT_DOUBLE|

#### API 3: `ConvertToTensor(const aclFloatArray *value, DataType dataType)`

| Parameter| Data Type|
| --- | --- |
| value | `aclFloatArray`, `float` data type|
| dataType | DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, and DT_DOUBLE|

#### API 4: `ConvertToTensor(const aclFp16Array *value, DataType dataType)`

| Parameter| Data Type|
| --- | --- |
| value | `aclFp16Array`, `fp16_t` data type|
| dataType | DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, and DT_DOUBLE|

#### API 5: `ConvertToTensor(const aclBf16Array *value, DataType dataType)`

| Parameter| Data Type|
| --- | --- |
| value | `aclBf16Array`, `bfloat16` data type|
| dataType | DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, and DT_DOUBLE|

#### API 6: `ConvertToTensor(const T *value, uint64_t size, DataType dataType)`

| Parameter| Data Type|
| --- | --- |
| value | The template parameter T can be int64_t, uint64_t, int32_t, uint32_t, int8_t, uint8_t, int16_t, uint16_t, float, double, bool, char, op::bfloat16, or op::fp16_t.|
| size | A `uint64_t` value to indicate the number of elements|
| dataType | DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, and DT_DOUBLE|

#### API 7: `ConvertToTensor(const aclScalar *value, DataType dataType)`

| Parameter| Data Type|
| --- | --- |
| value | `aclScalar`: DT_BOOL, DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64, DT_FLOAT, DT_DOUBLE, DT_FLOAT16, DT_BF16, DT_COMPLEX64, DT_COMPLEX128, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT8_E8M0, DT_FLOAT6_E3M2, DT_FLOAT6_E2M3, DT_FLOAT4_E2M1, DT_FLOAT4_E1M2, and DT_HIFLOAT8|
| dataType | DT_BOOL, DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64, DT_FLOAT, DT_DOUBLE, DT_FLOAT16, DT_BF16, DT_COMPLEX64, DT_COMPLEX128, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT8_E8M0, DT_FLOAT6_E3M2, DT_FLOAT6_E2M3, DT_FLOAT4_E2M1, DT_FLOAT4_E1M2, and DT_HIFLOAT8.|

## Examples

```cpp
// Convert an aclScalar and an int64_t to tensors on the host side.
void Func(aclOpExecutor *executor) {
    int64_t val = 5;
    aclScalar *scalar = executor->AllocScalar(val);
    const aclTensor *tensor = executor->ConvertToTensor(scalar, DT_INT64);
    tensor = executor->ConvertToTensor(&val, 1, DT_INT64);
}
```
