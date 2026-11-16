# AllocScalar

## Function

Allocates an `aclScalar` and assigns it a value. Multiple overloaded functions are used to support multiple data types.

## Prototype

```cpp
aclScalar *AllocScalar(const void *data, op::DataType dataType)
```

```cpp
aclScalar *AllocScalar(float value)
```

```cpp
aclScalar *AllocScalar(double value)
```

```cpp
aclScalar *AllocScalar(fp16_t value)
```

```cpp
aclScalar *AllocScalar(bfloat16 value)
```

```cpp
aclScalar *AllocScalar(int32_t value)
```

```cpp
aclScalar *AllocScalar(int64_t value)
```

```cpp
aclScalar *AllocScalar(int16_t value)
```

```cpp
aclScalar *AllocScalar(int8_t value)
```

```cpp
aclScalar *AllocScalar(uint32_t value)
```

```cpp
aclScalar *AllocScalar(uint64_t value)
```

```cpp
aclScalar *AllocScalar(uint16_t value)
```

```cpp
aclScalar *AllocScalar(uint8_t value)
```

```cpp
aclScalar *AllocScalar(bool value)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| data | Input| Pointer to the source data.|
| dataType | Input| Data type of the source data.|
| value | Input| Value of `aclScalar`.|

## Returns

Success: allocated `aclScalar`. Failure: `nullptr`.

## Restrictions

The input pointer cannot be null.

## Examples

```cpp
// Initialize an aclScalar with the value 5 and a data type of int64.
void Func(aclOpExecutor *executor) {
    int64_t val = 5;
    aclScalar *scalar = executor->AllocScalar(val);
    scalar = executor->AllocScalar(&val, DT_INT64);
}
```
