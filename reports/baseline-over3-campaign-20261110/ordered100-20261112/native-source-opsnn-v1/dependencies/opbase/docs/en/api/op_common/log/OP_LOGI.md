# OP\_LOGI

## Function

Prints operator INFO logs.

## Prototype

```cpp
OP_LOGI(opName, ...)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| opName | Input| Object to be printed. It can be an operator name, indicating that the operator information will be printed, or a function name, indicating that the information related to the function will be printed. The value can be either a `const char*` or a `std::string`.|

## Returns

None

## Restrictions

None

## Examples

The following code is for reference only and should not be copied directly for execution:

```cpp
bool BroadcastDim(int64_t& dim1, const int64_t dim2)
{
    if ((dim1 != 1) && (dim2 != 1)) {
        OP_LOGI("BroadcastDim", "%ld and %ld cannot broadcast!", dim1, dim2);
        return false;
    }

    return true;
}
```
