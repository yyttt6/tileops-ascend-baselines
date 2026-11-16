# GetPrimaryFormat

## Function

Obtains the root format, during which the C0 information is removed and only the format information is retained. For example, converts FORMAT\_FRACTAL\_Z\_C04 to FORMAT\_FRACTAL\_Z.

## Prototype

```cpp
int32_t GetPrimaryFormat(int32_t format)
```

```cpp
Format GetPrimaryFormat(Format format)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| format | Input| Format of the original data.|

## Returns

Format after the C0 information is removed.

## Restrictions

None

## Examples

```cpp
// If the storage format of the input is fractal Z, a return statement is executed.
void Func(const aclTensor *input) {
    if (GetPrimaryFormat(input->GetStorageFormat()) == FORMAT_FRACTAL_Z){
        return;
    }
}
```
