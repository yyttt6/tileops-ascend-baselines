# IsPrivateFormat

## Function

Checks whether the input format is a private format.

## Prototype

```cpp
bool IsPrivateFormat(Format format)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| format | Input| Data format to be checked. The data type is `op::Format` (`ge::Format`).|

## Returns

If format is a private format, true is returned. Otherwise, false is returned.

## Restrictions

None

## Examples

```cpp
// Check whether the storage format of the input is private. If yes, a return statement is executed.
void Func(const aclTensor *input) {
    if (IsPrivateFormat(input->GetStorageFormat())) {
        return;
    }
}
```
