# CREATE\_EXECUTOR

## Function

Creates a `UniqueExecutor` object, which is the factory class of `aclOpExecutor`.

## Prototype

```cpp
CREATE_EXECUTOR()
```

## Parameters

None

## Restrictions

- This macro must be called before `aclOpExecutor` is used.
- Before the L2 first-phase API returns, ReleaseTo must be called to pass executor to executor of the L2 API input parameter.

## Examples

```cpp
// Create an executor.
auto uniqueExecutor = CREATE_EXECUTOR();
```
