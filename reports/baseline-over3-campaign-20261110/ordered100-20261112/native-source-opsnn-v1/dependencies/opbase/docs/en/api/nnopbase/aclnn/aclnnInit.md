# aclnnInit

## Function

Initializes aclnnXxx in the single-operator API execution framework. Before calling such APIs, you must initialize aclnn resources (such as reading environment variables, parsing configuration files, and loading the resource library). Otherwise, an internal system error occurs, affecting service running.

> **Note**:
>Either aclnnInit or aclInit can be called to initialize resources. The difference is that aclnnInit initializes only aclnn-related resources, while aclInit initializes aclnn and other resources in the acl API. Therefore, aclnnInit is more lightweight than aclInit. If both APIs are called, no failure message is returned.

## Prototype

```cpp
aclnnStatus aclnnInit(const char *configPath)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| configPath | Input| Path (including the file name) of the aclnn initialization configuration file. You can enable the debugging capability of the aclnn API through this configuration. The default value is `NULL`.<br>The configuration file must be in JSON format. For example, the value of `configPath` is `/home/acl.json`.|

The following is a configuration example of `acl.json`:

```json
{   
    "op_debug_config":{
        "enable_debug_kernel":"on"
    }
}
```

`enable_debug_kernel` supports the following values:

- `on`: enables the debugging capability of the aclnn API. That is, during operator execution, the system checks whether the global memory overflows and whether the internal pipeline is synchronized.
- `off`: disables the debugging capability of the aclnn API. The default value is `off`.

## Returns

0 on success; else, failure. For details about the return codes, see [Common API Return Codes](common_api_return_codes.md).

## Restrictions

- This API must be used together with [aclnnFinalize](aclnnFinalize.md) to initialize and deinitialize aclnn resources.
- aclnnInit can be called only once in a process.

## Examples

The following code is for reference only and should not be copied directly for execution:

```cpp
// Initialize resources.
auto ret = aclnnInit("/home/acl.json");
...
// Create an operator API parameter object.
ret = aclCreate***(...);
...
// Call the two-phase operator APIs.
ret = aclnnXxxGetWorkspaceSize(...);
ret = aclnnXxx(...);
...
// Destroy the operator API parameter object.
ret = aclDestroy***();
...
// Deinitialize resources.
ret = aclnnFinalize();
```
