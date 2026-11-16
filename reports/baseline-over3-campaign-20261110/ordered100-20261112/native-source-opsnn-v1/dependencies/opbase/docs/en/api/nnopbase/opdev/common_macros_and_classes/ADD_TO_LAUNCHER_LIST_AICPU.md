# ADD\_TO\_LAUNCHER\_LIST\_AICPU

## Function

Creates an execution task of an AI CPU operator and places the task in the aclOpExecutor execution queue. The task is executed when the second-phase API aclnn_Xxx_ is called.

## Prototype

```cpp
ADD_TO_LAUNCHER_LIST_AICPU(KERNEL_NAME, attrNames, opArgs...)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| KERNEL_NAME | Input| Operator name, for example, `Add`.|
| attrNames | Input| Operator attribute name. For details, see [OP_ATTR_NAMES](OP_ATTR_NAMES.md).|
| opArgs... | Input| Operator parameters.|

## Restrictions

- If the operator requires `INFER_SHAPE`, this macro must be called after [INFER_SHAPE](INFER_SHAPE.md).
- The following APIs are called by the preceding macro definitions:

    ```text
    The following APIs are called by the preceding macro definitions:
    OP_ATTR_NAMES
    OP_INPUT(x...)
    OP_OUTPUT(x...)
    OP_ATTR(x...)
    ```

## Examples

```cpp
// Call ADD_TO_LAUNCHER_LIST_AICPU to create an execution task for the IndexPut operator. IndexPut is the operator name, accumulate is the operator attribute name and parameter, selfRef, values, masks, and indices are the operator input parameters, and out is the operator output parameter.
ADD_TO_LAUNCHER_LIST_AICPU(IndexPut, OP_ATTR_NAMES({"accumulate"}), OP_INPUT(selfRef, values, masks, indices), OP_OUTPUT(out), OP_ATTR(accumulate));
```
