# ADD\_TO\_LAUNCHER\_LIST\_AICORE

## Function

Creates an execution task of an AI Core operator and places the task in the aclOpExecutor execution queue. The task is executed when the second-phase API aclnn_Xxx_ is called.

## Prototype

```cpp
ADD_TO_LAUNCHER_LIST_AICORE(KERNEL_NAME, op_args...)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| KERNEL_NAME | Input| Operator name, for example, `Add`.|
| op_args... | Input| Operator parameters, including [OP_INPUT](OP_INPUT.md), [OP_OUTPUT](OP_OUTPUT.md), and [OP_ATTR](OP_ATTR.md).|

## Restrictions

- If the operator requires `INFER_SHAPE`, this macro must be called after [INFER_SHAPE](INFER_SHAPE.md).
- The following APIs are called by the preceding macro definitions:

    ```text
    OP_INPUT(x...)
    OP_OUTPUT(x...)
    OP_ATTR(x...)
    OP_WORKSPACE(x...)
    OP_OUTSHAPE(x...)
    OP_OPTION(x...)
    OP_EMPTY_ARG
    OP_MODE(x...)
    ```

## Examples

```cpp
// Call ADD_TO_LAUNCHER_LIST_AICORE to create an execution task for the add operator. Add is the operator name, self and other are operator input parameters, and addOut is the operator output parameter.
ADD_TO_LAUNCHER_LIST_AICORE(Add, OP_INPUT(self, other), OP_OUTPUT(addOut));
```
