# INFER\_SHAPE

## Function

Runs `InferShape` for a specified operator to infer its output shape.

## Prototype

```cpp
INFER_SHAPE(KERNEL_NAME, op_args...)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| KERNEL_NAME | Input| Operator name, for example, `Add`.|
| op_args... | Input| Operator parameters, including [OP_INPUT](OP_INPUT.md), [OP_OUTPUT](OP_OUTPUT.md), and [OP_ATTR](OP_ATTR.md).|

## Restrictions

- If an operator requires INFER\_SHAPE, this macro must be called before [ADD\_TO\_LAUNCHER\_LIST\_AICORE](ADD_TO_LAUNCHER_LIST_AICORE.md).
- The following are the APIs that will be called by the macro definition.

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
// Call INFER_SHAPE to infer the output shape of the BatchMatMul operator. BatchMatMulV3 is the operator name, OP_INPUT is the operator input parameter, OP_OUTPUT is the operator output parameter, and OP_ATTR is the operator attribute parameter.
INFER_SHAPE(BatchMatMulV3, OP_INPUT(x1, x2, bias, nullptr), OP_OUTPUT(bmmOut), OP_ATTR(adjX1, adjX2, offsetX, opImplModeEnum));
```
