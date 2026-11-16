# L2\_DFX\_PHASE\_1

## Function

Collects latency statistics on and prints input parameters of the L2 first-phase API `aclnn_Xxx_GetWorkspaceSize`. It must be called at the beginning of the first-phase API.

## Prototype

```cpp
L2_DFX_PHASE_1(APIName, IN, OUT)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| APIName | Input| Name of the L2 API on the host side, for example, aclnn*Xxx*.|
| IN | Input| Input parameters of an operator. It is encapsulated by [DFX_IN](DFX_IN.md).|
| OUT | Input| Output parameters of an operator. It is encapsulated by [DFX_OUT](DFX_OUT.md).|

## Restrictions

- This macro must be called at the beginning of the L2 first-phase API. Otherwise, the latency statistics may be inaccurate. The input parameters must be strictly consistent with those of the L2 API.
- The following are the APIs that will be called by the macro definition.

    ```text
    #define DFX_IN(...) std::make_tuple(__VA_ARGS__)
    #define DFX_OUT(...) std::make_tuple(__VA_ARGS__)
    ```

## Examples

```cpp
// Collect latency statistics on and print input parameters of the L2 first-phase API for the abs operator. self is the input of the abs operator, and out is the output of the abs operator.
L2_DFX_PHASE_1(aclnnAbs, DFX_IN(self), DFX_OUT(out));
```
