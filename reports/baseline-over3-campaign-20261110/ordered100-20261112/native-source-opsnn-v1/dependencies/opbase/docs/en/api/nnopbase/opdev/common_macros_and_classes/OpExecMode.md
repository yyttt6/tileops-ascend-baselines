# OpExecMode

The OpExecMode class indicates the run mode of an operator. The definition is as follows:

```cpp
enum class OpExecMode : uint32_t {
    // ExecMode support OR operation
    OP_EXEC_MODE_DEFAULT = 0,
    OP_EXEC_MODE_HF32T = 1,
    OP_EXEC_MODE_RESERVED = 0xFFFFFFFF
};
```
