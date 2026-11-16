# OpExecMode

OpExecMode枚举类表示算子的运行模式，具体定义如下：

```cpp
enum class OpExecMode : uint32_t {
    // ExecMode support OR operation
    OP_EXEC_MODE_DEFAULT = 0,
    OP_EXEC_MODE_HF32T = 1,
    OP_EXEC_MODE_RESERVED = 0xFFFFFFFF
}
```
