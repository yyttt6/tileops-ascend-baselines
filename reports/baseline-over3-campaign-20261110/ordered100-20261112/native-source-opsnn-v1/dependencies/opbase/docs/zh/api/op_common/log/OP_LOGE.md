# OP\_LOGE

## 功能说明

打印算子ERROR级别日志。

## 函数原型

```cpp
OP_LOGE(opName, ...)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| opName | 输入 | 待打印对象，可以是算子名，表示打印算子信息；也可以是某个函数名，表示打印函数内相关信息。支持const char*或std::string类型。 |

## 返回值说明

无

## 约束说明

无

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```cpp
bool BroadcastDim(int64_t& dim1, const int64_t dim2)
{
    if ((dim1 != 1) && (dim2 != 1)) {
        OP_LOGE("BroadcastDim", "%ld and %ld cannot broadcast!", dim1, dim2);
        return false;
    }

    return true;
}
```
