# GetBlockSize

## 功能说明

获取当前NPU平台上一个data block内存大小（单位Byte）。

## 函数原型

```cpp
int64_t GetBlockSize()
```

## 参数说明

无

## 返回值说明

返回int64\_t。

## 约束说明

无

## 调用示例

```cpp
// 获取当前NPU平台上一个data block所占的内存大小
void func() {
    int64_t size = GetCurrentPlatformInfo().GetBlockSize();
}
```
