# GetCurrentPlatformInfo

## 功能说明

获取NPU平台信息，包括AI处理器型号、参数等信息。

## 函数原型

```cpp
const PlatformInfo &GetCurrentPlatformInfo()
```

## 参数说明

无

## 返回值说明

返回PlatformInfo的引用。

## 约束说明

无

## 调用示例

```cpp
// 获取当前平台信息
void func() {
    const PlatformInfo npuInfo = GetCurrentPlatformInfo();
}
```
