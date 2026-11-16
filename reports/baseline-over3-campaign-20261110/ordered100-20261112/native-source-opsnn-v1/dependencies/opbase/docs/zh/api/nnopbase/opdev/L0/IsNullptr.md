# IsNullptr

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：不支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas 推理系列产品</term>：支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas 训练系列产品</term>：支持
<!-- end id6 -->

## 功能说明

判断输入的指针是否为空。若为空指针返回true、并打印错误日志，否则返回false。

## 函数原型

```cpp
static inline bool IsNullptr(const aclTensor *tensor, const char *name)
```

```cpp
static inline bool IsNullptr(const aclTensorList *tensorList, const char *name)
```

```cpp
static inline bool IsNullptr(const aclScalar *scalar, const char *name)
```

```cpp
static inline bool IsNullptr(const aclIntArray *intArr, const char *name)
```

```cpp
static inline bool IsNullptr(const aclBoolArray *boolArr, const char *name)
```

```cpp
static inline bool IsNullptr(const aclFloatArray *floatArr, const char *name)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| tensor | 输入 | 需要被检查的指针，支持aclTensor *、aclTensorList*、aclScalar *、aclIntArray*、aclBoolArray *、aclFloatArray*类型。 |
| name | 输入 | 被检查的指针的一个标识，如果被检查指针为空，则打印的错误日志里会输出此标识（name）。 |

## 返回值说明

返回bool类型，如果指针被判断为nullptr，返回true，否则返回false。

## 约束说明

无

## 调用示例

```cpp
#define OP_CHECK_NULL(param, retExpr) \
  if (IsNullptr(param, #param)) { \
    retExpr; \
  }
```
