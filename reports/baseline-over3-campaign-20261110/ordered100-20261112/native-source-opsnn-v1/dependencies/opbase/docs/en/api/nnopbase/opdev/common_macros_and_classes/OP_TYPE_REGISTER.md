# OP\_TYPE\_REGISTER

## Function

Registers an L0 operator and generates its operator name and `TypeID`.

## Prototype

```cpp
OP_TYPE_REGISTER(kernelName)
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| kernelName | Input| Operator name, for example, *Abs*.|

## Restrictions

This macro must be used at the beginning of an L0 API.

## Examples

```cpp
namespace l0op{
OP_TYPE_REGISTER(Abs);
...
}
```
