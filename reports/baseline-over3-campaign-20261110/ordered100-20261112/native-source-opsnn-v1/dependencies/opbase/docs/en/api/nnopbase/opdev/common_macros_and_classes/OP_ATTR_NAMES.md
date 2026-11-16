# OP\_ATTR\_NAMES

## Function

Encapsulates attribute names of an AI CPU operator.

## Prototype

```cpp
OP_ATTR_NAMES::op::FVector<std::string>
```

## Parameters

| Parameter| Input/Output| Description|
| --- | --- | --- |
| / | Input| Attribute names of an AI CPU operator, represented as an `FVector` of type `std::string`. The default value is an empty `FVector`, that is, `OP_ATTR_NAMES()`.|

## Restrictions

None

## Examples

```cpp
// Encapsulate three attribute names of an AI CPU operator: Tindices, T, and use_locking.
OP_ATTR_NAMES({"Tindices", "T", "use_locking"});
```
