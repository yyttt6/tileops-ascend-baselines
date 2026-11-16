# 预留接口

本章接口为预留接口，后续有可能变更或废弃，不建议开发者使用，开发者无需关注。

**表 1**  接口列表

| 接口名称 | 功能说明 |
| --- | --- |
| ToDataType(const std::string &dataTypeStr) | 将给定的字符串转换为op::DataType（即ge::DataType）。 |
| CalcShapeBytes(int64_t size, DataType dataType, bool ceil=false) | 用给定的shape大小和dtype计算总大小。 |
