# Reserved APIs

The APIs described in this section are reserved and may be changed or deprecated in the future. Therefore, you are not advised to pay attention to or use these APIs.

**Table 1** APIs

| API| Description|
| --- | --- |
| ToDataType(const std::string &dataTypeStr) | Converts the given string to `op::DataType` (`ge::DataType`).|
| CalcShapeBytes(int64_t size, DataType dataType, bool ceil=false) | Calculates the total size based on the given shape size and data type.|
| CheckType(const DataType dtype, const std::initializer_list\<DataType\> &valid_types) | Checks whether the data type is in the specified list.|
