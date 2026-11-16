# Reserved APIs

The APIs described in this section are reserved and may be changed or deprecated in the future. Therefore, you are not advised to pay attention to or use these APIs.

**Table 1** API list

| API| Description|
| --- | --- |
| ToFormat(const std::string &formatStr) | Converts a string to `Format`.|
| ToString(Format format) | Converts `Format` to a string.|
| GetSubFormat(int32_t format) | Obtains the `SubFormat` of a specified `Format`.|
| GetSubFormat(Format format) | Obtains the `SubFormat` of a specified `Format`.|
| HasSubFormat(int32_t format) | Checks if a `Format` has a `SubFormat`.|
| GetC0Format(int32_t format) | Obtains the `C0Format` of a specified `Format`.|
| HasC0Format(int32_t format) | Checks if a `Format` has a `C0Format`.|
| GetFormatFromSub(int32_t primaryFormat, int32_t subFormat) | Generates a `Format` using the specified `PrimaryFormat` and `SubFormat`.|
