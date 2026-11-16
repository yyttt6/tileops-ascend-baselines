# op\_def

The APIs described in this section are reserved and may be changed or deprecated in the future. Therefore, you are not advised to pay attention to or use these APIs.

**Table 1** API list

| API Definition| Description|
| --- | --- |
| ToOpImplMode(const std::string &implModeStr) | Converts `implModeStr` to `OpImplMode`.|
| ToString(OpImplMode implMode) | Converts `OpImplMode` to a string.|
| ImplModeToString(OpImplMode implMode) | Converts `implModeStr` to `OpImplMode`.|
| Add(uint32_t &id, const char *opName) | Registers `opName` into `OpTypeDict` and returns the operator ID.|
| ToOpType(const std::string &opName) | Searches for an operator ID in `OpTypeDict` based on `opName`.|
| ToString(uint32_t opType) | Searches for an operator name in `OpTypeDict` based on an operator ID.|
| GetAllOpTypeSize() | Obtains the number of registered operators in `OpTypeDict`.|
| ToOpTypeByConfigJson(const std::string &op_config_json) | Searches for an operator ID based on the operator's JSON file name.|
| UpdateConfigJsonPath(uint32_t opType, const std::string &opFile) | Updates the JSON file name of an operator based on the operator ID.|
| ReadFile2String(const char *filename, std::string &content) | Reads `filename` to `content`.|
| ReadDirBySuffix(const std::string &dir, const std::string &suffix, std::vector\<std::string\> &paths) | Obtains all files with the specified suffix in the `dir` directory.|
| ToIndex(OpImplMode implMode) | Obtains the index of `OpImplMode`.|
| ToIndexChar(OpImplMode implMode) | Obtains the character-form index of `OpImplMode`.|
| GetOpConfigJsonFileName(uint32_t opType) | Obtains the Config JSON file name of an operator based on the operator ID.|
