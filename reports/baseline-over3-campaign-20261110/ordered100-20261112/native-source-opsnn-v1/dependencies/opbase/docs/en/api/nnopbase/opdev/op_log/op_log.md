# op\_log

The APIs described in this section are reserved and may be changed or deprecated in the future. Therefore, you are not advised to pay attention to or use these APIs.

**Table 1** API list

| API Definition| Description|
| --- | --- |
| GetTid() | Obtains a thread ID.|
| CheckLogLevelInner(int32_t moduleId, int32_t logLevel) | Checks the log level.|
| DlogRecordInner(int32_t moduleId, int32_t level, const char *fmt, ...) | Prints logs.|
| ReportErrorMessageInner(const std::string &code, const char *fmt, ...) | Reports an internal error message.|
| ReportErrorMessage(const char \*code, const char \*fmt, Arguments &&... args) | Reports an error message.|
| GetLogApiInfo() | Obtains API information.|
| GetFileName(const char *path) | Obtains a file name.|
| GetOpName() | Obtains an operator name.|
