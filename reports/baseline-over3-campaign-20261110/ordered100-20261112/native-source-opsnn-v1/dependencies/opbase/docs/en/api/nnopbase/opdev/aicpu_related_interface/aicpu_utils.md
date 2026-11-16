# aicpu\_uitls

The APIs described in this section are reserved and may be changed or deprecated in the future. Therefore, you are not advised to pay attention to or use these APIs.

**Table 1** APIs

| API Definition| Description|
| --- | --- |
| RecordAicpuTime(const size_t index) | Current timestamp when an operator task is executed.|
| lk(gTimeStampMutex) | Mutex of the timestamp structure.|
| clock_gettime(CLOCK_MONOTONIC, &(gAicpuTimeStamp.tp[index])) | A system function used to obtain the current system timestamp.|
