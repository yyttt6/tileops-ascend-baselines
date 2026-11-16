# Reserved APIs

The APIs described in this section are reserved and may be changed or deprecated in the future. Therefore, you are not advised to pay attention to or use these APIs.

**Table 1** APIs

| API| Description|
| --- | --- |
| PlatformInfo() | Works as the constructor of `PlatformInfo`.|
| PlatformInfo(int32_t deviceId) | Works as the constructor of `PlatformInfo`.|
| PlatformInfo(const PlatformInfo &other) | Works as the constructor of `PlatformInfo` for copying data.|
| PlatformInfo(const PlatformInfo &&other) | Works as the constructor of `PlatformInfo` for moving data.|
| SetPlatformImpl(PlatformInfoImpl *impl) | Sets the `PlatformInfoImpl` object of `PlatformInfo`.|
| GetDeviceId() | Obtains `deviceId` from `PlatformInfo`.|
| CheckSupport(SocSpec socSpec, SocSpecAbility ability) | Obtains `socSpec` from `PlatformInfo`.|
| GetCubeCoreNum() | Obtains the number of Cube Cores from `PlatformInfo`.|
| GetVectorCoreNum() | Obtains the number of Vector Cores from `PlatformInfo`.|
| Valid() | Checks whether `PlatformInfo` is valid.|
| GetPlatformInfos() | Obtains NPU platform information, including the AI processor model and parameters.|
| ToString(SocVersion socVersion) | Converts `SocVersion` to a string.|
| GetFftsPlusMode() | Determines whether the ffts plus mode is supported.|
