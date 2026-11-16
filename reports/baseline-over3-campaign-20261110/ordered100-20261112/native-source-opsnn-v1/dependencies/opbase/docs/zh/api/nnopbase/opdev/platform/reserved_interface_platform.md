# 预留接口

本章接口为预留接口，后续有可能变更或废弃，不建议开发者使用，开发者无需关注。

**表 1**  接口列表

| 接口定义 | 功能说明 |
| --- | --- |
| PlatformInfo() | PlatformInfo的构造函数。 |
| PlatformInfo(int32_t deviceId) | PlatformInfo的构造函数。 |
| PlatformInfo(const PlatformInfo &other) | PlatformInfo拷贝构造函数。 |
| PlatformInfo(const PlatformInfo &&other) | PlatformInfo移动构造函数。 |
| SetPlatformImpl(PlatformInfoImpl *impl) | 设置PlatformInfo的PlatformInfoImpl对象。 |
| GetDeviceId() | 从PlatformInfo中获取deviceId。 |
| CheckSupport(SocSpec socSpec, SocSpecAbility ability) | 从PlatformInfo中获取socSpec。 |
| GetCubeCoreNum() | 从PlatformInfo中获取cube核数。 |
| GetVectorCoreNum() | 从PlatformInfo中获取vector核数。 |
| Valid() | 判断PlatformInfo是否valid。 |
| GetPlatformInfos() | 获取NPU平台信息，包括AI处理器型号、参数等信息。 |
| ToString(SocVersion socVersion) | 将SocVersion转换为字符串。 |
| GetFftsPlusMode() | 判断是否支持ffts plus模式。 |
