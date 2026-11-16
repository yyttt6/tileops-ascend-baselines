# aclnnInit

## 功能说明

单算子API执行框架中aclnnXxx接口的初始化函数，在调用该类算子接口前必须先进行aclnn相关资源的初始化（如读取环境变量、配置文件、加载资源库等），否则会导致系统内部出错，影响业务正常运行。

> **说明**： 
>调用aclnnInit或aclInit接口，均能实现资源初始化，二者区别在于aclnnInit仅完成aclnn相关资源初始化，而aclInit完成acl接口中各种资源（包含aclnn）的初始化。因此aclnnInit相对于aclInit更轻量一些。若两个接口都调用，也不返回失败。

## 函数原型

```cpp
aclnnStatus aclnnInit(const char *configPath)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| configPath | 输入 | aclnn的初始化配置文件所在路径（包含文件名），开发者可通过此配置开启aclnn接口的debug调试能力。默认为NULL。<br>配置文件需要为json格式，例如configPath的取值为“/home/acl.json”。 |

acl.json的配置示例如下：

```json
{   
    "op_debug_config":{
        "enable_debug_kernel":"on"
    }
}
```

配置项“enable_debug_kernel”支持的取值如下：

- on：开启aclnn接口的debug调试能力，即算子在执行过程中会检测Global Memory是否内存越界，内部流水线是否同步等操作。
- off：不开启aclnn接口的debug调试能力。 默认值为off。

## 返回值说明

返回0表示成功，返回其他值表示失败，返回码列表参见[公共接口返回码](public_interface_return_code.md)。

## 约束说明

- 本接口需与[aclnnFinalize](aclnnFinalize.md)接口配套使用，分别完成aclnn资源初始化与去初始化。
- 一个进程内只允许调用一次aclnnInit接口，不支持重复调用。

## 调用示例

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```cpp
// 资源初始化
auto ret = aclnnInit("/home/acl.json");
...
// 创建算子接口参数对象
ret = aclCreate***(...);
...
// 调用算子两段式接口
ret = aclnnXxxGetWorkspaceSize(...);
ret = aclnnXxx(...);
...
// 销毁算子接口参数对象
ret = aclDestroy***();
...
// 资源去初始化
ret = aclnnFinalize();
```
