# latest_manager模块说明

## 总体说明

latest_manager用于解决多版本latest目录下版本兼容性问题。

## 兼容分析

latest_manager存在3种兼容关系：

1. latest_manager对版本脚本兼容
2. 版本数据对latest_manager兼容
3. latest_manager新版本对老版本兼容

### latest_manager对版本脚本兼容

版本脚本会调用latest_manager的manager.sh脚本，向latest_manager发送消息。
manager.sh脚本需兼容老版本脚本的消息格式。需要保证在兼容周期内只新增参数，不删除参数。

### 版本数据对latest_manager兼容

latest_manager会读取版本包目录下文件：（这些文件的格式需要保证兼容性）

1. ascend_install.info
2. version.info
3. script/filelist.csv

会调用版本包目录下文件：（这些文件的调用方式需要保证兼容性）

1. script/[package]_custom_create_softlink.sh
2. script/[package]_custom_remove_softlink.sh

会间接调用版本包目录下文件：（这些文件的调用方式需要保证兼容性）

1. bin/prereq_check.*
2. bin/setenv.*

### latest_manager新版本对老版本兼容

1. 数据兼容
老版本latest_manager升级到新版本时，需要将老版本latest_manager**生成的数据**，转换到新版本上。
比如新版本将老版本的version.cfg修改为version.json，那么在升级版本时，需要做数据的自动迁移。

2. 功能兼容
latest_manager调用版本脚本的函数，新版本函数需要兼容老版本函数功能。

这样，对latest_manager的代码管理提出要求：

1. latest_manager生成新数据时，要经过设计，评审。只要生成新数据，就要处理兼容性。
2. 需要记录latest_manager生成数据的历史版本，以便处理数据迁移。
3. latest_manager调用版本脚本的函数，功能只增不减。
