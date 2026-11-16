# opbase

## 🔥Latest News

- [2025/12] 支持kirinx90，支持离线编译。
- [2025/09] opbase项目首次上线。

## 🚀概述

opbase是[CANN](https://hiascend.com/software/cann) （Compute Architecture for Neural Networks）算子库依赖的基础框架库，其提供基础的调度能力和公共依赖项，包括公共的头文件、结构体、调度框架等。关于CANN算子库请访问[ops-math](https://gitcode.com/cann/ops-math)、[ops-nn](https://gitcode.com/cann/ops-nn)、[ops-cv](https://gitcode.com/cann/ops-cv)、[ops-transformer](https://gitcode.com/cann/ops-transformer)获取算子源码实现详情。

<img src="docs/zh/figures/architecture.png" alt="架构图"  width="700px" height="320px">

## 📌版本配套

本项目源码会跟随CANN软件版本发布，关于CANN软件版本与本项目标签的对应关系请参阅[release仓库](https://gitcode.com/cann/release-management)中的相应版本说明。
请注意，为确保您的源码定制开发顺利进行，请选择配套的CANN版本与Gitcode标签源码，使用master分支可能存在版本不匹配的风险。

## 🛠️环境准备

[环境部署](docs/zh/install/quick_install.md)是体验本项目能力的前提，请先完成NPU驱动、CANN包安装等，确保环境正常。

## ⬇️源码下载

环境准备好后，下载与CANN版本配套的分支源码，命令如下，\$\{tag\_version\}替换为分支标签名。

> 说明：若环境中已存在配套分支源码，**可跳过本步骤**。

```bash
git clone -b ${tag_version} https://gitcode.com/cann/opbase.git
```

## 📖学习教程

- [快速入门](docs/QUICKSTART.md)：从零开始快速体验项目源码构建和功能本地验证（如UT）操作。
- [进阶教程](docs/README.md)：如需深入了解算子调用/开发场景下依赖的框架API、指南等，请查阅文档中心获取详细指引。

## 💬相关信息

- [目录结构](docs/zh/appendix/dir_structure.md)
- [贡献指南](CONTRIBUTING.md)
- [安全声明](SECURITY.md)
- [许可证](LICENSE)
- [所属SIG](https://gitcode.com/cann/community/tree/master/CANN/sigs/ops-basic)

------

本项目功能和文档正在持续更新和完善中，欢迎您关注最新版本。

- **问题反馈**：通过GitCode[【Issues】](https://gitcode.com/cann/opbase/issues)提交问题
- **社区互动**：通过GitCode[【讨论】](https://gitcode.com/cann/opbase/discussions)参与交流
