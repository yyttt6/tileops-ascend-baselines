# opbase

## 🔥Latest News

- [2025/12] Supported KirinX90 and offline compilation.
- [2025/09] Released the opbase project for the first time.

## 🚀 Overview

opbase is a basic framework library that the Compute Architecture for Neural Networks ([CANN](https://hiascend.com/software/cann)) operator library depends on. It provides basic scheduling capabilities and common dependencies, including common header files, structures, and scheduling frameworks. For details about the CANN operator library, visit [ops-math](https://gitcode.com/cann/ops-math), [ops-nn](https://gitcode.com/cann/ops-nn), [ops-cv](https://gitcode.com/cann/ops-cv), and [ops-transformer](https://gitcode.com/cann/ops-transformer) to obtain the operator source code.

<img src="docs/zh/figures/architecture.png" alt="Architecture " width="700px" height="320px">

## 📌 Version Mapping

The source code of this project is released with the CANN software version. For details about the mapping between the CANN software version and the project tag, see the version description in the [release repository](https://gitcode.com/cann/release-management).
To ensure smooth source code customization, select the matching CANN version and GitCode tag source code. Using the master branch may cause version mismatch.

## 🛠️ Environment Setup

Before using this project, you need to install the NPU driver, CANN package, and other required components. For details, see [Environment Deployment](docs/en/install/quick_install.md).

## ⬇️ Source Code Download

After the environment is set up, run the command below to download the branch source code that matches the CANN version. Replace \$\{tag\_version\} with the branch tag name.

> Note: If the matching branch source code already exists in the environment, **skip this step**. For example, CANNLab provides the source code for the latest commercial CANN version by default.

```bash
git clone -b ${tag_version} https://gitcode.com/cann/opbase.git
```

## 📖 Tutorials

- [Quick Start](docs/QUICKSTART.md): You can quickly experience project source code build and local verification (such as UT) from scratch.
- [Advanced Tutorials](docs/README.md): To learn more about the framework APIs, guidelines, and other information required for operator calling and development, visit the Documentation Center.

## 📝 Related Information

- Directory Structure
- [Contributions](CONTRIBUTING.md)
- [Security Statement](SECURITY.md)
- [Licenses](LICENSE)
- [SIG](https://gitcode.com/cann/community/tree/master/CANN/sigs/ops-basic)

------

This project's functionality and documentation are under continuous development and refinement. We invite you to follow the latest releases.

- **Feedback**: Submit queries or report bugs via GitCode [Issues](https://gitcode.com/cann/opbase/issues).
- **Interaction**: Participate in [discussion](https://gitcode.com/cann/opbase/discussions) on GitCode.
