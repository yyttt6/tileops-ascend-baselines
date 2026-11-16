# Environment Deployment

Before experiencing the features of this project, follow the steps below to complete the basic environment setup and ensure that the NPU driver, firmware, CANN software (`Ascend-cann-toolkit`), and other required components are installed.

## Environment Setup

This project provides multiple methods for setting up the Ascend environment. Select one as needed.

> **Note**: The compile-time and run-time mentioned in this document are described as below. You can use them based on your actual needs.
>
> - Compile-time: If you only need to compile this project and do not need to execute it, you only need to install the CANN toolkit.
> - Runtime: If you need to compile and execute (or only execute) this project, you need to install the driver, firmware, and CANN toolkit.

|  Installation Method |  Description |  Scenario |
| ----- | ------ | ------ |
|  CANNLab  | It is a one-stop development platform that provides an online Ascend environment. You do not need to manually install the environment.<br>Currently, single-node compute is provided. **The latest commercial CANN package is installed by default.**| This method is suitable for developers who do not have Ascend devices.|
|  Docker  | Docker images enable efficient deployment. The CANN package and required dependencies are pre-integrated in images.<br>Currently, this method is applicable only to Atlas A2 series with Ubuntu. **The latest commercial CANN package is installed by default.**|This method is suitable for developers who have Ascend devices and need to quickly set up an environment.|
|  Manual installation | Installing the CANN package and basic dependencies manually provides high flexibility.|This method is suitable for developers who have Ascend devices and want to manually install the CANN package or experience the latest master branch capabilities.|

### Method 1: CANNLab

If you do not have Ascend devices, you can use the CANNLab cloud development environment, a **one-stop operator development platform**. This platform provides an online Ascend environment. The required driver, firmware, software packages, and dependencies have been installed in the environment, and you do not need to manually install them.

> **Note**: By default, the environment comes with the latest commercial CANN package pre-installed. When downloading the source code, ensure that it matches the installed software version. For more information about the development platform, see [LINK](https://gitcode.com/org/cann/discussions/54).

1. Go to the open-source project and click `CANNLab` to log in with your authenticated Huawei Cloud account. If you have not signed up or authenticated, sign up and authenticate as prompted.

    <!-- <img src="../figures/cloudIDE.png" alt="CloudIDE" width="750px" height="85px"> -->

2. Create and start a cloud development environment as prompted. Click `Connect > WebIDE` to access the one-stop operator development platform. By default, the source code of the open-source project is stored in the `/mnt/workspace` directory.

    <!--   <img src="../figures/webIDE.png" alt="WebIDE" width="1000px" height="150px"> -->

### Method 2: Docker

If you have Ascend devices and want to quickly set up an Ascend environment, you can use Docker images for deployment.

> **Note:**
>
> - An image file is large, so it takes some time to download. For details about options of `docker` commands, see `docker --help`.
> - By default, install the latest commercial CANN package in the environment. When downloading the source code, ensure that it matches the software version.

1. **Install the driver and firmware (runtime dependencies).**

    For details about how to download and install the Ascend driver and firmware on the host, see "Software Package Preparation" and "NPU Driver and Firmware Installation" in [CANN Software Installation Guide](https://www.hiascend.com/document/redirect/CannCommunityInstWizard). The driver and firmware are runtime dependencies. If you only need to compile operators, installation is not required.

2. **Download an image.**

    - Step 1: Log in to the host as the **root** user. Ensure that the Docker engine (version 1.11.2 or later) has been installed on the host.
    - Step 2: Pull an image that has integrated the CANN software package and dependencies required by `opbase` from the [Ascend image repository](https://www.hiascend.com/developer/ascendhub/detail/17da20d1c2b6493cb38765adeba85884). You can select a command based on the architecture.

    ```bash
    # Example: Pulling a CANN development image of the Arm architecture
    docker pull --platform=arm64 swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.5.0-910b-ubuntu22.04-py3.10-ops
    # Example: Pulling a CANN development image of the x86 architecture
    docker pull --platform=amd64 swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.5.0-910b-ubuntu22.04-py3.10-ops
    ```

3. **Run Docker.**

    After pulling an image, you need to start the container with specific parameters so that the container can access Ascend devices on the host.

```bash
docker run --name cann_container --device /dev/davinci0 --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc -v /usr/local/dcmi:/usr/local/dcmi -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info -v /etc/ascend_install.info:/etc/ascend_install.info -it swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.5.0-910b-ubuntu22.04-py3.10-ops bash
```

| Parameter| Description| Remarks|
| :--- | :--- | :--- |
| `--name cann_container` | Defines a container name for easy management.| You can customize a name.|
| `--device /dev/davinci0` | Maps the host's NPUs into the container.| `davinci0` is NPU 0. This is an example only. To obtain the device ID (for example, `NPU 0` or `NPU 1`), run the `npu-smi info` command on the host.|
| `--device /dev/davinci_manager` | Maps the NPU device management interface.| - |
| `--device /dev/devmm_svm` | Maps the device memory management interface.| - |
| `--device /dev/hisi_hdc` | Maps the host-device communication interface.| - |
| `-v /usr/local/dcmi:/usr/local/dcmi` | Mounts the tools and libraries related to the Device Container Management Interface (DCMI).| - |
| `-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi` | Mounts the `npu-smi` tool.| This allows you to use this tool in the container to query the NPU status and performance.|
| `-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/` | (Critical) Mounts the host's NPU driver libraries into the container.| - |
| `-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info` | Mounts the driver version information file.| - |
| `-v /etc/ascend_install.info:/etc/ascend_install.info` | Mounts the CANN software installation information file.| - |
| `-it` | Specifies the combination of `-i` (interactive) and `-t` (pseudo-TTY).| - |
| `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.5.0-910b-ubuntu22.04-py3.10-ops` | Specifies the Docker image to be run.|Ensure that the image name and tag are the same as those of the image pulled using `docker pull`.|
| `bash` | Specifies the command to be executed immediately after the container is started.| - |

### Method 3: Manual Installation

If you have Ascend devices and want to manually set up an Ascend environment, perform the following operations:

#### Install software

- **Scenario 1: Experiencing the master branch or performing development based on the master branch**

    1. **Install the driver and firmware (runtime dependencies).**

        For details, see "Software Package Preparation" and "NPU Driver and Firmware Installation" in [CANN Software Installation Guide](https://www.hiascend.com/document/redirect/CannCommunityInstWizard). The driver and firmware are runtime dependencies. If you only need to compile operators, installation is not required.

    2. **Install the CANN toolkit.**

        Click [the download link](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master/), select the latest version, and download the corresponding package based on the product model and environment architecture. Run the commands below for installation. For more information, see [CANN Software Installation Guide](https://www.hiascend.com/document/redirect/CannCommunityInstWizard).

        ```bash
        # Ensure that the installation package is executable.
        chmod +x Ascend-cann-toolkit_${cann_version}_linux-${arch}.run
        # Installation command
        ./Ascend-cann-toolkit_${cann_version}_linux-${arch}.run --install --install-path=${install_path}
        ```

- **Scenario 2: Experiencing a released version or performing development based on a released version**

    Visit [CANN official website download center](https://www.hiascend.com/cann/download), select a released version (CANN 8.5.0 or later), download the corresponding package based on the product model and environment architecture, and run the commands provided on the web page to complete the installation.

#### Install basic dependencie

The basic dependencies of this project are listed below. Ensure that the version requirements are met.

- Python >= 3.7.0 (<= 3.10 is recommended)
- gcc >= 7.3.0
- cmake >= 3.16.0
- pigz (It is optional. It can accelerate packaging. The recommended version is 2.4 or later.)
- dos2unix
- make
- patch
- GoogleTest (It is required only when UT is executed. The recommended version is [release-1.11.0](https://github.com/google/googletest/releases/tag/release-1.11.0).)

You can use a project script to automate the installation of these dependencies. The procedure is as follows:

1. Download the source code.

    Download the branch source code that matches the CANN version. Replace \$\{tag\_version\} with the branch tag name.
   
    ```bash
    git clone -b ${tag_version} https://gitcode.com/cann/opbase.git
    ```
    
2. Install dependencies.

    Use the project script install\_deps.sh to install the dependencies. For a system that does not support the script, adapt it based on this section.

    ```bash
    bash install_deps.sh
    ```
## Environment Verification

After installing the CANN package, verify that the environment and driver are normal.

- **Check NPUs.**

    ```bash
    # Run the `npu-smi` command. If the device information is displayed properly, the driver is normal.
    npu-smi info
    ```

- **Check the CANN version.**

    ```bash
    # Check the version of the CANN toolkit (installed in the default path). For WebIDE, replace /usr/local with /home/developer.
    cat /usr/local/Ascend/cann/${arch}-linux/ascend_toolkit_install.info
    # Check the version of the CANN ops package (installed in the default path). For WebIDE, replace /usr/local with /home/developer.
    cat /usr/local/Ascend/cann/${arch}-linux/ascend_ops_install.info
    ```

## Environment Variable Configuration

Run the appropriate command to make the environment variables take effect.

```bash
# Default installation path (using the root user as an example; for a non-root user, replace /usr/local with ${HOME})
source /usr/local/Ascend/cann/set_env.sh
# Custom installation path
# source ${install_path}/cann/set_env.sh
```
