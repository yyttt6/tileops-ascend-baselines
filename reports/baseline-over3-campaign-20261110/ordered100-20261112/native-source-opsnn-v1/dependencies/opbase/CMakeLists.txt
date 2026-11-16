cmake_minimum_required(VERSION 3.16)
include(cmake/fetch_cann_cmake.cmake)
project(opbase VERSION 1.0.0)
init_cann_project()
add_cann_target_options()

set(PACKAGE_TYPE "run" CACHE STRING "package type: run/rpm/deb/all")
set_property(CACHE PACKAGE_TYPE PROPERTY STRINGS run rpm deb all)
set(SUPPORTED_PACKAGE_TYPES run rpm deb all)
if(NOT PACKAGE_TYPE IN_LIST SUPPORTED_PACKAGE_TYPES)
  message(FATAL_ERROR "PACKAGE_TYPE only supports run/rpm/deb/all, got: ${PACKAGE_TYPE}")
endif()

if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    message(STATUS "Build opbase with dependency cann pkgs installed")
    option(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG "Build opbase with dependency cann pkgs installed" TRUE)
else()
    message(STATUS "Build opbase with cann source exist")
    option(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG "Build opbase need cann source code" FALSE)
endif()

option(ENABLE_BUILD_DEVICE "Build opbase with device" TRUE)

if (NOT DEFINED CMAKE_MODULE_PATH)
    set(CMAKE_MODULE_PATH
        ${CANN_CMAKE_DIR}/modules
        ${CANN_3RD_LIB_PATH}/cmake/modules
    )
endif()

set(OPS_BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

include(cmake/config.cmake)
include(cmake/variables.cmake)
message(STATUS "Variables in ${CMAKE_PROJECT_NAME} project:")
cmake_print_variables(BUILD_WITH_DEPENDENCY_SOURCE)
cmake_print_variables(CANN_3RD_LIB_PATH)
cmake_print_variables(CANN_3RD_PKG_PATH)

if(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG)
    message(STATUS "ENV: $ENV{ASCEND_HOME_PATH}")
    if(DEFINED ENV{ASCEND_HOME_PATH} AND NOT "$ENV{ASCEND_HOME_PATH}" STREQUAL "")
        set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
    else()
        if ("$ENV{USER}" STREQUAL "root")
            set(ASCEND_HOME_PATH /usr/local/Ascend/cann)
        else()
            set(ASCEND_HOME_PATH $ENV{HOME}/Ascend/cann)
        endif() 
    endif()

    if(DEFINED ENV{TOOLCHAIN_DIR})
        set(TOOLCHAIN_DIR $ENV{TOOLCHAIN_DIR})
    else()
        if(MINRC)
            set(TOOLCHAIN_DIR /usr)
        else()
            set(TOOLCHAIN_DIR ${ASCEND_HOME_PATH}/toolkit/toolchain/hcc)
        endif()
    endif() 
    add_cann_third_party(protobuf)
    add_cann_third_party(eigen)
    add_cann_third_party(json)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/func.cmake)

    include(version.cmake)
    set(ASCEND_INSTALL_PATH "${ASCEND_HOME_PATH}")
    set(ASCEND_INCLUDE_DIR ${ASCEND_HOME_PATH}/include)

    # 设置 CMAKE_MODULE_PATH 和 CMAKE_PREFIX_PATH 以支持 find_cann_package
    list(APPEND CMAKE_MODULE_PATH
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules
    )
    list(APPEND CMAKE_PREFIX_PATH
        ${ASCEND_HOME_PATH}
    )

    find_cann_package(metadef MODULE REQUIRED)
    find_cann_package(graph MODULE REQUIRED)
    find_cann_package(tiling_api MODULE REQUIRED)
    find_cann_package(unified_dlog MODULE REQUIRED)
    find_cann_package(runtime MODULE REQUIRED)
    find_cann_package(securec MODULE REQUIRED)
    find_cann_package(error_manager MODULE REQUIRED)
    find_cann_package(ascend_dump MODULE REQUIRED)
    find_cann_package(acl_rt MODULE REQUIRED)
    find_cann_package(cann_version)
    find_cann_package(asc_devkit)
    find_cann_package(hccl)

    if(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG)
        check_cann_pkg_build_deps("opbase")
    endif()
    add_cann_version_info_targets()

    add_library(opbase_headers INTERFACE)
    target_include_directories(opbase_headers INTERFACE
        ${OPS_BASE_DIR}/include
        ${OPS_BASE_DIR}/include/op_common
    )

    # 本地头文件 target（pkg_inc 下的本地头文件优先于 CANN 安装目录）
    add_library(ops_base_internal_headers INTERFACE)
    target_include_directories(ops_base_internal_headers INTERFACE
        ${OPS_BASE_DIR}/include/op_common/op_graph
        ${OPS_BASE_INCLUDE}
        ${OPS_BASE_INCLUDE}/op_common
    )
    target_link_libraries(ops_base_internal_headers INTERFACE opbase_headers)

    add_library(ops_base SHARED ${OPS_BASE_SRC})
    target_compile_options(ops_base
        PRIVATE
        -fPIC
        -fvisibility=hidden
    )

    target_link_libraries(ops_base
        PRIVATE
        ops_base_internal_headers
        exe_graph
        graph_base
        graph
        -Wl,--no-as-needed
        register
        -Wl,--as-needed
        -Wl,--whole-archive
        tiling_api
        -Wl,--no-whole-archive
        $<BUILD_INTERFACE:intf_pub>
        $<BUILD_INTERFACE:error_manager_headers>
        $<BUILD_INTERFACE:c_sec_headers>
        $<BUILD_INTERFACE:asc_host_headers>
        unified_dlog
    )

    add_subdirectory(aicpu_common/context)
    target_compile_definitions(ops_base PRIVATE
        BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG
    )
    if(DEFINED PRODUCT_SIDE AND "${PRODUCT_SIDE}" STREQUAL "device")
        message(STATUS ">>> Detected PRODUCT_SIDE=device, enabling device-specific features")
        target_compile_definitions(ops_base PRIVATE PRODUCT_SIDE_IS_DEVICE=1)
    endif()

    if(ENABLE_UT OR ENABLE_ST)
        add_subdirectory(tests)
    else()
        add_subdirectory(src/nnopbase)
        include(cmake/package.cmake)
    endif()
else()
    message(STATUS "Build ops_base_objs only from other source code")

    add_library(ops_base_infer_objs OBJECT ${OPS_BASE_INFER_SRC})
    add_library(ops_base_tiling_objs OBJECT ${OPS_BASE_TILING_SRC})
    add_library(ops_base_util_objs OBJECT ${OPS_BASE_UTIL_SRC})

    set(OPBASE_METADEF_INC
        ${METADEF_INCLUDE}
        ${METADEF_INCLUDE}/common
        ${METADEF_INCLUDE}/common/util
        ${METADEF_INCLUDE}/external
        ${METADEF_INCLUDE}/external/graph
        ${METADEF_INCLUDE}/external/exe_graph
        ${METADEF_INCLUDE}/exe_graph
        ${TOP_DIR}/runtime/pkg_inc
        ${TOP_DIR}/runtime/pkg_inc/aicpu_sched/common
        ${TOP_DIR}/runtime/include/external
        )

    target_include_directories(ops_base_infer_objs
        PRIVATE
        ${OPS_BASE_INCLUDE}
        ${OPBASE_METADEF_INC}
        ${ASCEND_DIR}/include/toolchain
        ${ASCEND_DIR}/pkg_inc/base
        ${C_SEC_INCLUDE}
    )
    target_compile_options(ops_base_infer_objs
        PRIVATE
        -fPIC
        -fvisibility=hidden
    )
    target_link_libraries(ops_base_infer_objs
        PRIVATE $<BUILD_INTERFACE:intf_pub_cxx14>
        $<$<BOOL:${BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG}>:$<BUILD_INTERFACE:alog_headers>>
        $<$<NOT:$<BOOL:${BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG}>>:$<BUILD_INTERFACE:slog_headers>>
        alog)

    target_include_directories(ops_base_tiling_objs
        PRIVATE
        ${OPS_BASE_INCLUDE}
        ${OPBASE_METADEF_INC}
        ${ASCEND_DIR}/include/toolchain
        ${ASCEND_DIR}/pkg_inc/base
        ${C_SEC_INCLUDE}
    )
    target_compile_options(ops_base_tiling_objs
        PRIVATE
        -fPIC
        -fvisibility=hidden
    )
    target_link_libraries(ops_base_tiling_objs
        PRIVATE $<BUILD_INTERFACE:intf_pub_cxx14>
        $<$<BOOL:${BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG}>:$<BUILD_INTERFACE:alog_headers>>
        $<$<NOT:$<BOOL:${BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG}>>:$<BUILD_INTERFACE:slog_headers>>
        alog
        tiling_api)

    target_include_directories(ops_base_util_objs
        PRIVATE
        ${OPS_BASE_INCLUDE}
        ${OPBASE_METADEF_INC}
        ${ASCEND_DIR}/include/toolchain
        ${ASCEND_DIR}/pkg_inc/base
        ${C_SEC_INCLUDE}
    )
    target_compile_options(ops_base_util_objs
        PRIVATE
        -fPIC
        -fvisibility=hidden
    )
    target_link_libraries(ops_base_util_objs
        PRIVATE $<BUILD_INTERFACE:intf_pub_cxx14>
        $<$<BOOL:${BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG}>:$<BUILD_INTERFACE:alog_headers>>
        $<$<NOT:$<BOOL:${BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG}>>:$<BUILD_INTERFACE:slog_headers>>
        alog
        tiling_api)
endif()

add_cann_device_project(opbase)
