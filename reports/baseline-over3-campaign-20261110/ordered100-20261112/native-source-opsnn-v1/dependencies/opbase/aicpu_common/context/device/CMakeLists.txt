cmake_minimum_required(VERSION 3.14.1)
project(cpu_kernels_context_device)

protobuf_generate(aicpu_proto_device PROTO_SRCS PROTO_HDRS ${proto_src_files} TARGET)
set(aicpu_proto_include "${CMAKE_BINARY_DIR}/proto/aicpu_proto_device")

add_library(aicpu_context STATIC
  ${local_context_src_files}
  ${PROTO_SRCS}
)
add_dependencies(aicpu_context aicpu_proto_device)

target_include_directories(aicpu_context PRIVATE
  ${aicpu_proto_include}
)

target_compile_definitions(aicpu_context PRIVATE
  _FORTIFY_SOURCE=2
  google=ascend_private
  $<$<CONFIG:Release>:NDEBUG>
)

target_compile_options(aicpu_context PRIVATE
  -O2
  -ftrapv
  -fstack-protector-all
  -fPIC
  -Werror=sign-compare
  -Werror=unused-function
  -Werror=unused-variable
  -Wno-address
  -fvisibility-inlines-hidden
  -fvisibility=hidden
  -DEIGEN_MPL2_ONLY
  -DEigen=ascend_Eigen
)

target_link_libraries(aicpu_context PRIVATE
  aicpu_context_headers
  $<BUILD_INTERFACE:intf_pub>
  $<BUILD_INTERFACE:slog_headers>
  $<BUILD_INTERFACE:runtime_headers>
  $<BUILD_INTERFACE:c_sec_headers>
  Eigen3::Eigen
  ascend_protobuf_static
  -Wl,-z,relro,-z,now
  $<$<NOT:$<STREQUAL:${RESERVED_SYMBOL},true>>:-s>
  -ldl
  -shared
)

add_library(aicpu_nodedef STATIC
  ${nodedef_src_files}
)

add_dependencies(aicpu_nodedef aicpu_context)

target_include_directories(aicpu_nodedef PRIVATE
    ${aicpu_proto_include}
)

target_compile_definitions(aicpu_nodedef PRIVATE
  _FORTIFY_SOURCE=2
  google=ascend_private
)

target_compile_options(aicpu_nodedef PRIVATE
  -O2
  -std=c++17
  -ftrapv
  -fstack-protector-all
  -fPIC
  -Werror
  -DEIGEN_MPL2_ONLY
  -DEigen=ascend_Eigen
)

target_link_libraries(aicpu_nodedef PRIVATE
  $<BUILD_INTERFACE:runtime_headers>
  aicpu_context_headers
)

set (aicpu_cust_src_files
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpu_proto/node_def.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpu_proto/node_def_impl.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpu_proto/tensor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpu_proto/tensor_impl.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpu_proto/tensor_shape.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpu_proto/tensor_shape_impl.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpu_proto/attr_value.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpu_proto/attr_value_impl.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/host_sharder.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/device_sharder.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/eigen_threadpool.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/device.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/context.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/cpu_kernel_register.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/cpu_kernel_utils.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/cpu_kernel_cache.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/async_event_util.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/async_cpu_kernel.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../cust_op/cust_op_log.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../cust_op/cust_dlog_record.cc
    ${PROTO_SRCS}
)

add_library(aicpu_cust_log STATIC
  ${aicpu_cust_src_files}
)

add_dependencies(aicpu_cust_log aicpu_proto_device)

target_include_directories(aicpu_cust_log PRIVATE
  ${aicpu_proto_include}
)

target_link_libraries(aicpu_cust_log PRIVATE
  aicpu_context_headers
  $<BUILD_INTERFACE:intf_pub>
  $<BUILD_INTERFACE:slog_headers>
  $<BUILD_INTERFACE:runtime_headers>
  $<BUILD_INTERFACE:c_sec_headers>
  Eigen3::Eigen
  ascend_protobuf_static
  -Wl,-z,relro,-z,now
  $<$<NOT:$<STREQUAL:${RESERVED_SYMBOL},true>>:-s>
  -ldl
  -shared
)

target_compile_definitions(aicpu_cust_log PRIVATE
    _FORTIFY_SOURCE=2
    google=ascend_private
    $<$<CONFIG:Release>:NDEBUG>
)

target_compile_options(aicpu_cust_log PRIVATE
    -O2
    -std=c++17
    -ftrapv
    -fstack-protector-all
    -Werror=sign-compare
    -Werror=unused-function
    -Werror=unused-variable
    -Wno-address
    -fvisibility-inlines-hidden
    -fvisibility=hidden
    -fPIC
)

install(TARGETS aicpu_context aicpu_nodedef
    DESTINATION ${CMAKE_HOST_SYSTEM_PROCESSOR}-linux/lib64
    COMPONENT opbase
)

install(TARGETS aicpu_cust_log
    DESTINATION ${CMAKE_HOST_SYSTEM_PROCESSOR}-linux/lib64/device/lib64
    COMPONENT opbase
)
