cmake_minimum_required(VERSION 3.14.1)
project(cpu_kernels_context_host)

protobuf_generate(aicpu_proto_host PROTO_SRCS PROTO_HDRS ${proto_src_files} TARGET)
set(aicpu_proto_include "${CMAKE_BINARY_DIR}/proto/aicpu_proto_host")

add_library(aicpu_context_host STATIC
  ${local_context_src_files}
  ${PROTO_SRCS}
)
add_dependencies(aicpu_context_host aicpu_proto_host)

target_include_directories(aicpu_context_host PRIVATE
  ${aicpu_proto_include}
)

target_compile_definitions(aicpu_context_host PRIVATE
  _FORTIFY_SOURCE=2
  google=ascend_private
  _GLIBCXX_USE_CXX11_ABI=0
)

target_compile_options(aicpu_context_host PRIVATE
  -O2
  -ftrapv
  -fstack-protector-all
  -fPIC
  -Werror=sign-compare
  -Werror=unused-function
  -Werror=unused-variable
  -Wno-address
  -DEIGEN_MPL2_ONLY
  -DEigen=ascend_Eigen
)

target_link_libraries(aicpu_context_host PRIVATE
  aicpu_context_headers
  $<BUILD_INTERFACE:intf_pub>
  $<BUILD_INTERFACE:c_sec_headers>
  $<BUILD_INTERFACE:slog_headers>
  $<BUILD_INTERFACE:runtime_headers>
  runtime
  metadef
  Eigen3::Eigen
  ascend_protobuf_static
  -Wl,-z,relro,-z,now
  $<$<NOT:$<STREQUAL:${RESERVED_SYMBOL},true>>:-s>
  -ldl
  -shared
)

add_library(aicpu_nodedef_host STATIC
  ${nodedef_src_files}
)

add_dependencies(aicpu_nodedef_host aicpu_context_host)

target_include_directories(aicpu_nodedef_host PRIVATE
    ${aicpu_proto_include}
)

target_compile_definitions(aicpu_nodedef_host PRIVATE
  _FORTIFY_SOURCE=2
  google=ascend_private
  _GLIBCXX_USE_CXX11_ABI=0
)

target_compile_options(aicpu_nodedef_host PRIVATE
  -O2
  -std=c++17
  -ftrapv
  -fstack-protector-all
  -fPIC
  -Werror
  -DEIGEN_MPL2_ONLY
  -DEigen=ascend_Eigen
)

target_link_libraries(aicpu_nodedef_host PRIVATE
  $<BUILD_INTERFACE:runtime_headers>
  aicpu_context_headers
  runtime
  metadef
)
