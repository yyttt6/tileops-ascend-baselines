project(op_common_utest)

file(GLOB_RECURSE UT_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
include_directories(
    ${OPS_BASE_DIR}/include
    ${OPS_BASE_DIR}/include/op_common
    ${OPS_BASE_INCLUDE}
    ${OPS_BASE_INCLUDE}/op_common
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/include/base
    ${ASCEND_HOME_PATH}/include/toolchain
    ${ASCEND_HOME_PATH}/pkg_inc/base
)

if(ENABLE_UT)
    add_executable(op_common_utest ${UT_SOURCES})

    target_compile_options(op_common_utest PUBLIC
        -fPIE
        -g -O0
        )
    if(ENABLE_COVERAGE)
        target_compile_options(op_common_utest PRIVATE
        --coverage
        -fprofile-arcs
        -ftest-coverage)
    endif()

    target_link_libraries(op_common_utest
        PRIVATE
        intf_pub
        gtest
        gtest_main
        ${ASCEND_HOME_PATH}/lib64/libunified_dlog.so
        $<$<BOOL:${ENABLE_COVERAGE}>:gcov>
        )
endif()
