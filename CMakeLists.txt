cmake_minimum_required(VERSION 3.10)

project(tenncor C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(PACKAGE_TESTS "Build the tests" ON)

set(tenncor_INSTALL_default ON)
if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
   set(tenncor_INSTALL_default OFF)
endif()
set(tenncor_INSTALL ${tenncor_INSTALL_default} CACHE BOOL "Generate installation target")
set(tenncor_INSTALL_BINDIR "bin" CACHE STRING "Installation directory for executables")
set(tenncor_INSTALL_LIBDIR "lib" CACHE STRING "Installation directory for libraries")
set(tenncor_INSTALL_INCLUDEDIR "include" CACHE STRING "Installation directory for headers")
set(tenncor_INSTALL_CMAKEDIR "lib/cmake/${PACKAGE_NAME}" CACHE STRING "Installation directory for cmake config files")

set(tenncor_OPTIMIZE ON CACHE BOOL "Optimize tenncor libraries")

#### third_parties ####

set(CONAN_CMAKE_SILENT_OUTPUT OFF)
include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

find_package(gRPC CONFIG REQUIRED)

#### set optimization options after potential third party overrides ####

if(tenncor_OPTIMIZE)
    add_compile_options(-march=native -O3)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        add_compile_options(-openmp)
    else()
        add_compile_options(-lgomp -fopenmp)
    endif()
endif()

#### generate cpp files ####

set(EGEN_TOOL "${CMAKE_CURRENT_SOURCE_DIR}/tools/cmake-egen/egen")

get_filename_component(EIGEN_TYPE_CFG "cfg/fulltype.yml" ABSOLUTE)
get_filename_component(EIGEN_OP_CFG "cfg/ops.yml" ABSOLUTE)
set(EIGEN_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/internal/eigen/generated")
set(EIGEN_SRCS "${EIGEN_GENERATED_DIR}/dtype.cpp" "${EIGEN_GENERATED_DIR}/opcode.cpp")
set(EIGEN_HDRS "${EIGEN_GENERATED_DIR}/dtype.hpp" "${EIGEN_GENERATED_DIR}/opcode.hpp")
add_custom_command(
    OUTPUT ${EIGEN_SRCS} ${EIGEN_HDRS}
    COMMAND "${EGEN_TOOL}"
    ARGS --plugins plugins.dtypes:DTypesPlugin plugins.opcodes:OpcodesPlugin
        --out "${EIGEN_GENERATED_DIR}"
        --cfgs "${EIGEN_TYPE_CFG}" "${EIGEN_OP_CFG}"
    DEPENDS "${EIGEN_TYPE_CFG}" "${EIGEN_OP_CFG}"
)

get_filename_component(TENNCOR_CFG "cfg/tenncor.yml" ABSOLUTE)
set(TENNCOR_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/tenncor/generated")
set(TENNCOR_SRCS "${TENNCOR_GENERATED_DIR}/api.cpp")
set(TENNCOR_HDRS "${TENNCOR_GENERATED_DIR}/api.hpp")
set(PYTENNCOR_SRCS "${TENNCOR_GENERATED_DIR}/pyapi_tenncor.cpp")
set(PYTENNCOR_HDRS "${TENNCOR_GENERATED_DIR}/pyapi.hpp")
add_custom_command(
    OUTPUT "${TENNCOR_SRCS}" "${TENNCOR_HDRS}" "${PYTENNCOR_SRCS}" "${PYTENNCOR_HDRS}"
    COMMAND "${EGEN_TOOL}"
    ARGS --plugins plugins.apis:APIsPlugin plugins.pyapis:PyAPIsPlugin
        --out "${TENNCOR_GENERATED_DIR}"
        --ext_path "${CMAKE_CURRENT_SOURCE_DIR}/cfg/tenncor"
        --cfgs "${TENNCOR_CFG}"
    DEPENDS "${TENNCOR_CFG}"
)

#### generate proto files ####

find_program(_PROTOBUF_PROTOC protoc ${CONAN_BIN_DIRS_PROTOBUF} NO_DEFAULT_PATH)
set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:gRPC::grpc_cpp_plugin>)

# internal/onnx
get_filename_component(ONNX_PROTO "internal/onnx/onnx.proto" ABSOLUTE)
set(ONNX_PROTO_SRCS "${CMAKE_CURRENT_BINARY_DIR}/internal/onnx/onnx.pb.cc")
set(ONNX_PROTO_HDRS "${CMAKE_CURRENT_BINARY_DIR}/internal/onnx/onnx.pb.h")
add_custom_command(
    OUTPUT "${ONNX_PROTO_SRCS}" "${ONNX_PROTO_HDRS}"
    COMMAND ${_PROTOBUF_PROTOC}
    ARGS --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
        -I "${CMAKE_CURRENT_SOURCE_DIR}"
        "${ONNX_PROTO}"
    DEPENDS "${ONNX_PROTO}"
)
add_library(onnx_proto_obj OBJECT ${ONNX_PROTO_SRCS})

# internal/query
get_filename_component(QUERY_PROTO "internal/query/query.proto" ABSOLUTE)
set(QUERY_PROTO_SRCS "${CMAKE_CURRENT_BINARY_DIR}/internal/query/query.pb.cc")
set(QUERY_PROTO_HDRS "${CMAKE_CURRENT_BINARY_DIR}/internal/query/query.pb.h")
add_custom_command(
    OUTPUT "${QUERY_PROTO_SRCS}" "${QUERY_PROTO_HDRS}"
    COMMAND ${_PROTOBUF_PROTOC}
    ARGS --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
        -I "${CMAKE_CURRENT_SOURCE_DIR}"
        "${QUERY_PROTO}"
    DEPENDS "${QUERY_PROTO}"
)
add_library(query_proto_obj OBJECT ${QUERY_PROTO_SRCS})

# internal/opt
get_filename_component(OPT_PROTO "internal/opt/optimize.proto" ABSOLUTE)
set(OPT_PROTO_SRCS "${CMAKE_CURRENT_BINARY_DIR}/internal/opt/optimize.pb.cc")
set(OPT_PROTO_HDRS "${CMAKE_CURRENT_BINARY_DIR}/internal/opt/optimize.pb.h")
add_custom_command(
    OUTPUT "${OPT_PROTO_SRCS}" "${OPT_PROTO_HDRS}"
    COMMAND ${_PROTOBUF_PROTOC}
    ARGS --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
        -I "${CMAKE_CURRENT_SOURCE_DIR}"
        "${OPT_PROTO}"
    DEPENDS "${OPT_PROTO}"
)
add_library(opt_proto_obj OBJECT ${OPT_PROTO_SRCS})

# tenncor/distr/iosvc
get_filename_component(DISTR_IOSVC_PROTO "tenncor/distr/iosvc/distr.io.proto" ABSOLUTE)
set(DISTR_IOSVC_PROTO_SRCS "${CMAKE_CURRENT_BINARY_DIR}/tenncor/distr/iosvc/distr.io.pb.cc")
set(DISTR_IOSVC_PROTO_HDRS "${CMAKE_CURRENT_BINARY_DIR}/tenncor/distr/iosvc/distr.io.pb.h")
set(DISTR_IOSVC_GRPC_SRCS "${CMAKE_CURRENT_BINARY_DIR}/tenncor/distr/iosvc/distr.io.grpc.pb.cc")
set(DISTR_IOSVC_GRPC_HDRS "${CMAKE_CURRENT_BINARY_DIR}/tenncor/distr/iosvc/distr.io.grpc.pb.h")
add_custom_command(
    OUTPUT "${DISTR_IOSVC_PROTO_SRCS}" "${DISTR_IOSVC_PROTO_HDRS}" "${DISTR_IOSVC_GRPC_SRCS}" "${DISTR_IOSVC_GRPC_HDRS}"
    COMMAND ${_PROTOBUF_PROTOC}
    ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
        --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
        -I "${CMAKE_CURRENT_SOURCE_DIR}"
        --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
        "${DISTR_IOSVC_PROTO}"
    DEPENDS "${DISTR_IOSVC_PROTO}"
)
add_library(distr_iosvc_proto_obj OBJECT ${DISTR_IOSVC_PROTO_SRCS} ${DISTR_IOSVC_GRPC_SRCS})

# tenncor/eteq/opsvc
get_filename_component(DISTR_OPSVC_PROTO "tenncor/eteq/opsvc/distr.op.proto" ABSOLUTE)
set(DISTR_OPSVC_PROTO_SRCS "${CMAKE_CURRENT_BINARY_DIR}/tenncor/eteq/opsvc/distr.op.pb.cc")
set(DISTR_OPSVC_PROTO_HDRS "${CMAKE_CURRENT_BINARY_DIR}/tenncor/eteq/opsvc/distr.op.pb.h")
set(DISTR_OPSVC_GRPC_SRCS "${CMAKE_CURRENT_BINARY_DIR}/tenncor/eteq/opsvc/distr.op.grpc.pb.cc")
set(DISTR_OPSVC_GRPC_HDRS "${CMAKE_CURRENT_BINARY_DIR}/tenncor/eteq/opsvc/distr.op.grpc.pb.h")
add_custom_command(
    OUTPUT "${DISTR_OPSVC_PROTO_SRCS}" "${DISTR_OPSVC_PROTO_HDRS}" "${DISTR_OPSVC_GRPC_SRCS}" "${DISTR_OPSVC_GRPC_HDRS}"
    COMMAND ${_PROTOBUF_PROTOC}
    ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
        --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
        -I "${CMAKE_CURRENT_SOURCE_DIR}"
        --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
        "${DISTR_OPSVC_PROTO}"
    DEPENDS "${DISTR_OPSVC_PROTO}"
)
add_library(distr_opsvc_proto_obj OBJECT ${DISTR_OPSVC_PROTO_SRCS} ${DISTR_OPSVC_GRPC_SRCS})

# tenncor/find/lusvc
get_filename_component(DISTR_LUSVC_PROTO "tenncor/find/lusvc/distr.lu.proto" ABSOLUTE)
set(DISTR_LUSVC_PROTO_SRCS "${CMAKE_CURRENT_BINARY_DIR}/tenncor/find/lusvc/distr.lu.pb.cc")
set(DISTR_LUSVC_PROTO_HDRS "${CMAKE_CURRENT_BINARY_DIR}/tenncor/find/lusvc/distr.lu.pb.h")
set(DISTR_LUSVC_GRPC_SRCS "${CMAKE_CURRENT_BINARY_DIR}/tenncor/find/lusvc/distr.lu.grpc.pb.cc")
set(DISTR_LUSVC_GRPC_HDRS "${CMAKE_CURRENT_BINARY_DIR}/tenncor/find/lusvc/distr.lu.grpc.pb.h")
add_custom_command(
    OUTPUT "${DISTR_LUSVC_PROTO_SRCS}" "${DISTR_LUSVC_PROTO_HDRS}" "${DISTR_LUSVC_GRPC_SRCS}" "${DISTR_LUSVC_GRPC_HDRS}"
    COMMAND ${_PROTOBUF_PROTOC}
    ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
        --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
        -I "${CMAKE_CURRENT_SOURCE_DIR}"
        --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
        "${DISTR_LUSVC_PROTO}"
    DEPENDS "${DISTR_LUSVC_PROTO}"
)
add_library(distr_lusvc_proto_obj OBJECT ${DISTR_LUSVC_PROTO_SRCS} ${DISTR_LUSVC_GRPC_SRCS})

# tenncor/hone/hosvc
get_filename_component(DISTR_HOSVC_PROTO "tenncor/hone/hosvc/distr.ho.proto" ABSOLUTE)
set(DISTR_HOSVC_PROTO_SRCS "${CMAKE_CURRENT_BINARY_DIR}/tenncor/hone/hosvc/distr.ho.pb.cc")
set(DISTR_HOSVC_PROTO_HDRS "${CMAKE_CURRENT_BINARY_DIR}/tenncor/hone/hosvc/distr.ho.pb.h")
set(DISTR_HOSVC_GRPC_SRCS "${CMAKE_CURRENT_BINARY_DIR}/tenncor/hone/hosvc/distr.ho.grpc.pb.cc")
set(DISTR_HOSVC_GRPC_HDRS "${CMAKE_CURRENT_BINARY_DIR}/tenncor/hone/hosvc/distr.ho.grpc.pb.h")
add_custom_command(
    OUTPUT "${DISTR_HOSVC_PROTO_SRCS}" "${DISTR_HOSVC_PROTO_HDRS}" "${DISTR_HOSVC_GRPC_SRCS}" "${DISTR_HOSVC_GRPC_HDRS}"
    COMMAND ${_PROTOBUF_PROTOC}
    ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
        --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
        -I "${CMAKE_CURRENT_SOURCE_DIR}"
        --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
        "${DISTR_HOSVC_PROTO}"
    DEPENDS "${DISTR_HOSVC_PROTO}"
)
add_library(distr_hosvc_proto_obj OBJECT ${DISTR_HOSVC_PROTO_SRCS} ${DISTR_HOSVC_GRPC_SRCS})

# tenncor/serial/oxsvc
get_filename_component(DISTR_OXSVC_PROTO "tenncor/serial/oxsvc/distr.ox.proto" ABSOLUTE)
set(DISTR_OXSVC_PROTO_SRCS "${CMAKE_CURRENT_BINARY_DIR}/tenncor/serial/oxsvc/distr.ox.pb.cc")
set(DISTR_OXSVC_PROTO_HDRS "${CMAKE_CURRENT_BINARY_DIR}/tenncor/serial/oxsvc/distr.ox.pb.h")
set(DISTR_OXSVC_GRPC_SRCS "${CMAKE_CURRENT_BINARY_DIR}/tenncor/serial/oxsvc/distr.ox.grpc.pb.cc")
set(DISTR_OXSVC_GRPC_HDRS "${CMAKE_CURRENT_BINARY_DIR}/tenncor/serial/oxsvc/distr.ox.grpc.pb.h")
add_custom_command(
    OUTPUT "${DISTR_OXSVC_PROTO_SRCS}" "${DISTR_OXSVC_PROTO_HDRS}" "${DISTR_OXSVC_GRPC_SRCS}" "${DISTR_OXSVC_GRPC_HDRS}"
    COMMAND ${_PROTOBUF_PROTOC}
    ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
        --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
        -I "${CMAKE_CURRENT_SOURCE_DIR}"
        --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
        "${DISTR_OXSVC_PROTO}"
    DEPENDS "${DISTR_OXSVC_PROTO}"
)
add_library(distr_oxsvc_proto_obj OBJECT ${DISTR_OXSVC_PROTO_SRCS} ${DISTR_OXSVC_GRPC_SRCS})

# dbg/print/printsvc
get_filename_component(DISTR_PRINTSVC_PROTO "dbg/print/printsvc/distr.print.proto" ABSOLUTE)
set(DISTR_PRINTSVC_PROTO_SRCS "${CMAKE_CURRENT_BINARY_DIR}/dbg/print/printsvc/distr.print.pb.cc")
set(DISTR_PRINTSVC_PROTO_HDRS "${CMAKE_CURRENT_BINARY_DIR}/dbg/print/printsvc/distr.print.pb.h")
set(DISTR_PRINTSVC_GRPC_SRCS "${CMAKE_CURRENT_BINARY_DIR}/dbg/print/printsvc/distr.print.grpc.pb.cc")
set(DISTR_PRINTSVC_GRPC_HDRS "${CMAKE_CURRENT_BINARY_DIR}/dbg/print/printsvc/distr.print.grpc.pb.h")
add_custom_command(
    OUTPUT "${DISTR_PRINTSVC_PROTO_SRCS}" "${DISTR_PRINTSVC_PROTO_HDRS}" "${DISTR_PRINTSVC_GRPC_SRCS}" "${DISTR_PRINTSVC_GRPC_HDRS}"
    COMMAND ${_PROTOBUF_PROTOC}
    ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
        --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
        -I "${CMAKE_CURRENT_SOURCE_DIR}"
        --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
        "${DISTR_PRINTSVC_PROTO}"
    DEPENDS "${DISTR_PRINTSVC_PROTO}"
)
add_library(distr_printsvc_proto_obj OBJECT ${DISTR_PRINTSVC_PROTO_SRCS} ${DISTR_PRINTSVC_GRPC_SRCS})

# Include generated *.pb.h files
include_directories("${CMAKE_CURRENT_BINARY_DIR}")

#### local libraries ####

# internal/global
set(GLOBAL_LIB ${PROJECT_NAME}_global)
add_library(${GLOBAL_LIB}
    internal/global/src/config.cpp
    internal/global/src/g3logs.cpp
    internal/global/src/logs.cpp
    internal/global/src/random.cpp
)
target_include_directories(${GLOBAL_LIB} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/.>
    $<INSTALL_INTERFACE:.>
)
target_link_libraries(${GLOBAL_LIB} PUBLIC ${CONAN_LIBS_BOOST} ${CONAN_LIBS_CPPKG} ${CONAN_LIBS_G3LOG})

# internal/marsh
set(MARSH_LIB ${PROJECT_NAME}_marsh)
add_library(${MARSH_LIB} internal/marsh/src/objs.cpp)
target_include_directories(${MARSH_LIB} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/.>
    $<INSTALL_INTERFACE:.>
)
target_link_libraries(${MARSH_LIB} PUBLIC ${CONAN_LIBS_CPPKG})

# internal/teq
set(TEQ_LIB ${PROJECT_NAME}_teq)
add_library(${TEQ_LIB}
    internal/teq/src/derive.cpp
    internal/teq/src/evaluator.cpp
    internal/teq/src/ileaf.cpp
    internal/teq/src/shape.cpp
    internal/teq/src/traveler.cpp
)
target_include_directories(${TEQ_LIB} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/.>
    $<INSTALL_INTERFACE:.>
)
target_link_libraries(${TEQ_LIB} PUBLIC ${GLOBAL_LIB} ${MARSH_LIB})

# internal/eigen
set(EIGEN_LIB ${PROJECT_NAME}_eigen)
add_library(${EIGEN_LIB}
    internal/eigen/src/convert.cpp
    internal/eigen/src/packattr.cpp
    ${EIGEN_SRCS}
)
target_include_directories(${EIGEN_LIB} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/.>
    $<INSTALL_INTERFACE:.>
)
target_link_libraries(${EIGEN_LIB} PUBLIC ${TEQ_LIB} ${CONAN_LIBS_EIGEN})

# internal/onnx
set(ONNX_LIB ${PROJECT_NAME}_onnx)
add_library(${ONNX_LIB}
    internal/onnx/src/load.cpp
    internal/onnx/src/marshal.cpp
    $<TARGET_OBJECTS:onnx_proto_obj>
)
target_include_directories(${ONNX_LIB} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/.>
    $<INSTALL_INTERFACE:.>
)
target_link_libraries(${ONNX_LIB} PUBLIC ${TEQ_LIB} ${CONAN_LIBS_BOOST} protobuf::libprotobuf)

# internal/query
set(QUERY_LIB ${PROJECT_NAME}_query)
add_library(${QUERY_LIB}
    internal/query/src/parse.cpp
    internal/query/src/querier.cpp
    $<TARGET_OBJECTS:query_proto_obj>
)
target_include_directories(${QUERY_LIB} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/.>
    $<INSTALL_INTERFACE:.>
)
target_link_libraries(${QUERY_LIB} PUBLIC ${EIGEN_LIB} protobuf::libprotobuf)

# internal/opt
set(OPT_LIB ${PROJECT_NAME}_opt)
add_library(${OPT_LIB}
    internal/opt/src/apply.cpp
    internal/opt/src/parse.cpp
    $<TARGET_OBJECTS:opt_proto_obj>
)
target_include_directories(${OPT_LIB} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/.>
    $<INSTALL_INTERFACE:.>
)
target_link_libraries(${OPT_LIB} PUBLIC ${QUERY_LIB})

# tenncor/distr
set(DISTR_LIB ${PROJECT_NAME}_distr)
add_library(${DISTR_LIB}
    tenncor/distr/src/manager.cpp
    tenncor/distr/src/p2p.cpp
    tenncor/distr/src/reference.cpp
)
target_include_directories(${DISTR_LIB} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/.>
    $<INSTALL_INTERFACE:.>
)
target_link_libraries(${DISTR_LIB} PUBLIC ${EIGEN_LIB} ${CONAN_LIBS_CPPKG} ${CONAN_LIBS_PPCONSUL} protobuf::libprotobuf)

# tenncor/distr/iosvc
set(DISTR_IOSVC_LIB ${PROJECT_NAME}_distr_iosvc)
add_library(${DISTR_IOSVC_LIB}
    tenncor/distr/iosvc/src/pb_helper.cpp
    tenncor/distr/iosvc/src/service.cpp
    $<TARGET_OBJECTS:distr_iosvc_proto_obj>
)
target_include_directories(${DISTR_IOSVC_LIB} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/.>
    $<INSTALL_INTERFACE:.>
)
target_link_libraries(${DISTR_IOSVC_LIB} PUBLIC ${DISTR_LIB})

# tenncor/eteq
set(ETEQ_LIB ${PROJECT_NAME}_eteq)
add_library(${ETEQ_LIB}
    tenncor/eteq/src/etens.cpp
    tenncor/eteq/src/make.cpp
)
target_include_directories(${ETEQ_LIB} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/.>
    $<INSTALL_INTERFACE:.>
)
target_link_libraries(${ETEQ_LIB} PUBLIC ${EIGEN_LIB})

# tenncor/eteq/opsvc
set(DISTR_OPSVC_LIB ${PROJECT_NAME}_distr_opsvc)
add_library(${DISTR_OPSVC_LIB}
    tenncor/eteq/opsvc/src/service.cpp
    $<TARGET_OBJECTS:distr_opsvc_proto_obj>
)
target_include_directories(${DISTR_OPSVC_LIB} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/.>
    $<INSTALL_INTERFACE:.>
)
target_link_libraries(${DISTR_OPSVC_LIB} PUBLIC ${DISTR_IOSVC_LIB} ${ETEQ_LIB})

# tenncor/find/lusvc
set(DISTR_LUSVC_LIB ${PROJECT_NAME}_distr_lusvc)
add_library(${DISTR_LUSVC_LIB}
    tenncor/find/lusvc/src/service.cpp
    $<TARGET_OBJECTS:distr_lusvc_proto_obj>
)
target_include_directories(${DISTR_LUSVC_LIB} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/.>
    $<INSTALL_INTERFACE:.>
)
target_link_libraries(${DISTR_LUSVC_LIB} PUBLIC ${DISTR_IOSVC_LIB} ${QUERY_LIB})

# tenncor/hone
set(HONE_LIB ${PROJECT_NAME}_hone)
add_library(${HONE_LIB}
    tenncor/hone/src/cstrules.cpp
    tenncor/hone/src/duplicates.cpp
    tenncor/hone/src/optimize.cpp
)
target_include_directories(${HONE_LIB} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/.>
    $<INSTALL_INTERFACE:.>
)
target_link_libraries(${HONE_LIB} PUBLIC ${OPT_LIB} ${ETEQ_LIB})

# tenncor/hone/hosvc
set(DISTR_HOSVC_LIB ${PROJECT_NAME}_distr_hosvc)
add_library(${DISTR_HOSVC_LIB}
    tenncor/hone/hosvc/src/service.cpp
    $<TARGET_OBJECTS:distr_hosvc_proto_obj>
)
target_include_directories(${DISTR_HOSVC_LIB} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/.>
    $<INSTALL_INTERFACE:.>
)
target_link_libraries(${DISTR_HOSVC_LIB} PUBLIC ${DISTR_IOSVC_LIB} ${HONE_LIB})

# tenncor/layr
set(LAYR_LIB ${PROJECT_NAME}_layr)
add_library(${LAYR_LIB}
    tenncor/layr/src/layer.cpp
)
target_include_directories(${LAYR_LIB} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/.>
    $<INSTALL_INTERFACE:.>
)
target_link_libraries(${LAYR_LIB} PUBLIC ${ETEQ_LIB})

# tenncor/serial
set(SERIAL_LIB ${PROJECT_NAME}_serial)
add_library(${SERIAL_LIB}
    tenncor/serial/src/serialize.cpp
)
target_include_directories(${SERIAL_LIB} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/.>
    $<INSTALL_INTERFACE:.>
)
target_link_libraries(${SERIAL_LIB} PUBLIC ${ONNX_LIB} ${LAYR_LIB})

# tenncor/serial/oxsvc
set(DISTR_OXSVC_LIB ${PROJECT_NAME}_distr_oxsvc)
add_library(${DISTR_OXSVC_LIB}
    tenncor/serial/oxsvc/src/segment.cpp
    tenncor/serial/oxsvc/src/service.cpp
    tenncor/serial/oxsvc/src/topography.cpp
    tenncor/serial/oxsvc/src/util.cpp
    $<TARGET_OBJECTS:distr_oxsvc_proto_obj>
)
target_include_directories(${DISTR_OXSVC_LIB} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/.>
    $<INSTALL_INTERFACE:.>
)
target_link_libraries(${DISTR_OXSVC_LIB} PUBLIC ${DISTR_IOSVC_LIB} ${SERIAL_LIB})

# tenncor/trainer
set(TRAINER_LIB ${PROJECT_NAME}_trainer)
add_library(${TRAINER_LIB} INTERFACE)
target_include_directories(${TRAINER_LIB} INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/.>
    $<INSTALL_INTERFACE:.>
)
target_link_libraries(${TRAINER_LIB} INTERFACE ${LAYR_LIB})

# tenncor
set(TENNCOR_LIB c${PROJECT_NAME})
add_library(${TENNCOR_LIB}
    tenncor/src/distr.cpp
    tenncor/src/eteq.cpp
    tenncor/src/hone.cpp
    tenncor/src/layr.cpp
    tenncor/src/serial.cpp
    ${TENNCOR_SRCS}
)
target_include_directories(${TENNCOR_LIB} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/.>
    $<INSTALL_INTERFACE:.>
)
target_link_libraries(${TENNCOR_LIB} PUBLIC
    ${DISTR_OPSVC_LIB}
    ${DISTR_LUSVC_LIB}
    ${DISTR_HOSVC_LIB}
    ${DISTR_OXSVC_LIB}
    ${TRAINER_LIB}
)

if(tenncor_INSTALL)
    foreach(_PATH
        internal/global
        internal/marsh
        internal/teq
        internal/eigen
        internal/onnx
        internal/query
        internal/opt
        tenncor/distr
        tenncor/distr/iosvc
        tenncor/eteq
        tenncor/eteq/opsvc
        tenncor/find/lusvc
        tenncor/hone
        tenncor/hone/hosvc
        tenncor/layr
        tenncor/serial
        tenncor/serial/oxsvc
        tenncor/trainer
        tenncor
    )
        file(GLOB _HDRS ${_PATH}/*.hpp)
        install(FILES ${_HDRS} DESTINATION "${tenncor_INSTALL_INCLUDEDIR}/${_PATH}")
    endforeach()
    install(FILES internal/eigen/generated/dtype.hpp internal/eigen/generated/opcode.hpp DESTINATION "${tenncor_INSTALL_INCLUDEDIR}/internal/eigen/generated")
    install(FILES tenncor/generated/api.hpp DESTINATION "${tenncor_INSTALL_INCLUDEDIR}/tenncor/generated")

    set(TargetFile ${PROJECT_NAME}Targets)
    install(TARGETS
        ${GLOBAL_LIB}
        ${MARSH_LIB}
        ${TEQ_LIB}
        ${EIGEN_LIB}
        ${ONNX_LIB}
        ${QUERY_LIB}
        ${OPT_LIB}
        ${DISTR_LIB}
        ${DISTR_IOSVC_LIB}
        ${ETEQ_LIB}
        ${DISTR_OPSVC_LIB}
        ${DISTR_LUSVC_LIB}
        ${HONE_LIB}
        ${DISTR_HOSVC_LIB}
        ${LAYR_LIB}
        ${SERIAL_LIB}
        ${DISTR_OXSVC_LIB}
        ${TRAINER_LIB}
        ${TENNCOR_LIB}
        EXPORT ${TargetFile}
        RUNTIME DESTINATION ${tenncor_INSTALL_BINDIR}
        LIBRARY DESTINATION ${tenncor_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${tenncor_INSTALL_LIBDIR}
    )
    install(EXPORT ${TargetFile}
        DESTINATION ${tenncor_INSTALL_CMAKEDIR}
        NAMESPACE ${PROJECT_NAME}::
    )
endif()

# dbg/print
set(PRINT_LIB "${PROJECT_NAME}_print")
add_library(${PRINT_LIB}
    dbg/print/src/teq.cpp
)
target_include_directories(${PRINT_LIB} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/.>
    $<INSTALL_INTERFACE:.>
)
target_link_libraries(${PRINT_LIB} PUBLIC ${TEQ_LIB})

# dbg/print/printsvc
set(DISTR_PRINTSVC_LIB "${PROJECT_NAME}_dbg_printsvc")
add_library(${DISTR_PRINTSVC_LIB}
    dbg/print/printsvc/src/data.cpp
    dbg/print/printsvc/src/service.cpp
    $<TARGET_OBJECTS:distr_printsvc_proto_obj>
)
target_include_directories(${DISTR_PRINTSVC_LIB} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/.>
    $<INSTALL_INTERFACE:.>
)
target_link_libraries(${DISTR_PRINTSVC_LIB} PUBLIC ${DISTR_IOSVC_LIB} ${PRINT_LIB})

# tenncor/python
set(PYTENNCOR ${PROJECT_NAME})
pybind11_add_module(${PYTENNCOR}
    tenncor/python/distr_ext.cpp
    tenncor/python/eteq_ext.cpp
    tenncor/python/layr_ext.cpp
    tenncor/python/query_ext.cpp
    tenncor/pyutils/src/convert.cpp
    ${PYTENNCOR_SRCS}
)
target_include_directories(${PYTENNCOR} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/.>
    $<INSTALL_INTERFACE:.>
)
target_link_libraries(${PYTENNCOR} PUBLIC ${TENNCOR_LIB} ${DISTR_PRINTSVC_LIB})
