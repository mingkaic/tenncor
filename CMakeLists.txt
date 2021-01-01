cmake_minimum_required(VERSION 3.10)

project(tenncor C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

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

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

find_package(gRPC CONFIG REQUIRED)

#### set optimization options after potential third party overrides ####

if(tenncor_OPTIMIZE)
    message(STATUS "compiler option: ${CMAKE_CXX_COMPILER_ID}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -O3")
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -openmp")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
	add_link_options(-lgomp)
    endif()
endif()

#### generate cpp files ####

set(EGEN_TOOL "${CMAKE_CURRENT_SOURCE_DIR}/tools/egen/egen.py")

get_filename_component(EIGEN_TYPE_CFG "cfg/fulltype.yml" ABSOLUTE)
get_filename_component(EIGEN_OP_CFG "cfg/ops.yml" ABSOLUTE)
set(EIGEN_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/internal/eigen/generated")
set(EIGEN_SRCS "${EIGEN_GENERATED_DIR}/dtype.cpp" "${EIGEN_GENERATED_DIR}/opcode.cpp")
set(EIGEN_HDRS "${EIGEN_GENERATED_DIR}/dtype.hpp" "${EIGEN_GENERATED_DIR}/opcode.hpp")
add_custom_command(
    OUTPUT ${EIGEN_SRCS} ${EIGEN_HDRS}
    COMMAND "python3"
    ARGS ${EGEN_TOOL}
        --plugins plugins.dtypes:DTypesPlugin plugins.opcodes:OpcodesPlugin
        --out "${EIGEN_GENERATED_DIR}"
	--strip_prefix "${CMAKE_CURRENT_BINARY_DIR}"
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
    COMMAND "python3"
    ARGS ${EGEN_TOOL}
        --plugins plugins.apis:APIsPlugin plugins.pyapis:PyAPIsPlugin
        --out "${TENNCOR_GENERATED_DIR}"
	--strip_prefix "${CMAKE_CURRENT_BINARY_DIR}"
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

include_directories(
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/.>
    $<INSTALL_INTERFACE:.>
)

# internal/global
set(GLOBAL_LIB ${PROJECT_NAME}_global)
add_library(${GLOBAL_LIB}
    internal/global/src/config.cpp
    internal/global/src/g3logs.cpp
    internal/global/src/logs.cpp
    internal/global/src/random.cpp
)
target_link_libraries(${GLOBAL_LIB} PUBLIC ${CONAN_LIBS_BOOST} ${CONAN_LIBS_CPPKG} ${CONAN_LIBS_G3LOG})

# internal/marsh
set(MARSH_LIB ${PROJECT_NAME}_marsh)
add_library(${MARSH_LIB} internal/marsh/src/objs.cpp)
target_link_libraries(${MARSH_LIB} PUBLIC ${CONAN_LIBS_CPPKG} ${CONAN_LIBS_BOOST})

# internal/teq
set(TEQ_LIB ${PROJECT_NAME}_teq)
add_library(${TEQ_LIB}
    internal/teq/src/derive.cpp
    internal/teq/src/evaluator.cpp
    internal/teq/src/ileaf.cpp
    internal/teq/src/shape.cpp
    internal/teq/src/traveler.cpp
)
target_link_libraries(${TEQ_LIB} PUBLIC ${GLOBAL_LIB} ${MARSH_LIB})

# internal/eigen
set(EIGEN_LIB ${PROJECT_NAME}_eigen)
add_library(${EIGEN_LIB}
    internal/eigen/src/convert.cpp
    internal/eigen/src/packattr.cpp
    ${EIGEN_SRCS}
)
target_link_libraries(${EIGEN_LIB} PUBLIC ${TEQ_LIB} ${CONAN_LIBS_EIGEN})

# internal/onnx
set(ONNX_LIB ${PROJECT_NAME}_onnx)
add_library(${ONNX_LIB}
    internal/onnx/src/load.cpp
    internal/onnx/src/marshal.cpp
    $<TARGET_OBJECTS:onnx_proto_obj>
)
target_link_libraries(${ONNX_LIB} PUBLIC ${TEQ_LIB} ${CONAN_LIBS_BOOST} protobuf::libprotobuf)
target_compile_definitions(${ONNX_LIB} PUBLIC ORDERED_SAVE)

# internal/query
set(QUERY_LIB ${PROJECT_NAME}_query)
add_library(${QUERY_LIB}
    internal/query/src/parse.cpp
    internal/query/src/querier.cpp
    $<TARGET_OBJECTS:query_proto_obj>
)
target_link_libraries(${QUERY_LIB} PUBLIC ${EIGEN_LIB} protobuf::libprotobuf)

# internal/opt
set(OPT_LIB ${PROJECT_NAME}_opt)
add_library(${OPT_LIB}
    internal/opt/src/apply.cpp
    internal/opt/src/parse.cpp
    $<TARGET_OBJECTS:opt_proto_obj>
)
target_link_libraries(${OPT_LIB} PUBLIC ${QUERY_LIB})

# internal/util
set(UTILS_LIB ${PROJECT_NAME}_internal_utils)
add_library(${UTILS_LIB}
    internal/utils/coord/src/coord.cpp
)
target_link_libraries(${UTILS_LIB} PUBLIC ${TEQ_LIB})

# tenncor/distr
set(DISTR_LIB ${PROJECT_NAME}_distr)
add_library(${DISTR_LIB}
    tenncor/distr/src/manager.cpp
    tenncor/distr/src/p2p.cpp
    tenncor/distr/src/reference.cpp
)
target_link_libraries(${DISTR_LIB} PUBLIC ${EIGEN_LIB} ${CONAN_LIBS_CPPKG} ${CONAN_LIBS_PPCONSUL} gRPC::grpc++ gRPC::grpc++_unsecure)

# tenncor/distr/iosvc
set(DISTR_IOSVC_LIB ${PROJECT_NAME}_distr_iosvc)
add_library(${DISTR_IOSVC_LIB}
    tenncor/distr/iosvc/src/pb_helper.cpp
    tenncor/distr/iosvc/src/service.cpp
    $<TARGET_OBJECTS:distr_iosvc_proto_obj>
)
target_link_libraries(${DISTR_IOSVC_LIB} PUBLIC ${DISTR_LIB})

# tenncor/eteq
set(ETEQ_LIB ${PROJECT_NAME}_eteq)
add_library(${ETEQ_LIB}
    tenncor/eteq/src/etens.cpp
    tenncor/eteq/src/graphinfo.cpp
    tenncor/eteq/src/make.cpp
)
target_link_libraries(${ETEQ_LIB} PUBLIC ${EIGEN_LIB})
target_compile_definitions(${ETEQ_LIB} PUBLIC "SKIP_INIT")

# tenncor/eteq/opsvc
set(DISTR_OPSVC_LIB ${PROJECT_NAME}_distr_opsvc)
add_library(${DISTR_OPSVC_LIB}
    tenncor/eteq/opsvc/src/service.cpp
    $<TARGET_OBJECTS:distr_opsvc_proto_obj>
)
target_link_libraries(${DISTR_OPSVC_LIB} PUBLIC ${DISTR_IOSVC_LIB} ${ETEQ_LIB})

# tenncor/find/lusvc
set(DISTR_LUSVC_LIB ${PROJECT_NAME}_distr_lusvc)
add_library(${DISTR_LUSVC_LIB}
    tenncor/find/lusvc/src/service.cpp
    $<TARGET_OBJECTS:distr_lusvc_proto_obj>
)
target_link_libraries(${DISTR_LUSVC_LIB} PUBLIC ${DISTR_IOSVC_LIB} ${QUERY_LIB})

# tenncor/hone
set(HONE_LIB ${PROJECT_NAME}_hone)
add_library(${HONE_LIB}
    tenncor/hone/src/cstrules.cpp
    tenncor/hone/src/duplicates.cpp
    tenncor/hone/src/optimize.cpp
)
target_link_libraries(${HONE_LIB} PUBLIC ${OPT_LIB} ${ETEQ_LIB})

# tenncor/hone/hosvc
set(DISTR_HOSVC_LIB ${PROJECT_NAME}_distr_hosvc)
add_library(${DISTR_HOSVC_LIB}
    tenncor/hone/hosvc/src/service.cpp
    $<TARGET_OBJECTS:distr_hosvc_proto_obj>
)
target_link_libraries(${DISTR_HOSVC_LIB} PUBLIC ${DISTR_IOSVC_LIB} ${HONE_LIB})

# tenncor/layr
set(LAYR_LIB ${PROJECT_NAME}_layr)
add_library(${LAYR_LIB}
    tenncor/layr/src/layer.cpp
)
target_link_libraries(${LAYR_LIB} PUBLIC ${ETEQ_LIB})

# tenncor/serial
set(SERIAL_LIB ${PROJECT_NAME}_serial)
add_library(${SERIAL_LIB}
    tenncor/serial/src/serialize.cpp
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
target_link_libraries(${DISTR_OXSVC_LIB} PUBLIC ${DISTR_IOSVC_LIB} ${SERIAL_LIB})

# tenncor/trainer
set(TRAINER_LIB ${PROJECT_NAME}_trainer)
add_library(${TRAINER_LIB} INTERFACE)
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
target_link_libraries(${PRINT_LIB} PUBLIC ${TEQ_LIB})

# dbg/print/printsvc
set(DISTR_PRINTSVC_LIB "${PROJECT_NAME}_dbg_printsvc")
add_library(${DISTR_PRINTSVC_LIB}
    dbg/print/printsvc/src/data.cpp
    dbg/print/printsvc/src/service.cpp
    $<TARGET_OBJECTS:distr_printsvc_proto_obj>
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
target_link_libraries(${PYTENNCOR} PRIVATE ${TENNCOR_LIB} ${DISTR_PRINTSVC_LIB})

#### test mocks ####

# internal/teq
add_library(teq_mock
    internal/teq/mock/functor.cpp
    internal/teq/mock/leaf.cpp)
target_link_libraries(teq_mock PUBLIC ${TEQ_LIB} ${CONAN_LIBS_GTEST})

# internal/eigen
add_library(eigen_mock
    internal/eigen/mock/observable.cpp)
target_link_libraries(eigen_mock PUBLIC ${EIGEN_LIB} teq_mock)

# tenncor/distr
add_library(distr_mock tenncor/distr/mock/serverio.cpp)
target_link_libraries(distr_mock PUBLIC ${DISTR_LIB})

# tenncor/distr/iosvc
add_library(distr_iosvc_mock tenncor/distr/iosvc/mock/service.cpp)
target_link_libraries(distr_iosvc_mock PUBLIC distr_mock ${DISTR_IOSVC_LIB})

# tenncor/distr/opsvc
add_library(distr_opsvc_mock tenncor/eteq/opsvc/mock/service.cpp)
target_link_libraries(distr_opsvc_mock PUBLIC distr_mock ${DISTR_OPSVC_LIB})

# tenncor/find/lusvc
add_library(distr_lusvc_mock tenncor/find/lusvc/mock/service.cpp)
target_link_libraries(distr_lusvc_mock PUBLIC distr_mock ${DISTR_LUSVC_LIB})

# tenncor/hone/hosvc
add_library(distr_hosvc_mock tenncor/hone/hosvc/mock/service.cpp)
target_link_libraries(distr_hosvc_mock PUBLIC distr_mock ${DISTR_HOSVC_LIB})

# tenncor/serial/oxsvc
add_library(distr_oxsvc_mock tenncor/serial/oxsvc/mock/service.cpp)
target_link_libraries(distr_oxsvc_mock PUBLIC distr_mock ${DISTR_OXSVC_LIB})

# dbg/print/printsvc
add_library(distr_printsvc_mock dbg/print/printsvc/mock/service.cpp)
target_link_libraries(distr_printsvc_mock PUBLIC distr_mock ${DISTR_PRINTSVC_LIB})

#### tests ####

if(PACKAGE_TESTS)
    enable_testing()
endif()

set(_TESTUTIL ${PACKAGE_NAME}_testutil)
add_library(${_TESTUTIL}
    testutil/src/graph_comp.cpp
    testutil/src/port_check.cpp
)
target_link_libraries(${_TESTUTIL} ${PRINT_LIB} ${CONAN_LIBS_GTEST} ${CONAN_LIBS_CPPKG} ${CONAN_LIBS_BOOST})

# internal/global
set(GLOBAL_TEST global_test)
add_executable(${GLOBAL_TEST}
    internal/global/test/main.cpp
    internal/global/test/test_log.cpp
    internal/global/test/test_random.cpp)
target_link_libraries(${GLOBAL_TEST} ${_TESTUTIL} ${GLOBAL_LIB})
add_test(NAME ${GLOBAL_TEST} COMMAND ${GLOBAL_TEST})

# internal/marsh
set(MARSH_TEST marsh_test)
add_executable(${MARSH_TEST}
    internal/marsh/test/main.cpp
    internal/marsh/test/test_objs.cpp)
target_link_libraries(${MARSH_TEST} ${_TESTUTIL} ${MARSH_LIB})
add_test(NAME ${MARSH_TEST} COMMAND ${MARSH_TEST})

# internal/teq
set(TEQ_TEST teq_test)
add_executable(${TEQ_TEST}
    internal/teq/test/main.cpp
    internal/teq/test/test_evaluator.cpp
    internal/teq/test/test_grad.cpp
    internal/teq/test/test_leaf.cpp
    internal/teq/test/test_objs.cpp
    internal/teq/test/test_shape.cpp
    internal/teq/test/test_traveler.cpp)
target_link_libraries(${TEQ_TEST} ${_TESTUTIL} ${TEQ_LIB} teq_mock)
add_test(NAME ${TEQ_TEST} COMMAND ${TEQ_TEST})

# internal/eigen
set(EIGEN_TEST eigen_test)
add_executable(${EIGEN_TEST}
    internal/eigen/test/main.cpp
    internal/eigen/test/test_device.cpp
    internal/eigen/test/test_funcopt.cpp
    internal/eigen/test/test_meta.cpp
    internal/eigen/test/test_observable.cpp
    internal/eigen/test/test_operator.cpp
    internal/eigen/test/test_packer.cpp
    internal/eigen/test/test_shaper.cpp
    internal/eigen/test/test_typer.cpp)
target_link_libraries(${EIGEN_TEST} ${_TESTUTIL} ${EIGEN_LIB} eigen_mock)
add_test(NAME ${EIGEN_TEST} COMMAND ${EIGEN_TEST})

# internal/onnx
set(ONNX_TEST onnx_test)
add_executable(${ONNX_TEST}
    internal/onnx/test/main.cpp
    internal/onnx/test/test_load.cpp
    internal/onnx/test/test_save.cpp)
target_link_libraries(${ONNX_TEST} ${_TESTUTIL} ${ONNX_LIB} teq_mock)
add_test(NAME ${ONNX_TEST} COMMAND ${ONNX_TEST})
target_compile_definitions(${ONNX_TEST} PRIVATE CMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}/")

# internal/query
set(QUERY_TEST query_test)
add_executable(${QUERY_TEST}
    internal/query/test/main.cpp
    internal/query/test/test_attrs.cpp
    internal/query/test/test_base.cpp
    internal/query/test/test_path.cpp
    internal/query/test/test_query.cpp)
target_link_libraries(${QUERY_TEST} ${_TESTUTIL} ${QUERY_LIB} teq_mock)
add_test(NAME ${QUERY_TEST} COMMAND ${QUERY_TEST})

# internal/opt
set(OPT_TEST opt_test)
add_executable(${OPT_TEST}
    internal/opt/test/main.cpp
    internal/opt/test/test_apply.cpp
    internal/opt/test/test_graph.cpp
    internal/opt/test/test_parse.cpp
    internal/opt/test/test_parseattr.cpp)
target_link_libraries(${OPT_TEST} ${_TESTUTIL} ${OPT_LIB} teq_mock)
add_test(NAME ${OPT_TEST} COMMAND ${OPT_TEST})
target_compile_definitions(${OPT_TEST} PRIVATE CMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}/")

# tenncor/distr
set(DISTR_TEST distr_test)
add_executable(${DISTR_TEST}
    tenncor/distr/test/main.cpp
    tenncor/distr/test/test_manager.cpp
    tenncor/distr/test/test_peersvc.cpp
    tenncor/distr/test/test_reference.cpp)
target_link_libraries(${DISTR_TEST} ${_TESTUTIL} ${DISTR_LIB} teq_mock)
add_test(NAME ${DISTR_TEST} COMMAND ${DISTR_TEST})

# tenncor/distr/iosvc
set(DISTR_IOSVC_TEST distr_iosvc_test)
add_executable(${DISTR_IOSVC_TEST}
    tenncor/distr/iosvc/test/main.cpp
    tenncor/distr/iosvc/test/test_lookup.cpp
    tenncor/distr/iosvc/test/test_remote.cpp)
target_link_libraries(${DISTR_IOSVC_TEST} ${_TESTUTIL} distr_iosvc_mock teq_mock)
add_test(NAME ${DISTR_IOSVC_TEST} COMMAND ${DISTR_IOSVC_TEST})

# tenncor/eteq
set(ETEQ_TEST eteq_test)
add_executable(${ETEQ_TEST}
    tenncor/eteq/test/main.cpp
    tenncor/eteq/test/test_backprop.cpp
    tenncor/eteq/test/test_caster.cpp
    tenncor/eteq/test/test_constant.cpp
    tenncor/eteq/test/test_etens.cpp
    tenncor/eteq/test/test_functor.cpp
    tenncor/eteq/test/test_variable.cpp)
target_link_libraries(${ETEQ_TEST} ${_TESTUTIL} ${ETEQ_LIB} eigen_mock)
add_test(NAME ${ETEQ_TEST} COMMAND ${ETEQ_TEST})

# tenncor/eteq/opsvc
set(DISTR_OPSVC_TEST distr_opsvc_test)
add_executable(${DISTR_OPSVC_TEST}
    tenncor/eteq/opsvc/test/main.cpp
    tenncor/eteq/opsvc/test/test_derive.cpp
    tenncor/eteq/opsvc/test/test_evaluator.cpp
    tenncor/eteq/opsvc/test/test_reachable.cpp)
target_link_libraries(${DISTR_OPSVC_TEST} ${_TESTUTIL} distr_iosvc_mock distr_opsvc_mock teq_mock)
add_test(NAME ${DISTR_OPSVC_TEST} COMMAND ${DISTR_OPSVC_TEST})

# tenncor/find/lusvc
set(DISTR_LUSVC_TEST distr_lusvc_test)
add_executable(${DISTR_LUSVC_TEST}
    tenncor/find/lusvc/test/main.cpp
    tenncor/find/lusvc/test/test_lookup.cpp)
target_link_libraries(${DISTR_LUSVC_TEST} ${_TESTUTIL} distr_iosvc_mock distr_lusvc_mock teq_mock)
add_test(NAME ${DISTR_LUSVC_TEST} COMMAND ${DISTR_LUSVC_TEST})

# tenncor/hone
set(HONE_TEST hone_test)
add_executable(${HONE_TEST}
    tenncor/hone/test/main.cpp
    tenncor/hone/test/test_cstrules.cpp)
target_link_libraries(${HONE_TEST} ${_TESTUTIL} ${HONE_LIB})
add_test(NAME ${HONE_TEST} COMMAND ${HONE_TEST})
target_compile_definitions(${HONE_TEST} PRIVATE CMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}/")

# tenncor/hone/hosvc
set(DISTR_HOSVC_TEST distr_hosvc_test)
add_executable(${DISTR_HOSVC_TEST}
    tenncor/hone/hosvc/test/main.cpp
    tenncor/hone/hosvc/test/test_optimize.cpp)
target_link_libraries(${DISTR_HOSVC_TEST} ${_TESTUTIL} distr_iosvc_mock distr_hosvc_mock distr_printsvc_mock teq_mock)
add_test(NAME ${DISTR_HOSVC_TEST} COMMAND ${DISTR_HOSVC_TEST})

# tenncor/layr
set(LAYR_TEST layr_test)
add_executable(${LAYR_TEST}
    tenncor/layr/test/main.cpp
    tenncor/layr/test/test_api.cpp
    tenncor/layr/test/test_approx.cpp
    tenncor/layr/test/test_connect.cpp
    tenncor/layr/test/test_init.cpp)
target_link_libraries(${LAYR_TEST} ${_TESTUTIL} ${LAYR_LIB} teq_mock)
add_test(NAME ${LAYR_TEST} COMMAND ${LAYR_TEST})

# tenncor/serial
set(SERIAL_TEST serial_test)
add_executable(${SERIAL_TEST}
    tenncor/serial/test/main.cpp
    tenncor/serial/test/test_serialize.cpp)
target_link_libraries(${SERIAL_TEST} ${_TESTUTIL} ${SERIAL_LIB})
add_test(NAME ${SERIAL_TEST} COMMAND ${SERIAL_TEST})
target_compile_definitions(${SERIAL_TEST} PRIVATE CMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}/")

# tenncor/serial/oxsvc
set(DISTR_OXSVC_TEST distr_oxsvc_test)
add_executable(${DISTR_OXSVC_TEST}
    tenncor/serial/oxsvc/test/main.cpp
    tenncor/serial/oxsvc/test/test_load.cpp
    tenncor/serial/oxsvc/test/test_save.cpp
    tenncor/serial/oxsvc/test/test_segment.cpp)
target_link_libraries(${DISTR_OXSVC_TEST} ${_TESTUTIL} distr_iosvc_mock distr_oxsvc_mock distr_printsvc_mock teq_mock)
add_test(NAME ${DISTR_OXSVC_TEST} COMMAND ${DISTR_OXSVC_TEST})
target_compile_definitions(${DISTR_OXSVC_TEST} PRIVATE CMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}/")

# tenncor
set(ITEST integration_test)
add_executable(${ITEST}
    tenncor/test/main.cpp
    tenncor/test/test_api.cpp
    tenncor/test/test_approx.cpp
    tenncor/test/test_consul.cpp
    tenncor/test/test_distrib.cpp
    tenncor/test/test_equation.cpp
    tenncor/test/test_init.cpp
    tenncor/test/test_layer.cpp
    tenncor/test/test_opt.cpp
    tenncor/test/test_query.cpp
    tenncor/test/test_serialize.cpp)
target_link_libraries(${ITEST} ${_TESTUTIL} ${TENNCOR_LIB} ${UTILS_LIB} ${DISTR_PRINTSVC_LIB})
add_test(NAME ${ITEST} COMMAND ${ITEST})
target_compile_definitions(${ITEST} PRIVATE CMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}/")
