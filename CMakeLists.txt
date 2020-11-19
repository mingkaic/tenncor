cmake_minimum_required(VERSION 3.10)

project(Tenncor C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#### third_parties ####

# cppkg, boost, grpc
add_subdirectory(third_party/cmake_repos/cppkg EXCLUDE_FROM_ALL)

# eigen
if(NOT Eigen_FOUND AND NOT TARGET eigen)
    add_subdirectory(third_party/cmake_repos/eigen-git-mirror EXCLUDE_FROM_ALL)
endif()

# ppconsul
if(NOT Ppconsul_FOUND AND NOT TARGET ppconsul)
    add_subdirectory(third_party/cmake_repos/ppconsul EXCLUDE_FROM_ALL)
endif()

# g3log
if(NOT g3log_FOUND AND NOT TARGET g3log)
    set(G3_SHARED_LIB OFF CACHE BOOL "Build shared library." FORCE)
    set(CHANGE_G3LOG_DEBUG_TO_DBUG ON CACHE BOOL "Use DBUG logging level instead of DEBUG. By default DEBUG is the debugging level" FORCE)
    set(USE_DYNAMIC_LOGGING_LEVELS ON CACHE BOOL "Turn ON/OFF log levels. An disabled level will not push logs of that level to the sink. By default dynamic logging is disabled" FORCE)
    add_subdirectory(third_party/cmake_repos/g3log EXCLUDE_FROM_ALL)
    if (CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
        target_compile_options(g3logger PUBLIC -fvisibility=hidden -fvisibility-inlines-hidden)
    endif()
endif()

#### local libraries ####

# internal/global
set(GLOBAL_LIB ${PROJECT_NAME}_global)
add_library(${GLOBAL_LIB}
    internal/global/src/config.cpp
    internal/global/src/g3logs.cpp
    internal/global/src/logs.cpp
    internal/global/src/random.cpp
)
target_include_directories(${GLOBAL_LIB} PUBLIC .)
target_link_libraries(${GLOBAL_LIB} PUBLIC Boost::boost Cppkg_estd Cppkg_logs g3log)

# internal/marsh
set(MARSH_LIB ${PROJECT_NAME}_marsh)
add_library(${MARSH_LIB} internal/marsh/src/objs.cpp)
target_include_directories(${MARSH_LIB} PUBLIC .)
target_link_libraries(${MARSH_LIB} PUBLIC Cppkg_estd)

# internal/teq
set(TEQ_LIB ${PROJECT_NAME}_teq)
add_library(${TEQ_LIB}
    internal/teq/src/derive.cpp
    internal/teq/src/evaluator.cpp
    internal/teq/src/ileaf.cpp
    internal/teq/src/shape.cpp
    internal/teq/src/traveler.cpp
)
target_include_directories(${TEQ_LIB} PUBLIC .)
target_link_libraries(${TEQ_LIB} PUBLIC ${GLOBAL_LIB} ${MARSH_LIB})

# internal/eigen
set(EIGEN_LIB ${PROJECT_NAME}_eigen)
add_library(${EIGEN_LIB}
    internal/eigen/src/convert.cpp
    internal/eigen/src/packattr.cpp
)
target_include_directories(${EIGEN_LIB} PUBLIC .)
target_link_libraries(${EIGEN_LIB} PUBLIC ${TEQ_LIB} Eigen3::Eigen)

# internal/onnx
set(ONNX_LIB ${PROJECT_NAME}_onnx)
add_library(${ONNX_LIB}
    internal/onnx/onnx.pb.cc
    internal/onnx/src/load.cpp
    internal/onnx/src/marshal.cpp
)
target_include_directories(${ONNX_LIB} PUBLIC .)
target_link_libraries(${ONNX_LIB} PUBLIC ${TEQ_LIB} Boost::boost protobuf::libprotobuf)

# internal/query
set(QUERY_LIB ${PROJECT_NAME}_query)
add_library(${QUERY_LIB}
    internal/query/query.pb.cc
    internal/query/src/parse.cpp
    internal/query/src/querier.cpp
)
target_include_directories(${QUERY_LIB} PUBLIC .)
target_link_libraries(${QUERY_LIB} PUBLIC ${EIGEN_LIB} protobuf::libprotobuf)

# internal/opt
set(OPT_LIB ${PROJECT_NAME}_opt)
add_library(${OPT_LIB}
    internal/opt/optimize.pb.cc
    internal/opt/src/apply.cpp
    internal/opt/src/parse.cpp
)
target_include_directories(${OPT_LIB} PUBLIC .)
target_link_libraries(${OPT_LIB} PUBLIC ${QUERY_LIB})

## tenncor/distr
#set(DISTR_LIB ${PROJECT_NAME}_distr)
#add_library(${DISTR_LIB}
#    tenncor/distr/src/manager.cpp
#    tenncor/distr/src/p2p.cpp
#    tenncor/distr/src/reference.cpp
#)
#target_include_directories(${DISTR_LIB} PUBLIC .)
#target_link_libraries(${DISTR_LIB} PUBLIC ${EIGEN_LIB} ppconsul Cppkg_egrpc Cppkg_error)

## tenncor/distr/iosvc
#set(DISTR_IOSVC_LIB ${PROJECT_NAME}_distr_iosvc)
#add_library(${DISTR_IOSVC_LIB}
#)
#target_include_directory(${DISTR_IOSVC_LIB} PUBLIC .)
#target_link_libraries(${DISTR_IOSVC_LIB} PUBLIC ${DISTR_LIB})

# tenncor/eteq
set(ETEQ_LIB ${PROJECT_NAME}_eteq)
add_library(${ETEQ_LIB}
    tenncor/eteq/src/etens.cpp
    tenncor/eteq/src/make.cpp
)
target_include_directories(${ETEQ_LIB} PUBLIC .)
target_link_libraries(${ETEQ_LIB} PUBLIC ${EIGEN_LIB})

## tenncor/eteq/opsvc
#set(DISTR_OPSVC_LIB ${PROJECT_NAME}_distr_opsvc)
#add_library(${DISTR_OPSVC_LIB}
#    tenncor/eteq/opsvc/distr.op.grpc.pb.cc
#    tenncor/eteq/opsvc/distr.op.pb.cc
#    tenncor/eteq/opsvc/src/service.cpp
#)
#target_include_directories(${DISTR_OPSVC_LIB} PUBLIC .)
#target_link_libraries(${DISTR_OPSVC_LIB} PUBLIC ${DISTR_IOSVC_LIB} ${ETEQ_LIB})

## tenncor/find/lusvc
#set(DISTR_LUSVC_LIB ${PROJECT_NAME}_distr_lusvc)
#add_library(${DISTR_LUSVC_LIB}
#    tenncor/find/lusvc/distr.lu.grpc.pb.cc
#    tenncor/find/lusvc/distr.lu.pb.cc
#    tenncor/find/lusvc/src/service.cpp
#)
#target_include_directories(${DISTR_LUSVC_LIB} PUBLIC .)
#target_link_libraries(${DISTR_LUSVC_LIB} PUBLIC ${DISTR_IOSVC_LIB} ${QUERY_LIB})

# tenncor/hone
set(HONE_LIB ${PROJECT_NAME}_hone)
add_library(${HONE_LIB}
    tenncor/hone/src/cstrules.cpp
    tenncor/hone/src/duplicates.cpp
    tenncor/hone/src/optimize.cpp
)
target_include_directories(${HONE_LIB} PUBLIC .)
target_link_libraries(${HONE_LIB} PUBLIC ${OPT_LIB} ${ETEQ_LIB})

## tenncor/hone/hosvc
#set(DISTR_HOSVC_LIB ${PROJECT_NAME}_distr_hosvc)
#add_library(${DISTR_HOSVC_LIB}
#    tenncor/hone/hosvc/distr.ho.grpc.pb.cc
#    tenncor/hone/hosvc/distr.ho.pb.cc
#    tenncor/hone/hosvc/src/service.cpp
#)
#target_include_directories(${DISTR_HOSVC_LIB} PUBLIC .)
#target_link_libraries(${DISTR_HOSVC_LIB} PUBLIC ${DISTR_IOSVC_LIB} ${HONE_LIB})

# tenncor/layr
set(LAYR_LIB ${PROJECT_NAME}_layr)
add_library(${LAYR_LIB}
    tenncor/layr/src/layer.cpp
)
target_include_directories(${LAYR_LIB} PUBLIC .)
target_link_libraries(${LAYR_LIB} PUBLIC ${ETEQ_LIB})

# tenncor/serial
set(SERIAL_LIB ${PROJECT_NAME}_serial)
add_library(${SERIAL_LIB}
    tenncor/serial/src/serialize.cpp
)
target_include_directories(${SERIAL_LIB} PUBLIC .)
target_link_libraries(${SERIAL_LIB} PUBLIC ${ONNX_LIB} ${LAYR_LIB})

## tenncor/serial/oxsvc
#set(DISTR_OXSVC_LIB ${PROJECT_NAME}_distr_oxsvc)
#add_library(${DISTR_OXSVC_LIB}
#    tenncor/serial/oxsvc/distr.ox.grpc.pb.cc
#    tenncor/serial/oxsvc/distr.ox.pb.cc
#    tenncor/serial/oxsvc/src/segment.cpp
#    tenncor/serial/oxsvc/src/service.cpp
#    tenncor/serial/oxsvc/src/topography.cpp
#    tenncor/serial/oxsvc/src/util.cpp
#)
#target_include_directories(${DISTR_OXSVC_LIB} PUBLIC .)
#target_link_libraries(${DISTR_OXSVC_LIB} PUBLIC ${DISTR_IOSVC_LIB} ${SERIAL_LIB})

# tenncor/trainer
set(TRAINER_LIB ${PROJECT_NAME}_trainer)
add_library(${TRAINER_LIB} INTERFACE)
target_include_directories(${TRAINER_LIB} INTERFACE .)
target_link_libraries(${TRAINER_LIB} INTERFACE ${LAYR_LIB})
