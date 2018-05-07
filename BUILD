licenses(["notice"])

package(
    default_visibility = [ "//visibility:public" ],
)

#######################################
#           GENERAL SOURCE            #
#######################################

filegroup(
    name = "srcs",
    srcs = [
        "BUILD",
        "//:tenncor_hpp",
        "//:tenncor_cpp",
        "//:monitor_csv_hpp",
        "//:monitor_csv_cpp",
        "//:monitor_rpc_hpp",
        "//:monitor_rpc_cpp",
        "//proto:srcs",
        "//tests:srcs",
    ],
)

#######################################
#            MAIN SOURCE              #
#######################################

filegroup(
    name = "tenncor_hpp",
    srcs = glob([
        "include/graph/**/*.hpp",
        "include/operate/**/*.hpp",
        "include/tensor/**/*.hpp",
        "include/utils/**/*.hpp",
        "src/graph/**/*.ipp",
        "src/operate/**/*.ipp",
        "src/tensor/**/*.ipp",
        "src/utils/**/*.ipp",
    ]),
)

filegroup(
    name = "tenncor_cpp",
    srcs = glob([
        "src/graph/**/*.cpp",
        "src/operate/**/*.cpp",
        "src/tensor/**/*.cpp",
        "src/utils/**/*.cpp",
    ]),
)

#######################################
#           MONITOR SOURCE            #
#######################################

######### CSV MONITOR #########
filegroup(
    name = "monitor_csv_hpp",
    srcs = glob([
        "include/edgeinfo/*.hpp",
        "include/edgeinfo/csv_record/*.hpp", 
    ]),
)

filegroup(
    name = "monitor_csv_cpp",
    srcs = glob([
        "src/edgeinfo/*.cpp",
        "src/edgeinfo/csv_record/*.cpp",
    ]),
)

######### GRPC MONITOR #########
filegroup(
    name = "monitor_rpc_hpp",
    srcs = glob([
        "include/edgeinfo/*.hpp",
        "include/edgeinfo/rpc_record/*.hpp", 
        "include/thread/*.hpp",
        "src/thread/*.ipp",
    ]),
)

filegroup(
    name = "monitor_rpc_cpp",
    srcs = glob([
        "src/edgeinfo/*.cpp",
        "src/edgeinfo/rpc_record/*.cpp", 
        "src/thread/*.cpp",
    ]),
)

#######################################
#             MAIN LIBRARY            #
#######################################

cc_library(
    name = "tenncor",
    hdrs = [ ":tenncor_hpp" ],
    srcs = [ ":tenncor_cpp" ],
    includes = [ "include" ],
    deps = [
        "//proto:tenncor_serial_cc_proto",
    ],
    copts = [ "-std=c++14" ],
)

#######################################
#           MONITOR LIBRARY           #
#######################################

# ######### CSV MONITOR #########
# cc_library(
#     name = "tenncor_csv",
#     hdrs = [ ":tenncor_hpp", ":monitor_csv_hpp" ],
#     srcs = [ ":tenncor_cpp", ":monitor_csv_cpp" ],
#     includes = [ "include" ],
#     deps = [ 
#         "//proto:tenncor_serial_cc_proto",
#     ],
#     defines = [ "CSV_RCD" ],
#     copts = [ "-std=c++14" ],
# )

# ######### GRPC MONITOR #########
# cc_library(
#     name = "tenncor_rpc",
#     hdrs = [ ":tenncor_hpp", ":monitor_rpc_hpp" ],
#     srcs = [ ":tenncor_cpp", ":monitor_rpc_cpp" ],
#     includes = [ "include" ],
#     deps = [
#         "//proto:tenncor_serial_cc_proto",
#         "//proto:tenncor_monitor_grpc",
#         "@com_github_grpc_grpc//:grpc++",
#     ],
#     defines = [ "RPC_RCD" ],
#     copts = [ "-std=c++14" ],
# )
