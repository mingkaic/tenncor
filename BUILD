licenses(["notice"])

package(
    default_visibility = [ "//visibility:public" ],
)

#######################################
#           GENERAL SOURCE            #
#######################################

filegroup(
    name = "srcs",
    srcs = glob([
        "*util/*.hpp",
        "*util/**/*.cpp",
    ]) + [
        "BUILD",
        "//clay:srcs",
        "//mold:srcs",
        "//slip:srcs",
        "//kiln:srcs",
        "//wire:srcs",
        "//lead:srcs",
        "//regress:srcs",
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
#              LIBRARIES              #
#######################################

cc_library(
    name = "ioutil",
    hdrs = glob(["ioutil/*.hpp"]),
    srcs = glob(["ioutil/src/*.cpp"]),
    copts = ["-std=c++14"],
)

cc_library(
    name = "fuzzutil",
    hdrs = glob(["fuzzutil/*.hpp"]),
    srcs = glob(["fuzzutil/src/*.cpp"]),
    deps = [
        "//:ioutil",
        "@com_github_mingkaic_testify//:testify",
    ],
    copts = ["-std=c++14"],
    testonly = True,
)

cc_library(
    name = "regressutil",
    hdrs = glob([ "regressutil/*.hpp" ]),
    srcs = glob([ "regressutil/src/*.cpp" ]),
    copts = [ "-std=c++14" ],
    deps = [
        "//:ioutil",
        "//wire:wire",
        "//kiln:kiln",
        "@com_google_googletest//:gtest",
    ],
)

# cc_library(
#     name = "protoutil",
#     hdrs = glob(["protoutil/*.hpp"]),
#     srcs = glob(["protoutil/src/*.hpp"]),
#     deps = [
#         "//clay:clay",
#         "//lead:lead",
#         "@com_google_googletest//:gtest"
#     ],
#     copts = ["-std=c++14"],
#     testonly = True,
# )

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
