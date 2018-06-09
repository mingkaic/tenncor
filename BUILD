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
        "//lead:srcs",
        "//regress:srcs",
    ],
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
        "//kiln:kiln",
        "@com_google_googletest//:gtest",
    ],
)
