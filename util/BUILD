licenses(["notice"])

package(
    default_visibility = ["//visibility:public"],
)

filegroup(
    name = "srcs",
    srcs = glob([
        "*.hpp",
        "src/*.cpp",
        "test/*.cpp",
    ]) + ["BUILD"],
)

######### LIBRARY #########

cc_library(
    name = "util",
    hdrs = glob(["*.hpp"]),
    srcs = glob(["src/*.cpp"]),
    copts = ["-std=c++14"],
)

######### TEST #########

cc_test(
    name = "test_util",
    size = "small",
    srcs = glob(["test/*.cpp"]),
    deps = [
        "//util:util",
        "@com_google_googletest//:gtest",
    ],
    copts = ["-std=c++14"],
)
