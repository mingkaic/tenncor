licenses(["notice"])

package(
    default_visibility = ["//visibility:public"],
)

filegroup(
    name = "srcs",
    srcs = glob([
        "*.hpp",
        "src/*.cpp",
    ]) + ["BUILD"],
)

filegroup(
    name = "test_srcs",
    srcs = glob([
        "test/*.hpp",
        "test/*.cpp",
    ]),
)

######### LIBRARY #########

cc_library(
    name = "llo",
    hdrs = glob(["*.hpp"]),
    srcs = glob(["src/*.cpp"]),
    copts = ["-std=c++14"],
    deps = ["//ade:ade"],
)

######### TEST #########

cc_test(
    name = "test_llo",
    size = "small",
    srcs = glob(["test/*.hpp", "test/*.cpp"]),
    deps = [
        "//llo:llo",
        "//testutil:testutil",
    ],
    copts = ["-std=c++14"],
)
