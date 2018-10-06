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

filegroup(
    name = "test_data",
    srcs = glob(["test/data/*.txt"]),
)

######### LIBRARY #########

cc_library(
    name = "ade",
    hdrs = glob(["*.hpp"]),
    srcs = glob(["src/*.cpp"]),
    copts = ["-std=c++14"],
    deps = ["//util:util"],
)

######### TEST #########

cc_test(
    name = "test_dynamic",
    size = "small",
    srcs = [
        "test/dmain.cpp",
        "test/test_functor.cpp",
        "test/test_fwder.cpp",
        "test/test_shape.cpp",
        "test/test_tensor.cpp",
    ],
    deps = ["//testutil:testutil"],
    copts = ["-std=c++14"],
)

cc_test(
    name = "test_static",
    size = "small",
    srcs = glob([
        "test/smain.cpp",
        "test/test_grader.cpp",
    ]),
    deps = [
        "//dbg:dbg",
        "@com_google_googletest//:gtest",
    ],
    copts = ["-std=c++14"],
    data = ["//ade:test_data"],
)
