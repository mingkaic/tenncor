licenses(["notice"])

package(
    default_visibility = ["//visibility:public"],
)

filegroup(
    name = "srcs",
    srcs = glob([
        "*.hpp",
        "src/*.cpp",
        "test/*.hpp",
        "test/*.cpp",
    ]) + ["BUILD"],
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

filegroup(
    name = "testdata",
    srcs = glob(["test/data/*.txt"]),
)

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
        "//ade:ade",
        "//cli/util:util",
        "@com_google_googletest//:gtest",
    ],
    copts = ["-std=c++14"],
    data = ["//ade:testdata"],
)
