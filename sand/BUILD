licenses(["notice"])

package(
    default_visibility = ["//visibility:public"],
)

filegroup(
    name = "srcs",
    srcs = glob([
        "*.hpp",
        "include/*.hpp",
        "src/*.cpp",
        "test/*.cpp",
    ]) + ["BUILD"],
)

######### LIBRARY #########

cc_library(
    name = "sand",
    hdrs = glob([
        "*.hpp",
        "include/*.hpp",
    ]),
    srcs = glob(["src/*.cpp"]),
    copts = ["-std=c++14"],
    deps = ["//:util"],
)

######### TEST #########

cc_test(
    name = "test",
    size = "small",
    srcs = glob(["test/*.cpp"]),
    deps = [
        "//sand:sand",
        "@com_google_googletest//:gtest",
    ],
    linkstatic = 1,
    copts = ["-std=c++14"],
)
