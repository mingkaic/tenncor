licenses(["notice"])

package(
    default_visibility = ["//visibility:public"],
)

filegroup(
    name = "srcs",
    srcs = glob(["*.hpp", "*.cpp"]) + ["BUILD"],
)

######### LIBRARY #########

cc_library(
    name = "testutil",
    hdrs = ["common.hpp"],
    srcs = ["common.cpp"],
    deps = [
        "//ade:ade",
        "@com_github_mingkaic_testify//simple:simple",
    ],
    copts = ["-std=c++14"],
)
