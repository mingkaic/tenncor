licenses(["notice"])

package(
    default_visibility = ["//visibility:public"],
)

filegroup(
    name = "srcs",
    srcs = glob([
        "*.hpp",
        "*.cpp",
    ]) + ["BUILD"],
)

######### LIBRARY #########

cc_library(
    name = "dbg",
    hdrs = glob(["*.hpp"]),
    srcs = glob(["*.cpp"]),
    copts = ["-std=c++14"],
    deps = ["//ade:ade"],
)
