licenses(["notice"])

#######################################
#         TEST UTILITY FILES          #
#######################################

filegroup(
    name = "srcs",
    srcs = glob([ "include/*.hpp", "src/*.cpp" ]) + [ "BUILD" ],
    visibility = [ "//visibility:public" ],
)

#######################################
#           UTILITY LIBRARY           #
#######################################

# wrapper utility functions for testify
cc_library(
    name = "tenncor_testify",
    hdrs = glob(["include/*.hpp"]),
    srcs = glob(["src/*.cpp"]),
    deps = [ 
        "//:tenncor",
        "@com_github_mingkaic_testify//:testify",
    ],
    includes = [ "include" ],
    copts = [ "-std=c++14" ],
    visibility = [ "//tests/unit:__pkg__" ],
)
