licenses(["notice"])

package(
    default_visibility = [ "//visibility:public" ],
)

filegroup(
    name = "srcs",
    srcs = glob([
        "*.hpp",
        "src/*.cpp",
        "test/*.cpp",
    ]) + [ "BUILD" ],
)

######### LIBRARY #########

cc_library(
    name = "mold",
    hdrs = glob([ "*.hpp" ]),
    srcs = glob([ "src/*.cpp" ]),
    copts = [ "-std=c++14" ],
    deps = [ "//clay:clay" ],
)

######### TEST #########

cc_test(
    name = "test",
    size = "small",
    srcs = glob([ "test/*.cpp" ]),
    deps = [
        "//mold:mold",
        "//:fuzzutil",
    ],
    linkstatic = 1,
    copts = [ "-std=c++14" ],
)
