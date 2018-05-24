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
    name = "slip",
    hdrs = glob([ "*.hpp" ]),
    srcs = glob([ "src/*.cpp" ]),
    copts = [ "-std=c++14" ],
    deps = [
        "//:ioutil",
        "//mold:mold",
    ],
)

######### TEST #########

cc_test(
    name = "test",
    size = "small",
    srcs = glob([ "test/*.cpp" ]),
    deps = [
        "//slip:slip",
        "//:fuzzutil",
    ],
    linkstatic = 1,
    copts = [ "-std=c++14" ],
)
