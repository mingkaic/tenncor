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
    ]) + [
        ":hdrs",
        ":srcs",
        ":test_srcs",
        "BUILD",
    ],
)

######### LIBRARY #########

cc_library(
    name = "clay",
    hdrs = glob([ "*.hpp" ]),
    srcs = glob([ "src/*.cpp" ]),
    copts = [ "-std=c++14" ],
)

######### TEST #########

cc_library(
    name = "testutil",
    hdrs = glob(["testutil/*.hpp"]),
    srcs = glob(["testutil/src/*.cpp"]),
    deps = [
        "//clay:clay",
        "//:ioutil",
        "@com_github_mingkaic_testify//:testify",
    ],
    copts = [ "-std=c++14" ],
    testonly = True,
)

cc_test(
    name = "test",
    size = "small",
    srcs = glob([ "test/*.cpp" ]),
    deps = [
        "//clay:clay",
        "//clay:testutil",
        "@com_github_mingkaic_testify//:testify",
    ],
    linkstatic = 1,
    copts = [ "-std=c++14" ],
)
