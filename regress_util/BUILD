licenses(["notice"])

package(
    default_visibility = [ "//visibility:public" ],
)

filegroup(
    name = "srcs",
    srcs = glob([
        "*.hpp",
        "src/*.cpp",
    ]) + [ "BUILD" ],
)

######### LIBRARY #########

cc_library(
    name = "regress_util",
    hdrs = glob([ "*.hpp" ]),
    srcs = glob([ "src/*.cpp" ]),
    copts = [ "-std=c++14" ],
    deps = [
        "//:ioutil",
        "//wire:wire",
        "//kiln:kiln",
        "@com_google_googletest//:gtest",
    ],
)
