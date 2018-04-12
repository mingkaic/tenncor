licenses(["notice"])

#######################################
#        REGRESSION TEST FILES        #
#######################################

filegroup(
    name = "srcs",
    srcs = glob([ "**/*.hpp", "**/*.cpp", "**/*.py" ]) + [ "BUILD" ],
    visibility = [ "//visibility:public" ],
)

######### REGRESSION TEST #########

cc_test(
    name = "test",
    size = "enormous",
    srcs = glob([ "include/**/*.hpp", "src/**/*.cpp" ]),
    data = glob([ "samples/**" ]),
    deps = [
        "@com_google_googletest//:gtest",
        "//:tenncor",
    ],
    linkstatic = 1,
    copts = [ "-std=c++14" ]
)
