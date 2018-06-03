licenses(["notice"])

#######################################
#        REGRESSION TEST FILES        #
#######################################

filegroup(
    name = "srcs",
    srcs = glob([
        "*.hpp",
        "test/*.cpp",
        "tf_generate/*.py",
    ]) + [ "BUILD" ],
    visibility = [ "//visibility:public" ],
)

######### REGRESSION TEST #########

cc_test(
    name = "test",
    size = "enormous",
    srcs = glob([ "test/*.cpp" ]),
    data = glob([ "samples/**" ]),
    deps = [ "//:regressutil" ],
    linkstatic = 1,
    copts = [ "-std=c++14" ],
)
