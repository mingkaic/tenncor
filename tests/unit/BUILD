licenses(["notice"])

#######################################
#           UNIT TEST FILES           #
#######################################

filegroup(
    name = "srcs",
    srcs = glob([ "**/*.cpp" ]) + [ "BUILD" ],
    visibility = [ "//visibility:public" ],
)

UNITTEST_DEPS = [
    "//:tenncor",
    "//tests/utils:tenncor_testify",
    "@com_github_mingkaic_testify//:testify",
]

######### FUZZ TESTS #########

cc_test(
    name = "test_tensor",
    size = "small",
    srcs = glob([ "tensor/*.cpp", "main.cpp" ]),
    deps = UNITTEST_DEPS,
    linkstatic = 1,
    copts = [ "-std=c++14" ],
)

cc_test(
    name = "test_graph",
    size = "small",
    srcs = glob([ "graph/*.cpp", "main.cpp" ]),
    deps = UNITTEST_DEPS,
    linkstatic = 1,
    copts = [ "-std=c++14" ],
)

cc_test(
    name = "test_operate",
    size = "small",
    srcs = glob([ "operate/*.cpp", "main.cpp" ]),
    deps = UNITTEST_DEPS,
    linkstatic = 1,
    copts = [ "-std=c++14" ],
)

cc_test(
    name = "test_serialize",
    size = "small",
    srcs = glob([ "serialize/*.cpp", "main.cpp" ]),
    deps = UNITTEST_DEPS,
    data = glob([ "samples/**" ]),
    linkstatic = 1,
    copts = [ "-std=c++14" ],
)
