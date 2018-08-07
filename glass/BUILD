licenses(["notice"])

package(
    default_visibility = [ "//visibility:public" ],
)

load("@org_pubref_rules_protobuf//cpp:rules.bzl", "cc_proto_library")
load("@org_pubref_rules_protobuf//python:rules.bzl", "py_proto_library")

filegroup(
    name = "srcs",
    srcs = glob([
        "*.hpp",
        "src/*.cpp",
        "test/*.cpp",
    ]) + [
        "BUILD",
        ":protos",
    ],
)

filegroup(
    name = "protos",
    srcs = glob([ "*.proto" ]),
)

######### LIBRARY #########

proto_library(
    name = "proto",
    srcs = [ "//glass:protos" ],
    deps = [ "@com_google_protobuf//:any_proto" ],
)

cc_proto_library(
    name = "cc_proto",
    protos = [ "//glass:protos" ],
    imports = [ "external/com_google_protobuf/src" ],
    inputs = [ "@com_google_protobuf//:well_known_protos" ],
    with_grpc = False,
)

py_proto_library(
    name = "py_proto",
    protos = [ "//glass:protos" ],
    imports = [ "external/com_google_protobuf/src" ],
    inputs = [ "@com_google_protobuf//:well_known_protos" ],
    with_grpc = False,
)

cc_library(
    name = "glass",
    hdrs = glob([ "*.hpp" ]),
    srcs = glob([ "src/*.cpp" ]),
    copts = [ "-std=c++14" ],
    deps = [
        "//:util",
        "//soil:soil",
        "//glass:cc_proto",
    ],
)

######### TEST #########

cc_test(
    name = "test",
    size = "small",
    srcs = glob([ "test/*.cpp" ]),
    deps = [
        "//glass:glass",
        "//:fuzzutil",
    ],
    linkstatic = 1,
    copts = [ "-std=c++14" ],
)
