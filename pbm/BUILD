licenses(["notice"])

package(
    default_visibility = ["//visibility:public"],
)

load("@org_pubref_rules_protobuf//cpp:rules.bzl", "cc_proto_library")
load("@org_pubref_rules_protobuf//python:rules.bzl", "py_proto_library")

filegroup(
    name = "srcs",
    srcs = glob([
        "*.hpp",
        "src/*.cpp",
    ]) + [":protos", "BUILD"],
)

filegroup(
    name = "test_srcs",
    srcs = glob([
        "test/*.hpp",
        "test/*.cpp",
    ]),
)

filegroup(
    name = "protos",
    srcs = glob(["*.proto"]),
)

######### LIBRARY #########

proto_library(
    name = "proto_lib",
    srcs = ["//pbm:protos"],
    deps = [
        "@com_google_protobuf//:any_proto",
        "@com_google_protobuf//:empty_proto",
    ],
)

cc_proto_library(
    name = "cc_proto",
    protos = ["//pbm:protos"],
    imports = ["external/com_google_protobuf/src"],
    inputs = ["@com_google_protobuf//:well_known_protos"],
    with_grpc = False,
)

py_proto_library(
    name = "py_proto",
    protos = ["//pbm:protos"],
    imports = ["external/com_google_protobuf/src"],
    inputs = ["@com_google_protobuf//:well_known_protos"],
    with_grpc = False,
)

cc_library(
    name = "pbm",
    hdrs = glob(["*.hpp"]),
    srcs = glob(["src/*.cpp"]),
    copts = ["-std=c++14"],
    deps = ["//llo:llo", ":cc_proto"],
)

######### TEST #########

filegroup(
    name = "testdata",
    srcs = glob([
        "test/data/*.pb",
        "test/data/*.txt",
    ]),
)

cc_test(
    name = "test_pbm",
    size = "small",
    srcs = glob([
        "test/main.cpp",
        "test/test_load.cpp",
        "test/test_save.cpp",
    ]),
    deps = [
        "//pbm:pbm",
        "//cli/util:util",
        "@com_google_googletest//:gtest",
    ],
    copts = ["-std=c++14"],
    data = ["//pbm:testdata"],
)
