package(
    default_visibility = [ "//visibility:public" ],
)

licenses(["notice"])

load("@org_pubref_rules_protobuf//cpp:rules.bzl", "cc_proto_library")
load("@org_pubref_rules_protobuf//python:rules.bzl", "py_proto_library")

filegroup(
    name = "srcs",
    srcs = [ ":graphmgr", "BUILD" ],
)

filegroup(
    name = "graphmgr",
    srcs = [ "graphmgr.proto" ],
)

proto_library(
    name = "grpc",
    srcs = [ ":graphmgr" ],
    deps = [
        "//lead:proto",
        "@com_google_protobuf//:any_proto",
    ],
)

cc_proto_library(
    name = "cc_grpc",
    protos = [ ":graphmgr" ],
    imports = [ "external/com_google_protobuf/src" ],
    proto_deps = [ "//lead:cc_proto" ],
    inputs = [
        "//lead:proto",
        "@com_google_protobuf//:well_known_protos",
    ],
    with_grpc = True,
)

py_proto_library(
    name = "py_grpc",
    protos = [ ":graphmgr" ],
    imports = [ "external/com_google_protobuf/src" ],
    proto_deps = [ "//lead:py_proto" ],
    inputs = [
        "//lead:proto",
        "@com_google_protobuf//:well_known_protos",
    ],
    with_grpc = True,
)
