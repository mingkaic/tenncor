package(
    default_visibility = [ "//visibility:public" ],
)

licenses(["notice"])

load("@org_pubref_rules_protobuf//cpp:rules.bzl", "cc_proto_library")
load("@org_pubref_rules_protobuf//python:rules.bzl", "py_proto_library")
load("@io_bazel_rules_go//proto:def.bzl", "go_proto_library")

filegroup(
    name = "srcs",
    srcs = [ ":graphmgr", "BUILD" ],
)

filegroup(
    name = "graphmgr",
    srcs = [ "graphgen.proto" ],
)

proto_library(
    name = "graphmgr_grpc",
    srcs = [ ":graphmgr" ],
    deps = [
        "//proto:tenncor_serial_proto",
        "@com_google_protobuf//:any_proto",
    ],
)

cc_proto_library(
    name = "graphmgr_cc_grpc",
    protos = [ ":graphmgr" ],
    imports = [ "external/com_google_protobuf/src" ],
    proto_deps = [ "//proto:tenncor_serial_cc_proto" ],
    inputs = [
        "@com_google_protobuf//:well_known_protos",
        "//proto:serial",
    ],
    with_grpc = True,
)

py_proto_library(
    name = "graphmgr_py_grpc",
    protos = [ ":graphmgr" ],
    imports = [ "external/com_google_protobuf/src" ],
    proto_deps = [ "//proto:tenncor_serial_py_proto" ],
    inputs = [
        "@com_google_protobuf//:well_known_protos",
        "//proto:serial",
    ],
    with_grpc = True,
)

go_proto_library(
    name = "graphmgr_go_grpc",
    compilers = [ "@io_bazel_rules_go//proto:go_grpc" ],
    importpath = "github.com/mingkaic/tenncor/tests/graphmgr",
    proto = ":graphmgr_grpc",
    deps = [ "//proto:tenncor_serial_go_proto" ],
)
