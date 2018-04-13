package(
    default_visibility = [ "//visibility:public" ],
)

licenses(["notice"])

load("@org_pubref_rules_protobuf//cpp:rules.bzl", "cc_proto_library")
load("@org_pubref_rules_protobuf//go:rules.bzl", "go_proto_library")
load("@org_pubref_rules_protobuf//python:rules.bzl", "py_proto_library")

filegroup(
    name = "srcs",
    srcs = [ ":graphmgr", "BUILD" ],
)

filegroup(
    name = "graphmgr",
    srcs = glob([ "**/*.proto" ])
)

cc_proto_library(
    name = "graphmgr_cc_grpc",
    protos = glob([ "**/*.proto" ]),
    imports = [ "external/com_google_protobuf/src" ],
    proto_deps = [ "//proto:tenncor_serial_cc_proto" ],
    inputs = [
        "@com_google_protobuf//:well_known_protos",
        "//proto:serial",
    ],
    with_grpc = True,
)

go_proto_library(
    name = "graphmgr_go_grpc",
    protos = [ "//tests/graphmgr:graphmgr" ],
    imports = [ "external/com_google_protobuf/src" ],
    proto_deps = [ "//proto:tenncor_serial_go_proto" ],
    inputs = [ "//proto:serial" ],
    deps = [ "//proto:tenncor_serial_go_proto" ],
    with_grpc = True,
)

py_proto_library(
    name = "graphmgr_py_grpc",
    protos = glob([ "**/*.proto" ]),
    imports = [ "external/com_google_protobuf/src" ],
    proto_deps = [ "//proto:tenncor_serial_py_proto" ],
    inputs = [
        "@com_google_protobuf//:well_known_protos",
        "//proto:serial",
    ],
    with_grpc = True,
)
