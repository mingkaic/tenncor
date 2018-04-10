package(
    default_visibility = [ "//visibility:public" ],
)

licenses(["notice"])

load("@org_pubref_rules_protobuf//python:rules.bzl", "py_proto_library")

py_proto_library(
    name = "expect_pb",
    protos = [ "expect.proto" ],
    imports = [
        "external/com_google_protobuf/src",
        "//proto:tenncor_serial_py_proto",
    ],
    inputs = [
        "@com_google_protobuf//:well_known_protos",
        "//proto:serial",
    ],
    with_grpc = False,
)

py_library(
    name = "tenncorgen",
    srcs = [
        "save_ast.py",
        "save_data.py", 
        "tfgen.py",
        "utils.py",
    ],
    imports = [ "." ],
    deps = [
        ":expect_pb",
        "//proto:tenncor_serial_py_proto",
        "@com_github_mingkaic_testify//py:graphast",
    ],
)

py_test(
    name = "test",
    srcs = [
        "save_ast_test.py", 
        "test.py",
    ],
    deps = [ ":tenncorgen" ]
)

py_binary(
    name = "protogen",
    srcs = [ "protogen.py" ],
    deps = [ ":tenncorgen" ],
)
