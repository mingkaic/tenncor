package(
	default_visibility = [ "//visibility:public" ],
)

licenses(["notice"])

load("@org_pubref_rules_protobuf//cpp:rules.bzl", "cc_proto_library")
load("@org_pubref_rules_protobuf//python:rules.bzl", "py_proto_library")

#######################################
#           GENERAL PROTOS            #
#######################################

filegroup(
    name = "srcs",
    srcs = [ ":serial", ":monitor", "BUILD" ],
)

#######################################
#            LOCAL PROTOS             #
#######################################

filegroup(
    name = "serial",
    srcs = glob([ "serial/*.proto" ]),
)

filegroup(
    name = "monitor",
    srcs = glob([ "monitor/*.proto" ]),
)

#######################################
#             LIBRARIES               #
#######################################

# grpc monitoring library
cc_proto_library(
	name = "tenncor_monitor_grpc_proto",
	protos = [ ":monitor" ],
	with_grpc = True,
)

# serialization library
cc_proto_library(
	name = "tenncor_serial_cc_proto",
	protos = [ ":serial" ],
    imports = ["external/com_google_protobuf/src"],
    inputs = ["@com_google_protobuf//:well_known_protos"],
	with_grpc = False,
)

py_proto_library(
	name = "tenncor_serial_py_proto",
	protos = [ ":serial" ],
    imports = ["external/com_google_protobuf/src"],
    inputs = ["@com_google_protobuf//:well_known_protos"],
	with_grpc = False,
)
