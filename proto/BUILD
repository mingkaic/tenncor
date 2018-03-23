package(
	default_visibility = [ "//visibility:public" ],
)

licenses(["notice"])

load("//bazel:grpc_proto.bzl", "grpc_proto_library")

filegroup(
    name = "srcs",
    srcs = glob([ "**/*.proto" ]) + [ "BUILD" ],
)

# grpc monitoring library
grpc_proto_library(
	name = "tenncor_monitor_grpc_proto",
	src = "monitor/grpc_gui.proto",
)

# serialization library
cc_proto_library(
	name = "tenncor_serial_cc_proto",
	deps = [ ":tenncor_serial" ],
	visibility = [ "//:__pkg__" ],
)

# py_proto_library(
# 	name = "tenncor_serial_py_proto",
# 	deps = [ ":tenncor_serial" ],
# )

proto_library(
	name = "tenncor_serial",
	srcs = [ "serial/data.proto", "serial/graph.proto" ],
	deps = [ "@com_google_protobuf//:any_proto" ],
)
