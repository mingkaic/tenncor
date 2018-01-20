cc_proto_library(
	name = "tenncor_monitor_cc_proto",
	deps = [":tenncor_monitor"],
	visibility = ["//visibility:public"],
)

cc_proto_library(
	name = "tenncor_serial_cc_proto",
	deps = [":tenncor_serial"],
	visibility = ["//:__pkg__"],
)

proto_library(
	name = "tenncor_monitor",
	srcs = ["monitor/grpc_gui.proto"],
)

proto_library(
	name = "tenncor_serial",
	srcs = ["serial/tenncor.proto"],
)
