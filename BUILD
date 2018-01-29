package(
	default_visibility = [ "//visibility:public" ],
)

licenses(["notice"])

TCR_PUBLIC_HDRS = glob([
	"include/graph/**/*.hpp",
	"include/memory/**/*.hpp",
	"include/tensor/**/*.hpp",
	"include/utils/**/*.hpp",
	"src/graph/**/*.ipp",
	"src/memory/**/*.ipp",
	"src/tensor/**/*.ipp",
	"src/utils/**/*.ipp",
])

TCR_SRC = glob([
	"src/graph/**/*.cpp",
	"src/memory/**/*.cpp",
	"src/tensor/**/*.cpp",
	"src/utils/**/*.cpp",
])

######### Main Library #########

cc_library(
	name = "tenncor",
	hdrs = TCR_PUBLIC_HDRS,
	srcs = TCR_SRC,
	deps = [ "//proto:tenncor_serial_cc_proto" ],
	copts = [ "-std=c++14" ],
)

######### Monitor Library #########

cc_library(
	name = "tenncor_csv",
	hdrs = TCR_PUBLIC_HDRS + 
	glob([
		"include/edgeinfo/*.hpp",
		"include/edgeinfo/csv_record/*.hpp", 
	]),
	srcs = TCR_SRC + 
	glob([
		"src/edgeinfo/*.cpp",
		"src/edgeinfo/csv_record/*.cpp", 
	]),
	deps = [ "//proto:tenncor_serial_cc_proto" ],
	defines = [ "CSV_RCD" ],
	copts = [ "-std=c++14" ],
)

cc_library(
	name = "tenncor_rpc",
	hdrs = TCR_PUBLIC_HDRS + 
	glob([
		"include/edgeinfo/*.hpp",
		"include/edgeinfo/rpc_record/*.hpp", 
		"include/thread/*.hpp",
		"src/thread/*.ipp",
	]),
	srcs = TCR_SRC + 
	glob([
		"src/edgeinfo/*.cpp",
		"src/edgeinfo/rpc_record/*.cpp", 
		"src/thread/*.hpp",
	]),
	deps = [
		"//proto:tenncor_serial_cc_proto",
		"//proto:tenncor_monitor_grpc_proto",
		"@com_github_grpc_grpc//:grpc++",
	],
	defines = [ "RPC_RCD" ],
	copts = [ "-std=c++14" ],
)
