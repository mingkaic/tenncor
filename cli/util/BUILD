licenses(["notice"])

filegroup(
	name = "srcs",
	srcs = glob([
        "*.hpp",
        "*.cpp",
	]) + ["BUILD"],
	visibility = ["//cli"],
)

######### LIBRARY #########

cc_library(
	name = "util",
	hdrs = glob(["*.hpp"]),
	srcs = glob(["*.cpp"]),
	copts = ["-std=c++14"],
	visibility = ["//visibility:public"],
)
