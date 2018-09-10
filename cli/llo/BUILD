licenses(["notice"])

filegroup(
	name = "srcs",
	srcs = glob([
		"**/*.h", "*.c",
		"ast/*.hpp", "ast/*.cpp",
	]) + [
		"BUILD",
		"ascii/srcs",
	],
	visibility = ["//visibility:public"],
)

######### LIBRARY #########

cc_library(
	name = "ast",
	hdrs = glob(["ast/*.h"]),
	srcs = glob(["ast/*.cpp"]),
	copts = ["-std=c++14"],
    deps = [
		"//llo:llo",
		"//cli/util:util",
	],
)

######### BINARY #########

cc_binary(
	name = "cli",
	srcs = glob(["*.h", "*.c"]),
    deps = [":ast"],
)
