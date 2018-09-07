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
	name = "ade_ast",
	hdrs = glob([
		"ast/*.h",
		"ast/*.hpp",
	]),
	srcs = glob(["ast/*.cpp"]),
	copts = ["-std=c++14"],
    deps = ["//ade:ade"],
)

######### BINARY #########

cc_binary(
	name = "cli",
	srcs = glob(["*.h", "*.c"]),
    deps = [":ade_ast"],
)
