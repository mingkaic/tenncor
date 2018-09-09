licenses(["notice"])

package(
	default_visibility = ["//visibility:public"],
)

filegroup(
	name = "srcs",
	srcs = glob([
		"*.hpp",
		"src/*.cpp",
		"test/*.hpp",
		"test/*.cpp",
	]) + ["BUILD"],
)

######### LIBRARY #########

cc_library(
	name = "llo",
	hdrs = glob(["*.hpp"]),
	srcs = glob(["src/*.cpp"]),
	copts = ["-std=c++14"],
	deps = ["//ade:ade"],
)

######### TEST #########

cc_test(
	name = "test",
	size = "small",
	srcs = glob(["test/*.hpp", "test/*.cpp"]),
	deps = [
		"//llo:llo",
        "@com_google_googletest//:gtest",
	],
	linkstatic = 1,
	copts = ["-std=c++14"],
)
