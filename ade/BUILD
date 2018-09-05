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
	name = "ade",
	hdrs = glob(["*.hpp"]),
	srcs = glob(["src/*.cpp"]),
	copts = ["-std=c++14"],
	deps = ["//util:util"],
)

######### TEST #########

cc_test(
	name = "test",
	size = "small",
	srcs = glob(["test/*.hpp", "test/*.cpp"]),
	deps = [
		"//ade:ade",
        "@com_google_googletest//:gtest",
		#"@com_github_mingkaic_testify//simple:simple",
	],
	linkstatic = 1,
	copts = ["-std=c++14"],
)
