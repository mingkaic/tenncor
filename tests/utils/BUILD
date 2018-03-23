licenses(["notice"])

filegroup(
    name = "srcs",
    srcs = glob([ "include/*.hpp", "src/*.cpp" ]) + [
        "BUILD",
    ],
	visibility = ["//visibility:public"],
)

cc_library(
    name = "tenncor_testify",
	hdrs = glob(["include/*.hpp"]),
	srcs = glob(["src/*.cpp"]),
    deps = [ 
		"//:tenncor",
        "@com_github_mingkaic_testify//:testify",
    ],
	includes = [ "include" ],
	visibility = [ "//visibility:public" ],
	copts = [ "-std=c++14" ],
)
