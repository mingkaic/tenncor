licenses(["notice"])

package(
	default_visibility = [ "//visibility:public" ],
)

filegroup(
    name = "srcs",
    srcs = glob([ "include/*.hpp", "src/*.cpp" ]) + [
        "BUILD",
    ],
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
	copts = [ "-std=c++14" ],
)
