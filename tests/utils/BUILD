licenses(["notice"])

cc_library(
    name = "tenncor_testify",
	hdrs = glob(["**/*.hpp"]),
	srcs = glob(["**/*.cpp"]),
    deps = [ 
		"//:tenncor",
        "@com_github_mingkaic_testify//:testify",
    ],
	visibility = [ "//visibility:public" ],
	copts = [ "-std=c++14" ],
)
