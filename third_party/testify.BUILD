licenses(["notice"])

cc_library(
	name = "testify",
	hdrs = glob([ "include/**/*.hpp" ]),
	srcs = glob([ "src/**/*.cpp" ]),
	includes = [ "include" ],
	deps = [ "@com_google_googletest//:gtest" ],
	visibility = [ "//visibility:public" ],
	copts = [ "-std=c++14" ],
)
