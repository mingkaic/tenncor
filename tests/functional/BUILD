cc_test(
	name = "tenncor_verify",
	size = "enormous",
	srcs = glob([ "include/**/*.hpp", "src/**/*.cpp" ]),
	data = glob([ "samples/**" ]),
	deps = [
		"@com_google_googletest//:gtest_main",
		"//:tenncor",
	],
	linkstatic = 1,
	copts = [ "-std=c++14" ]
)
