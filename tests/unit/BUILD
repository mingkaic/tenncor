COMMON_SRC = glob([
	"include/utils/*.h",
	"include/utils/*.hpp",
	"include/mocks/mock_actor.h",
	"src/utils/*.cpp",
])

licenses(["notice"])

######### FUZZ TESTS #########

cc_test(
	name = "tenncor_tensor",
	size = "small",
	srcs = glob([ "tensor/*.cpp", "main.cpp" ]),
	deps = [
		"//:tenncor",
		"//tests/utils:tenncor_testify",
	],
	linkstatic = 1,
	copts = [ "-std=c++14" ],
)

cc_test(
	name = "tenncor_bases",
	size = "small",
	srcs = glob([ "bases/*.cpp", "main.cpp" ]),
	deps = [
		"//:tenncor",
		"//tests/utils:tenncor_testify",
	],
	linkstatic = 1,
	copts = [ "-std=c++14" ],
)

cc_test(
	name = "tenncor_funcs",
	size = "small",
	srcs = glob([ "funcs/*.cpp", "main.cpp" ]),
	deps = [
		"//:tenncor",
		"//tests/utils:tenncor_testify",
	],
	linkstatic = 1,
	copts = [ "-std=c++14" ],
)

cc_test(
	name = "tenncor_operation",
	size = "medium",
	srcs = glob([ "operations/*.cpp", "main.cpp" ]),
	deps = [
		"//:tenncor",
		"//tests/utils:tenncor_testify",
	],
	linkstatic = 1,
	copts = [ "-std=c++14" ],
)

cc_test(
	name = "tenncor_all",
	size = "enormous",
	srcs = glob([ "**/*.cpp" ]),
	deps = [
		"//:tenncor",
		"//tests/utils:tenncor_testify",
	],
	linkstatic = 1,
	copts = [ "-std=c++14" ],
)
