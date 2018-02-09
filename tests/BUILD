COMMON_SRC = glob([
	"include/utils/*.h",
	"include/mocks/mock_actor.h",
	"src/utils/*.cpp",
])

licenses(["notice"])

######### FUZZ TESTS #########
cc_test(
	name = "tenncor_connector",
	size = "small",
	srcs = COMMON_SRC + glob([
		"include/mocks/mock_connector.h",
		"include/mocks/mock_linear.h",
		"include/mocks/mock_itensor.h",
		"include/mocks/mock_node.h",
		"src/graph/connector/*.cpp",
	]),
	deps = [
		"//:tenncor",
		"@com_google_googletest//:gtest_main",
	],
	linkstatic = 1,
	copts = [ "-std=c++17" ],
)

cc_test(
	name = "tenncor_leaf",
	size = "small",
	srcs = COMMON_SRC + glob([
		"include/mocks/mock_connector.h",
		"include/mocks/mock_ivariable.h",
		"include/mocks/mock_leaf.h",
		"include/mocks/mock_node.h",
		"src/graph/leaf/*.cpp",
	]),
	deps = [
		"//:tenncor",
		"@com_google_googletest//:gtest_main",
	],
	linkstatic = 1,
	copts = [ "-std=c++17" ],
)

cc_test(
	name = "tenncor_nodes",
	size = "small",
	srcs = COMMON_SRC + glob([
		"include/mocks/mock_node.h",
		"include/mocks/mock_observer.h",
		"include/mocks/mock_subject.h",
		"src/graph/top_node/*.cpp"
	]),
	deps = [
		"//:tenncor",
		"@com_google_googletest//:gtest_main",
	],
	linkstatic = 1,
	copts = [ "-std=c++17" ],
)

cc_test(
	name = "tenncor_operation",
	size = "medium",
	srcs = COMMON_SRC + glob([
		"src/operations/*.cpp",
	]),
	deps = [
		"//:tenncor",
		"@com_google_googletest//:gtest_main",
	],
	linkstatic = 1,
	copts = [ "-std=c++17" ],
)

cc_test(
	name = "tenncor_tensor",
	size = "small",
	srcs = COMMON_SRC + glob([
		"include/mocks/mock_itensor.h",
		"include/mocks/mock_tensor.h",
		"src/tensor/*.cpp",
	]),
	deps = [
		"//:tenncor",
		"@com_google_googletest//:gtest_main",
	],
	linkstatic = 1,
	copts = [ "-std=c++17" ],
)

cc_test(
	name = "tenncor_all",
	size = "enormous",
	srcs = glob([ "include/**/*.h", "src/**/*.cpp" ]),
	deps = [
		"//:tenncor",
		"@com_google_googletest//:gtest_main",
	],
	linkstatic = 1,
	copts = [ "-std=c++17" ],
)
