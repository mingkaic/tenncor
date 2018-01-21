COMMON_SRC = glob([
	"include/utils/*.h",
	"include/mocks/mock_actor.h",
	"src/utils/*.cpp",
])

cc_test(
	name = "tenncor_connector",
	size = "small",
	srcs = COMMON_SRC + glob([
		"include/mocks/mock_connector.h",
		"include/mocks/mock_immutable.h",
		"include/mocks/mock_itensor.h",
		"include/mocks/mock_node.h",
		"src/graph/connector/*.cpp",
	]),
	deps = [
		"@googletest//:gtest_main",
		"//:tenncor",
	],
	linkstatic = 1,
	copts = [ "-std=c++14" ],
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
		"@googletest//:gtest_main",
		"//:tenncor",
	],
	linkstatic = 1,
	copts = [ "-std=c++14" ],
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
		"@googletest//:gtest_main",
		"//:tenncor",
	],
	linkstatic = 1,
	copts = [ "-std=c++14" ],
)

cc_test(
	name = "tenncor_memory",
	size = "small",
	srcs = COMMON_SRC + glob([
		"include/mocks/mock_allocator.h",
		"src/memory/*.cpp",
	]),
	deps = [
		"@googletest//:gtest_main",
		"//:tenncor",
	],
	linkstatic = 1,
	copts = [ "-std=c++14" ],
)

cc_test(
	name = "tenncor_operation",
	size = "medium",
	srcs = COMMON_SRC + glob([
		"src/operations/*.cpp",
	]),
	deps = [
		"@googletest//:gtest_main",
		"//:tenncor",
	],
	linkstatic = 1,
	copts = [ "-std=c++14" ],
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
		"@googletest//:gtest_main",
		"//:tenncor",
	],
	linkstatic = 1,
	copts = [ "-std=c++14" ],
)

cc_test(
	name = "tenncor_all",
	size = "enormous",
	srcs = glob([ "include/**/*.h", "src/**/*.cpp" ]),
	deps = [
		"@googletest//:gtest_main",
		"//:tenncor",
	],
	linkstatic = 1,
	copts = [ "-std=c++14" ],
)
