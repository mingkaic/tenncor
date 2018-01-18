cc_test(
	name = "tenncor_connector",
	size = "small",
	srcs = glob(["src/utils/*.cpp"]) +
		glob(["include/utils/*.h"]) + 
		glob(["src/graph/connector/*.cpp"]) + [

		"include/mocks/mock_actor.h",
	],
	deps = [
		"@googletest//:gtest_main",
		"//:tenncor",
	],
	linkstatic = 1,
	copts = ["-std=c++14"],
)

cc_test(
	name = "tenncor_leaf",
	size = "small",
	srcs = glob(["src/utils/*.cpp"]) +
		glob(["include/utils/*.h"]) + 
		glob(["src/graph/leaf/*.cpp"]) + [

		"include/mocks/mock_actor.h",
	],
	deps = [
		"@googletest//:gtest_main",
		"//:tenncor",
	],
	linkstatic = 1,
	copts = ["-std=c++14"],
)

cc_test(
	name = "tenncor_nodes",
	size = "small",
	srcs = glob(["src/utils/*.cpp"]) +
		glob(["include/utils/*.h"]) + 
		glob(["src/graph/top_node/*.cpp"]) + [
		"include/mocks/mock_subject.h",
		"include/mocks/mock_observer.h",
		"include/mocks/mock_node.h",
		"include/mocks/mock_actor.h",
	],
	deps = [
		"@googletest//:gtest_main",
		"//:tenncor",
	],
	linkstatic = 1,
	copts = ["-std=c++14"],
)

cc_test(
	name = "tenncor_memory",
	size = "small",
	srcs = glob(["src/utils/*.cpp"]) +
		glob(["include/utils/*.h"]) + 
		glob(["src/memory/*.cpp"]) + [
		"include/mocks/mock_allocator.h",
		"include/mocks/mock_actor.h",
	],
	deps = [
		"@googletest//:gtest_main",
		"//:tenncor",
	],
	linkstatic = 1,
	copts = ["-std=c++14"],
)

cc_test(
	name = "tenncor_operation",
	size = "small",
	srcs = glob(["src/utils/*.cpp"]) +
		glob(["include/utils/*.h"]) + 
		glob(["src/operations/*.cpp"]) + [

		"include/mocks/mock_actor.h",
	],
	deps = [
		"@googletest//:gtest_main",
		"//:tenncor",
	],
	linkstatic = 1,
	copts = ["-std=c++14"],
)

cc_test(
	name = "tenncor_tensor",
	size = "small",
	srcs = glob(["src/utils/*.cpp"]) +
		glob(["include/utils/*.h"]) + 
		glob(["src/tensor/*.cpp"]) + [
		"include/mocks/mock_itensor.h",
		"include/mocks/mock_tensor.h",
		"include/mocks/mock_actor.h",
	],
	deps = [
		"@googletest//:gtest_main",
		"//:tenncor",
	],
	linkstatic = 1,
	copts = ["-std=c++14"],
)
