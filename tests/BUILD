cc_test(
    name = "tenncor_test",
    size = "enormous",
    srcs = glob(["include/**/*.h"]) + glob(["src/**/*.cpp"]),
    deps = [
        "@googletest//:gtest_main",
        "//:tenncor",
    ],
	copts = ["-std=c++14"],
)
