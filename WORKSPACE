workspace(name = "com_github_mingkaic_tenncor")

# === import external dependencies ===

load("//:third_party/all.bzl", "dependencies")
dependencies()

# === test dependencies ===

load("@com_github_mingkaic_cppkg//:gtest.bzl", "gtest_repository")
gtest_repository(name = "gtest")

load("//third_party/repos:benchmark.bzl", "benchmark_repository")
benchmark_repository()

load("@com_github_mingkaic_cppkg//:cppkg.bzl", "dependencies")
dependencies()

load("@com_github_nelhage_rules_boost//:boost/boost.bzl", "boost_deps")
boost_deps()

# === local dependencies ===

load(
    "@com_github_stackb_rules_proto//cpp:deps.bzl", "cpp_proto_library",
    "cpp_grpc_library"
)
cpp_proto_library()
cpp_grpc_library()

load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")
grpc_deps()
