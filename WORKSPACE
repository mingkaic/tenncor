workspace(name = "com_github_mingkaic_tenncor")

# === import external dependencies ===

load("//third_party:all.bzl", tenncor_deps="dependencies")
tenncor_deps()

# === test dependencies ===

load("@com_github_mingkaic_cppkg//:gtest.bzl", "gtest_repository")
gtest_repository(name = "gtest")

load("//third_party:repos/benchmark.bzl", "benchmark_repository")
benchmark_repository()

# === more import external dependencies ===

load("@com_github_pybind_bazel//:python_configure.bzl", "python_configure")
python_configure(name="local_config_python")

load("@com_github_mingkaic_cppkg//:cppkg.bzl", cppkg_deps="dependencies")
cppkg_deps()

load("@com_github_nelhage_rules_boost//:boost/boost.bzl", "boost_deps")
boost_deps()

load("@rules_proto_grpc//:repositories.bzl", "rules_proto_grpc_toolchains", "rules_proto_grpc_repos")
rules_proto_grpc_toolchains()
rules_proto_grpc_repos()

load("@rules_proto_grpc//cpp:repositories.bzl", rules_proto_grpc_cpp_repos="cpp_repos")
rules_proto_grpc_cpp_repos()

load("@rules_proto_grpc//python:repositories.bzl", rules_proto_grpc_python_repos="python_repos")
rules_proto_grpc_python_repos()

load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")
grpc_deps()
