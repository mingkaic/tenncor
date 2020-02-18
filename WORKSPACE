workspace(name = "com_github_mingkaic_tenncor")

# === import external dependencies ===

load("//third_party:all.bzl", "dependencies")
dependencies()

load("@rules_proto//proto:repositories.bzl", "rules_proto_dependencies", "rules_proto_toolchains")
rules_proto_dependencies()
rules_proto_toolchains()

# ===== pybind11 dependencies ===
load("@com_github_pybind_bazel//:python_configure.bzl", "python_configure")
python_configure(name="local_config_python")

# === test dependencies ===

load("@com_github_mingkaic_cppkg//:gtest.bzl", "gtest_repository")
gtest_repository(name = "gtest")

load("//third_party:repos/benchmark.bzl", "benchmark_repository")
benchmark_repository()

load("@com_github_mingkaic_cppkg//:cppkg.bzl", cppkg_dependencies="dependencies")
cppkg_dependencies()

load("@com_github_nelhage_rules_boost//:boost/boost.bzl", "boost_deps")
boost_deps()

# === local dependencies ===

load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")
grpc_deps()

load("@com_github_grpc_grpc//bazel:grpc_extra_deps.bzl", "grpc_extra_deps")
grpc_extra_deps()
