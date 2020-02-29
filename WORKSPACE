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

# === additional external dependencies (remove after native cpp proto rules works) ===

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
    name = "com_google_protobuf_custom",
    sha256 = "a19dcfe9d156ae45d209b15e0faed5c7b5f109b6117bfc1974b6a7b98a850320",
    strip_prefix = "protobuf-3.7.0",
    urls = ["https://github.com/protocolbuffers/protobuf/archive/v3.7.0.tar.gz"],
)
