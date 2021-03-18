workspace(name = "com_github_mingkaic_tenncor")

# === load local dependencies ===

load("//third_party:all.bzl", tenncor_deps="dependencies")
tenncor_deps()

# === load cppkg dependencies ===

# override cppkg grpc version
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
	name = "rules_proto_grpc",
	urls = ["https://github.com/rules-proto-grpc/rules_proto_grpc/archive/3.0.0.tar.gz"],
	sha256 = "f9b50e672870fe5d60b8b2f3cd1731ceb89a9edef3513d81ba7f7c0d2991b51f",
	strip_prefix = "rules_proto_grpc-3.0.0",
)

load("@com_github_mingkaic_cppkg//third_party:all.bzl", cppkg_deps="dependencies")
cppkg_deps()

# === boost dependencies ===

load("@com_github_nelhage_rules_boost//:boost/boost.bzl", "boost_deps")
boost_deps()

# === load grpc depedencies ===

# common dependencies
load("@rules_proto_grpc//:repositories.bzl", "rules_proto_grpc_toolchains", "rules_proto_grpc_repos")
rules_proto_grpc_toolchains()
rules_proto_grpc_repos()
load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")
grpc_deps()

# c++ dependencies
load("@rules_proto_grpc//cpp:repositories.bzl", rules_proto_grpc_cpp_repos="cpp_repos")
rules_proto_grpc_cpp_repos()

# python dependencies
load("@rules_proto_grpc//python:repositories.bzl", rules_proto_grpc_python_repos="python_repos")
rules_proto_grpc_python_repos()

# === load pybind dependencies ===

load("@com_github_pybind_bazel//:python_configure.bzl", "python_configure")
python_configure(name="local_config_python")

# === load benchmark dependencies ===

load("//third_party:repos/benchmark.bzl", "benchmark_repository")
benchmark_repository()

# === development ===

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
git_repository(
    name = "com_grail_bazel_compdb",
	remote = "https://github.com/grailbio/bazel-compilation-database",
	tag = "0.4.5",
)
