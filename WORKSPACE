workspace(name = "com_github_mingkaic_tenncor")

# === import external dependencies ===

load("//third_party:all.bzl", "dependencies")
dependencies()

# == flex + bison rules ==
load("@rules_m4//m4:m4.bzl", "m4_register_toolchains")
m4_register_toolchains()
load("@rules_flex//flex:flex.bzl", "flex_register_toolchains")
flex_register_toolchains()
load("@rules_bison//bison:bison.bzl", "bison_register_toolchains")
bison_register_toolchains()

# === test dependencies ===

load("@com_github_mingkaic_cppkg//:gtest.bzl", "gtest_repository")
gtest_repository(name = "gtest")

load("//third_party:repos/benchmark.bzl", "benchmark_repository")
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

load("@rules_foreign_cc//:workspace_definitions.bzl", "rules_foreign_cc_dependencies")
rules_foreign_cc_dependencies(
    native_tools_toolchains=[
        "//third_party:built_cmake_toolchain",
        "//third_party:built_ninja_toolchain",
    ],
)
