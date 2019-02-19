workspace(name = "com_github_mingkaic_tenncor")

# local dependencies

load("//:third_party/all.bzl", "dependencies")
dependencies()

load("@protobuf_rules//cpp:deps.bzl", "cpp_proto_library")
cpp_proto_library()

# test dependencies

load("@com_github_mingkaic_cppkg//:gtest.bzl", "gtest_repository")
gtest_repository(name = "gtest")

load("//third_party/repos:benchmark.bzl", "benchmark_repository")
benchmark_repository()

# rocnnet dependencies

load("@com_github_mingkaic_cppkg//:cppkg.bzl", "dependencies")
dependencies()

load("@com_github_nelhage_rules_boost//:boost/boost.bzl", "boost_deps")
boost_deps()
