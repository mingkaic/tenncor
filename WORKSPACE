workspace(name = "com_github_mingkaic_tenncor")

# local dependencies

load("//:tenncor.bzl", "dependencies")
dependencies()

# test dependencies

load("@com_github_mingkaic_cppkg//:gtest.bzl", "gtest_repository")
gtest_repository(name = "gtest")
