workspace(name = "com_github_mingkaic_tenncor")

# local dependencies

load("//:tenncor.bzl", "dependencies")
dependencies()

# test dependencies

load("@com_github_mingkaic_cppkg//:gtest.bzl", "gtest_repository")
gtest_repository(name = "gtest")

git_repository(
    name = "com_github_google_benchmark",
    remote = "https://github.com/google/benchmark",
    commit = "e776aa0275e293707b6a0901e0e8d8a8a3679508",
)
