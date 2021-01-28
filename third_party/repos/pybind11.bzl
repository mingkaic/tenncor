load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

def pybind11_repository():
    new_git_repository(
        name = "pybind11",
        remote = "https://github.com/pybind/pybind11.git",
        tag = "v2.6.2",
        build_file = "@com_github_pybind_bazel//:pybind11.BUILD",
    )
