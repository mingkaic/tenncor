load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def pybind_bazel_repository():
    git_repository(
        name = "com_github_pybind_bazel",
        remote = "https://github.com/raggledodo/pybind11_bazel.git",
        commit = "efabeef76d32f9e9c904ee9d839ee69bec68bd0f",
    )
