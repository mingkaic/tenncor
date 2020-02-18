load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def pybind_bazel_repository():
    git_repository(
        name = "com_github_pybind_bazel",
        remote = "https://github.com/raggledodo/pybind11_bazel.git",
        commit = "c77a7d9b184578c924e81dae85e711d2bcbac8a9",
    )
