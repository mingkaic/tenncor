load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def dependencies():
    if "cppkg" not in native.existing_rules():
        git_repository(
            name = "cppkg",
            remote = "https://github.com/mingkaic/cppkg",
            commit = "699752065208a61375b261f2877ae8168dddaa52",
        )
