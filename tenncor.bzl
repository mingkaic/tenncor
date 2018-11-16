load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def dependencies():
    if "rules_protobuf" not in native.existing_rules():
        git_repository(
            name = "rules_protobuf",
            remote = "https://github.com/mingkaic/rules_protobuf",
            commit = "f5615fa9d544d0a69cd73d8716364d8bd310babe",
        )
