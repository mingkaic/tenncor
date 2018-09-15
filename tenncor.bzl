load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def dependencies():
    # protobuf dependency
    if "org_pubref_rules_protobuf" not in native.existing_rules():
        git_repository(
            name = "org_pubref_rules_protobuf",
            remote = "https://github.com/pubref/rules_protobuf",
            commit = "82dee5585aae7e0ed111b221187852a71ad8a71b",
        )

def test_dependencies():
    # test utility dependency
    if "com_github_mingkaic_testify" not in native.existing_rules():
        git_repository(
            name = "com_github_mingkaic_testify",
            remote = "https://github.com/raggledodo/testify",
            commit = "d51725bea2dd2837c69617548613a7a9c22ddadc",
        )
