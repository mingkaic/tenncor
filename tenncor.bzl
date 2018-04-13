load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def dependencies():
    # protobuf dependency
    if "org_pubref_rules_protobuf" not in native.existing_rules():
        git_repository(
            name = "org_pubref_rules_protobuf",
            remote = "https://github.com/mingkaic/rules_protobuf", # todo: revert to pubref after fix
            commit = "d60352ba20aa5eba5f86e1b93c1e048483372bf8",
        )

def test_dependencies():
    # test utility dependency
    if "com_github_mingkaic_testify" not in native.existing_rules():
        git_repository(
            name = "com_github_mingkaic_testify",
            remote = "https://github.com/raggledodo/testify",
            commit = "1d3e2b12511fdcbe01deff1fe53d9eecab978a5a",
        )
