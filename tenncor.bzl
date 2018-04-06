load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def dependencies():
    # test utility dependency
    if "com_github_mingkaic_testify" not in native.existing_rules():
        git_repository(
            name = "com_github_mingkaic_testify",
            remote = "https://github.com/raggledodo/testify",
            commit = "4f72f76775b0fa5e8305097f555b6fec6599022c",
        )

    # protobuf dependency
    if "org_pubref_rules_protobuf" not in native.existing_rules():
        git_repository(
            name = "org_pubref_rules_protobuf",
            remote = "https://github.com/pubref/rules_protobuf",
            commit = "7c8c80b61e3a0bc30fd61302d781a317524b0167",
        )