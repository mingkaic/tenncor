load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def dependencies():
    # test utility dependency
    if "com_github_mingkaic_testify" not in native.existing_rules():
        git_repository(
            name = "com_github_mingkaic_testify",
            remote = "https://github.com/raggledodo/testify",
            commit = "ef0d005db6e4fa10b8d479951901fb9df7b936c3",
        )

    # protobuf dependency
    if "org_pubref_rules_protobuf" not in native.existing_rules():
        git_repository(
            name = "org_pubref_rules_protobuf",
            remote = "https://github.com/pubref/rules_protobuf",
            commit = "7c8c80b61e3a0bc30fd61302d781a317524b0167",
        )
