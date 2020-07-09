load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

def consul_repository():
    new_git_repository(
        name = "com_github_oliora_ppconsul",
        remote = "https://github.com/oliora/ppconsul.git",
        commit = "c979fafe678e7149abc04dd9e7a6aa0db9d5c9fa",
        build_file = "//third_party:consul.BUILD",
    )
