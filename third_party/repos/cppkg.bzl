load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def cppkg_repository():
    git_repository(
        name = "com_github_mingkaic_cppkg",
        remote = "https://github.com/mingkaic/cppkg",
        commit = "01a20b79f8fea518f9927cdc2ebf2bfb5a625abd",
    )
