load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def cppkg_repository():
    git_repository(
        name = "com_github_mingkaic_cppkg",
        remote = "https://github.com/mingkaic/cppkg",
        commit = "19bf17bee52036c81e5b5ae885ea22347f678024",
    )
