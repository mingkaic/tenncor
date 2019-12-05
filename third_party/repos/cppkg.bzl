load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def cppkg_repository():
    git_repository(
        name = "com_github_mingkaic_cppkg",
        remote = "https://github.com/mingkaic/cppkg",
        commit = "0bb64cbc5607a2a02381999dcdb75d610df2c599",
    )
