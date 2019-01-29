load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def dependencies():
    if "com_github_mingkaic_cppkg" not in native.existing_rules():
        git_repository(
            name = "com_github_mingkaic_cppkg",
            remote = "https://github.com/mingkaic/cppkg",
            commit = "740dfa89acba7fa46c2f17aea8dad62274e4fc51",
        )
