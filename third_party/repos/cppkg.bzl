load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def cppkg_repository():
    git_repository(
        name = "com_github_mingkaic_cppkg",
        remote = "https://github.com/mingkaic/cppkg",
        commit = "f127a4f131c4a71625fa90baae0e98c8ccde8c38",
    )
