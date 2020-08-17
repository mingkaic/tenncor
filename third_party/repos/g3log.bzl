load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

def g3log_repository():
    new_git_repository(
        name = "com_github_kjellkod_g3log",
        remote = "https://github.com/KjellKod/g3log.git",
        commit = "f1eff42b9143af35effe2c78923d2a726119c71c",
        build_file = "@com_github_mingkaic_tenncor//third_party:g3log.BUILD",
    )
