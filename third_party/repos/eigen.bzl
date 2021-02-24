load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

def eigen_repository():
    new_git_repository(
        name = "com_github_eigenteam_eigen",
        remote = "https://gitlab.com/libeigen/eigen.git",
        tag = "3.3.9",
        build_file = "@com_github_mingkaic_tenncor//third_party:eigen.BUILD",
    )
