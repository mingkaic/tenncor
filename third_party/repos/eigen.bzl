load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

def eigen_repository():
    new_git_repository(
        name = "com_github_eigenteam_eigen",
        remote = "https://github.com/eigenteam/eigen-git-mirror.git",
        commit = "d787fcdabf6bd801f2d84c9138ef84fe525185b5",
        build_file = "@com_github_mingkaic_tenncor//third_party:eigen.BUILD",
    )
