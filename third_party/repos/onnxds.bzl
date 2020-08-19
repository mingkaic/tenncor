load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def onnxds_repository():
    git_repository(
        name = "com_github_mingkaic_onnxds",
        remote = "https://github.com/mingkaic/onnxds.git",
        commit = "231b777701bda1d8ec38936682487ceedb97570d",
    )
