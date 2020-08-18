load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def onnxds_repository():
    git_repository(
        name = "com_github_mingkaic_onnxds",
        remote = "https://github.com/mingkaic/onnxds.git",
        commit = "dd02f1b53be8e2f3e1a847c227a91debdd7bfea1",
    )
