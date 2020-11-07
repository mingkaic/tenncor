load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def onnxds_repository():
    git_repository(
        name = "com_github_mingkaic_onnxds",
        remote = "https://github.com/mingkaic/onnxds.git",
        commit = "cfe45e216ecacb7e651451bfb090c0ff66debffc",
    )
