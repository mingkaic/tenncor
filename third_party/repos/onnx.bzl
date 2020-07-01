load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

def onnx_repository():
    new_git_repository(
        name = "com_github_onnx_onnx",
        remote = "https://github.com/onnx/onnx.git",
        commit = "c08a7b76cf7c1555ae37186f12be4d62b2c39b3b",
        build_file = "//third_party:onnx.BUILD",
    )
