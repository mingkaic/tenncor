load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

_BUILD_CONTENT="""
load("@rules_proto//proto:defs.bzl", "proto_library")

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"]) # MIT

genrule(
    name = "proto3to2",
    srcs = ["onnx/onnx.proto3"],
    outs = ["onnx.proto"],
    cmd = "cp $(location onnx/onnx.proto3) $@",
)

proto_library(
    name = "onnx_proto",
    srcs = ["onnx.proto"],
    import_prefix = "onnx",
)
"""

def onnx_repository():
    new_git_repository(
        name = "com_github_onnx_onnx",
        remote = "https://github.com/onnx/onnx.git",
        commit = "c08a7b76cf7c1555ae37186f12be4d62b2c39b3b",
        build_file_content = _BUILD_CONTENT,
    )
