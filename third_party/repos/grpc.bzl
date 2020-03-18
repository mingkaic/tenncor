load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def grpc_rules_repository():
    http_archive(
        name = "com_github_grpc_grpc",
        urls = ["https://github.com/grpc/grpc/archive/v1.27.3.tar.gz"],
        strip_prefix = "grpc-1.27.3",
    )
