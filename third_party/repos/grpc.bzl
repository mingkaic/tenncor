load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def grpc_rules_repository():
    http_archive(
        name = "com_github_grpc_grpc",
        urls = ["https://github.com/grpc/grpc/archive/64fb7d47452e32e8746569bf0d1c19c5d1f1a1d9.tar.gz"],
        strip_prefix = "grpc-64fb7d47452e32e8746569bf0d1c19c5d1f1a1d9",
    )
