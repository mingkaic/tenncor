load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def grpc_rules_repository():
    http_archive(
        name = "com_github_grpc_grpc",
        urls = ["https://github.com/grpc/grpc/archive/bf22ccbcac73c89b2fd860d84bd0b8b0945fde07.tar.gz"],
        strip_prefix = "grpc-bf22ccbcac73c89b2fd860d84bd0b8b0945fde07",
    )
