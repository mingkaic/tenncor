load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def grpc_rules_repository():
    http_archive(
        name = "rules_proto_grpc",
        urls = ["https://github.com/rules-proto-grpc/rules_proto_grpc/archive/1.0.2.tar.gz"],
        sha256 = "5f0f2fc0199810c65a2de148a52ba0aff14d631d4e8202f41aff6a9d590a471b",
        strip_prefix = "rules_proto_grpc-1.0.2",
    )
