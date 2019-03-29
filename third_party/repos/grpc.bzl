load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def grpc_rules_repository(name):
    http_archive(
        name = name,
        urls = ["https://github.com/grpc/grpc/archive/109c570727c3089fef655edcdd0dd02cc5958010.tar.gz"],
        strip_prefix = "grpc-109c570727c3089fef655edcdd0dd02cc5958010",
    )
