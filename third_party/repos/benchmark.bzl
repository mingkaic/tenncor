load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def benchmark_repository():
    http_archive(
        name = "com_github_google_benchmark",
        urls = ["https://github.com/google/benchmark/archive/v1.5.2.tar.gz"],
        strip_prefix = "benchmark-1.5.2",
    )
