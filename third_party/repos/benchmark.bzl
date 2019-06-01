load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def benchmark_repository(name = "com_github_google_benchmark"):
    http_archive(
        name = name,
        urls = ["https://github.com/google/benchmark/archive/v1.5.0.tar.gz"],
        strip_prefix = "benchmark-1.5.0",
    )
