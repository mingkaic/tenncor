load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def benchmark_repository():
    git_repository(
        name = "com_github_google_benchmark",
        remote = "https://github.com/google/benchmark",
        commit = "e776aa0275e293707b6a0901e0e8d8a8a3679508",
    )
