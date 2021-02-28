load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def libgexf_repository():
    git_repository(
        name = "libgexf_unofficial",
        remote = "https://github.com/mingkaic/libgexf.git",
        commit = "9813bded83ccaec5734a1e9ae723bc0e31005805",
    )
