load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

def consul_repository():
    new_git_repository(
        name = "com_github_oliora_ppconsul",
        remote = "https://github.com/mingkaic/ppconsul.git",
        commit = "43ae2d3e64f85224ad2059804942760e89aa553b",
        build_file = "@com_github_mingkaic_tenncor//third_party:consul.BUILD",
    )
