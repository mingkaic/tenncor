load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

def consul_repository():
    new_git_repository(
        name = "com_github_oliora_ppconsul",
        remote = "https://github.com/mingkaic/ppconsul.git",
        commit = "c8f894481e90c8d5f0c14af23fbff040f30a1f37",
        build_file = "@com_github_mingkaic_tenncor//third_party:consul.BUILD",
    )
