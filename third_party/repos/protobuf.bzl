load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def pb_rules_repository():
    http_archive(
        name = "com_github_stackb_rules_proto",
        urls = ["https://github.com/stackb/rules_proto/archive/b93b544f851fdcd3fc5c3d47aee3b7ca158a8841.tar.gz"],
        sha256 = "c62f0b442e82a6152fcd5b1c0b7c4028233a9e314078952b6b04253421d56d61",
        strip_prefix = "rules_proto-b93b544f851fdcd3fc5c3d47aee3b7ca158a8841",
    )
