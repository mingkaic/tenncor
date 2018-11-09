load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

all_content = """filegroup(
    name = "all",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"]
)"""

json_parser = """cc_library(
    name = "json_parser",
    hdrs = glob(["include/**"]),
    copts = ["-std=c++14"],
    strip_include_prefix = "include",
    visibility = ["//visibility:public"],
)"""

def dependencies():
    if "rules_protobuf" not in native.existing_rules():
        git_repository(
            name = "rules_protobuf",
            remote = "https://github.com/mingkaic/rules_protobuf",
            commit = "f5615fa9d544d0a69cd73d8716364d8bd310babe",
        )

    # todo: use this in the future once it's stable
    # if "rules_foreign_cc" not in native.existing_rules():
    #    http_archive(
    #        name = "rules_foreign_cc",
    #        strip_prefix = "rules_foreign_cc-master",
    #        url = "https://github.com/bazelbuild/rules_foreign_cc/archive/master.zip",
    #    )

    if "com_github_nlohmann_json" not in native.existing_rules():
        http_archive(
            name = "com_github_nlohmann_json",
            build_file_content = json_parser,
            strip_prefix = "json-3.4.0",
            urls = ["https://github.com/nlohmann/json/archive/v3.4.0.tar.gz"],
        )

    # unused for now. will wrap around eigen to replace llo in the future
    if "com_github_xianyi_openblas" not in native.existing_rules():
        http_archive(
            name = "com_github_xianyi_openblas",
            build_file_content = all_content,
            strip_prefix = "OpenBLAS-0.3.2",
            urls = ["https://github.com/xianyi/OpenBLAS/archive/v0.3.2.tar.gz"],
        )

    if "com_github_eigenteam_eigen" not in native.existing_rules():
        http_archive(
            name = "com_github_eigenteam_eigen",
            build_file_content = all_content,
            strip_prefix = "eigen-git-mirror-3.3.5",
            urls = ["https://github.com/eigenteam/eigen-git-mirror/archive/3.3.5.tar.gz"],
        )

def test_dependencies():
    if "com_github_mingkaic_testify" not in native.existing_rules():
        git_repository(
            name = "com_github_mingkaic_testify",
            remote = "https://github.com/raggledodo/testify",
            commit = "e96e793b7082c3eb95f6177d5e7b0612ef6cd29c",
        )
