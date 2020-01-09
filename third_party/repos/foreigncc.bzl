load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def foreign_cc_repository():
    http_archive(
        name = "rules_foreign_cc",
        strip_prefix = "rules_foreign_cc-master",
        url = "https://github.com/bazelbuild/rules_foreign_cc/archive/master.zip",
    )
