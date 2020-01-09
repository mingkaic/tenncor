load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

_BUILD_CONTENT = """
filegroup(
    name = "all",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"]
)
"""

def gqlparser_repository():
    http_archive(
        name = "com_github_graphql_parser",
        build_file_content = _BUILD_CONTENT,
        strip_prefix = "libgraphqlparser-master",
        urls = ["https://github.com/graphql/libgraphqlparser/archive/master.zip"],
    )
