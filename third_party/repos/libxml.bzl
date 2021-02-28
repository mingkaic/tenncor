load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def libxml_repository():
    http_archive(
        name = "libxml_archive",
        build_file = "@libgexf_unofficial//:libxml.BUILD",
        sha256 = "f63c5e7d30362ed28b38bfa1ac6313f9a80230720b7fb6c80575eeab3ff5900c",
        strip_prefix = "libxml2-2.9.7",
        urls = [
            "https://mirror.bazel.build/xmlsoft.org/sources/libxml2-2.9.7.tar.gz",
            "http://xmlsoft.org/sources/libxml2-2.9.7.tar.gz",
        ],
    )
