load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

_BUILD_CONTENT = """
"""

def curl_repository():
    new_git_repository(
        name = "com_github_curl_curl",
        remote = "https://github.com/curl/curl.git",
        tag = "curl-7_69_1",
        build_file = "@com_github_mingkaic_tenncor//third_party:curl.BUILD",
    )
