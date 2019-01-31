load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def dependencies():
    if "com_github_mingkaic_cppkg" not in native.existing_rules():
        git_repository(
            name = "com_github_mingkaic_cppkg",
            remote = "https://github.com/mingkaic/cppkg",
            commit = "f9135d982b1f2420c834310a36e38af3a04e8d81",
        )
