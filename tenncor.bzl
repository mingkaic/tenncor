load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def dependencies():
    if "com_github_mingkaic_cppkg" not in native.existing_rules():
        git_repository(
            name = "com_github_mingkaic_cppkg",
            remote = "https://github.com/mingkaic/cppkg",
            commit = "fa37730e25bd189056df795d4a374a16e5cc39fe",
        )
