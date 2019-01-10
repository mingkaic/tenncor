load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def dependencies():
    if "com_github_mingkaic_cppkg" not in native.existing_rules():
        git_repository(
            name = "com_github_mingkaic_cppkg",
            remote = "https://github.com/mingkaic/cppkg",
            commit = "c866eaba28d9f34ce593c651cbf628f5325f2183",
        )
