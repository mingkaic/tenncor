load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def dependencies():
    if "cppkg" not in native.existing_rules():
        git_repository(
            name = "cppkg",
            remote = "https://github.com/mingkaic/cppkg",
            commit = "85d64ffeb46facc2d3f2348515f6babad503eff4",
        )
