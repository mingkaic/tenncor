load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def cppkg_repository():
    if "com_github_nelhage_rules_boost" not in native.existing_rules():
        git_repository(
            name = "com_github_nelhage_rules_boost",
            commit = "891e33a7cf4601d5e4187ec9b28d9472a3491032",
            remote = "https://github.com/mingkaic/rules_boost",
        )

    git_repository(
        name = "com_github_mingkaic_cppkg",
        remote = "https://github.com/mingkaic/cppkg",
        commit = "3ea2085a1d80a1120d87e16c6936037be0150197",
    )
