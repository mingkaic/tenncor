load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

_BUILD_CONTENT = """load(
    "@com_github_mingkaic_tenncor//third_party/drake_rules:install.bzl",
    "install",
)

licenses([
    "notice",  # BSD-3-Clause
    "reciprocal",  # MPL-2.0
    "unencumbered",  # Public-Domain
])

package(
    default_visibility = ["//visibility:public"],
)

filegroup(
    name = "srcs",
    srcs = glob(
        include = [
            "Eigen/*",
            "Eigen/**/*.h",
            "unsupported/Eigen/*",
            "unsupported/Eigen/**/*",
        ],
        exclude = ["**/CMakeLists.txt"],
    ),
)

cc_library(
    name = "eigen",
    hdrs = [":srcs"],
    defines = ["EIGEN_MPL2_ONLY"],
    includes = ["."],
)

install(
    name = "install",
    targets = [":eigen"],
    hdr_dest = "include/eigen3",
    guess_hdrs = "PACKAGE",
    docs = glob(["COPYING.*"]),
    doc_dest = "share/doc/eigen3",
)
"""

def eigen_repository():
    new_git_repository(
        name = "com_github_eigenteam_eigen",
        remote = "https://github.com/eigenteam/eigen-git-mirror.git",
        commit = "d787fcdabf6bd801f2d84c9138ef84fe525185b5",
        build_file_content = _BUILD_CONTENT,
    )
