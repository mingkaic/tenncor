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
