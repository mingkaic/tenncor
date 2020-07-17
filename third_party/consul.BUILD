package(
    default_visibility = ["//visibility:public"],
)

filegroup(
    name = "hdrs",
    srcs = glob(
        include = [
            "ext/**/*.h",
            "ext/**/*.hpp",
            "include/ppconsul/*",
            "include/ppconsul/**/*",
        ],
        exclude = ["**/CMakeLists.txt"],
    ),
)

filegroup(
    name = "srcs",
    srcs = glob(
        include = [
            "ext/**/*.c",
            "ext/**/*.cpp",
            "src/*",
            "src/**/*",
        ],
        exclude = ["**/CMakeLists.txt"],
    ) + [":hdrs"],
)

cc_library(
    name = "consul",
    hdrs = [":hdrs"],
    srcs = [":srcs"],
    deps = [
        "@boost//:variant",
        "@boost//:optional",
        "@boost//:fusion",
        "@com_github_curl_curl//:curl",
    ],
    includes = [
        "include",
        "src",
        "src/curl",
        "ext",
        "ext/json11",
    ],
)
