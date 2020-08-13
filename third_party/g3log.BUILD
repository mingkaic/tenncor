licenses([
    "unencumbered",  # Public-Domain
])

package(
    default_visibility = ["//visibility:public"],
)

filegroup(
    name = "hdrs",
    srcs = glob([
        "src/*.hpp",
        "src/**/*.hpp",
        "src/*.ipp",
        "src/**/*.ipp",
    ], exclude=["src/g3log/stacktrace_windows.hpp"]),
)

filegroup(
    name = "windows_hdrs",
    srcs = ["src/g3log/stacktrace_windows.hpp"],
)

filegroup(
    name = "srcs",
    srcs = glob([
        "src/*.cpp",
        "src/**/*.cpp",
    ], exclude=[
        "src/crashhandler_unix.cpp",
        "src/crashhandler_windows.cpp",
        "src/stacktrace_windows.cpp",
    ]),
)

filegroup(
    name = "windows_srcs",
    srcs = [
        "src/crashhandler_windows.cpp",
        "src/stacktrace_windows.cpp",
    ],
)

filegroup(
    name = "unix_srcs",
    srcs = ["src/crashhandler_unix.cpp"],
)

# translation of https://github.com/KjellKod/g3log/blob/69a0be4c9c0d4a4e29056f99c8ef0b053130136a/Options.cmake
genrule(
    name = "generated_definitions",
    srcs = [],
    outs = ["src/g3log/generated_definitions.hpp"],
    cmd = "\n".join([
        "cat <<'EOF' >$@",
        "#define G3LOG_DEBUG DBUG",
        "#define G3_DYNAMIC_LOGGING",
        # "#define CHANGE_G3LOG_DEBUG_TO_DBUG",
        "#define G3_DYNAMIC_MAX_MESSAGE_SIZE",
        "#define G3_LOG_FULL_FILENAME",
        # "#define DISABLE_FATAL_SIGNALHANDLING",
        # "#define DISABLE_VECTORED_EXCEPTIONHANDLING",
        "#define DEBUG_BREAK_AT_FATAL_SIGNAL",
        "EOF",
    ])
)

cc_library(
    name = "g3log",
    hdrs = [":hdrs", ":generated_definitions"] + select({
        "@com_github_mingkaic_tenncor//:windows": [":windows_hdrs"],
        "//conditions:default": [],
    }),
    srcs = [":srcs"] + select({
        "@com_github_mingkaic_tenncor//:windows": [":windows_srcs"],
        "//conditions:default": [":unix_srcs"],
    }),
    copts = ["-std=c++17"],
    includes = ["src"],
)
