licenses(["notice"])

load("//:yacc.bzl", "genyacc", "genlex")

filegroup(
    name = "srcs",
    srcs = glob([
        "**/*.h", "*.c",
        "ast/*.hpp", "ast/*.cpp",
    ]) + [
        "BUILD",
        "ascii/srcs",
    ],
    visibility = ["//visibility:public"],
)

######### LIBRARY #########

genyacc(
    name = "yacc",
    out_hdr = "ade.cli.h",
    out_src = "ade.cli.c",
    src = "calc.yacc",
)

genlex(
    name = "flex",
    out = "ade.lex.c",
    src = "calc.lex",
)

cc_library(
    name = "ast",
    hdrs = glob(["ast/*.h"]),
    srcs = glob(["ast/*.cpp"]),
    copts = ["-std=c++14"],
    deps = [
        "//dbg:dbg",
        "//llo:llo",
    ],
)

######### BINARY #########

cc_binary(
    name = "cli",
    srcs = [":flex", ":yacc"],
    deps = [":ast"],
)
