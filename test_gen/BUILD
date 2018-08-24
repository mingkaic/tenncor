licenses(["notice"])

package(
    default_visibility = ["//visibility:public"],
)

filegroup(
    name = "srcs",
    srcs = glob(["*.py"]) + ["BUILD"],
)

######### LIBRARY #########

py_binary(
    name = "tfgen",
    srcs = glob(["*.py"]),
    deps = ["@com_github_mingkaic_testify//retrop:retrop"],
)
