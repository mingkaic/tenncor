package(
    default_visibility = [ "//visibility:public" ],
)

licenses(["notice"])

filegroup(
    name = "srcs",
    srcs = glob([ "*.py" ]) + [ "BUILD" ],
)

py_library(
    name = "py_save",
    srcs = [
        "save_ast.py",
        "save_data.py",
    ],
    deps = [
        "//graphmgr:py_grpc",
        "//lead:py_proto",
        "@com_github_mingkaic_testify//testify_py:graphast",
    ],
)

py_test(
    name = "test",
    srcs = [
        "test.py",
        "save_ast_test.py",
        "save_data_test.py",
    ],
    deps = [ ":py_save" ],
)

py_binary(
    name = "protogen",
    srcs = [ "protogen.py" ],
    deps = [ ":py_save" ],
)
