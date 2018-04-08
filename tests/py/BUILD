package(
    default_visibility = [ "//visibility:public" ],
)

licenses(["notice"])

py_library(
    name = "tenncorgen",
    srcs = [
        "save_ast.py",
        "save_data.py", 
        "tfgen.py",
        "utils.py",
    ],
    imports = [ "." ],
    deps = [
        "//proto:tenncor_serial_py_proto",
        "@com_github_mingkaic_testify//py:graphast",
    ],
)

py_test(
    name = "test",
    srcs = [
        "save_ast_test.py", 
        "test.py",
    ],
    deps = [ ":tenncorgen" ]
)

py_binary(
    name = "protogen",
    srcs = [ "protogen.py" ],
    deps = [ ":tenncorgen" ],
)
