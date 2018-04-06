package(
    default_visibility = [ "//visibility:public" ],
)

licenses(["notice"])

py_library(
    name = "protogen",
    srcs = [ "protogen.py" ],
    imports = [ "." ],
    deps = [
        "//proto:tenncor_serial_py_proto",
        "@com_github_mingkaic_testify//py:graphast",
    ],
)

py_test(
    name = "test",
    srcs = [ "protogen_test.py", "test.py" ],
    deps = [ ":protogen" ]
)

py_binary(
    name = "rsynth",
    srcs = [ "rsynth.py", "tfsynth.py" ],
    deps = [
        "@com_github_mingkaic_testify//py:graphast",
    ],
)

py_binary(
    name = "dsynth",
    srcs = [ "dsynth.py", "tfsynth.py" ],
    deps = [
        "@com_github_mingkaic_testify//py:graphast",
    ],
)
