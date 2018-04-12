package(
    default_visibility = [ "//visibility:public" ],
)

licenses(["notice"])

filegroup(
    name = "srcs",
    srcs = glob([ "tenncorgen/*.py" ]) + [
        "//tests/py/test:srcs",
        "protogen.py",
        "BUILD",
    ],
)

py_library(
    name = "tenncorgen",
    srcs = glob([ "tenncorgen/*.py" ]),
    imports = [ "." ],
    deps = [
        "//tests/graphmgr:graphmgr_py_grpc",
        "//proto:tenncor_serial_py_proto",
        "@com_github_mingkaic_testify//py:graphast",
    ],
)

py_binary(
    name = "protogen",
    srcs = [ "protogen.py" ],
    deps = [ ":tenncorgen" ],
)
