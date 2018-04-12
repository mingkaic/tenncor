licenses(["notice"])

filegroup(
    name = "srcs",
    srcs = glob([ "*.py" ]) + [ "BUILD" ],
)

py_test(
    name = "test",
    srcs = [ ":srcs" ],
    deps = [ "//tests/py:tenncorgen" ],
)
