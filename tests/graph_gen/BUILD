package(
	default_visibility = [ "//visibility:public" ],
)

licenses(["notice"])

filegroup(
    name = "srcs",
    srcs = glob([ "*.py", "shapeclass/*.py" ]) + [
        "BUILD",
        "struct.yaml",
    ],
)

py_library(
    name = "graph_ast",
    srcs = [
        "conf.py",
        "nodes.py",
        "gen.py",
        "synth.py",
    ],
    deps = [ "//proto:tenncor_serial_py_proto" ],
)

py_binary(
    name = "rsynth",
    srcs = [ "rsynth.py" ],
    deps = [
        ":graph_ast",
    ],
)

py_binary(
    name = "dsynth",
    srcs = [ "dsynth.py" ],
    deps = [
        ":graph_ast",
    ],
)

py_test(
    name = "test",
    srcs = [
        "conf_test.py",
        "nodes_test.py",
        "gen_test.py",
        "synth_test.py",
        "test.py",
    ],
    deps = [ ":graph_ast" ],
)
