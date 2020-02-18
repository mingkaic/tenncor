load("@com_github_pybind_bazel//:build_defs.bzl", "pybind_extension")

def pybind_py_library(name,
        cc_so_name = None,
        copts = [],
        features = [],
        tags = [],
        cc_srcs = [],
        cc_deps = [],
        py_library_rule = native.py_library,
        py_srcs = [],
        py_deps = [],
        py_imports = [],
        visibility = None,
        testonly = None,
        **kwargs):

    if not cc_so_name:
        if name.endswith("_py"):
            cc_so_name = name[:-3]
        else:
            cc_so_name = name

    pybind_extension(cc_so_name, copts, features, tags,
        deps=cc_deps, srcs=cc_srcs,
        visibility=visibility, testonly=testonly, **kwargs)

    py_library_rule(
        name = name,
        data = [cc_so_name + '.so'],
        srcs = py_srcs,
        deps = py_deps,
        imports = py_imports,
        testonly = testonly,
        visibility = visibility,
    )
