load("//third_party/repos:eigen.bzl", "eigen_repository")
load("//third_party/repos:numpy.bzl", "numpy_repository")
load("//third_party/repos:protobuf.bzl", "protobuf_rules_repository")
load("//third_party/repos:pybind11.bzl", "pybind11_repository")
load("//third_party/repos:python.bzl", "python_repository")
load("//third_party/repos:cppkg.bzl", "cppkg_repository")

def dependencies(excludes = []):
    ignores = native.existing_rules().keys() + excludes
    if "numpy" not in ignores:
        numpy_repository(name = "numpy")

    if "python" not in ignores:
        python_repository(name = "python")

    if "com_github_pybind_pybind11" not in ignores:
        pybind11_repository()

    if "protobuf_rules" not in ignores:
        protobuf_rules_repository(name = "protobuf_rules")

    if "eigen" not in ignores:
        eigen_repository(name = "com_github_eigenteam_eigen")

    if "com_github_mingkaic_cppkg" not in ignores:
        cppkg_repository()