load("//third_party:repos/eigen.bzl", "eigen_repository")
load("//third_party:repos/numpy.bzl", "numpy_repository")
load("//third_party:repos/protobuf.bzl", "pb_rules_repository")
load("//third_party:repos/grpc.bzl", "grpc_rules_repository")
load("//third_party:repos/pybind11.bzl", "pybind11_repository")
load("//third_party:repos/python.bzl", "python_repository")
load("//third_party:repos/cppkg.bzl", "cppkg_repository")
load("//third_party:repos/flexison.bzl", "flexison_repository")
load("//third_party:repos/onnx.bzl", "onnx_repository")

def dependencies(excludes = []):
    ignores = native.existing_rules().keys() + excludes
    if "numpy" not in ignores:
        numpy_repository(name = "numpy")

    if "python" not in ignores:
        python_repository(name = "python")

    if "com_github_pybind_pybind11" not in ignores:
        pybind11_repository()

    if "com_github_stackb_rules_proto" not in ignores:
        pb_rules_repository()

    if "com_github_eigenteam_eigen" not in ignores:
        eigen_repository()

    if "com_github_mingkaic_cppkg" not in ignores:
        cppkg_repository()

    if "com_github_grpc_grpc" not in ignores:
        grpc_rules_repository()

    if "rules_m4" not in ignores and "rules_flex" not in ignores and "rules_bison" not in ignores:
        flexison_repository()

    if "com_github_onnx_onnx" not in ignores:
        onnx_repository()
