load("//third_party:repos/eigen.bzl", "eigen_repository")
load("//third_party:repos/grpc.bzl", "grpc_rules_repository")
load("//third_party:repos/pybind11.bzl", "pybind11_repository")
load("//third_party:repos/pybind_bazel.bzl", "pybind_bazel_repository")
load("//third_party:repos/cppkg.bzl", "cppkg_repository")
load("//third_party:repos/onnx.bzl", "onnx_repository")
load("//third_party:repos/benchmark.bzl", "benchmark_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def dependencies(excludes = []):
    ignores = native.existing_rules().keys() + excludes
    if "rules_proto" not in ignores:
        http_archive(
            name = "rules_proto",
            sha256 = "602e7161d9195e50246177e7c55b2f39950a9cf7366f74ed5f22fd45750cd208",
            strip_prefix = "rules_proto-97d8af4dc474595af3900dd85cb3a29ad28cc313",
            urls = [
                "https://mirror.bazel.build/github.com/bazelbuild/rules_proto/archive/97d8af4dc474595af3900dd85cb3a29ad28cc313.tar.gz",
                "https://github.com/bazelbuild/rules_proto/archive/97d8af4dc474595af3900dd85cb3a29ad28cc313.tar.gz",
            ],
        )

    if "com_github_pybind_pybind11" not in ignores:
        pybind11_repository()

    if "com_github_pybind_bazel" not in ignores:
        pybind_bazel_repository()

    if "com_github_eigenteam_eigen" not in ignores:
        eigen_repository()

    if "com_github_mingkaic_cppkg" not in ignores:
        cppkg_repository()

    if "com_github_grpc_grpc" not in ignores:
        grpc_rules_repository()

    if "com_github_onnx_onnx" not in ignores:
        onnx_repository()

    if "com_github_google_benchmark" not in ignores:
        benchmark_repository()
