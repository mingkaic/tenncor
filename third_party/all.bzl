load("//third_party:repos/eigen.bzl", "eigen_repository")
load("//third_party:repos/grpc.bzl", "grpc_rules_repository")
load("//third_party:repos/pybind11.bzl", "pybind11_repository")
load("//third_party:repos/pybind_bazel.bzl", "pybind_bazel_repository")
load("//third_party:repos/cppkg.bzl", "cppkg_repository")
load("//third_party:repos/onnx.bzl", "onnx_repository")
load("//third_party:repos/benchmark.bzl", "benchmark_repository")
load("//third_party:repos/consul.bzl", "consul_repository")
load("//third_party:repos/curl.bzl", "curl_repository")

def dependencies(excludes = []):
    ignores = native.existing_rules().keys() + excludes

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

    if "com_github_oliora_ppconsul" not in ignores:
        consul_repository()

    if "com_github_curl_curl" not in ignores:
        curl_repository()
