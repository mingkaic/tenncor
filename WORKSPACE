workspace(name = "com_github_mingkaic_tenncor")

load("//:tenncor.bzl", "dependencies", "test_dependencies")
dependencies()
test_dependencies()

load("@com_github_mingkaic_testify//:testify.bzl", "dependencies")
dependencies()

load("@org_pubref_rules_protobuf//cpp:rules.bzl", "cpp_proto_repositories")
cpp_proto_repositories()

load("@org_pubref_rules_protobuf//python:rules.bzl", "py_proto_repositories")
py_proto_repositories()
