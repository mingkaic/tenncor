workspace(name = "com_github_mingkaic_tenncor")

# gtest dependency
new_git_repository(
	name = "googletest",
	build_file = "BUILD.gmock",
	remote = "https://github.com/google/googletest",
	tag = "release-1.8.0",
)

# grpc dependency for monitoring
git_repository(
	name = 'com_github_grpc_grpc',
	remote = 'https://github.com/grpc/grpc.git',
	commit = '5b48dc737151464c1d863df6e4318ff3d766ddbc',
)
load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")
grpc_deps()
