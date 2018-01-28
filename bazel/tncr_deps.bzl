def tncr_deps():
    native.bind(
        name = "grpc",
		actual = "@com_github_grpc_grpc//:grpc++",
    )

    native.bind(
        name = "gtest",
        actual = "@com_github_google_googletest//:gtest",
    )

    native.bind(
        name = "gtest_main",
        actual = "@com_google_googletest//:gtest_main",
    )

    native.bind(
        name = "protobuf",
        actual = "@com_google_protobuf//:protobuf",
    )

    native.bind(
        name = "protobuf_clib",
        actual = "@com_google_protobuf//:protoc_lib",
    )

    native.bind(
        name = "protobuf_headers",
        actual = "@com_google_protobuf//:protobuf_headers",
    )

    native.bind(
        name = "protocol_compiler",
        actual = "@com_google_protobuf//:protoc",
    )

    native.bind(
        name = "grpc_cpp_plugin",
        actual = "@com_github_grpc_grpc//:grpc_cpp_plugin"
    )

    # grpc dependency for monitoring
    if "com_github_grpc_grpc" not in native.existing_rules():
        native.http_archive(
            name = "com_github_grpc_grpc",
            strip_prefix = "grpc-123547c9625c56fdf5cb4ddd1df55ae0c785fa60",
            url = "https://github.com/grpc/grpc/archive/123547c9625c56fdf5cb4ddd1df55ae0c785fa60.tar.gz",
        )

    # protobuf dependency for serialization and monitoring
    if "com_google_protobuf" not in native.existing_rules():
        native.http_archive(
            name = "com_google_protobuf",
            strip_prefix = "protobuf-2761122b810fe8861004ae785cc3ab39f384d342",
            url = "https://github.com/google/protobuf/archive/2761122b810fe8861004ae785cc3ab39f384d342.tar.gz",
        )

    # gtest dependency
    if "com_google_googletest" not in native.existing_rules():
        native.new_http_archive(
            name = "com_google_googletest",
            build_file = "@com_github_mingkaic_tenncor//third_party:gtest.BUILD",
            strip_prefix = "googletest-ec44c6c1675c25b9827aacd08c02433cccde7780",
            url = "https://github.com/google/googletest/archive/ec44c6c1675c25b9827aacd08c02433cccde7780.tar.gz",
        )
