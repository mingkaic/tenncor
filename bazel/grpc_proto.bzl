proto_filetype = FileType([".proto"])

def grpc_proto_library(name, src, has_service = True):
    basename = src[0:-5]

    command = "$(location @com_google_protobuf//:protoc) --cpp_out=$(GENDIR)/"
    command += " $(location %s);" % (src)

    cc_proto_name = name + "_cc_proto"
    header_outputs = [basename + "pb.h"]
    outputs = header_outputs + [basename + "pb.cc"]

    native.genrule(
        name = cc_proto_name,
        srcs = [ src ],
        cmd = command,
        outs = outputs,
        tools = ['@com_google_protobuf//:protoc'],
    )

    if has_service:
        grpc_command = "$(location @com_google_protobuf//:protoc) --grpc_out=$(GENDIR)/"
        grpc_command += " --plugin=protoc-gen-grpc=$(location @com_github_grpc_grpc//:grpc_cpp_plugin)"
        grpc_command += " $(location %s);" % (src)

        grpc_cc_proto_name = name + "_cc_proto_service"
        grpc_header_outputs = [basename + "grpc.pb.h"]
        grpc_outputs = grpc_header_outputs + [basename + "grpc.pb.cc"]
        grpc_extra_srcs = [':' + grpc_cc_proto_name]
        grpc_extra_deps = [
            '@com_github_grpc_grpc//:grpc++',
        ]

        native.genrule(
            name = grpc_cc_proto_name,
            srcs = [ src ],
            cmd = grpc_command,
            outs = grpc_outputs,
            tools = [
                '@com_google_protobuf//:protoc',
                '@com_github_grpc_grpc//:grpc_cpp_plugin',
            ],
        )
    else:
        grpc_header_outputs = []
        grpc_extra_srcs = []
        grpc_extra_deps = []

    native.cc_library(
        name = name,
        hdrs = header_outputs + grpc_header_outputs,
        srcs = [
            ":" + cc_proto_name
        ] + grpc_extra_srcs,
        deps = [
            '@com_google_protobuf//:protobuf',
        ] + grpc_extra_deps,
    )