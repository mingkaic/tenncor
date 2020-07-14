
sprefix_tool = "@com_github_mingkaic_tenncor//third_party:strip_prefix"
saffix_tool = "@com_github_mingkaic_tenncor//third_party:strip_affix"

def py_proto_library(name, srcs, package,
    protoc = "@com_google_protobuf//:protoc",
    deps = [],
    **kwargs):

    if any([False == proto.endswith('.proto') for proto in srcs]):
        print('protofiles must end with .proto: {}'.format(srcs))

    basenames = [proto[:-len('.proto')] for proto in srcs]

    tools = [protoc, saffix_tool]
    command = "$(location {})".format(protoc)
    sources = [proto + '_pb2.py' for proto in basenames]

    command_affix = ""
    for proto in srcs:
        command_affix += " $(locations %s)" % (proto)

    proto_command = command + \
        " --python_out=$$($(location {}) $(RULEDIR) {})".format(
            saffix_tool, package) + command_affix

    native.genrule(
        name = "pbgen_" + name,
        srcs = srcs,
        cmd = proto_command,
        outs = sources,
        tools = tools,
    )

    native.py_library(
        name = name,
        srcs = sources,
        deps = deps,
        **kwargs,
    )

def cc_proto_library(name, srcs, package,
    with_service = False,
    protoc = "@com_google_protobuf//:protoc",
    cc_grpc_plugin = "@com_github_grpc_grpc//src/compiler:grpc_cpp_plugin",
    deps = [],
    proto_deps = [],
    proto_paths = [],
    **kwargs):

    if any([False == proto.endswith('.proto') for proto in srcs]):
        print('protofiles must end with .proto: {}'.format(srcs))

    basenames = [proto[:-len('proto')] for proto in srcs]

    tools = [protoc, sprefix_tool, saffix_tool]
    command = "$(location {})".format(protoc)
    headers = [proto + 'pb.h' for proto in basenames]
    sources = [proto + 'pb.cc' for proto in basenames]

    if len(''.join(proto_paths)) > 0:
        for proto_path in proto_paths:
            if len(proto_path) > 0:
                command += " --proto_path {}".format(proto_path)

    command_affix = ""
    for proto in srcs:
        command_affix += " $$($(location {}) $(location {}) {})".format(
            sprefix_tool, proto, ' '.join(proto_paths))

    proto_inputs = srcs + proto_deps
    proto_command = command + \
        " --cpp_out=$$($(location {}) $(RULEDIR) {})".format(
            saffix_tool, package) + \
        command_affix

    native.genrule(
        name = "pbgen_" + name,
        srcs = proto_inputs,
        cmd = proto_command,
        outs = headers + sources,
        tools = tools,
    )

    if with_service:
        tools += [cc_grpc_plugin]

        grpc_headers = [proto + 'grpc.pb.h' for proto in basenames]
        grpc_sources = [proto + 'grpc.pb.cc' for proto in basenames]

        grpc_command = command + \
            " --grpc_out=$$($(location {}) $(RULEDIR) {})".format(
                saffix_tool, package) + \
            " --plugin=protoc-gen-grpc=$(location {})".format(
                cc_grpc_plugin) + \
            command_affix

        native.genrule(
            name = "grpcgen_" + name,
            srcs = proto_inputs,
            cmd = grpc_command,
            outs = grpc_headers + grpc_sources,
            tools = tools,
        )

        grpc_package = str(cc_grpc_plugin).split("/")[0]
        extra_deps = [grpc_package + "//:grpc++"]
    else:
        grpc_headers = []
        grpc_sources = []

        proto_package = str(protoc).split("/")[0]
        extra_deps = [proto_package + "//:protobuf"]

    native.cc_library(
        name = name,
        hdrs = headers + grpc_headers,
        srcs = sources + grpc_sources,
        deps = deps + extra_deps,
        **kwargs,
    )
