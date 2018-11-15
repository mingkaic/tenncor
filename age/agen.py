import argparse
import json
import os.path

import age.templates.api_tmpl as api
import age.templates.codes_tmpl as codes
import age.templates.grader_tmpl as grader
import age.templates.opera_tmpl as opera
import age.templates.runtime_tmpl as runtime

prog_description = 'Generate c++ glue layer mapping ADE and some data-processing library.'
hdr_postfix = ".hpp"
src_postfix = ".cpp"

api_filename = "api"
codes_filename = "code"
grader_filename = "grader"
opera_filename = "opmap"
runtime_filename = "runtime"

class Fields:
    def __init__(self, fields):
        self.fields = fields

    def unmarshal_json(self, jobj):
        outs = {}
        for field in self.fields:
            outtype, outholder = self.fields[field]
            if field not in jobj:
                continue
            entry = jobj[field]
            gottype = type(entry)
            if gottype != outtype:
                raise Exception("cannot read {} of type {} as type {}".format(\
                    field, gottype.__name__, outtype.__name__))
            if str == type(outholder):
                outs[outholder] = entry
            else:
                outs.update(outholder.unmarshal_json(entry))
        return outs

root = Fields({
    "opcodes": (dict, "opcodes"),
    "dtypes": (dict, "dtypes"),
    "data": (dict, Fields({
        "sum": (unicode, "sum"),
        "prod": (unicode, "prod"),
        "data_in": (unicode, "data_in"),
        "data_out": (unicode, "data_out"),
        "scalarize": (unicode, "scalarize"),
    })),
    "apis": (list, "apis")
})

def parse(cfg_str):
    args = json.loads(cfg_str)
    if type(args) != dict:
        raise Exception("cannot parse non-root object {}".format(cfg_str))

    if 'includes' in args:
        includes = args['includes']
    else:
        includes = {}
    return root.unmarshal_json(args), includes

def format_include(includes):
    return '\n'.join(["#include " + include for include in includes]) + '\n\n'

def make_dir(fields, includes, includepath):
    opcodes = fields["opcodes"]

    code_fields = {
        "opcodes": opcodes.keys(),
        "dtypes": fields["dtypes"]
    }
    codes_header = codes_filename + hdr_postfix
    codes_source = codes_filename + src_postfix
    codes_hdr_path = os.path.join(includepath, codes_header)

    codes_header_include = ["<string>"]
    codes_source_include = ["<unordered_map>", '"err/log.hpp"', '"' + codes_hdr_path + '"']
    if codes_header in includes:
        codes_header_include += includes[codes_header]
    if codes_source in includes:
        codes_source_include += includes[codes_source]

    runtime_fields = {
        "sum": fields["sum"],
        "prod": fields["prod"],
        "scalarize": fields["scalarize"],
    }
    runtime_header = runtime_filename + hdr_postfix
    runtime_source = runtime_filename + src_postfix
    runtime_hdr_path = os.path.join(includepath, runtime_header)

    runtime_header_include = ['"age/runtime/grader.hpp"']
    runtime_source_include = ['"' + codes_hdr_path + '"', '"' + runtime_hdr_path + '"']
    if runtime_header in includes:
        runtime_header_include += includes[runtime_header]
    if runtime_source in includes:
        runtime_source_include += includes[runtime_source]

    api_header = api_filename + hdr_postfix
    api_source = api_filename + src_postfix
    api_hdr_path = os.path.join(includepath, api_header)

    api_header_include = ['"age/runtime/grader.hpp"', '"ade/functor.hpp"']
    api_source_include = ['"' + codes_hdr_path + '"', '"' + api_hdr_path + '"']
    if api_header in includes:
        api_header_include += includes[api_header]
    if api_source in includes:
        api_source_include += includes[api_source]

    grader_fields = {
        "grads": {code: opcodes[code]["derivative"] for code in opcodes}
    }
    grader_header = grader_filename + hdr_postfix
    grader_source = grader_filename + src_postfix
    grader_hdr_path = os.path.join(includepath, grader_header)

    grader_header_include = ['"age/runtime/grader.hpp"']
    grader_source_include = ['"' + codes_hdr_path + '"', '"' + api_hdr_path + '"', '"' + grader_hdr_path + '"']
    if grader_header in includes:
        grader_header_include += includes[grader_header]
    if grader_source in includes:
        grader_source_include += includes[grader_source]

    opera_fields = {
        "data_out": fields["data_out"],
        "data_in": fields["data_in"],
        "types": fields["dtypes"],
        "ops": {code: opcodes[code]["operation"] for code in opcodes}
    }
    opera_header = opera_filename + hdr_postfix
    opera_source = opera_filename + src_postfix
    opera_hdr_path = os.path.join(includepath, opera_header)

    opera_header_include = ['"ade/functor.hpp"', '"' + codes_hdr_path + '"', ]
    opera_source_include = ['"' + opera_hdr_path + '"']
    if opera_header in includes:
        opera_header_include += includes[opera_header]
    if opera_source in includes:
        opera_source_include += includes[opera_source]

    return {
        (api_header,
            format_include(api_header_include) + api.header.repr(fields)),
        (api_source,
            format_include(api_source_include) + api.source.repr(fields)),
        (codes_header,
            format_include(codes_header_include) + codes.header.repr(code_fields)),
        (codes_source,
            format_include(codes_source_include) + codes.source.repr(code_fields)),
        (grader_header,
            format_include(grader_header_include) + grader.header.repr(grader_fields)),
        (grader_source,
            format_include(grader_source_include) + grader.source.repr(grader_fields)),
        (opera_header,
            format_include(opera_header_include) + opera.header.repr(opera_fields)),
        (opera_source,
            format_include(opera_source_include) + opera.source.repr(opera_fields)),
        (runtime_header,
            format_include(runtime_header_include) + runtime.header.repr(runtime_fields)),
        (runtime_source,
            format_include(runtime_source_include) + runtime.source.repr(runtime_fields)),
    }

def main(cfgpath = None,
    outpath = None,
    strip_prefix = ''):

    includepath = outpath
    if includepath and includepath.startswith(strip_prefix):
        includepath = includepath[len(strip_prefix):]

    if cfgpath:
        with open(str(cfgpath), 'r') as cfg:
            cfg_str = cfg.read()
        if cfg_str == None:
            raise Exception("cannot read from cfg file {}".format(cfgpath))
    else:
        cfg_str = sys.stdin.read()
    fields, includes = parse(cfg_str)

    directory = make_dir(fields, includes, includepath)

    if outpath:
        for fname, content in directory:
            with open(os.path.join(outpath, fname), 'w') as out:
                out.write(content)
    else:
        for fname, content in directory:
            print("============== %s ==============" % fname)
            print(content)

if '__main__' == __name__:
    parser = argparse.ArgumentParser(description=prog_description)
    parser.add_argument('--cfg', dest='cfgpath', nargs='?',
        help='Configuration json file on mapping info (default: read from stdin)')
    parser.add_argument('--out', dest='outpath', nargs='?',
        help='Directory path to dump output files (default: write to stdin)')
    parser.add_argument('--strip_prefix', dest='strip_prefix', nargs='?', default='',
        help='Directory path to dump output files (default: write to stdin)')
    args = parser.parse_args()

    main(cfgpath = args.cfgpath,
        outpath = args.outpath,
        strip_prefix = args.strip_prefix)
