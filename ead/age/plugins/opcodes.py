import logging

try: # this is a workaround (todo: remove)
    from gen.plugin_base import PluginBase
except:
    from gen.plugin_base2 import PluginBase

from gen.file_rep import FileRep

from ead.age.plugins.template import build_template

_header_template = '''
#ifndef _GENERATED_OPCODES_HPP
#define _GENERATED_OPCODES_HPP

namespace age
{{

enum _GENERATED_OPCODE
{{
    BAD_OP = 0,
    //>>> opcodes
    {opcodes}
    _N_GENERATED_OPCODES,
}};

std::string name_op (_GENERATED_OPCODE code);

_GENERATED_OPCODE get_op (std::string name);

template <typename T>
void typed_exec (_GENERATED_OPCODE opcode, {params})
{{
    //>>> ^ params
    switch (opcode)
    {{
        //>>> ops
        {ops}
        default: logs::fatal("unknown opcode");
    }}
}}

}}

#endif // _GENERATED_OPCODES_HPP
'''

_source_template = '''
#ifdef _GENERATED_OPCODES_HPP

namespace age
{{

struct EnumHash
{{
    template <typename T>
    size_t operator() (T e) const
    {{
        return static_cast<size_t>(e);
    }}
}};

static std::unordered_map<_GENERATED_OPCODE,std::string,EnumHash> code2name =
{{
    //>>> code2names
    {code2names}
}};

static std::unordered_map<std::string,_GENERATED_OPCODE> name2code =
{{
    //>>> name2codes
    {name2codes}
}};

std::string name_op (_GENERATED_OPCODE code)
{{
    return estd::try_get(code2name, code, "BAD_OP");
}}

_GENERATED_OPCODE get_op (std::string name)
{{
    return estd::try_get(name2code, name, BAD_OP);
}}

}}

#endif
'''

def _handle_opcodes(params, opcalls):
    assert(len(opcalls))
    dopcodes = list(opcalls.keys())
    return ',\n    '.join(dopcodes) + ','

def _handle_params(params, opcalls):
    return params.strip()

def _handle_ops(params, opcalls):
    _opcode_case_tmp = 'case {code}: {stmt} break;'
    return '\n        '.join([
        _opcode_case_tmp.format(code=opcode, stmt=opcalls[opcode])
        for opcode in opcalls
    ])

def _handle_code2names(params, opcalls):
    _code2names_tmp = '{{ {code}, "{code}" }}'
    return ',\n    '.join([
        _code2names_tmp.format(code=code)
        for code in opcalls
    ])

def _handle_name2codes(params, opcalls):
    _name2codes_tmp = '{{ "{code}", {code} }}'
    return ',\n    '.join([
        _name2codes_tmp.format(code=code)
        for code in opcalls
    ])

_plugin_id = "OPCODE"

class OpcodesPlugin:

    def plugin_id(self):
        return _plugin_id

    def process(self, generated_files, arguments):
        _hdr_file = 'opcode.hpp'
        _src_file = 'opcode.cpp'
        plugin_key = 'opcode'
        if plugin_key not in arguments:
            logging.warning(
                'no relevant arguments found for plugin %s', _plugin_id)
            return

        module = globals()
        opcodes = arguments[plugin_key]

        operator_include = []
        if 'operator_path' in opcodes:
            operator_include.append(
                '"' + opcodes['operator_path'].strip() + '"')

        if 'params' not in opcodes:
            raise Exception('params not in opcodes. keys:' + str(opcodes.keys()))

        if 'opcalls' not in opcodes:
            raise Exception('opcalls not in opcodes. keys:' + str(opcodes.keys()))

        generated_files[_hdr_file] = FileRep(
            build_template(_header_template, module,
                opcodes['params'], opcodes['opcalls']),
            user_includes=['<string>', '"logs/logs.hpp"'] + operator_include,
            internal_refs=[])

        generated_files[_src_file] = FileRep(
            build_template(_source_template, module,
                opcodes['params'], opcodes['opcalls']),
            user_includes=['"estd/estd.hpp"'],
            internal_refs=[_hdr_file])

        return generated_files

PluginBase.register(OpcodesPlugin)
