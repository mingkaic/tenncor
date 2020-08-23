import logging

from tools.gen.plugin_base import PluginBase
from tools.gen.file_rep import FileRep

from plugins.template import build_template

_header_template = '''
#ifndef _GENERATED_OPCODES_HPP
#define _GENERATED_OPCODES_HPP

namespace egen
{{

enum _GENERATED_OPCODE
{{
    BAD_OP = 0,
    //>>> opcodes
    {opcodes}
    _N_GENERATED_OPCODES,
}};

std::string name_op (_GENERATED_OPCODE code);

_GENERATED_OPCODE get_op (const std::string& name);

bool is_commutative (_GENERATED_OPCODE code);

bool is_commutative (const std::string& name);

bool is_idempotent (_GENERATED_OPCODE code);

bool is_idempotent (const std::string& name);

template <typename T>
void typed_exec (_GENERATED_OPCODE opcode, {params})
{{
    //>>> ^ params
    switch (opcode)
    {{
        //>>> ops
        {ops}
        default: global::fatal("unknown opcode");
    }}
}}

// GENERIC_MACRO must accept a static opcode as an argument.
// e.g.:
// #define GENERIC_MACRO(COMPILE_OPCODE) run<COMPILE_OPCODE>(args...);
// ...
// OPCODE_LOOKUP(GENERIC_MACRO, rt_opcode)
// this is used for mapping compile-time ops using runtime opcode variable
#define OPCODE_LOOKUP(GENERIC_MACRO, OPCODE)\\
switch (OPCODE)\\
{{\\
    {cases}\\
    default: global::fatal("executing bad op");\\
}}
//>>> ^ cases

}}

#endif // _GENERATED_OPCODES_HPP
'''

_source_template = '''
#ifdef _GENERATED_OPCODES_HPP

namespace egen
{{

static const std::unordered_map<_GENERATED_OPCODE,std::string,estd::EnumHash> code2name =
{{
    //>>> code2names
    {code2names}
}};

static const std::unordered_map<std::string,_GENERATED_OPCODE> name2code =
{{
    //>>> name2codes
    {name2codes}
}};

static const std::unordered_set<_GENERATED_OPCODE,estd::EnumHash> commutatives =
{{
    //>>> commcodes
    {commcodes}
}};

static const std::unordered_set<_GENERATED_OPCODE,estd::EnumHash> idempotents =
{{
    //>>> idemcodes
    {idemcodes}
}};

std::string name_op (_GENERATED_OPCODE code)
{{
    return estd::try_get(code2name, code, "BAD_OP");
}}

_GENERATED_OPCODE get_op (const std::string& name)
{{
    return estd::try_get(name2code, name, BAD_OP);
}}

bool is_commutative (_GENERATED_OPCODE code)
{{
    return estd::has(commutatives, code);
}}

bool is_commutative (const std::string& name)
{{
    if (estd::has(name2code, name))
    {{
        return is_commutative(name2code.at(name));
    }}
    return false;
}}

bool is_idempotent (_GENERATED_OPCODE code)
{{
    return estd::has(idempotents, code);
}}

bool is_idempotent (const std::string& name)
{{
    if (estd::has(name2code, name))
    {{
        return is_idempotent(name2code.at(name));
    }}
    return false;
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
        _opcode_case_tmp.format(code=opcode, stmt=opcalls[opcode]['stmt'])
        for opcode in opcalls
    ])

def _handle_cases(params, opcalls):
    _lookup_case_tmp = 'case egen::{code}: GENERIC_MACRO(::egen::{code}) break;'
    return '\\\n    '.join([
        _lookup_case_tmp.format(code=opcode)
        for opcode in opcalls
    ])

def _handle_code2names(params, opcalls):
    _code2names_tmp = '{{ {code}, "{code}" }}'
    return ',\n    '.join([
        _code2names_tmp.format(code=code)
        for code in list(opcalls.keys())
    ])

def _handle_name2codes(params, opcalls):
    _name2codes_tmp = '{{ "{code}", {code} }}'
    return ',\n    '.join([
        _name2codes_tmp.format(code=code)
        for code in list(opcalls.keys())
    ])

def _handle_commcodes(params, opcalls):
    return ',\n    '.join(filter(
        lambda code: opcalls[code].get("commutative", False),
        list(opcalls.keys())))

def _handle_idemcodes(params, opcalls):
    return ',\n    '.join(filter(
        lambda code: opcalls[code].get("idempotent", True),
        list(opcalls.keys())))

_plugin_id = "OPCODE"

@PluginBase.register
class OpcodesPlugin:

    def plugin_id(self):
        return _plugin_id

    def process(self, generated_files, arguments, **kwargs):
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
            user_includes=['"estd/contain.hpp"'],
            internal_refs=[_hdr_file])

        return generated_files