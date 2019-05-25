import logging

from gen.plugin_base2 import PluginBase
from gen.file_rep import FileRep

from ead.generator.plugins.template import build_template

_header_template = '''
#ifndef _GENERATED_DTYPES_HPP
#define _GENERATED_DTYPES_HPP

namespace age
{{

enum _GENERATED_DTYPE
{{
    BAD_TYPE = 0,
    //>>> enumeration
    {enumeration}
    _N_GENERATED_DTYPES,
}};

std::string name_type (_GENERATED_DTYPE type);

uint8_t type_size (_GENERATED_DTYPE type);

_GENERATED_DTYPE get_type (std::string name);

template <typename T>
_GENERATED_DTYPE get_type (void)
{{
    return BAD_TYPE;
}}

//>>> mapping
{mapping}

// uses std containers for type conversion
template <typename OUTTYPE>
void type_convert (std::vector<OUTTYPE>& out, void* input,
    _GENERATED_DTYPE intype, size_t nelems)
{{
    switch (intype)
    {{
        //>>> conversions
        {conversions}
        default:
            logs::fatalf("invalid input type %s",
                name_type(intype).c_str());
    }}
}}

// GENERIC_MACRO must accept a real type as an argument.
// e.g.:
// #define GENERIC_MACRO(REAL_TYPE) run<REAL_TYPE>(args...);
// ...
// TYPE_LOOKUP(GENERIC_MACRO, type_code)
#define TYPE_LOOKUP(GENERIC_MACRO, DTYPE)\\
switch (DTYPE)\\
{{\\
    {cases}\\
    default: logs::fatal("executing bad type");\\
}}
//>>> ^ cases

}}

#endif // _GENERATED_DTYPES_HPP
'''

_source_template = '''
#ifdef _GENERATED_DTYPES_HPP

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

static std::unordered_map<_GENERATED_DTYPE,std::string,EnumHash> type2name =
{{
    //>>> type2names
    {type2names}
}};

static std::unordered_map<std::string,_GENERATED_DTYPE> name2type =
{{
    //>>> name2types
    {name2types}
}};

std::string name_type (_GENERATED_DTYPE type)
{{
    auto it = type2name.find(type);
    if (type2name.end() == it)
    {{
        return "BAD_TYPE";
    }}
    return it->second;
}}

_GENERATED_DTYPE get_type (std::string name)
{{
    auto it = name2type.find(name);
    if (name2type.end() == it)
    {{
        return BAD_TYPE;
    }}
    return it->second;
}}

uint8_t type_size (_GENERATED_DTYPE type)
{{
    switch (type)
    {{
        //>>> typesizes
        {typesizes}
        default: logs::fatal("cannot get size of bad type");
    }}
    return 0;
}}

//>>> get_types
{get_types}

}}

#endif
'''

def _handle_enumeration(dtypes):
    assert(len(dtypes))
    dtype_codes = list(dtypes.keys())
    return ',\n    '.join(dtype_codes) + ','

def _handle_mapping(dtypes):
    _dtype_mapping_tmp = 'template <>\n'+\
        '_GENERATED_DTYPE get_type<{dtype}> (void);'
    return '\n\n'.join([
        _dtype_mapping_tmp.format(dtype=dtypes[code])
        for code in dtypes
    ])

_convert_tmp = '''case {code}:
            out = std::vector<OUTTYPE>(({dtype}*) input, ({dtype}*) input + nelems);
            break;'''
def _handle_conversions(dtypes):
    return '\n        '.join([
        _convert_tmp.format(code=code, dtype=dtypes[code])
        for code in dtypes
    ])

def _handle_cases(dtypes):
    _dtype_case_tmp = 'case age::{code}: GENERIC_MACRO({dtype}) break;'
    return '\\\n    '.join([
        _dtype_case_tmp.format(code=code, dtype=dtypes[code])
        for code in dtypes
    ])

def _handle_type2names(dtypes):
    _dtype2names_tmp = '{{ {code}, "{code}" }}'
    return ',\n    '.join([
        _dtype2names_tmp.format(code=code)
        for code in dtypes
    ])

def _handle_name2types(dtypes):
    _names2dtype_tmp = '{{ "{code}", {code} }}'
    return ',\n    '.join([
        _names2dtype_tmp.format(code=code)
        for code in dtypes
    ])

def _handle_typesizes(dtypes):
    _size_case_tmp = 'case age::{code}: return sizeof({dtype});'
    return '\n    '.join([
        _size_case_tmp.format(code=code, dtype=dtypes[code])
        for code in dtypes
    ])


_get_type_tmp = '''
template <>
_GENERATED_DTYPE get_type<{dtype}> (void)
{{
    return {code};
}}
'''
def _handle_get_types(dtypes):
    return ''.join([
        _get_type_tmp.format(code=code, dtype=dtypes[code])
        for code in dtypes
    ])

_plugin_id = "DTYPE"

class DTypesPlugin:

    def plugin_id(self):
        return _plugin_id

    def process(self, generated_files, arguments):
        _hdr_file = 'dtype.hpp'
        _src_file = 'dtype.cpp'
        plugin_key = 'dtype'
        if plugin_key not in arguments:
            logging.warning(
                'no relevant arguments found for plugin %s', _plugin_id)
            return

        module = globals()
        dtypes = arguments[plugin_key]

        generated_files[_hdr_file] = FileRep(
            build_template(_header_template, module, dtypes),
            user_includes=['<string>', '"logs/logs.hpp"'],
            internal_refs=[])

        generated_files[_src_file] = FileRep(
            build_template(_source_template, module, dtypes),
            user_includes=['<unordered_map>'],
            internal_refs=[_hdr_file])

        return generated_files

PluginBase.register(DTypesPlugin)
