import logging

from tools.gen.plugin_base import PluginBase
from tools.gen.file_rep import FileRep

from plugins.template import build_template

_header_template = '''
#ifndef _GENERATED_DTYPES_HPP
#define _GENERATED_DTYPES_HPP

namespace egen
{{

{config_defines}

enum _GENERATED_DTYPE
{{
    BAD_TYPE = 0,
    //>>> enumeration
    {enumeration}
    _N_GENERATED_DTYPES,
}};

const _GENERATED_DTYPE default_dtype = {default_dtype};

std::string name_type (_GENERATED_DTYPE type);

uint8_t type_size (_GENERATED_DTYPE type);

size_t type_precision (_GENERATED_DTYPE type);

_GENERATED_DTYPE get_type (const std::string& name);

template <typename T>
_GENERATED_DTYPE get_type (void)
{{
    return BAD_TYPE;
}}

template <typename T>
struct TypeInfo
{{
    static const _GENERATED_DTYPE type = BAD_TYPE;

    TypeInfo (void) = delete;
}};

//>>> mapping
{mapping}

// converts from input to output type
template <typename OUTTYPE>
void type_convert (OUTTYPE* out, const void* input,
    _GENERATED_DTYPE intype, size_t nelems)
{{
    switch (intype)
    {{
        //>>> conversions
        {conversions}
        default:
            global::fatalf("invalid input type %s",
                name_type(intype).c_str());
    }}
}}

#define EVERY_TYPE(GENERIC_MACRO)\\
{apply_everytype}

// GENERIC_MACRO must accept a real type as an argument.
// e.g.:
// #define GENERIC_MACRO(REAL_TYPE) run<REAL_TYPE>(args...);
// ...
// TYPE_LOOKUP(GENERIC_MACRO, type_code)
#define TYPE_LOOKUP(GENERIC_MACRO, DTYPE)\\
switch (DTYPE)\\
{{\\
    {cases}\\
    default: global::fatal("executing bad type");\\
}}
//>>> ^ cases

}}

#endif // _GENERATED_DTYPES_HPP
'''

_source_template = '''
#ifdef _GENERATED_DTYPES_HPP

namespace egen
{{

static const std::unordered_map<_GENERATED_DTYPE,std::string,estd::EnumHash> type2name =
{{
    //>>> type2names
    {type2names}
}};

static const std::unordered_map<std::string,_GENERATED_DTYPE> name2type =
{{
    //>>> name2types
    {name2types}
}};

std::string name_type (_GENERATED_DTYPE type)
{{
    return estd::try_get(type2name, type, "BAD_DTYPE");
}}

_GENERATED_DTYPE get_type (const std::string& name)
{{
    return estd::try_get(name2type, name, BAD_TYPE);
}}

uint8_t type_size (_GENERATED_DTYPE type)
{{
    switch (type)
    {{
        //>>> typesizes
        {typesizes}
        default: global::fatal("cannot get size of bad type");
    }}
    return 0;
}}


size_t type_precision (_GENERATED_DTYPE type)
{{
    switch (type)
    {{
        //>>> precisions
        {precisions}
        default: break;
    }}
    return 0;
}}

//>>> get_types
{get_types}

}}

#endif
'''

dtype_key = 'dtype'

def _handle_config_defines(arguments):
    defns = arguments.get('defines', None)
    if defns:
        return '\n'.join(['#define {}'.format(defn) for defn in defns])
    return ''

def _handle_enumeration(arguments):
    dtypes = arguments[dtype_key]
    assert(len(dtypes))
    dtype_codes = list(dtypes.keys())
    return ',\n    '.join(dtype_codes) + ','

_dtype_mapping_tmp = '''template <>
_GENERATED_DTYPE get_type<{dtype}> (void);

template <>
struct TypeInfo<{dtype}>
{{
    static const _GENERATED_DTYPE type = {code};

    TypeInfo (void) = delete;
}};'''

def _handle_mapping(arguments):
    dtypes = arguments[dtype_key]
    return '\n\n'.join([
        _dtype_mapping_tmp.format(code=code, dtype=dtypes[code]['ctype'])
        for code in dtypes
    ])

_convert_tmp = '''case {code}:
        {{
            std::vector<OUTTYPE> temp(({dtype}*) input, ({dtype}*) input + nelems);
            std::memcpy(out, temp.data(), sizeof(OUTTYPE) * nelems);
        }}
            break;'''
def _handle_conversions(arguments):
    dtypes = arguments[dtype_key]
    return '\n        '.join([
        _convert_tmp.format(code=code, dtype=dtypes[code]['ctype'])
        for code in dtypes
    ])

def _handle_cases(arguments):
    dtypes = arguments[dtype_key]
    _dtype_case_tmp = 'case egen::{code}: GENERIC_MACRO({dtype}) break;'
    return '\\\n    '.join([
        _dtype_case_tmp.format(code=code, dtype=dtypes[code]['ctype'])
        for code in dtypes
    ])

def _handle_type2names(arguments):
    dtypes = arguments[dtype_key]
    _dtype2names_tmp = '{{ {code}, "{code}" }}'
    return ',\n    '.join([
        _dtype2names_tmp.format(code=code)
        for code in dtypes
    ])

def _handle_name2types(arguments):
    dtypes = arguments[dtype_key]
    _names2dtype_tmp = '{{ "{code}", {code} }}'
    return ',\n    '.join([
        _names2dtype_tmp.format(code=code)
        for code in dtypes
    ])

def _handle_typesizes(arguments):
    dtypes = arguments[dtype_key]
    _size_case_tmp = 'case egen::{code}: return sizeof({dtype});'
    return '\n    '.join([
        _size_case_tmp.format(code=code, dtype=dtypes[code]['ctype'])
        for code in dtypes
    ])

def _handle_precisions(arguments):
    dtypes = arguments[dtype_key]
    _precision_case_tmp = 'case egen::{code}: return {precision};'
    return '\n    '.join([
        _precision_case_tmp.format(code=code,
            precision=dtypes[code].get('precision', 0))
        for code in dtypes
    ])

_get_type_tmp = '''
template <>
_GENERATED_DTYPE get_type<{dtype}> (void)
{{
    return {code};
}}
'''
def _handle_get_types(arguments):
    dtypes = arguments[dtype_key]
    return ''.join([
        _get_type_tmp.format(code=code, dtype=dtypes[code]['ctype'])
        for code in dtypes
    ])

def _handle_default_dtype(arguments):
    deftype = arguments.get('default_type',
        list(arguments[dtype_key].keys())[0])
    return deftype

def _handle_apply_everytype(arguments):
    dtypes = arguments[dtype_key]
    return '\\\n'.join([
        'GENERIC_MACRO({})'.format(dtypes[code]['ctype'])
        for code in dtypes
    ])

_plugin_id = "DTYPE"

@PluginBase.register
class DTypesPlugin:

    def plugin_id(self):
        return _plugin_id

    def process(self, generated_files, arguments, **kwargs):
        _hdr_file = 'dtype.hpp'
        _src_file = 'dtype.cpp'
        dtype_key = 'dtype'
        if dtype_key not in arguments:
            logging.warning(
                'no relevant arguments found for plugin %s', _plugin_id)
            return

        module = globals()

        generated_files[_hdr_file] = FileRep(
            build_template(_header_template, module, arguments),
            user_includes=['<string>', '<cstring>', '"internal/global/global.hpp"'],
            internal_refs=[])

        generated_files[_src_file] = FileRep(
            build_template(_source_template, module, arguments),
            user_includes=['"estd/contain.hpp"'],
            internal_refs=[_hdr_file])

        return generated_files
