''' Representation of operation mapping files '''

import age.templates.template as template

FILENAME = 'opmap'

# EXPORT
header = template.AGE_FILE(FILENAME, template.HEADER_EXT,
'''#ifndef _GENERATED_OPERA_HPP
#define _GENERATED_OPERA_HPP

namespace age
{{

template <typename T>
{signature}_GENERATED_OPCODE opcode, ade::Shape shape, {data_in} in)
{{
    switch (opcode)
    {{
{ops}
        default: logs::fatal("unknown opcode");
    }}
}}

// GENERIC_MACRO must accept a real type as an argument.
// e.g.:
// #define GENERIC_MACRO(REAL_TYPE) run<REAL_TYPE>(args...);
// ...
// TYPE_LOOKUP(GENERIC_MACRO, type_code)
#define TYPE_LOOKUP(GENERIC_MACRO, DTYPE)\\
switch (DTYPE) {{\\
{generic_macros}\\
    default: logs::fatal("executing bad type");\\
}}

}}

#endif // _GENERATED_OPERA_HPP
''')

ref_signature = 'void typed_exec ({data_out} out, '

ret_signature = '{data_out} typed_exec ('

def parse_signature(data_out):
    fmt = ref_signature
    if isinstance(data_out, dict):
        if 'return' in data_out and data_out['return']:
            fmt = ret_signature
        data_out = data_out['type']
    return fmt.format(data_out=data_out)

header.data_in = ('data.data_in', lambda data_in: data_in)

header.signature = ('data.data_out', parse_signature)

header.ops = ('opcodes', lambda opcodes: '\n'.join(['''        case {code}:
            {retval}; break;'''.format(\
    code = code, retval = opcodes[code]['operation']) for code in template.sortkey(opcodes)]))

header.generic_macros = ('dtypes', lambda dtypes: '\\\n'.join([
    '    case age::{dtype}: GENERIC_MACRO({real_type}) break;'.format(\
    dtype = dtype, real_type = dtypes[dtype]) for dtype in template.sortkey(dtypes)]))
