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
void typed_exec (_GENERATED_OPCODE opcode,
    {data_out} out, ade::Shape shape, {data_in} in)
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

header.data_in = ('data.data_in', lambda data_in: data_in)

header.data_out = ('data.data_out', lambda data_out: data_out)

header.ops = ('opcodes', lambda opcodes: '\n'.join(['''        case {code}:
            {retval}; break;'''.format(\
    code = code, retval = opcodes[code]['operation']) for code in template.sortkey(opcodes)]))

header.generic_macros = ('dtypes', lambda dtypes: '\\\n'.join([
    '    case {dtype}: GENERIC_MACRO({real_type}) break;'.format(\
    dtype = dtype, real_type = dtypes[dtype]) for dtype in template.sortkey(dtypes)]))
