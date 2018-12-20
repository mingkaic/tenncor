''' Representation of operation mapping files '''

import template

FILENAME = 'opmap'

def sortkey(dic):
    arr = dic.keys()
    arr.sort()
    return arr

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

void op_exec (_GENERATED_OPCODE opcode, _GENERATED_DTYPE dtype,
    {data_out} out, ade::Shape shape, {data_in} in);

}}

#endif // _GENERATED_OPERA_HPP
''')

header.data_in = ('data.data_in', lambda data_in: data_in)

header.data_out = ('data.data_out', lambda data_out: data_out)

header.ops = ('opcodes', lambda opcodes: '\n'.join(['''        case {code}:
            {retval}; break;'''.format(\
    code = code, retval = opcodes[code]['operation']) for code in sortkey(opcodes)]))

# EXPORT
source = template.AGE_FILE(FILENAME, template.SOURCE_EXT,
'''#ifdef _GENERATED_OPERA_HPP

namespace age
{{

void op_exec (_GENERATED_OPCODE opcode, _GENERATED_DTYPE dtype,
    {data_out} out, ade::Shape shape, {data_in} in)
{{
    switch (dtype)
    {{
{types}
        default: logs::fatal("executing bad type");
    }}
}}

}}

#endif
''')

source.data_in = ('data.data_in', lambda data_in: data_in)

source.data_out = ('data.data_out', lambda data_out: data_out)

source.types = ('dtypes', lambda dtypes: '\n'.join(['''        case {dtype}:
            typed_exec<{real_type}>(opcode, out, shape, in); break;'''.format(\
    dtype = dtype, real_type = dtypes[dtype]) for dtype in sortkey(dtypes)]))
