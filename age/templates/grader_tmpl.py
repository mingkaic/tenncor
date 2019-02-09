''' Representation of gradient mapping files '''

import age.templates.template as template

FILENAME = 'grader'

# EXPORT
header = template.AGE_FILE(FILENAME, template.HEADER_EXT,
'''#ifndef _GENERATED_GRADER_HPP
#define _GENERATED_GRADER_HPP

namespace age
{{

ade::TensptrT chain_rule (ade::iFunctor* fwd,
    ade::FuncArg bwd, ade::TensT args, size_t idx);

}}

#endif // _GENERATED_GRADER_HPP
''')

header.scalarize = ('data.scalarize', lambda scalarize: scalarize)

header.sum = ('data.sum', lambda sum: sum)

# EXPORT
source = template.AGE_FILE(FILENAME, template.SOURCE_EXT,
'''#ifdef _GENERATED_GRADER_HPP

namespace age
{{

ade::TensptrT chain_rule (ade::iFunctor* fwd,
    ade::FuncArg bwd, ade::TensT args, size_t idx)
{{
    switch (fwd->get_opcode().code_)
    {{
{gradops}
        default: logs::fatal("no gradient rule for unknown opcode");
    }}
}}

}}

#endif
''')

source.gradops = ('opcodes', lambda opcodes: '\n'.join([
    '        case {code}: return {retval};'.format(\
    code = code, retval = opcodes[code]['derivative']) for code in template.sortkey(opcodes)]))
