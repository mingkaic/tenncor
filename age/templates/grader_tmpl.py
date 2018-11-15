''' Representation of gradient mapping files '''

import repr

# EXPORT
header = repr.FILE_REPR("""#ifndef _GENERATED_GRADER_HPP
#define _GENERATED_GRADER_HPP

namespace age
{{

ade::Tensorptr grad_rule (size_t code,TensT args,size_t idx);

}}

#endif // _GENERATED_GRADER_HPP
""")

# EXPORT
source = repr.FILE_REPR("""#ifdef _GENERATED_GRADER_HPP

namespace age
{{

ade::Tensorptr grad_rule (size_t code,TensT args,size_t idx)
{{
	switch (code)
	{{
{gradops}
		default: err::fatal("no gradient rule for unknown opcode");
	}}
}}

}}

#endif
""")

source.gradops = ("grads", lambda gradmap: '\n'.join(["\t\tcase {code}: return {retval};".format(\
	code = code, retval = gradmap[code]) for code in gradmap]))
