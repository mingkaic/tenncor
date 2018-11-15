''' Representation of operation mapping files '''

import repr

# EXPORT
header = repr.FILE_REPR("""#ifndef _GENERATED_OPERA_HPP
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
		default: err::fatal("unknown opcode");
	}}
}}

void op_exec (_GENERATED_OPCODE opcode, _GENERATED_DTYPE dtype,
	{data_out} out, ade::Shape shape, {data_in} in);

}}

#endif // _GENERATED_OPERA_HPP
""")

header.data_in = ("data_in", lambda data_in: data_in)

header.data_out = ("data_out", lambda data_out: data_out)

header.ops = ("ops", lambda ops: '\n'.join(["""		case {code}:
			{retval}; break;""".format(\
	code = code, retval = ops[code]) for code in ops]))

# EXPORT
source = repr.FILE_REPR("""#ifdef _GENERATED_OPERA_HPP

namespace age
{{

void op_exec (_GENERATED_OPCODE opcode, _GENERATED_DTYPE dtype,
	{data_out} out, ade::Shape shape, {data_in} in)
{{
	switch (dtype)
	{{
{types}
		default: err::fatal("executing bad type");
	}}
}}

}}

#endif
""")

source.data_in = ("data_in", lambda data_in: data_in)

source.data_out = ("data_out", lambda data_out: data_out)

source.types = ("types", lambda dtypes: '\n'.join(["""		case {dtype}:
			typed_exec<{real_type}>(opcode, out, shape, in); break;""".format(\
	dtype = dtype, real_type = dtypes[dtype]) for dtype in dtypes]))
