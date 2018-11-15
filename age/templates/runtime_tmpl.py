''' Representation of runtime extension files '''

import repr

# EXPORT
header = repr.FILE_REPR("""#ifndef _GENERATED_RUNTIME_HPP
#define _GENERATED_RUNTIME_HPP

namespace age
{{

template <typename T>
ade::Tensor* data (T scalar, ade::Shape shape)
{{
	return {scalarize};
}}

ade::Opcode sum_opcode (void);

ade::Opcode prod_opcode (void);

}}

#endif // _GENERATED_RUNTIME_HPP
""")

header.scalarize = ("scalarize", lambda scalarize: scalarize)

# EXPORT
source = repr.FILE_REPR("""#ifdef _GENERATED_RUNTIME_HPP

namespace age
{{

ade::Opcode sum_opcode (void)
{{
	return ade::Opcode{{"{sum}", {sum}}};
}}

ade::Opcode prod_opcode (void)
{{
	return ade::Opcode{{"{prod}", {prod}}};
}}

}}

#endif
""")

source.sum = ("sum", lambda sum: sum)

source.prod = ("prod", lambda prod: prod)
