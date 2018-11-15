#include "llo/generated/opmap.hpp"

#ifdef _GENERATED_OPERA_HPP

namespace age
{

void op_exec (_GENERATED_OPCODE opcode, _GENERATED_DTYPE dtype,
	char* out, ade::Shape shape, llo::DataArgsT& in)
{
	switch (dtype)
	{
		case UINT64:
			typed_exec<uint64_t>(opcode, out, shape, in); break;
		case INT32:
			typed_exec<int32_t>(opcode, out, shape, in); break;
		case INT16:
			typed_exec<int16_t>(opcode, out, shape, in); break;
		case DOUBLE:
			typed_exec<double>(opcode, out, shape, in); break;
		case FLOAT:
			typed_exec<float>(opcode, out, shape, in); break;
		case UINT8:
			typed_exec<uint8_t>(opcode, out, shape, in); break;
		case UINT32:
			typed_exec<uint32_t>(opcode, out, shape, in); break;
		case UINT16:
			typed_exec<uint16_t>(opcode, out, shape, in); break;
		case INT64:
			typed_exec<int64_t>(opcode, out, shape, in); break;
		case INT8:
			typed_exec<int8_t>(opcode, out, shape, in); break;
		default: err::fatal("executing bad type");
	}
}

}

#endif
