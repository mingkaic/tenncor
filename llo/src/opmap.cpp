#include "llo/opmap.hpp"

#ifdef LLO_OPMAP_HPP

namespace llo
{

void op_exec (age::_GENERATED_OPCODES opcode, GenericData& out, DataArgsT& data)
{
	switch (out.dtype_)
	{
		case DOUBLE:
			typed_exec<double>(opcode, out, data);
		break;
		case FLOAT:
			typed_exec<float>(opcode, out, data);
		break;
		case INT8:
			typed_exec<int8_t>(opcode, out, data);
		break;
		case INT16:
			typed_exec<int16_t>(opcode, out, data);
		break;
		case INT32:
			typed_exec<int32_t>(opcode, out, data);
		break;
		case INT64:
			typed_exec<int64_t>(opcode, out, data);
		break;
		case UINT8:
			typed_exec<uint8_t>(opcode, out, data);
		break;
		case UINT16:
			typed_exec<uint16_t>(opcode, out, data);
		break;
		case UINT32:
			typed_exec<uint32_t>(opcode, out, data);
		break;
		case UINT64:
			typed_exec<uint64_t>(opcode, out, data);
		break;
		default:
			err::fatal("executing bad type");
	}
}

}

#endif
