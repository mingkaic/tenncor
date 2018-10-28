#include "llo/opmap.hpp"

#ifdef LLO_OPMAP_HPP

namespace llo
{

void op_exec (ade::OPCODE opcode, GenericData& out, DataArgsT& data)
{
	switch (opcode)
	{
		case ade::COPY:
			exec<ade::COPY>(out, data);
		break;
		case ade::ABS:
			exec<ade::ABS>(out, data);
		break;
		case ade::NEG:
			exec<ade::NEG>(out, data);
		break;
		case ade::SIN:
			exec<ade::SIN>(out, data);
		break;
		case ade::COS:
			exec<ade::COS>(out, data);
		break;
		case ade::TAN:
			exec<ade::TAN>(out, data);
		break;
		case ade::EXP:
			exec<ade::EXP>(out, data);
		break;
		case ade::LOG:
			exec<ade::LOG>(out, data);
		break;
		case ade::SQRT:
			exec<ade::SQRT>(out, data);
		break;
		case ade::ROUND:
			exec<ade::ROUND>(out, data);
		break;
		case ade::POW:
			exec<ade::POW>(out, data);
		break;
		case ade::ADD:
			exec<ade::ADD>(out, data);
		break;
		case ade::SUB:
			exec<ade::SUB>(out, data);
		break;
		case ade::MUL:
			exec<ade::MUL>(out, data);
		break;
		case ade::DIV:
			exec<ade::DIV>(out, data);
		break;
		case ade::EQ:
			exec<ade::EQ>(out, data);
		break;
		case ade::NE:
			exec<ade::NE>(out, data);
		break;
		case ade::LT:
			exec<ade::LT>(out, data);
		break;
		case ade::GT:
			exec<ade::GT>(out, data);
		break;
		case ade::MIN:
			exec<ade::MIN>(out, data);
		break;
		case ade::MAX:
			exec<ade::MAX>(out, data);
		break;
		case ade::RAND_BINO:
			exec<ade::RAND_BINO>(out, data);
		break;
		case ade::RAND_UNIF:
			exec<ade::RAND_UNIF>(out, data);
		break;
		case ade::RAND_NORM:
			exec<ade::RAND_NORM>(out, data);
		break;
		default:
			ade::fatal("unknown opcode");
	}
}

}

#endif
