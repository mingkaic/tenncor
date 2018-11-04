#include "llo/opmap.hpp"

#ifdef LLO_OPMAP_HPP

namespace llo
{

void op_exec (age::OPCODE opcode, GenericData& out, DataArgsT& data)
{
	switch (opcode)
	{
		case age::COPY:
			exec<age::COPY>(out, data);
		break;
		case age::ABS:
			exec<age::ABS>(out, data);
		break;
		case age::NEG:
			exec<age::NEG>(out, data);
		break;
		case age::SIN:
			exec<age::SIN>(out, data);
		break;
		case age::COS:
			exec<age::COS>(out, data);
		break;
		case age::TAN:
			exec<age::TAN>(out, data);
		break;
		case age::EXP:
			exec<age::EXP>(out, data);
		break;
		case age::LOG:
			exec<age::LOG>(out, data);
		break;
		case age::SQRT:
			exec<age::SQRT>(out, data);
		break;
		case age::ROUND:
			exec<age::ROUND>(out, data);
		break;
		case age::POW:
			exec<age::POW>(out, data);
		break;
		case age::ADD:
			exec<age::ADD>(out, data);
		break;
		case age::SUB:
			exec<age::SUB>(out, data);
		break;
		case age::MUL:
			exec<age::MUL>(out, data);
		break;
		case age::DIV:
			exec<age::DIV>(out, data);
		break;
		case age::EQ:
			exec<age::EQ>(out, data);
		break;
		case age::NE:
			exec<age::NE>(out, data);
		break;
		case age::LT:
			exec<age::LT>(out, data);
		break;
		case age::GT:
			exec<age::GT>(out, data);
		break;
		case age::MIN:
			exec<age::MIN>(out, data);
		break;
		case age::MAX:
			exec<age::MAX>(out, data);
		break;
		case age::RAND_BINO:
			exec<age::RAND_BINO>(out, data);
		break;
		case age::RAND_UNIF:
			exec<age::RAND_UNIF>(out, data);
		break;
		case age::RAND_NORM:
			exec<age::RAND_NORM>(out, data);
		break;
		default:
			err::fatal("unknown opcode");
	}
}

}

#endif
