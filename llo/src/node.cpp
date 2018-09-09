#include "llo/node.hpp"

GenericData evaluate (DTYPE dtype, ade::iTensor* tens)
{
	if (Evaluable* ev = dynamic_cast<Evaluable*>(tens))
	{
		return ev->evaluate(dtype);
	}
	ade::iFunctor* f = static_cast<ade::iFunctor*>(tens);
	ade::OPCODE opcode = f->get_code();

	std::vector<ade::iTensor*> refs = f->get_refs();
	uint8_t nargs = refs.size();

	GenericData out(f->shape(), dtype);
	std::vector<GenericData> argdata(nargs);
	if (opcode == ade::BINO)
	{
		if (nargs != 2)
		{
			util::handle_error("BINO op does not have 2 args",
				util::ErrArg<size_t>("nargs", nargs));
		}
		argdata[0] = evaluate(dtype, refs[0]);
		argdata[1] = evaluate(DOUBLE, refs[1]);
	}
	else
	{
		for (uint8_t i = 0; i < nargs; ++i)
		{
			argdata[i] = evaluate(dtype, refs[i]);
		}
	}
	op_exec(opcode, out, argdata);
	return out;
}
