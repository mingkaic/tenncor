#include "llo/node.hpp"

#ifdef LLO_NODE_HPP

namespace llo
{

#define FILL_ONE(TYPE){\
TYPE* ptr = (TYPE*) cptr;\
std::fill(ptr, ptr + n, (TYPE) 1); } break;

static void fill_one (char* cptr, size_t n, DTYPE dtype)
{
	switch (dtype)
	{
		case DOUBLE:
			FILL_ONE(double)
		case FLOAT:
			FILL_ONE(float)
		case INT8:
			FILL_ONE(int8_t)
		case INT16:
			FILL_ONE(int16_t)
		case INT32:
			FILL_ONE(int32_t)
		case INT64:
			FILL_ONE(int64_t)
		case UINT8:
			FILL_ONE(uint8_t)
		case UINT16:
			FILL_ONE(uint16_t)
		case UINT32:
			FILL_ONE(uint32_t)
		case UINT64:
			FILL_ONE(uint64_t)
		default:
			util::handle_error("evaluating unknown type");
	}
}

#undef FILL_ONE

GenericData evaluate (DTYPE dtype, ade::iTensor* tens)
{
	if (tens == ade::Tensor::SYMBOLIC_ONE.get())
	{
		GenericData out(ade::Shape(), dtype);
		fill_one(out.data_.get(), 1, dtype);
		return out;
	}
	else if (tens == ade::Tensor::SYMBOLIC_ZERO.get())
	{
		GenericData out(ade::Shape(), dtype);
		std::memset(out.data_.get(), 0, type_size(dtype));
		return out;
	}
	else if (iEvaluable* ev = dynamic_cast<iEvaluable*>(tens))
	{
		return ev->evaluate(dtype);
	}
	ade::iFunctor* f = static_cast<ade::iFunctor*>(tens);
	ade::OPCODE opcode = f->get_code();

	std::vector<ade::iTensor*> refs = f->get_children();
	uint8_t nargs = refs.size();

	GenericData out(f->shape(), dtype);
	std::vector<GenericData> argdata(nargs);
	if (opcode == ade::RAND_BINO)
	{
		if (nargs != 2)
		{
			util::handle_error("RAND_BINO op does not have 2 args",
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
	switch (opcode)
	{
		case ade::MATMUL:
		{
			if (auto mf = dynamic_cast<ade::Functor<
				ade::MATMUL,uint8_t,uint8_t>*>(f))
			{
				op_exec(opcode, out, argdata,
					std::get<0>(mf->meta()), std::get<1>(mf->meta()));
			}
			else
			{
				op_exec(opcode, out, argdata);
			}
		}
		break;
		case ade::PERMUTE:
		{
			auto pf = static_cast<ade::Functor<
				ade::PERMUTE,std::vector<uint8_t>>*>(f);
			op_exec(opcode, out, argdata, std::get<0>(pf->meta()));
		}
		break;
		default:
			op_exec(opcode, out, argdata);
	}
	return out;
}

}

#endif
