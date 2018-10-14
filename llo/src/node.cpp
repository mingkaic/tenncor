#include "llo/node.hpp"

#ifdef LLO_NODE_HPP

namespace llo
{

#define FILL_ONE(TYPE){\
TYPE* ptr = (TYPE*) cptr;\
std::fill(ptr, ptr + n, (TYPE) 1); } break;

void fill_one (char* cptr, size_t n, DTYPE dtype)
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
			ade::fatal("evaluating unknown type");
	}
}

void get_func_children (std::vector<GenericData>& out,
	const EvalCtx& ctx, DTYPE dtype, ade::iFunctor* func)
{
	std::vector<ade::iTensor*> children = func->get_children();
	uint8_t nargs = children.size();
	out = std::vector<GenericData>(nargs);
	if (func->get_code() == ade::RAND_BINO)
	{
		if (nargs != 2)
		{
			ade::fatalf("cannot RAND_BINO without 2 arguments: "
				"using %d arguments", nargs);
		}
		Evaluator left_eval(ctx, dtype);
		children[0]->accept(left_eval);
		out[0] = left_eval.out_;

		Evaluator right_eval(ctx, DOUBLE);
		children[1]->accept(right_eval);
		out[1] = right_eval.out_;
	}
	else
	{
		for (uint8_t i = 0; i < nargs; ++i)
		{
			Evaluator evaler(ctx, dtype);
			children[i]->accept(evaler);
			out[i] = evaler.out_;
		}
	}
}

#undef FILL_ONE

}

#endif
