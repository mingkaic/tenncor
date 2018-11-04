#include "age/operation.hpp"

#ifdef AGE_OPERATION_HPP

namespace age
{

#define CODE_CASE(CODE)case CODE: return MAKE_CODE(CODE);

ade::OpPtrT make_code (OPCODE opcode)
{
	switch (opcode)
	{
		CODE_CASE(COPY)
		CODE_CASE(ABS)
		CODE_CASE(NEG)
		CODE_CASE(SIN)
		CODE_CASE(COS)
		CODE_CASE(TAN)
		CODE_CASE(EXP)
		CODE_CASE(LOG)
		CODE_CASE(SQRT)
		CODE_CASE(ROUND)
		CODE_CASE(POW)
		CODE_CASE(ADD)
		CODE_CASE(SUB)
		CODE_CASE(MUL)
		CODE_CASE(DIV)
		CODE_CASE(MIN)
		CODE_CASE(MAX)
		CODE_CASE(EQ)
		CODE_CASE(NE)
		CODE_CASE(GT)
		CODE_CASE(LT)
		CODE_CASE(RAND_BINO)
		CODE_CASE(RAND_UNIF)
		CODE_CASE(RAND_NORM)
		default:
			return MAKE_CODE(_BAD_OP);
	}
}

#undef CODE_CASE

ade::Tensorptr shaped_one (ade::Shape shape)
{
	return ade::Functor::get(MAKE_CODE(COPY), {
		{ade::extend(0, std::vector<ade::DimT>(shape.begin(), shape.end())),
		ade::Tensor::SYMBOLIC_ONE}
	});
}

ade::Tensorptr shaped_zero (ade::Shape shape)
{
	return ade::Functor::get(MAKE_CODE(COPY), {
		{ade::extend(0, std::vector<ade::DimT>(shape.begin(), shape.end())),
		ade::Tensor::SYMBOLIC_ZERO}
	});
}

}

#endif
