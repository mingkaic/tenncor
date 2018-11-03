#include "age/grader.hpp"

#ifdef ADE_GRADER_HPP

namespace ade
{

Tensorptr shaped_one (Shape shape)
{
	return Functor::get(MAKE_CODE(COPY), {
		{extend(0, std::vector<DimT>(shape.begin(), shape.end())),
		Tensor::SYMBOLIC_ONE}
	});
}

Tensorptr shaped_zero (Shape shape)
{
	return Functor::get(MAKE_CODE(COPY), {
		{extend(0, std::vector<DimT>(shape.begin(), shape.end())),
		Tensor::SYMBOLIC_ZERO}
	});
}

#define CODE_CASE(CODE)case CODE: return new Opcode<CODE>();

iOpcode* make_code (OPCODE opcode)
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
			return new Opcode<_BAD_OP>();
	}
}

#undef CODE_CASE

#define GRAD_SIGNATURE(CODE)template <>\
Tensorptr Opcode<CODE>::gradient (ArgsT args, size_t gradidx) const

#define CHECK_UNARY(CODE)if (1 != args.size())\
{ fatalf("cannot %s gradient without 1 argument: using %d argument(s)",\
#CODE, args.size()); } if (gradidx > 0)\
return shaped_zero(args[0].shape());

#define CHECK_BINARY(CODE)if (2 != args.size())\
{ fatalf("cannot %s gradient without 2 arguments: using %d argument(s)",\
#CODE, args.size()); } if (gradidx > 1)\
return shaped_zero(args[0].shape());

#define CHECK_NNARY(CODE)if (0 == args.size())\
{ fatalf("cannot %s gradient with no arguments", #CODE); }\
if (gradidx >= args.size()) return shaped_zero(args[0].shape());

#define ZERO_GRAD(CODE)GRAD_SIGNATURE(CODE)\
{ return shaped_zero(args[0].shape()); }

#define ONE_GRAD(CODE)GRAD_SIGNATURE(CODE)\
{ CHECK_UNARY(CODE) return shaped_one(args[0].shape()); }

#define HANDLE_BINARY(LEFT, RIGHT)return gradidx == 0 ? LEFT : RIGHT;

// grad(copy(f), f) = 1
ONE_GRAD(COPY)

// grad(abs(f), f) = f / abs(f)
GRAD_SIGNATURE(ABS)
{
	CHECK_UNARY(ABS)
	return Functor::get(MAKE_CODE(DIV), {
		args[0],
		{identity, Functor::get(MAKE_CODE(ABS), args)}});
}

// grad(neg(f), f) = -1
GRAD_SIGNATURE(NEG)
{
	CHECK_UNARY(NEG)
	return Functor::get(MAKE_CODE(NEG), {
		{identity, shaped_one(args[0].shape())}});
}

// grad(sin(f), f) = cos(f)
GRAD_SIGNATURE(SIN)
{
	CHECK_UNARY(SIN)
	return Functor::get(MAKE_CODE(COS), args);
}

// grad(cos(f), f) = -sin(f)
GRAD_SIGNATURE(COS)
{
	CHECK_UNARY(COS)
	return Functor::get(MAKE_CODE(NEG), {
		{identity, Functor::get(MAKE_CODE(SIN), args)}
	});
}

// grad(tan(f), f) = sec^2(f)
// 		= 1/cos^2(f)
GRAD_SIGNATURE(TAN)
{
	CHECK_UNARY(TAN)
	Tensorptr denom = Functor::get(MAKE_CODE(COS), args);
	return Functor::get(MAKE_CODE(DIV), {
		{identity, Functor::get(MAKE_CODE(DIV), {
			{identity, shaped_one(args[0].shape())},
			{identity, denom},
		})},
		{identity, denom},
	});
}

// grad(exp(f), f) = exp(f)
GRAD_SIGNATURE(EXP)
{
	CHECK_UNARY(EXP)
	return Functor::get(MAKE_CODE(EXP), args);
}

// grad(log(f), f) = 1 / f
GRAD_SIGNATURE(LOG)
{
	CHECK_UNARY(LOG)
	return Functor::get(MAKE_CODE(DIV), {
		{identity, shaped_one(args[0].shape())},
		args[0]
	});
}

// grad(sqrt(f), f) = 1/(2*sqrt(f))
GRAD_SIGNATURE(SQRT)
{
	CHECK_UNARY(SQRT)
	Tensorptr denom = Functor::get(MAKE_CODE(SQRT), args);
	return Functor::get(MAKE_CODE(DIV), {
		{identity, shaped_one(args[0].shape())},
		{identity, Functor::get(MAKE_CODE(ADD), {
			{identity, denom}, {identity, denom},
		})},
	});
}

// grad(round(f), f) = 1
ONE_GRAD(ROUND)

// grad(pow(f, g), f) = g * pow(f, g - 1)
// grad(pow(f, g), g) = pow(f, g) * log(f)
GRAD_SIGNATURE(POW)
{
	CHECK_BINARY(POW)
	HANDLE_BINARY(
		Functor::get(MAKE_CODE(MUL), {
			args[1],
			{identity, Functor::get(MAKE_CODE(POW), {
				args[0],
				{identity, Functor::get(MAKE_CODE(SUB), {
					args[1],
					{identity, shaped_one(args[0].shape())},
				})}
			})}
		}),
		Functor::get(MAKE_CODE(MUL), {
			{identity, Functor::get(MAKE_CODE(POW), args)},
			{identity, Functor::get(MAKE_CODE(LOG), {args[0]})},
		}))
}

// grad(sub(f, g), f) = 1
// grad(sub(f, g), g) = -1
GRAD_SIGNATURE(SUB)
{
	CHECK_BINARY(SUB)
	HANDLE_BINARY(
		shaped_one(args[0].shape()),
		Functor::get(MAKE_CODE(NEG), {
			{identity, shaped_one(args[0].shape())}
		}))
}

// grad(div(f, g), f) = 1 / g
// grad(div(f, g), g) = -f / g^2
GRAD_SIGNATURE(DIV)
{
	CHECK_BINARY(DIV)
	HANDLE_BINARY(
		Functor::get(MAKE_CODE(DIV), {
			{identity, shaped_one(args[0].shape())}, args[1]
		}),
		Functor::get(MAKE_CODE(DIV), {
			{identity, Functor::get(MAKE_CODE(DIV), {
				{identity, Functor::get(MAKE_CODE(NEG), {args[0]})},
				args[1],
			})}, args[1],
		}))
}

// grad(sum(f, g, ...), argi) = 1 if (argi is in args)
GRAD_SIGNATURE(ADD)
{
	CHECK_NNARY(ADD)
	return shaped_one(args[0].shape());
}

// grad(prod(f, g, ...), argi) = prod(argj if j is not i for j=0:n)
GRAD_SIGNATURE(MUL)
{
	CHECK_NNARY(MUL)
	args.erase(args.begin() + gradidx);
	return Functor::get(MAKE_CODE(MUL), args);
}

// grad(min(f, g, ...), argi) = min(f, g, ...) == argi
GRAD_SIGNATURE(MIN)
{
	CHECK_NNARY(MIN)
	return Functor::get(MAKE_CODE(EQ), {
		{identity, Functor::get(MAKE_CODE(MIN), args)},
		args[gradidx],
	});
}

// grad(min(f, g, ...), argi) = max(f, g, ...) == argi
GRAD_SIGNATURE(MAX)
{
	CHECK_NNARY(MAX)
	return Functor::get(MAKE_CODE(EQ), {
		{identity, Functor::get(MAKE_CODE(MAX), args)},
		args[gradidx],
	});
}

ZERO_GRAD(EQ)

ZERO_GRAD(NE)

ZERO_GRAD(LT)

ZERO_GRAD(GT)

ZERO_GRAD(RAND_BINO)

ZERO_GRAD(RAND_UNIF)

ZERO_GRAD(RAND_NORM)

#undef GRAD_SIGNATURE

#undef CHECK_UNARY

#undef CHECK_BINARY

#undef CHECK_NNARY

#undef ZERO_GRAD

#undef ONE_GRAD

#undef HANDLE_BINARY

}

#endif
