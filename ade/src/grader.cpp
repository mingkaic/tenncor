#include "ade/functor.hpp"

namespace ade
{

Tensorptr shaped_one (Shape shape)
{
	return Functor::get(COPY, {{
		extend(0, std::vector<DimT>(shape.begin(), shape.end())),
		Tensor::SYMBOLIC_ONE
	}});
}

Tensorptr shaped_zero (Shape shape)
{
	return Functor::get(COPY, {{
		extend(0, std::vector<DimT>(shape.begin(), shape.end())),
		Tensor::SYMBOLIC_ZERO
	}});
}

#ifdef ADE_GRADER_HPP

#define GRAD_SIGNATURE(CODE)template <>\
Tensorptr grader<CODE> (ArgsT args, size_t gradidx)

#define CHECK_UNARY(CODE)if (1 != args.size())\
{ fatalf("cannot %s gradient without 1 argument: using %d argument(s)",\
#CODE, args.size()); } if (gradidx > 0)\
return shaped_zero(args.front().shape());

#define CHECK_BINARY(CODE)if (2 != args.size())\
{ fatalf("cannot %s gradient without 2 arguments: using %d argument(s)",\
#CODE, args.size()); } if (gradidx > 1)\
return shaped_zero(args.front().shape());

#define CHECK_NNARY(CODE)if (0 == args.size())\
{ fatalf("cannot %s gradient with no arguments", #CODE); }\
if (gradidx >= args.size()) return shaped_zero(args.front().shape());

#define ZERO_GRAD(CODE)GRAD_SIGNATURE(CODE)\
{ return shaped_zero(args.front().shape()); }

#define ONE_GRAD(CODE)GRAD_SIGNATURE(CODE)\
{ CHECK_UNARY(CODE) return shaped_one(args.front().shape()); }

#define HANDLE_BINARY(LEFT, RIGHT)return gradidx == 0 ? LEFT : RIGHT;

// grad(copy(f), f) = 1
ONE_GRAD(COPY)

// grad(abs(f), f) = f / abs(f)
GRAD_SIGNATURE(ABS)
{
	CHECK_UNARY(ABS)
	return Functor::get(DIV, {
		args.front(),
		{identity, Functor::get(ABS, args)}});
}

// grad(neg(f), f) = -1
GRAD_SIGNATURE(NEG)
{
	CHECK_UNARY(NEG)
	return Functor::get(NEG, {
		{identity, shaped_one(args.front().shape())}});
}

// grad(sin(f), f) = cos(f)
GRAD_SIGNATURE(SIN)
{
	CHECK_UNARY(SIN)
	return Functor::get(COS, args);
}

// grad(cos(f), f) = -sin(f)
GRAD_SIGNATURE(COS)
{
	CHECK_UNARY(COS)
	return Functor::get(NEG, {{identity, Functor::get(SIN, args)}});
}

// grad(tan(f), f) = sec^2(f)
// 		= 1/cos^2(f)
GRAD_SIGNATURE(TAN)
{
	CHECK_UNARY(TAN)
	Tensorptr denom = Functor::get(COS, args);
	return Functor::get(DIV, {
		{identity, Functor::get(DIV, {
			{identity, shaped_one(args.front().shape())},
			{identity, denom},
		})},
		{identity, denom},
	});
}

// grad(exp(f), f) = exp(f)
GRAD_SIGNATURE(EXP)
{
	CHECK_UNARY(EXP)
	return Functor::get(EXP, args);
}

// grad(log(f), f) = 1 / f
GRAD_SIGNATURE(LOG)
{
	CHECK_UNARY(LOG)
	return Functor::get(DIV, {
		{identity, shaped_one(args.front().shape())},
		args.front()
	});
}

// grad(sqrt(f), f) = 1/(2*sqrt(f))
GRAD_SIGNATURE(SQRT)
{
	CHECK_UNARY(SQRT)
	Tensorptr denom = Functor::get(SQRT, args);
	return Functor::get(DIV, {
		{identity, shaped_one(args.front().shape())},
		{identity, Functor::get(ADD, {
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
		Functor::get(MUL, {
			args[1],
			{identity, Functor::get(POW, {
				args[0],
				{identity, Functor::get(SUB, {
					args[1],
					{identity, shaped_one(args[0].shape())},
				})}
			})}
		}),
		Functor::get(MUL, {
			{identity, Functor::get(POW, args)},
			{identity, Functor::get(LOG, {args[0]})},
		}))
}

// grad(sub(f, g), f) = 1
// grad(sub(f, g), g) = -1
GRAD_SIGNATURE(SUB)
{
	CHECK_BINARY(SUB)
	HANDLE_BINARY(
		shaped_one(args[0].shape()),
		Functor::get(NEG, {
			{identity, shaped_one(args[0].shape())}
		}))
}

// grad(div(f, g), f) = 1 / g
// grad(div(f, g), g) = -f / g^2
GRAD_SIGNATURE(DIV)
{
	CHECK_BINARY(DIV)
	HANDLE_BINARY(
		Functor::get(DIV, {
			{identity, shaped_one(args[0].shape())}, args[1]
		}),
		Functor::get(DIV, {
			{identity, Functor::get(DIV, {
				{identity, Functor::get(NEG, {args[0]})}, args[1],
			})}, args[1],
		}))
}

// grad(sum(f, g, ...), argi) = 1 if (argi is in args)
GRAD_SIGNATURE(ADD)
{
	CHECK_NNARY(ADD)
	return shaped_one(args.front().shape());
}

// grad(prod(f, g, ...), argi) = prod(argj if j is not i for j=0:n)
GRAD_SIGNATURE(MUL)
{
	CHECK_NNARY(MUL)
	args.erase(args.begin() + gradidx);
	return Functor::get(MUL, args);
}

// grad(min(f, g, ...), argi) = min(f, g, ...) == argi
GRAD_SIGNATURE(MIN)
{
	CHECK_NNARY(MIN)
	return Functor::get(EQ, {
		{identity, Functor::get(MIN, args)},
		args[gradidx],
	});
}

// grad(min(f, g, ...), argi) = max(f, g, ...) == argi
GRAD_SIGNATURE(MAX)
{
	CHECK_NNARY(MAX)
	return Functor::get(EQ, {
		{identity, Functor::get(MAX, args)},
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

#undef ZERO_GRAD

#define CALL_GRAD(OP)case OP: return grader<OP>(args, gradidx);

Tensorptr gradmap (OPCODE op, ArgsT args, size_t gradidx)
{
	switch (op)
	{
		CALL_GRAD(COPY)
		CALL_GRAD(ABS)
		CALL_GRAD(NEG)
		CALL_GRAD(SIN)
		CALL_GRAD(COS)
		CALL_GRAD(TAN)
		CALL_GRAD(EXP)
		CALL_GRAD(LOG)
		CALL_GRAD(SQRT)
		CALL_GRAD(ROUND)
		CALL_GRAD(POW)
		CALL_GRAD(ADD)
		CALL_GRAD(SUB)
		CALL_GRAD(MUL)
		CALL_GRAD(DIV)
		CALL_GRAD(EQ)
		CALL_GRAD(NE)
		CALL_GRAD(LT)
		CALL_GRAD(GT)
		CALL_GRAD(MIN)
		CALL_GRAD(MAX)
		CALL_GRAD(RAND_BINO)
		CALL_GRAD(RAND_UNIF)
		CALL_GRAD(RAND_NORM)
		default: break;
	}
}

}

#endif
