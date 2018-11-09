#include "adhoc/age/grader.hpp"

#ifdef AGE_GRADER_HPP

namespace age
{

ade::Opcode make_code (OPCODE opcode)
{
	return ade::Opcode{opname(opcode), opcode};
}

ade::Tensorptr shaped_one (ade::Shape shape)
{
	return ade::Functor::get(make_code(COPY), {
		{ade::extend(0, std::vector<ade::DimT>(shape.begin(), shape.end())),
		ade::Tensor::SYMBOLIC_ONE}
	});
}

ade::Tensorptr shaped_zero (ade::Shape shape)
{
	return ade::Functor::get(make_code(COPY), {
		{ade::extend(0, std::vector<ade::DimT>(shape.begin(), shape.end())),
		ade::Tensor::SYMBOLIC_ZERO}
	});
}

#define GRAD_SIGNATURE(CODE)template <>\
ade::Tensorptr gradient<CODE> (\
ade::ArgsT args, size_t gradidx)

#define CHECK_UNARY(CODE)if (1 != args.size())\
{ err::fatalf("cannot %s gradient without 1 argument: using %d argument(s)",\
#CODE, args.size()); } if (gradidx > 0)\
return shaped_zero(args[0].shape());

#define CHECK_BINARY(CODE)if (2 != args.size())\
{ err::fatalf("cannot %s gradient without 2 arguments: using %d argument(s)",\
#CODE, args.size()); } if (gradidx > 1)\
return shaped_zero(args[0].shape());

#define CHECK_NNARY(CODE)if (0 == args.size())\
{ err::fatalf("cannot %s gradient with no arguments", #CODE); }\
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
	return ade::Functor::get(make_code(DIV), {
		args[0],
		{ade::identity, ade::Functor::get(make_code(ABS), args)}});
}

// grad(neg(f), f) = -1
GRAD_SIGNATURE(NEG)
{
	CHECK_UNARY(NEG)
	return ade::Functor::get(make_code(NEG), {
		{ade::identity, shaped_one(args[0].shape())}});
}

// grad(sin(f), f) = cos(f)
GRAD_SIGNATURE(SIN)
{
	CHECK_UNARY(SIN)
	return ade::Functor::get(make_code(COS), args);
}

// grad(cos(f), f) = -sin(f)
GRAD_SIGNATURE(COS)
{
	CHECK_UNARY(COS)
	return ade::Functor::get(make_code(NEG), {
		{ade::identity, ade::Functor::get(make_code(SIN), args)}
	});
}

// grad(tan(f), f) = sec^2(f)
// 		= 1/cos^2(f)
GRAD_SIGNATURE(TAN)
{
	CHECK_UNARY(TAN)
	ade::Tensorptr denom = ade::Functor::get(make_code(COS), args);
	return ade::Functor::get(make_code(DIV), {
		{ade::identity, ade::Functor::get(make_code(DIV), {
			{ade::identity, shaped_one(args[0].shape())},
			{ade::identity, denom},
		})},
		{ade::identity, denom},
	});
}

// grad(exp(f), f) = exp(f)
GRAD_SIGNATURE(EXP)
{
	CHECK_UNARY(EXP)
	return ade::Functor::get(make_code(EXP), args);
}

// grad(log(f), f) = 1 / f
GRAD_SIGNATURE(LOG)
{
	CHECK_UNARY(LOG)
	return ade::Functor::get(make_code(DIV), {
		{ade::identity, shaped_one(args[0].shape())},
		args[0]
	});
}

// grad(sqrt(f), f) = 1/(2*sqrt(f))
GRAD_SIGNATURE(SQRT)
{
	CHECK_UNARY(SQRT)
	ade::Tensorptr denom = ade::Functor::get(make_code(SQRT), args);
	return ade::Functor::get(make_code(DIV), {
		{ade::identity, shaped_one(args[0].shape())},
		{ade::identity, ade::Functor::get(make_code(ADD), {
			{ade::identity, denom}, {ade::identity, denom},
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
		ade::Functor::get(make_code(MUL), {
			args[1],
			{ade::identity, ade::Functor::get(make_code(POW), {
				args[0],
				{ade::identity, ade::Functor::get(make_code(SUB), {
					args[1],
					{ade::identity, shaped_one(args[0].shape())},
				})}
			})}
		}),
		ade::Functor::get(make_code(MUL), {
			{ade::identity, ade::Functor::get(make_code(POW), args)},
			{ade::identity, ade::Functor::get(make_code(LOG), {args[0]})},
		}))
}

// grad(sub(f, g), f) = 1
// grad(sub(f, g), g) = -1
GRAD_SIGNATURE(SUB)
{
	CHECK_BINARY(SUB)
	HANDLE_BINARY(
		shaped_one(args[0].shape()),
		ade::Functor::get(make_code(NEG), {
			{ade::identity, shaped_one(args[0].shape())}
		}))
}

// grad(div(f, g), f) = 1 / g
// grad(div(f, g), g) = -f / g^2
GRAD_SIGNATURE(DIV)
{
	CHECK_BINARY(DIV)
	HANDLE_BINARY(
		ade::Functor::get(make_code(DIV), {
			{ade::identity, shaped_one(args[0].shape())}, args[1]
		}),
		ade::Functor::get(make_code(DIV), {
			{ade::identity, ade::Functor::get(make_code(DIV), {
				{ade::identity, ade::Functor::get(make_code(NEG), {args[0]})},
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
	return ade::Functor::get(make_code(MUL), args);
}

// grad(min(f, g, ...), argi) = min(f, g, ...) == argi
GRAD_SIGNATURE(MIN)
{
	CHECK_NNARY(MIN)
	return ade::Functor::get(make_code(EQ), {
		{ade::identity, ade::Functor::get(make_code(MIN), args)},
		args[gradidx],
	});
}

// grad(min(f, g, ...), argi) = max(f, g, ...) == argi
GRAD_SIGNATURE(MAX)
{
	CHECK_NNARY(MAX)
	return ade::Functor::get(make_code(EQ), {
		{ade::identity, ade::Functor::get(make_code(MAX), args)},
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

ade::Tensorptr gradient (OPCODE opcode, ade::ArgsT args, size_t gradidx)
{
    switch (opcode)
    {
        CASE_DEFN(COPY)
        CASE_DEFN(ABS)
        CASE_DEFN(NEG)
        CASE_DEFN(SIN)
        CASE_DEFN(COS)
        CASE_DEFN(TAN)
        CASE_DEFN(EXP)
        CASE_DEFN(LOG)
        CASE_DEFN(SQRT)
        CASE_DEFN(ROUND)
        CASE_DEFN(POW)
        CASE_DEFN(ADD)
        CASE_DEFN(SUB)
        CASE_DEFN(MUL)
        CASE_DEFN(DIV)
        CASE_DEFN(MIN)
        CASE_DEFN(MAX)
        CASE_DEFN(EQ)
        CASE_DEFN(NE)
        CASE_DEFN(LT)
        CASE_DEFN(GT)
        CASE_DEFN(RAND_BINO)
        CASE_DEFN(RAND_UNIF)
        CASE_DEFN(RAND_NORM)
    }
}

}

#endif
