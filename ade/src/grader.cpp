#include "ade/functor.hpp"

namespace ade
{

#ifdef ADE_GRADER_HPP

#define GRAD_SIGNATURE(CODE)template <>\
Tensorptr grader<CODE> (ArgsT& args, ArgsT& grads)

#define ZERO_GRAD(CODE)GRAD_SIGNATURE(CODE)\
{ Shape shape = map_shape(args[0].first, args[0].second->shape());\
return shaped_zero(shape); }

static void check_unary (const char* op, ArgsT& args)
{
	if (1 != args.size())
	{
		fatalf("cannot %s for non-single argument(s): using %d argument(s)",
			op, args.size());
	}
}

static void check_binary (const char* op, ArgsT& args)
{
	if (2 != args.size())
	{
		fatalf("cannot %s for non-binary argument(s): using %d argument(s)",
			op, args.size());
	}
}

static void check_nnary (const char* op, ArgsT& args)
{
	if (0 == args.size())
	{
		fatalf("cannot %s with arguments", op);
	}
}

GRAD_SIGNATURE(COPY)
{
	check_unary("COPY", args);
	return Functor::get(COPY, {grads.front()});
}

GRAD_SIGNATURE(ABS)
{
	check_unary("ABS", args);
	// abs'(f) = f * f' / abs(f)
	return Functor::get(DIV, {
		{identity, Functor::get(MUL, {args.front(), grads.front()})},
		{identity, Functor::get(ABS, args)},
	});
}

GRAD_SIGNATURE(NEG)
{
	check_unary("NEG", args);
	// neg'(f) = -f'
	return Functor::get(NEG, {grads.front()});
}

GRAD_SIGNATURE(SIN)
{
	check_unary("SIN", args);
	// sin'(f) = f'*cos(f)
	return Functor::get(MUL, {grads.front(),
		{identity, Functor::get(COS, args)},
	});
}

GRAD_SIGNATURE(COS)
{
	check_unary("COS", args);
	// cos'(f) = -f'*sin(f)
	return Functor::get(MUL, {
		{identity, Functor::get(NEG, {grads.front()})},
		{identity, Functor::get(SIN, args)},
	});
}

GRAD_SIGNATURE(TAN)
{
	check_unary("TAN", args);
	// tan'(f) = f'*sec^2(f)
	// 		= f'/cos^2(f)
	Tensorptr denom = Functor::get(COS, args);
	return Functor::get(DIV, {
		{identity, Functor::get(DIV, {grads.front(),
			{identity, denom},
		})},
		{identity, denom},
	});
}

GRAD_SIGNATURE(EXP)
{
	check_unary("EXP", args);
	// exp'(f) = f'*exp(f)
	return Functor::get(MUL, {grads.front(),
		{identity, Functor::get(EXP, args)},
	});
}

GRAD_SIGNATURE(LOG)
{
	check_unary("LOG", args);
	// log'(f) = f' / f
	return Functor::get(DIV, {grads.front(), args.front()});
}

GRAD_SIGNATURE(SQRT)
{
	check_unary("SQRT", args);
	// sqrt'(f) = f'/(2*sqrt(f))
	Tensorptr fwd = Functor::get(SQRT, args);
	return Functor::get(DIV, {grads.front(),
		{identity, Functor::get(ADD, {
			{identity, fwd}, {identity, fwd},
		})},
	});
}

GRAD_SIGNATURE(ROUND)
{
	check_unary("ROUND", args);
	// round'(f) = round(f')
	return Functor::get(ROUND, {grads.front()});
}

GRAD_SIGNATURE(POW)
{
	check_binary("POW", args);
	// pow'(f, g) = f' * g * pow(f, g - 1) + g' * pow(f, g) * log(f)
	//			= pow(f, g - 1) * (f' * g + g' * f * log(f))
	Tensorptr lhs = Functor::get(ADD, {
		{identity, Functor::get(MUL, {
			grads[0], args[1]
		})},
		{identity, Functor::get(MUL, {
			grads[1], args[0],
			{identity, Functor::get(LOG, {args[0]})},
		})},
	});
	return Functor::get(MUL, {
		{identity, Functor::get(POW, {args[0],
			{identity, Functor::get(SUB, {
				args[1], {identity, shaped_one(args[1].second->shape())},
			})},
		})},
		{identity, lhs},
	});
}

GRAD_SIGNATURE(ADD)
{
	check_nnary("ADD", args);
	// h'(f, g, ...) = f' + g' + ...
	return Functor::get(ADD, grads);
}

GRAD_SIGNATURE(SUB)
{
	check_binary("SUB", args);
	// h'(f, g) = f' - g'
	return Functor::get(SUB, grads);
}

GRAD_SIGNATURE(MUL)
{
	check_nnary("MUL", args);
	// h'(f, g) = f' * g * ... + f * g' * ... + ...
	ArgsT gargs;
	for (size_t i = 0, n = args.size(); i < n; ++i)
	{
		// f' * g * ...
		auto child = args[i];
		args[i] = grads[i];
		gargs.push_back({identity, Functor::get(MUL, args)});
		args[i] = child;
	}
	return Functor::get(ADD, gargs);
}

GRAD_SIGNATURE(DIV)
{
	check_binary("DIV", args);
	// h'(f, g) = (f' * g - g' * f) / g^2
	//			= f' / g - ((g' * f) / g) / g
	return Functor::get(SUB, {
		{identity, Functor::get(DIV, {grads[0], args[1]})},
		{identity, Functor::get(DIV, {
			{identity, Functor::get(DIV, {
				{identity, Functor::get(MUL, {grads[1], args[0]})},
				args[1],
			})},
			args[1],
		})},
	});
}

GRAD_SIGNATURE(MIN)
{
	check_nnary("MIN", args);
	ArgsT gargs;
	for (size_t i = 0, n = args.size(); i < n; ++i)
	{
		gargs.push_back({identity, Functor::get(MUL, {
			{identity, Functor::get(EQ, {
				{identity, Functor::get(MIN, args)}, args[i],
			})}, grads[i],
		})});
	}
	return Functor::get(ADD, gargs);
}

GRAD_SIGNATURE(MAX)
{
	check_nnary("MAX", args);
	ArgsT gargs;
	for (size_t i = 0, n = args.size(); i < n; ++i)
	{
		gargs.push_back({identity, Functor::get(MUL, {
			{identity, Functor::get(EQ, {
				{identity, Functor::get(MAX, args)}, args[i],
			})}, grads[i],
		})});
	}
	return Functor::get(ADD, gargs);
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

#define CALL_GRAD(OP)case OP: return grader<OP>(args, grads);

Tensorptr gradmap (OPCODE op, ArgsT args, ArgsT& grads)
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
	return Tensorptr(nullptr);
}

}

#endif
